"""
Evaluate LVB-style MCQ (schema-specific to provided JSONL):

Three accuracy lenses:
  1) lmms_eval_accuracy:
       Compare model letter from lvb_acc.parsed_pred to GT letter (from doc.correct_choice or lvb_acc.answer).
  2) answer_parsed_accuracy:
       LLM-parse the <answer>...</answer> block into a letter and compare to GT.
  3) thinking_parsed_accuracy:
       LLM-extract a letter from <think>...</think> given question + options and compare to GT.

Input schema (per JSONL line):
{
  "doc": {
    "video_id": "...",
    "id": "86CxyhFV9MI_0",
    "video_path": "...mp4",
    "subtitle_path": "...json",
    "correct_choice": 1,                    # 0-based index of correct option (A=0, B=1, ...)
    "question": "In the video, which ...?",
    "option0": "text A", "option1": "text B", "option2": "text C", "option3": "text D", "option4": "N/A",
    "duration": 190.16,
    "duration_group": 600,
    "question_category": "TOS",
    "topic_category": "NP-News-Programs",
    "starting_timestamp_for_subtitles": 0.0
  },
  "lvb_acc": {"id":"86CxyhFV9MI_0", "duration_group":600, "question_category":"TOS", "answer":"B", "parsed_pred":"C"},
  "submission": {"86CxyhFV9MI_0":"C. some option text"},       # optional
  "filtered_resps": "<think>...</think>\n<answer>\nC. ...\n</answer>",  # or nested under "resps"
  ...
}

Notes
-----
- Deterministic LLM parsing (temperature=0, top_p=1.0, top_k=-1).
- Batch size: -1 => one vLLM call for all prompts; else chunked by --batch_size.
- Only MCQ letters evaluated here (A-D; ignore option4 if it's "N/A").
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from _utils import *


# -----------------------------
# Regex helpers
# -----------------------------
RE_THINK   = re.compile(r"<think>(.*?)</think>",   re.DOTALL | re.IGNORECASE)
RE_ANSWER  = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
RE_LETTER  = re.compile(r"\b([A-Z])\s*[\.\)|:,-]?\b")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                rows.append(json.loads(s.rstrip(",")))
    return rows

def extract_last_block(text: str, pattern: re.Pattern) -> Optional[str]:
    if not text:
        return None
    matches = list(pattern.finditer(text))
    return matches[-1].group(1).strip() if matches else None

def extract_letter(text: str, num_options: Optional[int] = None) -> Optional[str]:
    if not text:
        return None
    # Prefer leading "(C) ..." or "C. ..." or "C) ..."
    m0 = re.match(r"^\s*\(?\s*([A-Z])\s*[\.\)|:,-]\s*", text)
    if m0:
        L = m0.group(1).upper()
        if (num_options is None) or (L in [chr(ord('A') + i) for i in range(num_options)]):
            return L
    letters = [m.group(1).upper() for m in RE_LETTER.finditer(text)]
    if not letters:
        return None
    if num_options:
        valid = [chr(ord('A') + i) for i in range(num_options)]
        letters = [L for L in letters if L in valid] or letters
    return letters[-1]


def batch_apply_chat_template(tokenizer, list_of_messages: List[List[Dict[str, str]]]) -> List[str]:
    prompts = []
    for messages in list_of_messages:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompts.append(prompt)
    return prompts


def run_llm(
    llm: LLM,
    tokenizer,
    base_msgs_builder,
    questions: List[str],
    options_list: List[List[str]],
    payloads: List[str],
    batch_size: int,
) -> List[str]:
    msgs_all = []
    for q, opts, payload in zip(questions, options_list, payloads):
        context_text = payload.strip() if payload else ""
        msgs = base_msgs_builder(opts, context_text)
        msgs_all.append(msgs)

    prompts = batch_apply_chat_template(tokenizer, msgs_all)
    sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=64)

    outputs: List[str] = []
    def _consume(batch_prompts: List[str]):
        if not batch_prompts:
            return
        gens = llm.generate(batch_prompts, sp)
        for g in gens:
            text = g.outputs[0].text.strip() if g.outputs and g.outputs[0].text is not None else ""
            outputs.append(text)

    if batch_size == -1:
        pbar = tqdm(total=len(prompts), desc="LLM calls (single)")
        _consume(prompts); pbar.update(len(prompts)); pbar.close()
    else:
        for i in tqdm(range(0, len(prompts), batch_size), desc="LLM calls"):
            _consume(prompts[i:i+batch_size])
    return outputs

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate LVB-style MCQ via LLM parsing of <think>/<answer>")
    parser.add_argument("--sample_jsonl", type=str, required=True)
    parser.add_argument("--output_dir",  type=str, required=True)
    parser.add_argument("--inference_model", type=str, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_jsonl(args.sample_jsonl)

    print(f"Loading model for parsing/extraction: {args.inference_model} (tp={args.tp}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.inference_model, trust_remote_code=True)
    llm = LLM(model=args.inference_model, tensor_parallel_size=args.tp, dtype="auto")

    per_samples: List[Dict[str, Any]] = []
    n_total = 0
    lmms_correct = answer_correct = thinking_correct = 0
    combo_counts = Counter()

    questions: List[str] = []
    options_list: List[List[str]] = []
    think_payloads: List[str] = []
    answer_payloads: List[str] = []
    temp_slots: List[Dict[str, Any]] = []

    for rec in tqdm(rows, desc="Processing samples"):
        n_total += 1
        doc = rec.get("doc", {}) or {}
        question = doc.get("question") or rec.get("input") or ""

        # Build options list from option0..option4 (skip "N/A")
        raw_opts = []
        for k in ["option0", "option1", "option2", "option3", "option4"]:
            if k in doc and doc[k] not in (None, ""):
                txt = str(doc[k])
                if txt.strip().upper() == "N/A":
                    continue
                raw_opts.append(txt.strip())
        options = raw_opts

        # Ground-truth letter: prefer doc.correct_choice (0-based index), else lvb_acc.answer
        gt_letter = ""
        cc = doc.get("correct_choice")
        if isinstance(cc, int):
            if 0 <= cc < len(options):
                gt_letter = chr(ord('A') + cc)
        elif isinstance(cc, str) and cc.isdigit():
            idx = int(cc)
            if 0 <= idx < len(options):
                gt_letter = chr(ord('A') + idx)
        if not gt_letter:
            gt_letter = ((rec.get("lvb_acc") or {}).get("answer") or "").strip().upper()

        # Raw response: prefer filtered_resps string; else nested in resps
        raw = ""
        fr = rec.get("filtered_resps")
        rs = rec.get("resps")
        if isinstance(fr, str):
            raw = fr
        elif isinstance(fr, list) and fr:
            if isinstance(fr[0], list) and fr[0]:
                raw = fr[0][0]
            elif isinstance(fr[0], str):
                raw = fr[0]
        elif isinstance(rs, list) and rs:
            if isinstance(rs[0], list) and rs[0]:
                raw = rs[0][0]
            elif isinstance(rs[0], str):
                raw = rs[0]
        elif isinstance(rs, str):
            raw = rs
        else:
            raw = ""

        think_txt = extract_last_block(raw, RE_THINK)  or ""
        ans_txt   = extract_last_block(raw, RE_ANSWER) or ""

        # Model letter for lmms_eval from lvb_acc.parsed_pred (already a letter)
        lvb = rec.get("lvb_acc") or {}
        parsed_pred_letter = (lvb.get("parsed_pred") or "").strip().upper() or None
        lmms_is_correct = (parsed_pred_letter == gt_letter) if parsed_pred_letter and gt_letter else False

        # Collect for batched LLM
        questions.append(question)
        options_list.append(options)
        think_payloads.append(think_txt)
        answer_payloads.append(ans_txt)

        temp_slots.append({
            "rec": rec,
            "question": question,
            "options": options,
            "gt_letter": gt_letter,
            "raw": raw,
            "think_txt": think_txt,
            "ans_txt": ans_txt,
            "model_letter_pred": parsed_pred_letter,
            "lmms_is_correct": lmms_is_correct,
            "meta": {
                "video_id": doc.get("video_id"),
                "clip_id": doc.get("id"),
                "video_path": doc.get("video_path"),
                "subtitle_path": doc.get("subtitle_path"),
                "question_category": doc.get("question_category"),
                "topic_category": doc.get("topic_category"),
                "duration": doc.get("duration"),
                "duration_group": doc.get("duration_group"),
                "start_ts_for_subs": doc.get("starting_timestamp_for_subtitles"),
            },
        })

    # Batched LLM calls
    think_outs = run_llm(
        llm, tokenizer, build_thinking_extraction_message,
        questions, options_list, think_payloads,
        batch_size=args.batch_size,
    )
    answer_outs = run_llm(
        llm, tokenizer, build_answer_parsing_message,
        questions, options_list, answer_payloads,
        batch_size=args.batch_size,
    )

    # Tally & outputs
    for i, slot in enumerate(temp_slots):
        gt_letter = slot["gt_letter"]
        nopts = len(slot["options"])

        t_letter = extract_letter(think_outs[i] if i < len(think_outs) else "", num_options=nopts)
        a_letter = extract_letter(answer_outs[i] if i < len(answer_outs) else "", num_options=nopts)

        thinking_is_correct = (t_letter == gt_letter) if t_letter and gt_letter else False
        answer_is_correct   = (a_letter == gt_letter) if a_letter and gt_letter else False
        lmms_is_correct     = slot["lmms_is_correct"]

        lmms_correct    += int(lmms_is_correct)
        answer_correct  += int(answer_is_correct)
        thinking_correct+= int(thinking_is_correct)
        combo_counts[(int(lmms_is_correct), int(answer_is_correct), int(thinking_is_correct))] += 1

        per_samples.append({
            "doc_id": slot["rec"].get("doc_id"),
            "question": slot["question"],
            "options": slot["options"],
            "ground_truth_letter": gt_letter,
            "model_letter_pred": slot["model_letter_pred"],
            "thinking_parsed_pred": t_letter,
            "answer_parsed_pred": a_letter,
            "lmms_eval_correct": lmms_is_correct,
            "thinking_parsed_correct": thinking_is_correct,
            "answer_parsed_correct": answer_is_correct,
            "raw_response": slot["raw"],
            "thinking_text": slot["think_txt"],
            "final_answer_text": slot["ans_txt"],
            "meta": slot["meta"],
        })

    # Agreement metric (answer vs thinking both match or both mismatch)
    answer_thinking_same = (
        combo_counts.get((0, 0, 0), 0) + combo_counts.get((1, 0, 0), 0) +
        combo_counts.get((0, 1, 1), 0) + combo_counts.get((1, 1, 1), 0)
    )
    answer_to_thinking_correlation = (answer_thinking_same / n_total) if n_total else 0.0

    summary = {
        "n_total": n_total,
        "lmms_eval_accuracy": (lmms_correct / n_total) if n_total else 0.0,
        "answer_parsed_accuracy": (answer_correct / n_total) if n_total else 0.0,
        "thinking_parsed_accuracy": (thinking_correct / n_total) if n_total else 0.0,
        "answer_to_thinking_correlation": answer_to_thinking_correlation,
        "lmms_eval_correct": lmms_correct,
        "answer_parsed_correct": answer_correct,
        "thinking_parsed_correct": thinking_correct,
        "combinations": {f"lmms{a}_ans{b}_think{c}": cnt for (a, b, c), cnt in sorted(combo_counts.items())},
        "discrepancies": {
            "answer_correct_thinking_wrong": combo_counts.get((1, 1, 0), 0) + combo_counts.get((0, 1, 0), 0),
            "answer_wrong_thinking_correct": combo_counts.get((1, 0, 1), 0) + combo_counts.get((0, 0, 1), 0),
            "lmms_vs_answer_mismatch": sum(cnt for (a, b, c), cnt in combo_counts.items() if a != b),
        },
    }

    # Write outputs
    os.makedirs(args.output_dir, exist_ok=True)
    per_sample_path = os.path.join(args.output_dir, "per_sample.json")
    summary_path    = os.path.join(args.output_dir, "summary.json")
    with open(per_sample_path, "w", encoding="utf-8") as f:
        json.dump(per_samples, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved per-sample to {per_sample_path}")
    print(f"Saved summary to {summary_path}")
    print(f"answer_to_thinking_correlation: {answer_to_thinking_correlation}")

if __name__ == "__main__":
    main()
