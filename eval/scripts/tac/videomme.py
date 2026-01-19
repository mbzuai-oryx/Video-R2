"""
Evaluate VideoMME-style MCQ with three lenses (schema-specific):
  1) lmms_eval_accuracy: compare videomme_perception_score.pred_answer (letter) to GT letter (doc.answer / target).
  2) answer_parsed_accuracy: LLM-parse <answer>...</answer> into a letter and compare to GT.
  3) thinking_parsed_accuracy: LLM-extract a letter from <think>...</think> given question+options and compare to GT.

Expected JSONL schema per line (example):
{
  "doc_id": 4,
  "doc": {
    "video_id": "002",
    "duration": "short",
    "domain": "Knowledge",
    "sub_category": "Humanity & History",
    "url": "https://www.youtube.com/watch?v=N1cdUjctpG8",
    "videoID": "N1cdUjctpG8",
    "question_id": "002-2",
    "task_type": "Action Reasoning",
    "question": "...",
    "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "answer": "D"
  },
  "target": "D",
  "resps": [["<think>...</think>\n<answer>A</answer>"]],
  "filtered_resps": ["<think>...</think>\n<answer>A</answer>"],
  "videomme_perception_score": {"pred_answer": "A", "answer": "D", ...}
}

Notes
-----
- Deterministic LLM parsing (temperature=0, top_p=1.0, top_k=-1).
- Batch size: -1 => one vLLM call for all prompts; else chunk by --batch_size.
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
RE_OPTLINE = re.compile(r"^\s*\(?\s*([A-Z])\s*[\.\):-]\s*(.*)$", re.IGNORECASE)

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

def parse_options(option_lines: List[str]) -> List[str]:
    """Strip leading letters from ['A. foo', 'B. bar', ...] -> ['foo','bar',...]; keep order."""
    out = []
    for s in option_lines:
        s = str(s)
        m = RE_OPTLINE.match(s)
        out.append(m.group(2).strip() if m else s.strip())
    return out

def letters_from_options(option_lines: List[str]) -> List[str]:
    """Extract letters ['A','B',...] from option lines; fallback to sequential if missing."""
    letters = []
    nxt = 0
    for s in option_lines:
        m = RE_OPTLINE.match(str(s))
        letters.append(m.group(1).upper() if m else chr(ord('A') + nxt))
        nxt += 1
    return letters


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
    parser = argparse.ArgumentParser(description="Evaluate VideoMME-style MCQ via LLM parsing of <think>/<answer>")
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
        option_lines = list(doc.get("options") or [])
        options = parse_options(option_lines)
        opt_letters = letters_from_options(option_lines)

        # GT letter: prefer doc.answer, fallback to top-level "target"
        gt_letter = (doc.get("answer") or rec.get("target") or "").strip().upper()

        # Raw response: prefer filtered_resps string; else nested in resps
        raw = ""
        fr = rec.get("filtered_resps")
        rs = rec.get("resps")
        if isinstance(fr, str):
            raw = fr
        elif isinstance(fr, list) and fr:
            raw = fr[0][0] if isinstance(fr[0], list) and fr[0] else (fr[0] if isinstance(fr[0], str) else "")
        elif isinstance(rs, list) and rs:
            raw = rs[0][0] if isinstance(rs[0], list) and rs[0] else (rs[0] if isinstance(rs[0], str) else "")
        elif isinstance(rs, str):
            raw = rs
        else:
            raw = ""

        think_txt = extract_last_block(raw, RE_THINK)  or ""
        ans_txt   = extract_last_block(raw, RE_ANSWER) or ""

        # Model's letter for lmms_eval from videomme_perception_score.pred_answer (fallback to minerva if present)
        mps = rec.get("videomme_perception_score") or rec.get("minerva_perception_score") or {}
        parsed_pred_letter = (mps.get("pred_answer") or "").strip().upper() or None
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
            "opt_letters": opt_letters,
            "gt_letter": gt_letter,
            "raw": raw,
            "think_txt": think_txt,
            "ans_txt": ans_txt,
            "model_letter_pred": parsed_pred_letter,
            "lmms_is_correct": lmms_is_correct,
            "meta": {
                "doc_id": rec.get("doc_id"),
                "video_id": doc.get("video_id") or doc.get("videoID"),
                "videoID": doc.get("videoID"),
                "url": doc.get("url"),
                "question_id": doc.get("question_id"),
                "domain": doc.get("domain"),
                "category": doc.get("category"),           # if present
                "sub_category": doc.get("sub_category"),
                "task_type": doc.get("task_type"),
                "length": doc.get("length"),
                "duration": doc.get("duration"),
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
            "doc_id": slot["meta"]["doc_id"],
            "question": slot["question"],
            "options": slot["options"],
            "option_letters": slot["opt_letters"],
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

    # Agreement metric (answer vs thinking both match or both mismatch relative to GT)
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
