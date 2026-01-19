"""
Evaluate TempCompass-style MCQ with three lenses:
  1) lmms_eval_accuracy: compare tempcompass_perception_score.pred_answer (letter) to GT letter (doc.answer).
  2) answer_parsed_accuracy: LLM-parse <answer>...</answer> into a letter and compare to GT.
  3) thinking_parsed_accuracy: LLM-extract a letter from <think>...</think> given question+options and compare to GT.

Expected JSONL schema per line:
{
  "doc": {
    "uuid": "...",
    "video_id": "...",
    "question": "...",
    "options": ["A. ...", "B. ...", ...],
    "answer": "B"
  },
  "tempcompass_perception_score": {"uuid":"...", "pred_answer":"B", "answer":"B"},
  "filtered_resps": "<think>...</think>\n<answer>\nB\n</answer>",  # or nested in "resps"
  ...
}

Notes
-----
- Deterministic LLM parsing (temperature=0, top_p=1.0, top_k=-1).
- Batch size: -1 => one vLLM call for all prompts; else chunk by --batch_size.
- Only MCQ letters are evaluated here.
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
RE_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
RE_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
RE_LETTER = re.compile(r"\b([A-Z])\s*[\.|\)|:|-]?\b")
RE_OPT_LINE = re.compile(r"^\s*([A-Z])\s*[.\):-]\s*(.*)$", re.IGNORECASE)

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                data.append(json.loads(line.rstrip(",")))
    return data

def extract_last_block(text: str, pattern: re.Pattern) -> Optional[str]:
    if not text:
        return None
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()

def extract_letter(text: str, num_options: Optional[int] = None) -> Optional[str]:
    if not text:
        return None
    # Prefer a leading option label like "B. ..." at the start
    m0 = re.match(r"^\s*([A-Z])\s*[\.\)|:,-]", text)
    if m0:
        L = m0.group(1).upper()
        if (num_options is None) or (L in [chr(ord('A') + i) for i in range(num_options)]):
            return L
    # Fallback: scan anywhere
    letters = [m.group(1).upper() for m in RE_LETTER.finditer(text)]
    if not letters:
        return None
    if num_options:
        valid = [chr(ord('A') + i) for i in range(num_options)]
        letters = [L for L in letters if L in valid] or letters
    return letters[-1]

def parse_options_list(option_lines: List[str]) -> List[str]:
    """
    From ["A. dribbling", "B. dunking"] -> ["dribbling", "dunking"], preserving order.
    Non-matching lines are kept as-is.
    """
    out = []
    for s in option_lines:
        if not isinstance(s, str):
            out.append(str(s))
            continue
        m = RE_OPT_LINE.match(s)
        out.append(m.group(2).strip() if m else s.strip())
    return out

def letters_from_options(option_lines: List[str]) -> List[str]:
    """
    Extract letters in order from ["A. ...", "B. ..."] -> ["A","B"].
    If a line lacks a letter, assign incrementally from A..Z.
    """
    letters = []
    nxt = 0
    for s in option_lines:
        m = RE_OPT_LINE.match(s) if isinstance(s, str) else None
        if m:
            letters.append(m.group(1).upper())
        else:
            letters.append(chr(ord('A') + nxt))
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
    base_msgs_builder,               # function(question, options) -> messages with user prefix
    questions: List[str],
    options_list: List[List[str]],
    payloads: List[str],             # appended to the last user message content
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
            _consume(prompts[i : i + batch_size])
    return outputs

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate TempCompass-style MCQ with LLM parsing of <think>/<answer>")
    parser.add_argument("--sample_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--inference_model", type=str, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records = load_jsonl(args.sample_jsonl)

    print(f"Loading model for parsing/extraction: {args.inference_model} (tp={args.tp}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.inference_model, trust_remote_code=True)
    llm = LLM(model=args.inference_model, tensor_parallel_size=args.tp, dtype="auto")

    per_samples: List[Dict[str, Any]] = []
    n_total = 0
    lmms_correct = 0
    answer_correct = 0
    thinking_correct = 0
    combo_counts = Counter()

    # For batched LLM
    questions: List[str] = []
    options_list: List[List[str]] = []
    think_payloads: List[str] = []
    answer_payloads: List[str] = []
    temp_slots: List[Dict[str, Any]] = []

    for rec in tqdm(records, desc="Processing samples"):
        n_total += 1
        doc = rec.get("doc", {}) or {}
        question = doc.get("question") or rec.get("input") or ""
        option_lines = doc.get("options") or []
        options = parse_options_list(option_lines)            # pure texts in original order
        opt_letters = letters_from_options(option_lines)      # ["A","B",...]

        gt_letter = (doc.get("answer") or rec.get("target") or "").strip().upper()

        # Raw response: prefer filtered_resps string; else look under resps (nested list) if present
        raw_response = ""
        fr = rec.get("filtered_resps")
        rs = rec.get("resps")
        if isinstance(fr, str):
            raw_response = fr
        elif isinstance(fr, list) and fr:
            if isinstance(fr[0], list) and fr[0]:
                raw_response = fr[0][0]
            elif isinstance(fr[0], str):
                raw_response = fr[0]
        elif isinstance(rs, list) and rs:
            if isinstance(rs[0], list) and rs[0]:
                raw_response = rs[0][0]
            elif isinstance(rs[0], str):
                raw_response = rs[0]
        elif isinstance(rs, str):
            raw_response = rs
        else:
            raw_response = ""

        think_txt = extract_last_block(raw_response, RE_THINK) or ""
        ans_txt   = extract_last_block(raw_response, RE_ANSWER) or ""

        # Model's letter (for lmms_eval) comes from tempcompass_perception_score.pred_answer
        parsed_pred_letter = ((rec.get("tempcompass_perception_score") or {}).get("pred_answer") or "").strip().upper() or None
        lmms_is_correct = (parsed_pred_letter == gt_letter) if parsed_pred_letter else False

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
            "raw_response": raw_response,
            "thinking_text": think_txt,
            "final_answer_text": ans_txt,
            "model_letter_pred": parsed_pred_letter,
            "lmms_is_correct": lmms_is_correct,
            "meta": {
                "uuid": doc.get("uuid"),
                "video_id": doc.get("video_id"),
            },
        })

    # Batched LLM calls
    extracted_from_think = run_llm(
        llm, tokenizer, build_thinking_extraction_message,
        questions, options_list, think_payloads,
        batch_size=args.batch_size,
    )
    parsed_from_answer = run_llm(
        llm, tokenizer, build_answer_parsing_message,
        questions, options_list, answer_payloads,
        batch_size=args.batch_size,
    )

    # Tally & outputs
    for i, slot in enumerate(temp_slots):
        gt_letter = slot["gt_letter"]
        num_opts = len(slot["options"])

        t_out = extracted_from_think[i] if i < len(extracted_from_think) else ""
        a_out = parsed_from_answer[i] if i < len(parsed_from_answer) else ""

        t_letter = extract_letter(t_out, num_options=num_opts)
        a_letter = extract_letter(a_out, num_options=num_opts)

        thinking_is_correct = (t_letter == gt_letter) if t_letter else False
        answer_is_correct   = (a_letter == gt_letter) if a_letter else False
        lmms_is_correct     = slot["lmms_is_correct"]

        lmms_correct    += int(lmms_is_correct)
        answer_correct  += int(answer_is_correct)
        thinking_correct+= int(thinking_is_correct)
        combo_counts[(int(lmms_is_correct), int(answer_is_correct), int(thinking_is_correct))] += 1

        per_samples.append({
            "doc_id": slot["rec"].get("doc_id"),
            "question": slot["question"],
            "options": slot["options"],                 # texts in order
            "option_letters": slot["opt_letters"],      # e.g., ["A","B"]
            "ground_truth_letter": gt_letter,
            "model_letter_pred": slot["model_letter_pred"],
            "thinking_parsed_pred": t_letter,
            "answer_parsed_pred": a_letter,
            "lmms_eval_correct": lmms_is_correct,
            "thinking_parsed_correct": thinking_is_correct,
            "answer_parsed_correct": answer_is_correct,
            "raw_response": slot["raw_response"],
            "thinking_text": slot["thinking_text"],
            "final_answer_text": slot["final_answer_text"],
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
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(per_sample_path, "w", encoding="utf-8") as f:
        json.dump(per_samples, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved per-sample to {per_sample_path}")
    print(f"Saved summary to {summary_path}")
    print(f"answer_to_thinking_correlation: {answer_to_thinking_correlation}")

if __name__ == "__main__":
    main()
