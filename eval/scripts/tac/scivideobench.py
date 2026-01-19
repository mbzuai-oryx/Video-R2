"""
Evaluate SciVideoBench-style video QA with three lenses:
  1) lmms_eval_accuracy: compare model letter from scivideobench_acc.pred_answer to GT letter (doc.answer).
  2) answer_parsed_accuracy: LLM-parse <answer>...</answer> and compare to GT.
  3) thinking_parsed_accuracy: LLM-extract from <think>...</think> given question+options, compare to GT.

Inputs (per JSONL line):
{
  "doc": {
    "question": "...",
    "options": {"A":"...", "B":"...", ...},
    "answer": "A",
    "question_type": "...",
    "discipline": "...",
    "subject": "...",
    "category": "..."
  },
  "scivideobench_acc": {
    "pred_answer": "E",
    "raw_output": "<think>...</think><answer>E</answer>"
  },
  "filtered_resps": "...",  # optional; if absent, fall back to scivideobench_acc.raw_output
  ...
}

Notes
-----
- Deterministic generation for parsing/extraction (temperature=0, top_p=1.0, top_k=-1).
- Batch size: -1 => single vLLM call for all prompts; else chunked by --batch_size.
- Letters validated against available options A..Z only.
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

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
RE_NUMBER = re.compile(r"[-+]?\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\$?\s*\d+(?:\.\d+)?")

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

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[,$%€£₹]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

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
    m0 = re.match(r"^\s*([A-Z])\s*[\.\)|:,-]", text)
    if m0:
        L = m0.group(1)
        if (num_options is None) or (L in [chr(ord('A') + i) for i in range(num_options)]):
            return L
    letters = [m.group(1) for m in RE_LETTER.finditer(text)]
    if not letters:
        return None
    if num_options:
        valid = [chr(ord('A') + i) for i in range(num_options)]
        letters = [L for L in letters if L in valid] or letters
    return letters[-1]

def parse_number(text: str) -> Optional[float]:
    if text is None:
        return None
    m = RE_NUMBER.search(text)
    if not m:
        return None
    token = m.group(0).replace("$", "").replace(",", "").strip()
    try:
        return float(token)
    except Exception:
        return None

def eq_numeric(a: Union[str, float, int], b: Union[str, float, int], places: int = 2) -> bool:
    try:
        fa = float(a)
    except Exception:
        fa = parse_number(str(a))
    try:
        fb = float(b)
    except Exception:
        fb = parse_number(str(b))
    if fa is None or fb is None:
        return False
    return round(fa, places) == round(fb, places)

def text_match(pred: str, gt: str) -> bool:
    if pred is None or gt is None:
        return False
    p, g = normalize_text(str(pred)), normalize_text(str(gt))
    if not p or not g:
        return False
    return p == g or (p in g) or (g in p)

def compare_mcq(pred_letter: Optional[str], gt_letter: Optional[str]) -> bool:
    if not pred_letter or not gt_letter:
        return False
    return pred_letter.strip().upper() == gt_letter.strip().upper()


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
    build_msgs_fn,
    questions: List[str],
    options_list: List[List[str]],
    payloads: List[str],
    batch_size: int,
) -> List[str]:
    msgs = [build_msgs_fn(opts, payload) for q, opts, payload in zip(questions, options_list, payloads)]
    prompts = batch_apply_chat_template(tokenizer, msgs)
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
        pbar = tqdm(total=len(prompts), desc="LLM call (single)")
        _consume(prompts); pbar.update(len(prompts)); pbar.close()
    else:
        for i in tqdm(range(0, len(prompts), batch_size), desc="LLM calls"):
            _consume(prompts[i : i + batch_size])
    return outputs

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate SciVideoBench-style Video QA (MCQ letters via scivideobench_acc)")
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

    questions: List[str] = []
    options_list: List[List[str]] = []
    thinking_texts: List[str] = []
    answer_texts: List[str] = []
    temp_slots: List[Dict[str, Any]] = []

    for rec in tqdm(records, desc="Processing samples"):
        n_total += 1
        doc = rec.get("doc", {}) or {}
        question = doc.get("question") or ""
        options_map = doc.get("options") or {}
        # Keep A..Z only and in order
        letters_sorted = [L for L in sorted(options_map.keys()) if isinstance(L, str) and len(L) == 1 and "A" <= L <= "Z"]
        options = [options_map[L] for L in letters_sorted]

        gt_letter = (doc.get("answer") or rec.get("target") or "").strip().upper()

        # Raw response: prefer filtered_resps string, else scivideobench_acc.raw_output
        raw_response = ""
        fr = rec.get("filtered_resps")
        if isinstance(fr, str):
            raw_response = fr
        elif isinstance(fr, list) and fr:
            # handle legacy nested list if ever present
            raw_response = fr[0][0] if isinstance(fr[0], list) and fr[0] else (fr[0] if isinstance(fr[0], str) else "")
        elif isinstance(rec.get("resps"), str):
            raw_response = rec["resps"]
        elif isinstance(rec.get("scivideobench_acc", {}).get("raw_output"), str):
            raw_response = rec["scivideobench_acc"]["raw_output"]
        else:
            raw_response = ""

        final_answer_text = extract_last_block(raw_response, RE_ANSWER) or ""
        thinking_text = extract_last_block(raw_response, RE_THINK) or ""

        # Parsed pred for lmms_eval from scivideobench_acc.pred_answer
        parsed_pred_letter = (rec.get("scivideobench_acc") or {}).get("pred_answer")
        parsed_pred_letter = (parsed_pred_letter or "").strip().upper() or None

        lmms_is_correct = compare_mcq(parsed_pred_letter, gt_letter)

        questions.append(question)
        options_list.append(options)
        thinking_texts.append(thinking_text)
        answer_texts.append(final_answer_text)

        temp_slots.append({
            "rec": rec,
            "question": question,
            "options": options,
            "options_map": options_map,
            "gt_letter": gt_letter,
            "raw_response": raw_response,
            "final_answer_text": final_answer_text,
            "thinking_text": thinking_text,
            "parsed_pred_letter": parsed_pred_letter,
            "lmms_is_correct": lmms_is_correct,
            "meta": {
                "question_type": doc.get("question_type"),
                "discipline": doc.get("discipline"),
                "subject": doc.get("subject"),
                "category": doc.get("category"),
            },
        })

    # Batched LLM calls
    extracted_letters = run_llm(
        llm, tokenizer, build_thinking_extraction_message,
        questions, options_list, thinking_texts,
        batch_size=args.batch_size,
    )
    parsed_answer_letters = run_llm(
        llm, tokenizer, build_answer_parsing_message,
        questions, options_list, answer_texts,
        batch_size=args.batch_size,
    )

    # Tally per-sample & metrics
    for i, slot in enumerate(temp_slots):
        gt_letter = slot["gt_letter"]

        think_out = extracted_letters[i] if i < len(extracted_letters) else ""
        think_letter = extract_letter(think_out, num_options=len(slot["options"]))
        thinking_is_correct = compare_mcq(think_letter, gt_letter)

        ans_out = parsed_answer_letters[i] if i < len(parsed_answer_letters) else ""
        ans_letter = extract_letter(ans_out, num_options=len(slot["options"]))
        answer_is_correct = compare_mcq(ans_letter, gt_letter)

        lmms_is_correct = slot["lmms_is_correct"]

        lmms_correct += int(lmms_is_correct)
        answer_correct += int(answer_is_correct)
        thinking_correct += int(thinking_is_correct)
        combo_counts[(int(lmms_is_correct), int(answer_is_correct), int(thinking_is_correct))] += 1

        per_samples.append({
            "doc_id": slot["rec"].get("doc_id"),
            "question": slot["question"],
            "options": slot["options"],
            "options_map": slot["options_map"],
            "ground_truth_letter": gt_letter,
            "raw_response": slot["raw_response"],
            "final_answer_text": slot["final_answer_text"],
            "thinking_text": slot["thinking_text"],
            "model_letter_pred": slot["parsed_pred_letter"],
            "answer_parsed_pred": ans_letter,
            "thinking_parsed_pred": think_letter,
            "lmms_eval_correct": lmms_is_correct,
            "answer_parsed_correct": answer_is_correct,
            "thinking_parsed_correct": thinking_is_correct,
            # Useful metadata (kept lightweight)
            "meta": slot["meta"],
        })

    # Agreement metric (answer vs thinking both match/mismatch)
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
