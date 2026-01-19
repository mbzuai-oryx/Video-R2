"""
Evaluate MVBench-style MCQ with three lenses, now reading multiple JSONL shards
and saving the exact LLM prompts per sample for both thinking and answer parsing.

Inputs
------
--input_dir: a directory to scan (recursively). We collect all files that:
  - start with: "mvbench_think"
  - contain: "samples_mvbench"
  - end with: ".jsonl"
These are typically 20 files; we concatenate them (sorted by path) and treat the
combined rows as the dataset.

Lenses
------
  1) lmms_eval_accuracy: compare mvbench_accuracy.pred_answer (letter) to GT.
  2) answer_parsed_accuracy: LLM-parse <answer>...</answer> to a letter vs GT.
  3) thinking_parsed_accuracy: LLM-extract a letter from <think>...</think> with question+options vs GT.

Per-sample outputs additionally include:
  - thinking_prompt: the exact prompt string used for thinking extraction
  - answer_prompt:   the exact prompt string used for answer parsing

Notes
-----
- Deterministic LLM parsing (temperature=0, top_p=1.0, top_k=-1).
- Batch size: -1 => single vLLM call for all prompts; else chunk by --batch_size.
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

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

def collect_rows_from_dir(input_dir: str) -> List[Dict[str, Any]]:
    """Recursively scan input_dir for JSONL files that match the naming rule,
    concatenate their rows, and return the combined list (stable, sorted)."""
    matched_paths: List[str] = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            fn_lower = fn.lower()
            if (
                fn_lower.startswith("mvbench_think") and
                "samples_mvbench" in fn_lower and
                fn_lower.endswith(".jsonl")
            ):
                matched_paths.append(os.path.join(root, fn))
    matched_paths.sort()  # stable order

    if not matched_paths:
        raise FileNotFoundError(
            f"No matching JSONL files found in '{input_dir}'. "
            "Expected files that start with 'mvbench_think', contain 'samples_mvbench', and end with '.jsonl'."
        )

    print(f"Found {len(matched_paths)} matching JSONL files:")
    for p in matched_paths:
        print(f"  - {p}")

    all_rows: List[Dict[str, Any]] = []
    for p in matched_paths:
        rs = load_jsonl(p)
        all_rows.extend(rs)

    print(f"Total concatenated rows: {len(all_rows)}")
    return all_rows

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_last_block(text: str, pattern: re.Pattern) -> Optional[str]:
    if not text:
        return None
    matches = list(pattern.finditer(text))
    return matches[-1].group(1).strip() if matches else None

def extract_letter(text: str, num_options: Optional[int] = None) -> Optional[str]:
    if not text:
        return None
    # Prefer a leading label like "(B) ...", "B. ...", "B) ..."
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

def text_to_letter_from_candidates(answer_text: str, candidates: List[str]) -> Optional[str]:
    """Map a gold/free-text answer to its letter by fuzzy-normalized equality/substring."""
    if answer_text is None:
        return None
    g = normalize_text(answer_text)
    # exact normalized match first
    for i, c in enumerate(candidates):
        if normalize_text(str(c)) == g:
            return chr(ord('A') + i)
    # lenient containment either way
    for i, c in enumerate(candidates):
        cn = normalize_text(str(c))
        if g in cn or cn in g:
            return chr(ord('A') + i)
    return None


def batch_apply_chat_template(tokenizer, list_of_messages: List[List[Dict[str, str]]]) -> List[str]:
    prompts = []
    for messages in list_of_messages:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompts.append(prompt)
    return prompts

def build_prompts_and_outputs(
    llm: LLM,
    tokenizer,
    base_msgs_builder,
    questions: List[str],
    options_list: List[List[str]],
    payloads: List[str],
    batch_size: int,
) -> Tuple[List[str], List[str]]:
    """
    Returns:
      outputs:  model generations (one per sample)
      prompts:  the exact rendered prompts sent to the model (one per sample)
    """
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

    return outputs, prompts

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate MVBench-style MCQ via LLM parsing of <think>/<answer>")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing mvbench_think* samples_mvbench*.jsonl shards")
    parser.add_argument("--output_dir",  type=str, required=True)
    parser.add_argument("--inference_model", type=str, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Gather + concatenate shards
    print(f"Scanning directory: {args.input_dir}")
    rows = collect_rows_from_dir(args.input_dir)

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
        question   = doc.get("question") or rec.get("input") or ""
        candidates = list(doc.get("candidates") or [])
        options    = [str(x) for x in candidates]  # already pure text; no letters included

        # Ground-truth letter
        mv = rec.get("mvbench_accuracy") or {}
        gt_letter = (mv.get("gt_answer") or "").strip().upper()
        if not gt_letter:
            gt_letter = text_to_letter_from_candidates(doc.get("answer"), options) or ""

        # Raw response
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

        # Model's letter (for lmms_eval) from mvbench_accuracy.pred_answer, which may be "(B) text"
        parsed_pred_letter = extract_letter(mv.get("pred_answer", ""), num_options=len(options))
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
                "video": doc.get("video"),
            },
        })

    # Batched LLM calls — now also returning the exact prompts used
    think_outs, think_prompts = build_prompts_and_outputs(
        llm, tokenizer, build_thinking_extraction_message,
        questions, options_list, think_payloads,
        batch_size=args.batch_size,
    )
    answer_outs, answer_prompts = build_prompts_and_outputs(
        llm, tokenizer, build_answer_parsing_message,
        questions, options_list, answer_payloads,
        batch_size=args.batch_size,
    )

    # Tally & outputs
    per_samples = []
    lmms_correct = answer_correct = thinking_correct = 0
    combo_counts = Counter()

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
            # NEW: store the exact prompts given to the LLM
            "thinking_prompt": think_prompts[i] if i < len(think_prompts) else "",
            "answer_prompt":   answer_prompts[i] if i < len(answer_prompts) else "",
            "meta": slot["meta"],
        })

    # Agreement metric (answer vs thinking both match or both mismatch)
    n_total = len(temp_slots)
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
