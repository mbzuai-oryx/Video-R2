"""
Evaluate MMMU-style video QA outputs with three accuracy lenses:
  1) lmms_eval_accuracy: compare mmmu_acc.parsed_pred to ground truth.
  2) answer_parsed_accuracy: **LLM-parse** the <answer>...</answer> text and compare to GT.
  3) thinking_parsed_accuracy: ask an LLM to extract an answer from <think>...</think> given the question/options, compare to GT.

Outputs:
  - per_sample.json in --output_dir: detailed per-sample fields & booleans.
  - summary.json in --output_dir: aggregate accuracies and agreement stats.

Notes:
  - Deterministic LLM extraction (temperature=0, top_p=1.0, top_k=-1).
  - Batch size: default -1 means process all prompts at once with a single vLLM call; otherwise chunk by --batch_size.
  - For MCQ, compare letters A–Z (preferring a leading label like "F. ...").
  - For open-ended numeric, compare to two decimals (rounded). For text, do lenient normalized substring/equality matching.
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
# Single capital letter optionally followed by punctuation/paren/colon
RE_LETTER = re.compile(r"\b([A-Z])\s*[\.|\)|:|-]?\b")
# Numeric with optional commas, decimal, sign
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
                try:
                    data.append(json.loads(line.rstrip(",")))
                except Exception:
                    raise
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
    # 1) Prefer a leading option label like "F. ..."
    m0 = re.match(r"^\s*([A-Z])\s*[\.\)|:,-]", text)
    if m0:
        L = m0.group(1)
        if (num_options is None) or (L in [chr(ord('A') + i) for i in range(num_options)]):
            return L
    # 2) Fallback to scanning all letters (previous behavior)
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
    token = m.group(0)
    token = token.replace("$", "").replace(",", "").strip()
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
    p = normalize_text(str(pred))
    g = normalize_text(str(gt))
    if not p or not g:
        return False
    return p == g or (p in g) or (g in p)


def is_mcq(options: Any) -> bool:
    return isinstance(options, list) and len(options) > 0


def gt_letter_from_answer(gt_answer: str, options: Optional[List[str]]) -> Optional[str]:
    if gt_answer is None:
        return None
    L = extract_letter(gt_answer, num_options=len(options) if options else None)
    if L:
        return L
    if options:
        gt_norm = normalize_text(gt_answer)
        for i, opt in enumerate(options):
            if text_match(gt_norm, opt):
                return chr(ord('A') + i)
    return None


def compare_mcq(pred_letter: Optional[str], gt_letter: Optional[str]) -> bool:
    if not pred_letter or not gt_letter:
        return False
    return pred_letter.strip().upper() == gt_letter.strip().upper()


def compare_open_ended(pred: str, gt: str) -> bool:
    if eq_numeric(pred, gt, places=2):
        return True
    return text_match(pred, gt)


def best_from_parsed_pred(parsed_pred: Any, options: Optional[List[str]], gt_answer: str) -> Tuple[Optional[str], Optional[str]]:
    if parsed_pred is None:
        return (None, None)
    if isinstance(parsed_pred, list):
        candidates = [str(x) for x in parsed_pred]
    else:
        candidates = [str(parsed_pred)]

    if is_mcq(options):
        for c in candidates[::-1]:
            L = extract_letter(c, num_options=len(options))
            if L:
                return (L, None)
        for c in candidates[::-1]:
            if options:
                for i, opt in enumerate(options):
                    if text_match(c, opt):
                        return (chr(ord('A') + i), None)
        return (None, None)
    else:
        for c in candidates:
            if parse_number(c) is not None:
                return (None, c)
        return (None, candidates[-1] if candidates else None)


def batch_apply_chat_template(tokenizer, list_of_messages: List[List[Dict[str, str]]]) -> List[str]:
    prompts = []
    for messages in list_of_messages:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt)
    return prompts


def run_llm_extraction(
    llm: LLM,
    tokenizer,
    questions: List[str],
    options_list: List[Optional[List[str]]],
    thinkings: List[str],
    is_mcq_flags: List[bool],
    batch_size: int,
) -> List[str]:
    msgs = [build_thinking_extraction_message(opts, th, is_mcq) for q, opts, th, is_mcq in zip(questions, options_list, thinkings, is_mcq_flags)]
    prompts = batch_apply_chat_template(tokenizer, msgs)

    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=128,
    )

    outputs: List[str] = []

    def _consume(batch_prompts: List[str]):
        if not batch_prompts:
            return
        gens = llm.generate(batch_prompts, sp)
        for g in gens:
            text = g.outputs[0].text.strip() if g.outputs and g.outputs[0].text is not None else ""
            outputs.append(text)

    if batch_size == -1:
        pbar = tqdm(total=len(prompts), desc="LLM extraction (single call)")
        _consume(prompts)
        pbar.update(len(prompts))
        pbar.close()
    else:
        for i in tqdm(range(0, len(prompts), batch_size), desc="LLM extraction"):
            _consume(prompts[i : i + batch_size])

    return outputs


def run_llm_parse_text(
    llm: LLM,
    tokenizer,
    questions: List[str],
    options_list: List[Optional[List[str]]],
    texts: List[str],
    is_mcq_flags: List[bool],
    batch_size: int,
) -> List[str]:
    msgs = [build_answer_parsing_message(opts, txt, is_mcq)
            for q, opts, txt, is_mcq in zip(questions, options_list, texts, is_mcq_flags)]
    prompts = batch_apply_chat_template(tokenizer, msgs)

    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=128,
    )

    outputs: List[str] = []

    def _consume(batch_prompts: List[str]):
        if not batch_prompts:
            return
        gens = llm.generate(batch_prompts, sp)
        for g in gens:
            text = g.outputs[0].text.strip() if g.outputs and g.outputs[0].text is not None else ""
            outputs.append(text)

    if batch_size == -1:
        pbar = tqdm(total=len(prompts), desc="LLM answer-parse (single call)")
        _consume(prompts)
        pbar.update(len(prompts))
        pbar.close()
    else:
        for i in tqdm(range(0, len(prompts), batch_size), desc="LLM answer-parse"):
            _consume(prompts[i: i + batch_size])

    return outputs


# -----------------------------
# Main evaluation
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate MMMU-style Video QA with multiple parsing lenses")
    parser.add_argument("--sample_jsonl", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write per_sample.json and summary.json")
    parser.add_argument("--inference_model", type=str, required=True, help="Model name/path used to extract answers from <think> and to parse <answer>")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size for vLLM (default=1)")
    parser.add_argument("--batch_size", type=int, default=-1, help="-1 to process all prompts at once; else chunk size for LLM.generate")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    records = load_jsonl(args.sample_jsonl)

    print(f"Loading model for parsing/extraction: {args.inference_model} (tp={args.tp}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.inference_model, trust_remote_code=True)
    llm = LLM(model=args.inference_model, tensor_parallel_size=args.tp, dtype="auto")

    per_samples: List[Dict[str, Any]] = []

    n_total = 0
    n_mcq = 0
    n_oe = 0

    lmms_correct = 0
    answer_correct = 0
    thinking_correct = 0

    combo_counts = Counter()

    questions: List[str] = []
    options_list: List[Optional[List[str]]] = []
    thinkings: List[str] = []
    answer_texts: List[str] = []
    is_mcq_flags: List[bool] = []

    temp_slots: List[Dict[str, Any]] = []

    for rec in tqdm(records, desc="Processing samples"):
        n_total += 1
        doc = rec.get("doc", {})
        question = doc.get("question") or rec.get("input") or ""
        options = doc.get("options") or []
        gt_answer = doc.get("answer") or rec.get("target")
        input_text = rec.get("input", "")

        responses = rec.get("filtered_resps") or rec.get("resps") or []
        raw_response = None
        if isinstance(responses, list) and responses:
            if isinstance(responses[0], list) and responses[0]:
                raw_response = responses[0][0]
            elif isinstance(responses[0], str):
                raw_response = responses[0]
        if raw_response is None:
            raw_response = ""

        final_answer_text = extract_last_block(raw_response, RE_ANSWER) or ""
        thinking_text = extract_last_block(raw_response, RE_THINK) or ""

        mcq_flag = is_mcq(options)
        if mcq_flag:
            n_mcq += 1
        else:
            n_oe += 1

        gt_letter = gt_letter_from_answer(str(gt_answer) if gt_answer is not None else "", options) if mcq_flag else None

        parsed_pred_any = rec.get("mmmu_acc", {}).get("parsed_pred")
        lmms_is_correct = False
        lmms_pred_letter = None
        lmms_pred_text = None
        if mcq_flag:
            lmms_pred_letter, _ = best_from_parsed_pred(parsed_pred_any, options, str(gt_answer))
            lmms_is_correct = compare_mcq(lmms_pred_letter, gt_letter)
        else:
            _, lmms_pred_text = best_from_parsed_pred(parsed_pred_any, options, str(gt_answer))
            lmms_is_correct = compare_open_ended(lmms_pred_text or "", str(gt_answer))

        # We'll compute answer_parsed_accuracy later via LLM parsing of <answer> text
        answer_is_correct = False
        answer_parsed_letter = None

        questions.append(question or input_text)
        options_list.append(options if mcq_flag else None)
        thinkings.append(thinking_text)
        answer_texts.append(final_answer_text or "")
        is_mcq_flags.append(mcq_flag)

        temp_slots.append({
            "rec": rec,
            "doc": doc,
            "question": question,
            "options": options,
            "gt_answer": gt_answer,
            "gt_letter": gt_letter,
            "input_text": input_text,
            "raw_response": raw_response,
            "final_answer_text": final_answer_text,
            "thinking_text": thinking_text,
            "mcq_flag": mcq_flag,
            "lmms_pred_letter": lmms_pred_letter,
            "lmms_pred_text": lmms_pred_text,
            "lmms_is_correct": lmms_is_correct,
        })

    # Run LLM extraction on <think>
    extracted = run_llm_extraction(
        llm, tokenizer, questions, options_list, thinkings, is_mcq_flags, batch_size=args.batch_size
    )

    # Run LLM parsing on <answer>
    ans_extracted = run_llm_parse_text(
        llm, tokenizer, questions, options_list, answer_texts, is_mcq_flags, batch_size=args.batch_size
    )

    for i, (slot, ext) in enumerate(zip(temp_slots, extracted)):
        mcq_flag = slot["mcq_flag"]
        gt_answer = slot["gt_answer"]
        gt_letter = slot["gt_letter"]

        thinking_pred_letter = None
        thinking_pred_text = None
        if mcq_flag:
            thinking_pred_letter = extract_letter(ext, num_options=len(slot["options"]))
            thinking_is_correct = compare_mcq(thinking_pred_letter, gt_letter)
        else:
            thinking_pred_text = ext.strip()
            thinking_is_correct = compare_open_ended(thinking_pred_text, str(gt_answer))

        # Answer parsing via LLM from <answer> block
        ans_ext = ans_extracted[i]
        if mcq_flag:
            answer_parsed_letter = extract_letter(ans_ext, num_options=len(slot["options"]))
            answer_is_correct = compare_mcq(answer_parsed_letter, gt_letter)
            answer_parsed_pred_out = answer_parsed_letter
        else:
            answer_parsed_pred_out = ans_ext.strip()
            answer_is_correct = compare_open_ended(answer_parsed_pred_out, str(gt_answer))

        lmms_is_correct = slot["lmms_is_correct"]
        lmms_correct += int(lmms_is_correct)
        answer_correct += int(answer_is_correct)
        thinking_correct += int(thinking_is_correct)

        combo_counts[(int(lmms_is_correct), int(answer_is_correct), int(thinking_is_correct))] += 1

        per_samples.append({
            "doc_id": slot["rec"].get("doc_id"),
            "doc": slot["doc"].get("id"),
            "question": slot["question"],
            "options": slot["options"],
            "ground_truth": gt_answer,
            "input": slot["input_text"],
            "raw_response": slot["raw_response"],
            "final_answer_text": slot["final_answer_text"],
            "thinking_text": slot["thinking_text"],
            "mmmu_parsed_pred": slot["rec"].get("mmmu_acc", {}).get("parsed_pred"),
            "answer_parsed_pred": answer_parsed_pred_out,
            "thinking_parsed_pred": thinking_pred_letter if mcq_flag else thinking_pred_text,
            "lmms_eval_correct": lmms_is_correct,
            "answer_parsed_correct": answer_is_correct,
            "thinking_parsed_correct": thinking_is_correct,
        })

    answer_thinking_same = (
        combo_counts.get((0, 0, 0), 0) + combo_counts.get((1, 0, 0), 0) +
        combo_counts.get((0, 1, 1), 0) + combo_counts.get((1, 1, 1), 0)
    )
    answer_to_thinking_correlation = (answer_thinking_same / n_total) if n_total else 0.0
    
    # Build summary
    summary: Dict[str, Any] = {
        "n_total": n_total,
        "n_mcq": n_mcq,
        "n_open_ended": n_oe,
        # Flat accuracies
        "lmms_eval_accuracy": (lmms_correct / n_total) if n_total else 0.0,
        "answer_parsed_accuracy": (answer_correct / n_total) if n_total else 0.0,
        "thinking_parsed_accuracy": (thinking_correct / n_total) if n_total else 0.0,
        "answer_to_thinking_correlation": answer_to_thinking_correlation,
        # Explicit counts for verification
        "lmms_eval_correct": lmms_correct,
        "lmms_eval_total": n_total,
        "answer_parsed_correct": answer_correct,
        "answer_parsed_total": n_total,
        "thinking_parsed_correct": thinking_correct,
        "thinking_parsed_total": n_total,
        "combinations": {
            f"lmms{a}_ans{b}_think{c}": cnt for (a, b, c), cnt in sorted(combo_counts.items())
        },
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
