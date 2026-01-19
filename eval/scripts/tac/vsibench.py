"""
Evaluate VSIBench-style mixed MCA/NA open-ended QA.

Now type-aware:
  • NA types -> MRA scoring (object_abs_distance, object_counting, object_size_estimation, room_size_estimation)
  • MCA types -> exact-match accuracy on option letter
And still:
  • Always LLM-parse from <think> and <answer> (numeric for NA, letter for MCA)
  • Report three top-line metrics:
      - lmms_eval_accuracy: model's own prediction scored type-appropriately
      - answer_parsed_accuracy: LLM parse of <answer> scored type-appropriately
      - thinking_parsed_accuracy: LLM parse of <think> scored type-appropriately
  • Also reports answer↔thinking agreement (booleanized) and combo breakdowns.

Expected JSONL per line (schema) remains as before.
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Union

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from _utils import *

# -----------------------------
# Regex helpers
# -----------------------------
RE_THINK  = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
RE_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
# Numeric token with optional commas/decimal/sign/currency (we strip symbols)
RE_NUMBER = re.compile(
    r"[-+]?[\$€£]?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?\s*%?"
)

# -----------------------------
# Task type lists
# -----------------------------
MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

# -----------------------------
# JSONL loader
# -----------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSONL at line {ln}: {e}\nLine: {s[:200]}...")
    return rows

# -----------------------------
# Extractors & parsing
# -----------------------------
def extract_last_block(text: str, pattern: re.Pattern) -> Optional[str]:
    if not text:
        return None
    matches = list(pattern.finditer(text))
    return matches[-1].group(1).strip() if matches else None

def parse_number_token(token: str) -> Optional[float]:
    t = token.replace(",", "").replace("$", "").replace("€", "").replace("£", "").strip()
    if t.endswith("%"):
        t = t[:-1].strip()
    try:
        return float(t)
    except Exception:
        return None

def parse_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = RE_NUMBER.search(text)
    if not m:
        return None
    return parse_number_token(m.group(0))

def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v
    except Exception:
        return parse_number(str(x)) if x is not None else None

def extract_option_letter(pred_text: str, options: Optional[List[str]] = None) -> Optional[str]:
    """Deterministic letter normalization (A-D), used for model 'prediction' only."""
    if not pred_text:
        return None
    t = pred_text.strip()
    m = re.match(r"^\s*([A-Da-d])(\b|[\.\):\-\s])", t)
    if m:
        return m.group(1).upper()
    if len(t) == 1 and t.upper() in ["A", "B", "C", "D"]:
        return t.upper()
    if options:
        letter_to_body = {}
        for opt in options:
            m2 = re.match(r"^\s*([A-Da-d])\s*[\.\):-]\s*(.*)$", (opt or "").strip())
            if m2:
                letter_to_body[m2.group(1).upper()] = m2.group(2).strip()
        lower_pred = t.lower()
        for letter, body in letter_to_body.items():
            if body and body.lower() in lower_pred:
                return letter
        def norm(s: str) -> str:
            return re.sub(r"[\W_]+", "", s.lower())
        npred = norm(lower_pred)
        for letter, body in letter_to_body.items():
            if body and norm(body) in npred:
                return letter
    return None

# -----------------------------
# Metrics
# -----------------------------
def mean_relative_accuracy(
    pred: Optional[float],
    target: Optional[float],
    start: float = 0.5,
    end: float = 0.95,
    interval: float = 0.05,
) -> float:
    """Proportion of c in [start, end] s.t. |pred-target|/|target| <= (1 - c)."""
    if pred is None or target is None:
        return 0.0
    if target == 0:
        return 0.0
    c_vals = []
    c = start
    while c <= end + 1e-9:
        c_vals.append(c)
        c += interval
    err = abs(pred - target) / abs(target)
    ok = sum(1 for c in c_vals if err <= (1.0 - c))
    return ok / len(c_vals)

def mra_bool_at_5pct(pred: Optional[float], target: Optional[float]) -> Optional[bool]:
    """Booleanize MRA at tightest 5% band (c=0.95)."""
    if pred is None or target is None or target == 0:
        return None
    return abs(pred - target) / abs(target) <= 0.05

# -----------------------------
# LLM helpers (numeric-only outputs for NA; letter-only outputs for MCA)
# -----------------------------
def build_extraction_message_numeric(question: str, payload_text: str) -> List[Dict[str, str]]:
    sys_prompt = (
        "You are a careful extractor. Based ONLY on the provided content, "
        "return the final numeric answer with no extra words."
    )
    user = (
        "Text to parse:\n" + (payload_text.strip() if payload_text else "") + "\n\n"
        "Return ONLY the number (digits, optional decimal) on a single line."
    )
    return [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": user}]


def batch_apply_chat_template(tokenizer, list_of_messages: List[List[Dict[str, str]]]) -> List[str]:
    prompts = []
    for messages in list_of_messages:
        # Be conservative about kwargs; many templates support these two
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    return prompts

def run_llm_simple(
    llm: LLM,
    tokenizer,
    messages_list: List[List[Dict[str, str]]],
    batch_size: int,
    max_tokens: int = 32,
) -> List[str]:
    prompts = batch_apply_chat_template(tokenizer, messages_list)
    sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=max_tokens)

    outs: List[str] = []
    def _consume(batch_prompts: List[str]):
        if not batch_prompts:
            return
        gens = llm.generate(batch_prompts, sp)
        for g in gens:
            txt = g.outputs[0].text.strip() if g.outputs and g.outputs[0].text is not None else ""
            outs.append(txt)

    if batch_size == -1:
        pbar = tqdm(total=len(prompts), desc="LLM parse (single batch)")
        _consume(prompts); pbar.update(len(prompts)); pbar.close()
    else:
        for i in tqdm(range(0, len(prompts), batch_size), desc="LLM parse (batches)"):
            _consume(prompts[i:i+batch_size])
    return outs

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate VSIBench-style QA with task-aware metrics")
    parser.add_argument("--sample_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--inference_model", type=str, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--mra_start", type=float, default=0.5)
    parser.add_argument("--mra_end", type=float, default=0.95)
    parser.add_argument("--mra_interval", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_jsonl(args.sample_jsonl)

    print(f"Loading model for parsing/extraction: {args.inference_model} (tp={args.tp}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.inference_model, trust_remote_code=True)
    llm = LLM(model=args.inference_model, tensor_parallel_size=args.tp, dtype="auto")

    per_samples: List[Dict[str, Any]] = []

    # Global tallies (mixed MCA/NA; each sample contributes a type-appropriate score)
    n_total = 0
    lmms_pool: List[float] = []
    answer_pool: List[float] = []
    thinking_pool: List[float] = []

    # For agreement/mismatch booleans
    combo_counts = Counter()

    # Pre-collect messages for batched LLM runs (we ALWAYS LLM-parse think/answer)
    think_msgs_numeric, ans_msgs_numeric = [], []
    think_msgs_letter,  ans_msgs_letter  = [], []
    think_slots, ans_slots = [], []  # store index mapping to later merge outputs

    temp_slots: List[Dict[str, Any]] = []
    for rec in tqdm(rows, desc="Processing samples"):
        n_total += 1
        doc = rec.get("doc", {}) or {}
        question = doc.get("question") or rec.get("input") or ""
        qtype = doc.get("question_type")
        options = doc.get("options") or []
        dataset = doc.get("dataset")
        scene_name = doc.get("scene_name")

        # Ground truth
        gt_raw = doc.get("ground_truth") or rec.get("target")
        gt_str = str(gt_raw) if gt_raw is not None else ""

        # Model prediction (free-form)
        model_pred = None
        if isinstance(rec.get("vsibench_score"), dict) and rec["vsibench_score"].get("prediction") is not None:
            model_pred = rec["vsibench_score"]["prediction"]
        elif doc.get("prediction") is not None:
            model_pred = doc.get("prediction")

        # Response text selection
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

        slot = {
            "rec": rec,
            "question": question,
            "qtype": qtype,
            "options": options,
            "gt_str": gt_str,
            "model_pred": model_pred,
            "raw": raw,
            "think_txt": think_txt,
            "ans_txt": ans_txt,
            "meta": {
                "dataset": dataset,
                "scene_name": scene_name,
                "question_type": qtype,
            },
        }
        temp_slots.append(slot)

        # Queue LLM parsing according to type
        if qtype in NA_QUESTION_TYPES:
            think_msgs_numeric.append(build_extraction_message_numeric(question, think_txt))
            ans_msgs_numeric.append(build_extraction_message_numeric(question, ans_txt))
            think_slots.append(("NA", len(temp_slots)-1))
            ans_slots.append(("NA", len(temp_slots)-1))
        elif qtype in MCA_QUESTION_TYPES:
            think_msgs_letter.append(build_thinking_extraction_message(options, think_txt))
            ans_msgs_letter.append(build_answer_parsing_message(options, ans_txt))
            think_slots.append(("MCA", len(temp_slots)-1))
            ans_slots.append(("MCA", len(temp_slots)-1))
        else:
            # Unknown type: treat as NA fallback (numeric), but mark unknown
            think_msgs_numeric.append(build_extraction_message_numeric(question, think_txt))
            ans_msgs_numeric.append(build_extraction_message_numeric(question, ans_txt))
            think_slots.append(("UNKNOWN", len(temp_slots)-1))
            ans_slots.append(("UNKNOWN", len(temp_slots)-1))

    # Run LLM for numeric and letter batches
    think_out_num = run_llm_simple(llm, tokenizer, think_msgs_numeric, args.batch_size, max_tokens=32) if think_msgs_numeric else []
    ans_out_num   = run_llm_simple(llm, tokenizer, ans_msgs_numeric,   args.batch_size, max_tokens=32) if ans_msgs_numeric else []
    think_out_let = run_llm_simple(llm, tokenizer, think_msgs_letter,  args.batch_size, max_tokens=8 ) if think_msgs_letter else []
    ans_out_let   = run_llm_simple(llm, tokenizer, ans_msgs_letter,    args.batch_size, max_tokens=8 ) if ans_msgs_letter else []

    # Stitch parsed values back by slot order
    it_num_t = iter(think_out_num); it_num_a = iter(ans_out_num)
    it_let_t = iter(think_out_let); it_let_a = iter(ans_out_let)

    for i, slot in enumerate(temp_slots):
        qtype = slot["qtype"]
        gt_str = slot["gt_str"]
        options = slot["options"]

        # Ground-truth normalization
        if qtype in MCA_QUESTION_TYPES:
            m = re.match(r"^\s*([A-Da-d])", gt_str or "")
            gt_letter = m.group(1).upper() if m else None
            gt_float = None
        else:
            gt_letter = None
            gt_float = safe_float(gt_str)

        # Parsed from LLM (think/answer)
        if qtype in MCA_QUESTION_TYPES:
            t_out = next(it_let_t, "")
            a_out = next(it_let_a, "")
            t_letter = extract_option_letter(t_out or "", options)
            a_letter = extract_option_letter(a_out or "", options)
            t_num = a_num = None
        else:
            t_out = next(it_num_t, "")
            a_out = next(it_num_a, "")
            t_num = parse_number(t_out)
            a_num = parse_number(a_out)
            t_letter = a_letter = None

        # Model prediction normalization (for lmms_eval)
        if qtype in MCA_QUESTION_TYPES:
            pred_letter = extract_option_letter(str(slot["model_pred"] or ""), options)
            lmms_score = 1.0 if (pred_letter and gt_letter and pred_letter == gt_letter) else 0.0
            ans_score  = 1.0 if (a_letter and gt_letter and a_letter == gt_letter) else 0.0
            think_score= 1.0 if (t_letter and gt_letter and t_letter == gt_letter) else 0.0

            # Booleanized for combos
            b_lmms  = lmms_score == 1.0
            b_ans   = ans_score == 1.0
            b_think = think_score == 1.0

        else:
            pred_num = safe_float(slot["model_pred"])
            lmms_score = mean_relative_accuracy(pred_num, gt_float, args.mra_start, args.mra_end, args.mra_interval)
            ans_score  = mean_relative_accuracy(a_num,   gt_float, args.mra_start, args.mra_end, args.mra_interval)
            think_score= mean_relative_accuracy(t_num,   gt_float, args.mra_start, args.mra_end, args.mra_interval)

            # Booleanized at 5% for combos
            b_lmms  = mra_bool_at_5pct(pred_num, gt_float) is True
            b_ans   = mra_bool_at_5pct(a_num,    gt_float) is True
            b_think = mra_bool_at_5pct(t_num,    gt_float) is True

        lmms_pool.append(lmms_score)
        answer_pool.append(ans_score)
        thinking_pool.append(think_score)
        combo_counts[(int(b_lmms), int(b_ans), int(b_think))] += 1

        per_samples.append({
            "doc_id": slot["rec"].get("doc_id") or slot["rec"].get("id") or slot["rec"].get("doc", {}).get("id"),
            "question": slot["question"],
            "ground_truth": gt_str,
            "question_type": qtype,
            "options": options,
            "model_prediction": slot["model_pred"],
            # LLM parsed outputs
            "thinking_parsed_pred": t_num if qtype in NA_QUESTION_TYPES else t_letter,
            "answer_parsed_pred":   a_num if qtype in NA_QUESTION_TYPES else a_letter,
            # Scores (type-appropriate)
            "lmms_eval_score": lmms_score,
            "answer_parsed_score": ans_score,
            "thinking_parsed_score": think_score,
            # Raw text for auditing
            "raw_response": slot["raw"],
            "thinking_text": slot["think_txt"],
            "final_answer_text": slot["ans_txt"],
            "meta": slot["meta"],
        })

    # Agreement (answer vs thinking parity of boolean correctness)
    n_total = len(temp_slots) if temp_slots else 0
    answer_thinking_same = sum(cnt for (a,b,c), cnt in combo_counts.items() if b == c)
    answer_to_thinking_agreement = (answer_thinking_same / n_total) if n_total else 0.0

    summary = {
        "n_total": n_total,
        # Mixed metrics: each sample contributes MRA (NA) or Accuracy (MCA)
        "lmms_eval_accuracy": (sum(lmms_pool) / n_total) if n_total else 0.0,
        "answer_parsed_accuracy": (sum(answer_pool) / n_total) if n_total else 0.0,
        "thinking_parsed_accuracy": (sum(thinking_pool) / n_total) if n_total else 0.0,
        "answer_to_thinking_correlation": answer_to_thinking_agreement,
        # Booleanized counts at 5% (NA) or exact (MCA)
        "combinations": {f"lmms{a}_ans{b}_think{c}": cnt for (a, b, c), cnt in sorted(combo_counts.items())},
        "mra_params": {
            "start": args.mra_start,
            "end": args.mra_end,
            "interval": args.mra_interval,
            "boolean_threshold": "5% rel. error for NA",
        },
    }

    # Write outputs (same filenames as original)
    per_sample_path = os.path.join(args.output_dir, "per_sample.json")
    summary_path    = os.path.join(args.output_dir, "summary.json")
    with open(per_sample_path, "w", encoding="utf-8") as f:
        json.dump(per_samples, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved per-sample to {per_sample_path}")
    print(f"Saved summary to {summary_path}")
    print(f"answer_to_thinking_agreement: {answer_to_thinking_agreement:.4f}")

if __name__ == "__main__":
    main()
