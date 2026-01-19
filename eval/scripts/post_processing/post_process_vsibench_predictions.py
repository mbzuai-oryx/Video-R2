"""
evaluate_predictions_vsibench.py

- Correct vLLM chat usage with:
  tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
- NA: converts free-form predictions to floats via LLM (handles number words), logs raw + parsed LLM outputs.
- MCA: exact-match on option letter; if --use_llm_for_MCA and mismatch, **batch LLM inference that returns only a letter**,
       then compare that letter to GT. Logs raw + parsed LLM outputs.
- Summary: per your spec (6 MCA + avg, 4 NA + avg, Overall), plus which LLM model was used.
- Batch control: --batch_size -1 (default) sends all prompts at once; otherwise chunked.
- Outputs: per_sample.jsonl and summary.json (no CSV).
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Load heavy deps when needed
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
except Exception:
    LLM = None
    SamplingParams = None
    AutoTokenizer = None

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
# Utilities & metrics
# -----------------------------
def extract_option_letter(
    text: str, options: Optional[List[str]] = None
) -> Optional[str]:
    """Normalize prediction into option letter A-D."""
    if not text:
        return None
    t = text.strip()

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
        # direct substring
        for letter, body in letter_to_body.items():
            if body and body.lower() in lower_pred:
                return letter

        # soft normalization
        def norm(s: str) -> str:
            return re.sub(r"[\W_]+", "", s.lower())

        npred = norm(lower_pred)
        for letter, body in letter_to_body.items():
            if not body:
                continue
            if norm(body) and norm(body) in npred:
                return letter
    return None


def mean_relative_accuracy(
    pred: Optional[float],
    target: Optional[float],
    start: float = 0.5,
    end: float = 0.95,
    interval: float = 0.05,
) -> float:
    if pred is None or target is None:
        return 0.0
    if not math.isfinite(pred) or not math.isfinite(target) or target == 0:
        return 0.0
    num_pts = int((end - start) / interval) + 1
    c_vals = np.linspace(start, end, num_pts)
    err = abs(pred - target) / abs(target)
    return float((err <= (1.0 - c_vals)).mean())


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def regex_number_fallback(text: str) -> Optional[float]:
    """Fallback: pick first float-looking number in text."""
    if not text:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return safe_float(m.group(0)) if m else None


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def flatten_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Handle inputs that store fields in `vsibench_score`/`doc` or at top-level."""
    vs = rec.get("vsibench_score", {})
    doc = rec.get("doc", {})
    out = {}

    def g(*keys, default=None):
        for k in keys:
            if k in vs:
                return vs[k]
            if k in doc:
                return doc[k]
            if k in rec:
                return rec[k]
        return default

    out["id"] = g("id", "doc_id", "qid")
    out["dataset"] = g("dataset")
    out["scene_name"] = g("scene_name")
    out["question_type"] = g("question_type")
    out["question"] = g("question")
    out["ground_truth"] = g("ground_truth")
    out["options"] = g("options")
    out["prediction"] = g("prediction", "pred")
    out["raw_record"] = rec
    return out


# -----------------------------
# LLM / chat helpers
# -----------------------------
def build_prompt_na_to_float(question: Optional[str], prediction_text: str) -> str:
    """
    Strong instruction: parse digits AND spelled-out numbers.
    Ensures 'There are two tables' -> 2, strict JSON only.
    """
    ctx = question or ""
    return (
        "You convert free-form answers into a single numeric value.\n"
        "Rules:\n"
        '1) Output STRICT JSON only: {"value": <number or null>} (no extra text).\n'
        "2) Extract a number from the answer text, even if written as a word.\n"
        "3) Interpret number words: zero=0, one=1, two=2, three=3, four=4, five=5, "
        "six=6, seven=7, eight=8, nine=9, ten=10, eleven=11, twelve=12.\n"
        "4) If the text clearly states a count (e.g., 'There are two tables'), return that count as a number.\n"
        "5) If multiple numbers exist, pick the single most relevant to the answer.\n"
        "6) If no clear numeric value, return null.\n\n"
        "Examples:\n"
        "Answer text: 'There are two tables in the room.' → {\"value\": 2}\n"
        "Answer text: 'Approximately ten meters.' → {\"value\": 10}\n"
        "Answer text: 'No idea.' → {\"value\": null}\n\n"
        f"Question (context): {ctx}\n"
        f"Answer text: {prediction_text}\n\n"
        "Return the JSON now."
    )


def build_prompt_mca_letter(
    question: str, options: List[str], prediction_text: str
) -> str:
    """
    Ask LLM to output ONLY the option letter (A/B/C/D) that best matches the prediction,
    based on the provided options. Strict JSON response.
    """
    return (
        "You are mapping a free-form prediction to one of the multiple-choice options.\n"
        "Return ONLY the letter of the best matching option.\n"
        "Rules:\n"
        '1) Output STRICT JSON only: {"letter": <"A"|"B"|"C"|"D"|null>} (no extra text).\n'
        "2) Choose the single most likely option letter that matches the prediction text.\n"
        "3) If you cannot determine the option confidently, return null.\n\n"
        f"Question: {question}\n"
        f"Options:\n{chr(10).join(options or [])}\n"
        f"Model prediction (free-form): {prediction_text}\n\n"
        "Return the JSON now."
    )


def run_vllm(model_path: str, tp_size: int) -> Tuple[Any, Any]:
    if LLM is None or AutoTokenizer is None or SamplingParams is None:
        raise RuntimeError("vLLM / transformers not available in this environment.")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        dtype="auto",
        tensor_parallel_size=tp_size,
        max_model_len=32768,
    )
    return llm, tok


def apply_chat_template(tok, user_texts: List[str]) -> List[str]:
    """Apply chat template with enable_thinking=False."""
    prompts = []
    for ut in user_texts:
        messages = [{"role": "user", "content": ut}]
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompts.append(prompt)
    return prompts


def generate_texts(llm, tok, user_prompts: List[str], batch_size: int) -> List[str]:
    """
    user_prompts: list of *plain* user strings (we apply chat template here).
    """
    prompts = apply_chat_template(tok, user_prompts)
    sp = SamplingParams(temperature=0.0, max_tokens=64)

    if batch_size == -1:
        results = llm.generate(prompts, sp, use_tqdm=False)
        return [r.outputs[0].text.strip() for r in results]

    outs: List[str] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="vLLM batches"):
        chunk = prompts[i : i + batch_size]
        results = llm.generate(chunk, sp, use_tqdm=False)
        outs.extend(r.outputs[0].text.strip() for r in results)
    return outs


def parse_strict_json(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VSIBench-style predictions with LLM post-processing for NA (and optional MCA)."
    )
    parser.add_argument(
        "--pred_jsonl", type=str, required=True, help="Path to predictions JSONL."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to write results."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Qwen3 checkpoint (required if NA present or --use_llm_for_MCA).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=-1,
        help="Batch size for vLLM (-1 = send all prompts at once).",
    )
    parser.add_argument(
        "--tp_size", type=int, default=1, help="Tensor parallel size for vLLM."
    )
    parser.add_argument(
        "--use_llm_for_MCA",
        action="store_true",
        help="Use LLM to letter-normalize MCA mismatches.",
    )
    parser.add_argument("--mra_start", type=float, default=0.5)
    parser.add_argument("--mra_end", type=float, default=0.95)
    parser.add_argument("--mra_interval", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    per_sample_path = os.path.join(args.output_dir, "per_sample.jsonl")
    summary_path = os.path.join(args.output_dir, "summary.json")

    records = [flatten_record(r) for r in read_jsonl(args.pred_jsonl)]

    need_llm_na = any(r.get("question_type") in NA_QUESTION_TYPES for r in records)
    need_llm_mca = args.use_llm_for_MCA
    need_llm = need_llm_na or need_llm_mca

    llm = tok = None
    if need_llm:
        if not args.model_path:
            print(
                "ERROR: --model_path is required because NA questions or --use_llm_for_MCA are present.",
                file=sys.stderr,
            )
            sys.exit(1)
        llm, tok = run_vllm(args.model_path, args.tp_size)

    # -------- NA prompts (batch) --------
    na_indices: List[int] = []
    na_user_texts: List[str] = []
    if need_llm_na:
        for idx, r in enumerate(records):
            if r.get("question_type") in NA_QUESTION_TYPES:
                na_indices.append(idx)
                na_user_texts.append(
                    build_prompt_na_to_float(
                        r.get("question") or "", str(r.get("prediction", "") or "")
                    )
                )

    na_llm_value: Dict[int, Optional[float]] = {}
    na_llm_raw: Dict[int, Optional[str]] = {}
    na_llm_json: Dict[int, Optional[dict]] = {}
    na_value_src: Dict[int, Optional[str]] = {}

    if na_user_texts:
        outs = generate_texts(llm, tok, na_user_texts, args.batch_size)
        for i, out in enumerate(outs):
            idx = na_indices[i]
            na_llm_raw[idx] = out
            obj = parse_strict_json(out)
            na_llm_json[idx] = obj
            val = None
            if obj and "value" in obj:
                val = safe_float(obj["value"])
            if val is not None:
                na_llm_value[idx] = val
                na_value_src[idx] = "llm"
            else:
                # fallback to regex on the raw prediction
                val = regex_number_fallback(records[idx].get("prediction", ""))
                na_llm_value[idx] = val
                na_value_src[idx] = "regex_fallback"

    # -------- First pass rows; collect MCA mismatches for batch LLM letter extraction --------
    rows: List[Dict[str, Any]] = []
    mca_pending_row_indices: List[int] = []  # indices into rows list
    mca_pending_user_texts: List[str] = []  # prompts to send to LLM

    for ridx, r in enumerate(records):
        qtype = r.get("question_type")
        gt = r.get("ground_truth")
        pred = r.get("prediction", "")
        options = r.get("options") or []

        row = {
            "id": r.get("id"),
            "dataset": r.get("dataset"),
            "scene_name": r.get("scene_name"),
            "question_type": qtype,
            "ground_truth": gt,
            "prediction_raw": pred,
            "metric": None,
            "metric_value": None,
        }

        if qtype in MCA_QUESTION_TYPES:
            m_gt = re.match(r"^\s*([A-Da-d])\b", str(gt) or "")
            gt_letter = m_gt.group(1).upper() if m_gt else None
            pred_letter = extract_option_letter(str(pred), options)
            match_exact = (
                pred_letter is not None
                and gt_letter is not None
                and pred_letter == gt_letter
            )

            row.update(
                {
                    "metric": "accuracy",
                    "gt_letter": gt_letter,
                    "parsed_pred_letter": pred_letter,  # may be replaced by LLM letter
                    "mca_llm_used": False,
                    "mca_llm_raw": None,
                    "mca_llm_json": None,
                }
            )

            if match_exact:
                row["metric_value"] = 1.0
            else:
                if args.use_llm_for_MCA and need_llm:
                    mca_pending_row_indices.append(len(rows))
                    mca_pending_user_texts.append(
                        build_prompt_mca_letter(
                            r.get("question", "") or "", options, str(pred)
                        )
                    )
                else:
                    row["metric_value"] = 0.0

        elif qtype in NA_QUESTION_TYPES:
            gt_float = safe_float(gt)
            pred_float = na_llm_value.get(ridx)
            mra = mean_relative_accuracy(
                pred_float,
                gt_float,
                start=args.mra_start,
                end=args.mra_end,
                interval=args.mra_interval,
            )
            row.update(
                {
                    "metric": f"MRA:{args.mra_start}:{args.mra_end}:{args.mra_interval}",
                    "gt_float": gt_float,
                    "parsed_pred_float": pred_float,
                    "metric_value": mra,
                    "na_llm_used": True,
                    "na_llm_raw": na_llm_raw.get(ridx),
                    "na_llm_json": na_llm_json.get(ridx),
                    "na_value_source": na_value_src.get(ridx),
                }
            )
        else:
            row.update(
                {
                    "metric": "unknown",
                    "metric_value": None,
                    "note": f"Unknown question_type={qtype}",
                }
            )

        rows.append(row)

    # -------- Batch-run MCA letter extraction and update rows --------
    if mca_pending_user_texts:
        outs = generate_texts(llm, tok, mca_pending_user_texts, args.batch_size)
        for i, out in enumerate(outs):
            row_idx = mca_pending_row_indices[i]
            row = rows[row_idx]
            obj = parse_strict_json(out)

            row["mca_llm_used"] = True
            row["mca_llm_raw"] = out
            row["mca_llm_json"] = obj

            # Pull letter and compute accuracy vs GT
            gt_letter = row.get("gt_letter")
            llm_letter = None
            if obj and isinstance(obj.get("letter"), (str, type(None))):
                cand = obj.get("letter")
                if isinstance(cand, str) and cand.upper() in ["A", "B", "C", "D"]:
                    llm_letter = cand.upper()

            # If LLM gave a letter, overwrite parsed_pred_letter with it
            if llm_letter:
                row["parsed_pred_letter"] = llm_letter

            # Final accuracy comparison
            if llm_letter and gt_letter:
                row["metric_value"] = 1.0 if llm_letter == gt_letter else 0.0
            else:
                # If we still can't determine, mark incorrect (0.0)
                row["metric_value"] = 0.0

    # -------- Aggregate metrics --------
    per_type_vals: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        qtype = row.get("question_type")
        val = row.get("metric_value")
        if qtype in (MCA_QUESTION_TYPES + NA_QUESTION_TYPES) and isinstance(
            val, (int, float)
        ):
            per_type_vals[qtype].append(float(val))

    def mean_or_none(vals: List[float]) -> Optional[float]:
        return float(np.mean(vals)) if vals else None

    per_type_mean: Dict[str, Optional[float]] = {
        t: mean_or_none(per_type_vals.get(t, []))
        for t in set(MCA_QUESTION_TYPES + NA_QUESTION_TYPES)
    }

    mca_vals = [
        per_type_mean[t] for t in MCA_QUESTION_TYPES if per_type_mean.get(t) is not None
    ]
    na_vals = [
        per_type_mean[t] for t in NA_QUESTION_TYPES if per_type_mean.get(t) is not None
    ]
    mca_avg = float(np.mean(mca_vals)) if mca_vals else None
    na_avg = float(np.mean(na_vals)) if na_vals else None
    overall_pool = [
        per_type_mean[t]
        for t in (MCA_QUESTION_TYPES + NA_QUESTION_TYPES)
        if per_type_mean.get(t) is not None
    ]
    overall = float(np.mean(overall_pool)) if overall_pool else None

    summary = {
        # Individual MCA accuracies
        "object_rel_direction_easy": per_type_mean.get("object_rel_direction_easy"),
        "object_rel_direction_medium": per_type_mean.get("object_rel_direction_medium"),
        "object_rel_direction_hard": per_type_mean.get("object_rel_direction_hard"),
        "object_rel_distance": per_type_mean.get("object_rel_distance"),
        "route_planning": per_type_mean.get("route_planning"),
        "obj_appearance_order": per_type_mean.get("obj_appearance_order"),
        # MCA average
        "MCA_QUESTION_TYPES": mca_avg,
        # Individual NA accuracies (MRA)
        "object_abs_distance": per_type_mean.get("object_abs_distance"),
        "object_counting": per_type_mean.get("object_counting"),
        "object_size_estimation": per_type_mean.get("object_size_estimation"),
        "room_size_estimation": per_type_mean.get("room_size_estimation"),
        # NA average
        "NA_QUESTION_TYPES": na_avg,
        # Overall across all 10
        "Overall": overall,
        # LLM used (if any)
        "llm_model": args.model_path if (need_llm) else None,
        # Repro
        "mra_params": {
            "start": args.mra_start,
            "end": args.mra_end,
            "interval": args.mra_interval,
        },
        "use_llm_for_MCA": args.use_llm_for_MCA,
        "batch_size_effective": args.batch_size,
    }

    # -------- Write outputs --------
    with open(per_sample_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("✅ Done.")
    print(f"• Per-sample JSONL: {per_sample_path}")
    print(f"• Summary JSON:     {summary_path}")
    print(f"MCA_QUESTION_TYPES: {summary['MCA_QUESTION_TYPES']}")
    print(f"NA_QUESTION_TYPES: {summary['NA_QUESTION_TYPES']}")
    print(f"Overall: {summary['Overall']}")


if __name__ == "__main__":
    main()
