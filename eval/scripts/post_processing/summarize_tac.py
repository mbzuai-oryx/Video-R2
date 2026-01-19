#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate per-benchmark summaries (recursive) into a single top-level summary.json.

What this script does
---------------------
- Recursively find every run folder that contains a `summary.json`
  (including cases like mvbench/summary.json directly under the dataset folder).
- Infer a canonical dataset name from the run folder:
    1) Prefer `samples_<name>[_think]` at the end of the folder name.
    2) Else any `samples_<name>_` occurrence.
    3) Else a `<dataset>_think_*` prefix (strip the `_think...` suffix).
    4) Else just the folder name (works for mvbench/summary.json).
- If multiple runs exist for the same dataset, pick the **latest** by mtime ("latest wins").
- Compute and export the following per dataset:
    * lmms_eval_accuracy
    * answer_parsed_accuracy
    * thinking_parsed_accuracy
    * answer_to_thinking_correlation  (from file if present; else Po=(TP+TN)/N)
    * consistency_score               = (T - A) / min(A, T)            # your original (unbounded, directional)
    * signed_disagreement             = sign(T-A) * (FP + FN) / N      # bounded in [-1,1], 0 = high consistency
    * signed_disagreement_mixer       = sign(T-A) * [ α*(1 - corr^γ) + (1-α)*(1 - corr) ]  # bounded in [-1,1]
    * disagreement_rate               = (FP + FN) / N                  # magnitude only, [0,1]
    * mixer_magnitude                 = α*(1 - corr^γ) + (1-α)*(1 - corr)                   # magnitude only, [0,1]
    * consistency_error               = FN / (TP + FN)                 # among answer-correct, thinking disagreed
    * consistency_accuracy            = 1 - consistency_error = TP / (TP + FN)

Proposed mixer (bounded, signed)
--------------------------------
We amplify the penalty for low agreement while keeping sign(T−A):

    C_mix = sign(T - A) * [ α * (1 - corr^γ) + (1 - α) * (1 - corr) ]

Where:
- corr ∈ [0,1] is the observed agreement between Answer and Thinking labels
  (by default Po = (TP + TN) / N). If the JSON’s "answer_to_thinking_correlation"
  equals Po, we’ll use that; otherwise we compute Po from the confusion.
- α ∈ [0,1] blends a **nonlinear** penalty (1 - corr^γ) and the linear one (1 - corr).
- γ > 1 increases curvature to punish mediocre/low corr more strongly.
- Range: [-1, 1]; sign indicates whether Thinking (+) or Answer (−) is better; 0 near perfect agreement.
- Good defaults: α = 0.7, γ = 2.0.

Outputs
-------
- Writes aggregated summary to:  <input_dir>/summary.json
  {
    "overall": {
      "num_datasets": ...,
      "average_consistency_score": ...,
      "average_answer_to_thinking_correlation": ...,
      "average_signed_disagreement": ...,
      "average_signed_disagreement_mixer": ...,
      "average_disagreement_rate": ...,
      "average_mixer_magnitude": ...,
      "average_consistency_error": ...,
      "average_consistency_accuracy": ...,
      "mixer_hyperparams": {"alpha": α, "gamma": γ}
    },
    "datasets": {
      "<dataset_name>": {
        ...metrics...,
        "source_path": ".../summary.json",
        "mtime": <epoch_seconds>
      }
    }
  }
- Prints a compact table with:
  consistency_score, answer_to_thinking_correlation, signed_disagreement,
  signed_disagreement_mixer, disagreement_rate, consistency_error, consistency_accuracy, mixer_magnitude.

Usage
-----
    python aggregate_summaries_recursive.py --input_dir /path/to/root_dir [--alpha 0.7 --gamma 2.0]

"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Keys we always pull through from child summaries (if present)
FOUR_KEYS = [
    "lmms_eval_accuracy",
    "answer_parsed_accuracy",
    "thinking_parsed_accuracy",
    "answer_to_thinking_correlation",
]

# ---- Core metrics -----------------------------------------------------------

def compute_consistency(a: Optional[float], t: Optional[float]) -> Optional[float]:
    """Your original metric: (T - A) / min(A, T); returns None if not computable."""
    if not isinstance(a, (int, float)) or not isinstance(t, (int, float)):
        return None
    if a <= 0 or t <= 0:
        return None  # avoid division by zero/negatives
    denom = min(a, t)
    if denom == 0:
        return None
    return (t - a) / denom

def parse_ans_think_confusion(combos: Dict[str, int]) -> Optional[Dict[str, int]]:
    """
    Collapse LMMS dimension and return TP,TN,FP,FN for Answer vs Thinking:
      TP: ans1_think1 (both correct)
      TN: ans0_think0 (both wrong)
      FP: ans0_think1 (thinking correct, answer wrong)
      FN: ans1_think0 (answer correct, thinking wrong)
    `combos` keys look like 'lmms0_ans1_think0' etc.
    """
    if not isinstance(combos, dict):
        return None

    TP = TN = FP = FN = 0
    for k, v in combos.items():
        if not isinstance(v, int):
            continue
        key = str(k).lower()
        if "ans1_think1" in key:
            TP += v
        elif "ans0_think0" in key:
            TN += v
        elif "ans0_think1" in key:
            FP += v
        elif "ans1_think0" in key:
            FN += v

    total = TP + TN + FP + FN
    if total == 0:
        return None
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN, "N": total}

def compute_signed_disagreement(a: Optional[float], t: Optional[float],
                                confusion: Optional[Dict[str, int]]) -> Optional[float]:
    """
    C_D = sign(T-A) * (FP + FN) / N
    Uses provided accuracies for the sign; if missing, derive A,T from confusion.
    """
    if confusion is None:
        return None
    TP, TN, FP, FN, N = (confusion[k] for k in ("TP","TN","FP","FN","N"))
    if N <= 0:
        return None

    # sign
    if isinstance(a, (int, float)) and isinstance(t, (int, float)):
        s = 1.0 if (t - a) > 0 else (-1.0 if (t - a) < 0 else 0.0)
    else:
        A_hat = (TP + FN) / N
        T_hat = (TP + FP) / N
        s = 1.0 if (T_hat - A_hat) > 0 else (-1.0 if (T_hat - A_hat) < 0 else 0.0)

    return s * (FP + FN) / N

def compute_disagreement_rate(confusion: Optional[Dict[str, int]]) -> Optional[float]:
    """Unsigned magnitude: (FP + FN) / N in [0,1]."""
    if confusion is None:
        return None
    TP, TN, FP, FN, N = (confusion[k] for k in ("TP","TN","FP","FN","N"))
    return (FP + FN) / N if N > 0 else None

def compute_consistency_error(confusion: Optional[Dict[str, int]]) -> Optional[float]:
    """
    Among answer-correct samples, fraction where thinking disagrees:
        consistency_error = FN / (TP + FN)
    Returns None if there are no answer-correct samples (TP + FN == 0) or no confusion.
    """
    if confusion is None:
        return None
    TP, TN, FP, FN, N = (confusion[k] for k in ("TP","TN","FP","FN","N"))
    denom = TP + FN
    return (FN / denom) if denom > 0 else None

def compute_consistency_accuracy(confusion: Optional[Dict[str, int]]) -> Optional[float]:
    """
    Complement of consistency_error, i.e.,
        consistency_accuracy = TP / (TP + FN)
    """
    if confusion is None:
        return None
    TP, TN, FP, FN, N = (confusion[k] for k in ("TP","TN","FP","FN","N"))
    denom = TP + FN
    return (TP / denom) if denom > 0 else None

def compute_signed_disagreement_mixer(
    a: Optional[float],
    t: Optional[float],
    confusion: Optional[Dict[str, int]],
    corr: Optional[float],
    alpha: float = 0.7,
    gamma: float = 2.0,
) -> Optional[float]:
    """
    Proposed mixer (bounded, signed):
        C_mix = sign(T - A) * [ alpha * (1 - corr**gamma) + (1 - alpha) * (1 - corr) ]

    - corr is the observed agreement in [0,1] between Answer and Thinking.
      By default we take corr = Po = (TP + TN) / N from the confusion.
      If the dataset JSON's 'answer_to_thinking_correlation' is present, we
      use that as corr; otherwise we compute Po from combos.
    - alpha in [0,1] blends nonlinear and linear penalties (default 0.7).
    - gamma > 1 sharpens the penalty for mediocre/low corr (default 2.0).
    - Returns value in [-1, 1]; sign indicates whether thinking (+) or answer (−) is better.
    """
    if confusion is None:
        return None
    TP, TN, FP, FN, N = (confusion[k] for k in ("TP","TN","FP","FN","N"))
    if N <= 0:
        return None

    # sign
    if isinstance(a, (int, float)) and isinstance(t, (int, float)):
        s = 1.0 if (t - a) > 0 else (-1.0 if (t - a) < 0 else 0.0)
    else:
        A_hat = (TP + FN) / N
        T_hat = (TP + FP) / N
        s = 1.0 if (T_hat - A_hat) > 0 else (-1.0 if (T_hat - A_hat) < 0 else 0.0)

    # corr (Po) if not supplied
    if not isinstance(corr, (int, float)):
        corr = (TP + TN) / N
    corr = max(0.0, min(1.0, float(corr)))  # clamp

    mix_mag = alpha * (1.0 - corr**gamma) + (1.0 - alpha) * (1.0 - corr)
    return s * mix_mag

def compute_mixer_magnitude(
    corr: Optional[float],
    confusion: Optional[Dict[str, int]],
    alpha: float = 0.7,
    gamma: float = 2.0,
) -> Optional[float]:
    """
    Unsigned mixer magnitude in [0,1]:
        M_mix = α*(1 - corr**γ) + (1 - α)*(1 - corr)
    Uses provided corr if available; otherwise computes Po from confusion.
    """
    if confusion is None and not isinstance(corr, (int, float)):
        return None
    if not isinstance(corr, (int, float)) and confusion is not None:
        TP, TN, *_ = (confusion[k] for k in ("TP","TN","FP","FN","N"))
        N = confusion["N"]
        corr = (TP + TN) / N if N > 0 else None
    if not isinstance(corr, (int, float)):
        return None
    corr = max(0.0, min(1.0, float(corr)))
    return alpha * (1.0 - corr**gamma) + (1.0 - alpha) * (1.0 - corr)

# ---- I/O helpers -----------------------------------------------------------

def infer_dataset_name(run_dir: Path) -> str:
    """
    Infer a canonical dataset name from the directory that contains summary.json.
    Priority:
      1) 'samples_<name>[_think]' at the end of folder name.
      2) Any 'samples_<name>_' occurrence.
      3) Pattern '<dataset>_think_*' without 'samples_'.
      4) Fallback: folder name itself (covers mvbench/summary.json).
    """
    name = run_dir.name

    m = re.search(r"samples_([^/]+?)(?:_think)?$", name)
    if m:
        n = m.group(1)
        if n.endswith("_think"):
            n = n[:-6]
        return n

    m2 = re.search(r"samples_([^_][^/]*?)(?:_|$)", name)
    if m2:
        n = m2.group(1)
        if n.endswith("_think"):
            n = n[:-6]
        return n

    m3 = re.search(r"^(.*?)(?:_think.*)$", name)
    if m3:
        return m3.group(1)

    return name

def read_dataset_summary(summary_path: Path, mixer_alpha: float, mixer_gamma: float) -> Dict[str, Any]:
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    extracted = {k: data.get(k, None) for k in FOUR_KEYS}

    # original consistency
    a = extracted.get("answer_parsed_accuracy")
    t = extracted.get("thinking_parsed_accuracy")
    extracted["consistency_score"] = compute_consistency(a, t)

    # confusion from combinations (collapsed over LMMS)
    confusion = parse_ans_think_confusion(data.get("combinations", {}))

    # raw agreement (Po) as corr if missing; otherwise use provided correlation
    if isinstance(extracted.get("answer_to_thinking_correlation"), (int, float)):
        corr_used = float(extracted["answer_to_thinking_correlation"])
    else:
        corr_used = None  # compute from confusion inside the mixer/magnitude

    # signed metrics
    extracted["signed_disagreement"] = compute_signed_disagreement(a, t, confusion)
    extracted["signed_disagreement_mixer"] = compute_signed_disagreement_mixer(
        a, t, confusion, corr=corr_used, alpha=mixer_alpha, gamma=mixer_gamma
    )

    # magnitudes (always useful, especially on ties)
    extracted["disagreement_rate"] = compute_disagreement_rate(confusion)
    extracted["mixer_magnitude"] = compute_mixer_magnitude(
        corr_used, confusion, alpha=mixer_alpha, gamma=mixer_gamma
    )

    # consistency error/accuracy among answer-correct
    extracted["consistency_error"] = compute_consistency_error(confusion)
    extracted["consistency_accuracy"] = compute_consistency_accuracy(confusion)

    return extracted

# ---- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory that contains benchmark folders and run subfolders with summary.json.",
    )
    parser.add_argument("--alpha", type=float, default=0.7, help="Mixer α in [0,1] (default 0.7).")
    parser.add_argument("--gamma", type=float, default=2.0, help="Mixer γ > 1 (default 2.0).")
    args = parser.parse_args()

    root = Path(args.input_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--input_dir not found or not a directory: {root}")

    output_summary_path = root / "summary.json"

    # 1) Find all summary.json files recursively; skip the aggregate we write.
    candidate_summaries: List[Path] = []
    for p in root.rglob("summary.json"):
        if p.resolve() == output_summary_path.resolve():
            continue  # don't read our own output
        candidate_summaries.append(p)

    if not candidate_summaries:
        print("[WARN] No summary.json files found under:", root)

    # 2) Group by inferred dataset name; choose latest by mtime
    grouped: Dict[str, List[Tuple[Path, float]]] = {}
    for sp in candidate_summaries:
        run_dir = sp.parent
        ds_name = infer_dataset_name(run_dir)
        mtime = sp.stat().st_mtime
        grouped.setdefault(ds_name, []).append((sp, mtime))

    chosen: Dict[str, Path] = {}
    for ds_name, items in grouped.items():
        latest = max(items, key=lambda it: it[1])  # by mtime
        chosen[ds_name] = latest[0]

    # 3) Build datasets dict with metrics + provenance
    datasets: Dict[str, Dict[str, Any]] = {}
    # Collectors for overall averages
    buckets: Dict[str, List[float]] = {
        "consistency_score": [],
        "answer_to_thinking_correlation": [],
        "signed_disagreement": [],
        "signed_disagreement_mixer": [],
        "disagreement_rate": [],
        "mixer_magnitude": [],
        "consistency_error": [],
        "consistency_accuracy": [],
    }

    for ds_name, spath in sorted(chosen.items(), key=lambda kv: kv[0].lower()):
        try:
            metrics = read_dataset_summary(spath, mixer_alpha=args.alpha, mixer_gamma=args.gamma)
        except Exception as e:
            print(f"[WARN] Failed to read {spath}: {e}")
            continue

        # provenance
        metrics["source_path"] = str(spath)
        metrics["mtime"] = spath.stat().st_mtime

        datasets[ds_name] = metrics

        # collect for overall stats
        for key in buckets.keys():
            val = metrics.get(key)
            if isinstance(val, (int, float)):
                buckets[key].append(float(val))

    def avg(xs: List[float]) -> Optional[float]:
        return float(sum(xs) / len(xs)) if xs else None

    aggregated = {
        "overall": {
            "num_datasets": len(datasets),
            "average_consistency_score": avg(buckets["consistency_score"]),
            "average_answer_to_thinking_correlation": avg(buckets["answer_to_thinking_correlation"]),
            "average_signed_disagreement": avg(buckets["signed_disagreement"]),
            "average_signed_disagreement_mixer": avg(buckets["signed_disagreement_mixer"]),
            "average_disagreement_rate": avg(buckets["disagreement_rate"]),
            "average_mixer_magnitude": avg(buckets["mixer_magnitude"]),
            "average_consistency_error": avg(buckets["consistency_error"]),
            "average_consistency_accuracy": avg(buckets["consistency_accuracy"]),
            "mixer_hyperparams": {"alpha": args.alpha, "gamma": args.gamma},
        },
        "datasets": datasets,
    }

    # 4) Write consolidated summary.json
    with open(output_summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)

    # 5) Terminal output
    hdr = (
        "  {name:>28} | {cons:^12} | {corr:^12} | {sd:^14} | {smix:^18} | {dr:^12} | {cerr:^12} | {cacc:^12} | {mm:^12}"
        .format(
            name="Dataset",
            cons="Consistency",
            corr="Corr (A↔T)",
            sd="SignedDisagree",
            smix=f"SignedMix(α={args.alpha:g},γ={args.gamma:g})",
            dr="DisagreeRate",
            cerr="ConsErr(A✓)",
            cacc="ConsAcc(A✓)",
            mm="MixMagnitude",
        )
    )
    print("\nPer-dataset metrics:")
    print(hdr)
    print("  " + "-" * 28 + "-+-" + "-" * 12 + "-+-" + "-" * 12 + "-+-" + "-" * 14 + "-+-" + "-" * 18 + "-+-" + "-" * 12 + "-+-" + "-" * 12 + "-+-" + "-" * 12)
    def fmt(x: Any) -> str:
        return "NA" if (x is None or not isinstance(x, (int, float))) else f"{x:.4f}"
    for name in sorted(datasets.keys(), key=str.lower):
        m = datasets[name]
        print(
            f"  {name:>28} | {fmt(m.get('consistency_score')):^12} | "
            f"{fmt(m.get('answer_to_thinking_correlation')):^12} | "
            f"{fmt(m.get('signed_disagreement')):^14} | "
            f"{fmt(m.get('signed_disagreement_mixer')):^18} | "
            f"{fmt(m.get('disagreement_rate')):^12} | "
            f"{fmt(m.get('consistency_error')):^12} | "
            f"{fmt(m.get('consistency_accuracy')):^12} | "
            f"{fmt(m.get('mixer_magnitude')):^12}"
        )

    def fmt_overall(x: Optional[float]) -> str:
        return "NA" if x is None else f"{x:.4f}"

    print("\nOverall averages:")
    print(f"  - consistency_score (yours)          : {fmt_overall(aggregated['overall']['average_consistency_score'])}")
    print(f"  - answer_to_thinking_correlation     : {fmt_overall(aggregated['overall']['average_answer_to_thinking_correlation'])}")
    print(f"  - signed_disagreement                : {fmt_overall(aggregated['overall']['average_signed_disagreement'])}")
    print(f"  - signed_disagreement_mixer (α,γ)    : {fmt_overall(aggregated['overall']['average_signed_disagreement_mixer'])}")
    print(f"  - disagreement_rate                  : {fmt_overall(aggregated['overall']['average_disagreement_rate'])}")
    print(f"  - mixer_magnitude (α,γ)              : {fmt_overall(aggregated['overall']['average_mixer_magnitude'])}")
    print(f"  - consistency_error (A✓ subset)      : {fmt_overall(aggregated['overall']['average_consistency_error'])}")
    print(f"  - consistency_accuracy (A✓ subset)   : {fmt_overall(aggregated['overall']['average_consistency_accuracy'])}")

    print(f"\nWrote aggregated summary to: {output_summary_path}")

if __name__ == "__main__":
    main()
