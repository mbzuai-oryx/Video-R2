"""
run_all_benchmarks.py

Drive evaluation for multiple video QA benchmarks from a single predictions folder.

Usage:
    python run_all_benchmarks.py \
        --input_dir /path/to/preds \
        --output_dir /path/to/results \
        --model "/vast/users/salman.khan/video_reasoning/checkpoints/Qwen/Qwen3-Next-80B-A3B-Instruct" \
        --tp 4 \
        --batch_size 8

Notes:
- Only files that contain "samples" and end with ".jsonl" are considered.
- MVBench is handled specially: we pass --input_dir to mvbench.py once.
- Video-MMMU-v2 is routed to video_mmmu.py.
- Skips already-completed runs (summary.json exists) unless --overwrite is set.
- Creates a consolidated all_summaries.json at the end (best-effort).
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess


# -----------------------
# Routing / Normalization
# -----------------------
BENCHMAP = {
    "longvideobench": {
        "script": "longvideobench.py",
        "mode": "per_file",  # pass --sample_jsonl
        "patterns": [r"^longvideobench", r"longvideobench"],
    },
    "minerva": {
        "script": "minerva.py",
        "mode": "per_file",
        "patterns": [r"^minerva", r"minerva"],
    },
    "mlvu": {
        "script": "mlvu.py",
        "mode": "per_file",
        "patterns": [r"^mlvu", r"mlvu"],
    },
    "mmvu": {
        "script": "mmvu.py",
        "mode": "per_file",
        "patterns": [r"^mmvu", r"mmvu"],
    },
    "mvbench": {
        "script": "mvbench.py",
        "mode": "directory",  # pass --input_dir once
        "patterns": [r"^mvbench", r"mvbench"],
    },
    "scivideobench": {
        "script": "scivideobench.py",
        "mode": "per_file",
        "patterns": [r"^scivideobench", r"scivideobench"],
    },
    "tempcompass": {
        "script": "tempcompass.py",
        "mode": "per_file",
        "patterns": [r"^tempcompass", r"tempcompass"],
    },
    "video_mmmu": {
        "script": "video_mmmu.py",
        "mode": "per_file",
        "patterns": [r"^video_mmmu", r"video_mmmu"],  # includes v2 variants too
        "normalize": lambda name: "video_mmmu",  # route v2 -> video_mmmu
    },
    "videomathqa": {
        "script": "videomathqa.py",
        "mode": "per_file",
        "patterns": [r"^videomathqa", r"videomathqa"],
    },
    "vsibench": {
        "script": "vsibench.py",
        "mode": "per_file",
        "patterns": [r"^vsibench", r"vsibench"],
    },
    "videomme": {
        "script": "videomme.py",
        "mode": "per_file",
        "patterns": [r"^videomme", r"videomme"],
    },
}


def normalize_benchmark_key(filename: str) -> str:
    base = filename.lower()
    for bench, meta in BENCHMAP.items():
        for pat in meta["patterns"]:
            if re.search(pat, base):
                # Special case: route video_mmmu_v2 to video_mmmu
                if bench == "video_mmmu":
                    return "video_mmmu"
                return bench
    return ""


def discover_samples(input_dir: Path) -> Tuple[Dict[str, List[Path]], bool]:
    """
    Return:
      - dict: benchmark -> list of jsonl paths (only for per_file benchmarks)
      - mvbench_present: whether mvbench shards exist in input_dir
    """
    per_file: Dict[str, List[Path]] = {k: [] for k, v in BENCHMAP.items() if v["mode"] == "per_file"}
    mvbench_present = False

    for p in input_dir.rglob("*.jsonl"):
        name = p.name.lower()
        if "samples" not in name:
            continue
        bench = normalize_benchmark_key(name)
        if not bench:
            continue
        meta = BENCHMAP[bench]
        if meta["mode"] == "directory":
            if "mvbench" in bench:
                mvbench_present = True
            continue
        per_file[bench].append(p)

    # Clean empty keys
    per_file = {k: v for k, v in per_file.items() if v}
    return per_file, mvbench_present


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def has_completed(output_dir: Path) -> bool:
    return (output_dir / "summary.json").exists()


def run_cmd(cmd: List[str], dry_run=False) -> int:
    if dry_run:
        print("[DRY-RUN]", " ".join(cmd))
        return 0
    print("[RUN]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--model", default=None, help="Forwarded to evaluators as --inference_model")
    ap.add_argument("--tp", default=None, help="Forwarded to evaluators as --tp")
    ap.add_argument("--batch_size", default=None, help="Forwarded to evaluators as --batch_size")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    per_file, mvbench_present = discover_samples(args.input_dir)

    print("== Discovery ==")
    for bench, files in per_file.items():
        print(f"{bench}: {len(files)} jsonl")
    print(f"mvbench_present: {mvbench_present}")
    print()

    # ---------
    # Run per-file benchmarks
    # ---------
    results_index = []
    for bench, files in per_file.items():
        script = BENCHMAP[bench]["script"]
        for jsonl_path in sorted(files):
            stem = jsonl_path.stem
            out_dir = args.output_dir / bench / stem
            ensure_dir(out_dir)

            if has_completed(out_dir) and not args.overwrite:
                print(f"[SKIP] {bench} :: {stem} (summary.json exists)")
                continue

            cmd = [sys.executable, script,
                   "--sample_jsonl", str(jsonl_path),
                   "--output_dir", str(out_dir)]
            if args.model:
                cmd += ["--inference_model", args.model]
            if args.tp:
                cmd += ["--tp", str(args.tp)]
            if args.batch_size:
                cmd += ["--batch_size", str(args.batch_size)]

            rc = run_cmd(cmd, dry_run=args.dry_run)
            results_index.append({
                "benchmark": bench,
                "input": str(jsonl_path),
                "output_dir": str(out_dir),
                "returncode": rc,
            })

    # ---------
    # Run MVBench once (directory mode)
    # ---------
    if mvbench_present:
        bench = "mvbench"
        script = BENCHMAP[bench]["script"]
        out_dir = args.output_dir / bench
        ensure_dir(out_dir)

        if has_completed(out_dir) and not args.overwrite:
            print(f"[SKIP] mvbench (summary.json exists)")
        else:
            cmd = [sys.executable, script,
                   "--input_dir", str(args.input_dir),
                   "--output_dir", str(out_dir)]
            if args.model:
                cmd += ["--inference_model", args.model]
            if args.tp:
                cmd += ["--tp", str(args.tp)]
            if args.batch_size:
                cmd += ["--batch_size", str(args.batch_size)]

            rc = run_cmd(cmd, dry_run=args.dry_run)
            results_index.append({
                "benchmark": bench,
                "input": str(args.input_dir),
                "output_dir": str(out_dir),
                "returncode": rc,
            })

    # ---------
    # Consolidate summaries (best-effort)
    # ---------
    all_summaries = []
    for bench_dir in (args.output_dir).glob("*"):
        if not bench_dir.is_dir():
            continue
        # mvbench may write summary.json at bench root; per-file benches may have subfolders
        candidates = []
        root_summary = bench_dir / "summary.json"
        if root_summary.exists():
            candidates.append(root_summary)
        for sub in bench_dir.glob("*/summary.json"):
            candidates.append(sub)

        for s in candidates:
            try:
                with open(s, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                data = {"_error": str(e)}
            all_summaries.append({
                "benchmark": bench_dir.name,
                "summary_path": str(s),
                "summary": data
            })

    index_path = args.output_dir / "all_summaries.json"
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump({
                "runs": results_index,
                "summaries": all_summaries,
            }, f, ensure_ascii=False, indent=2)
        print(f"\nWrote consolidated index: {index_path}")
    except Exception as e:
        print(f"[WARN] Failed to write consolidated index: {e}")


if __name__ == "__main__":
    main()
