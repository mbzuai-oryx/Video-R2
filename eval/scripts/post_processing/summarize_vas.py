#!/usr/bin/env python3
import argparse, json, os, glob

def find_key(d, target):
    """Recursively find all values for key `target` in nested dicts/lists."""
    if isinstance(d, dict):
        for k, v in d.items():
            if k == target:
                yield v
            else:
                yield from find_key(v, target)
    elif isinstance(d, list):
        for item in d:
            yield from find_key(item, target)

def main():
    ap = argparse.ArgumentParser(description="Compute average __attention_to_video_score per JSONL file.")
    ap.add_argument("--input_dir", required=True, help="Directory containing per_sample*.jsonl files")
    args = ap.parse_args()

    pattern = os.path.join(args.input_dir, "per_sample*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        print("No files found matching per_sample*.jsonl in", args.input_dir)
        return

    for fp in files:
        scores = []
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # collect all occurrences; fall back to top-level if present
                vals = list(find_key(obj, "__attention_to_video_score"))
                if not vals and "__attention_to_video_score" in obj:
                    vals = [obj["__attention_to_video_score"]]
                for v in vals:
                    try:
                        scores.append(float(v))
                    except (TypeError, ValueError):
                        pass

        if scores:
            avg = sum(scores) / len(scores)
            print(f"{os.path.basename(fp)}\t{avg:.6f}  (n={len(scores)})")
        else:
            print(f"{os.path.basename(fp)}\tNo __attention_to_video_score found")

if __name__ == "__main__":
    main()
