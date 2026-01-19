"""
Evaluate 'attention_to_video' using only the dataset 'input' and model 'response'.

Additions in this version:
- mvbench concatenation: all mvbench *samples*.jsonl are concatenated and treated as a single task/file.
- --dry_run: prints what the script would do (files, counts, batch sizes, outputs) without calling the LLM.
- Sampling params: temperature=0.7, top_p=0.8, top_k=20, max_tokens=256.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

LLM = None
SamplingParams = None
AutoTokenizer = None

JSON_OBJ_RE = re.compile(r"\{[\s\S]*?\}")

@dataclass
class UnifiedSample:
    bench: str
    file_path: str
    raw: Dict[str, Any]
    uid: str
    input_text: str
    response_text: str

def find_sample_files(input_dir: str) -> List[str]:
    out = []
    for fn in os.listdir(input_dir):
        if not fn.endswith('.jsonl'):
            continue
        if 'samples' not in fn:
            continue
        out.append(os.path.join(input_dir, fn))
    return sorted(out)

KNOWN_PREFIXES = [
    'longvideobench', 'minerva', 'mlvu', 'mmvu', 'mvbench', 'scivideobench',
    'tempcompass', 'video_mmmu', 'videomathqa', 'vsibench', 'videomme'
]

def infer_bench_from_filename(basename: str) -> str:
    name = basename.replace('.jsonl', '')
    if "video_mmmu_v2" in name:
        return "video_mmmu_v2"
    parts = name.split('_')
    if len(parts) >= 2 and parts[0] == 'video' and parts[1] == 'mmmu':
        return 'video_mmmu'
    first = parts[0]
    for p in KNOWN_PREFIXES:
        if first.startswith(p) or name.startswith(p):
            return p
    if 'mvbench' in name: return 'mvbench'
    if 'videomathqa' in name: return 'videomathqa'
    if 'videomme' in name: return 'videomme'
    return first

def _extract_uid(row: Dict[str, Any]) -> str:
    for key in ['id', 'qid', 'question_id', 'doc_id', 'uid']:
        if key in row:
            return str(row[key])
    if 'doc' in row and isinstance(row['doc'], dict):
        for key in ['id', 'question_id']:
            if key in row['doc']:
                return str(row['doc'][key])
    return ''

def _deep_first_string(x) -> str:
    seen = set()
    while isinstance(x, list) and x:
        oid = id(x)
        if oid in seen:
            break
        seen.add(oid)
        x = x[0]
    return x if isinstance(x, str) else ""

def _extract_response(row: Dict[str, Any]) -> str:
    for key in ['filtered_resps', 'resps']:
        if key in row:
            val = row[key]
            if isinstance(val, str):
                return val
            if isinstance(val, list):
                s = _deep_first_string(val)
                if s:
                    return s
    for key in ['response', 'prediction', 'raw_output', 'pred_answer']:
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""

def parse_row_to_unified(bench: str, file_path: str, row: Dict[str, Any]) -> Optional[UnifiedSample]:
    input_text = row.get('input')
    if not isinstance(input_text, str):
        return None
    response_text = _extract_response(row)
    uid = _extract_uid(row)
    return UnifiedSample(
        bench=bench,
        file_path=file_path,
        raw=row,
        uid=uid,
        input_text=input_text,
        response_text=response_text,
    )

SYSTEM_PROMPT = """You are an expert judge of **claimed visual grounding** in video QA chain-of-thought.
You will receive an Input (question/options/prompt) and a model Response that includes <think>…</think> and <answer>…</answer> tags.

Your task: Evaluate how much the reasoning inside <think> and </think> only *claims* to rely on concrete visual evidence from the video versus using text patterns, prior knowledge, or guessing. You do not have the video; score what the reasoning asserts, not whether those assertions are true or the final answer is correct.

What counts as visual grounding signals (positive cues):
- Concrete, specific references to visible entities and attributes (colors, positions, counts, on-screen text/OCR).
- Spatio-temporal descriptions tied to the video (“first/then,” “left/right,” “foreground/background,” scene changes).
- Actions/interactions clearly described as seen (“picks up,” “points at,” “walks behind”).
- Visual logic chained to the question (“Because the sign reads ‘Gate B’ and the arrow points left…”).
- Mentions like “the video shows,” “the video states,” or “the video indicates…” count as *moderate visual grounding* **if** they reference specific factual content (e.g., numbers, object names, or events), even without spatial detail.

What counts as non-visual or weak grounding (negative cues):
- Generic mentions of “the video” with no specific content (“the video talks about cars”).
- Option pattern matching, stereotypes/common knowledge, or generic phrases (“based on the context,” “usually…”).
- Restating the question or options with “I see” but no concrete visual detail.
- Timestamp name-dropping without linked content, irrelevant flourish, or contradictions.

Edge cases:
- If multiple <think> blocks exist, judge the first.
- If <think> is missing or empty → score 0.
- Ignore <answer> when scoring.

Scoring (0–10, integer):
- 10 (Excellent): Dense, highly specific, step-by-step visual reasoning directly addressing the question.
- 7–9 (Strong): Clearly video-based with several specific claims tied to what’s shown, even if phrased verbally (“the video states…”).
- 4–6 (Moderate): Claims to recall or cite specific video facts (numbers, entities, or comparisons) but lacks spatial or temporal detail.
- 1–3 (Weak): Minimal visual content; mostly generic or option-pattern reasoning with only token mentions of the video.
- 0 (None): No visual grounding; guessing, meta-reasoning only, or <think> absent.

Output: Return JSON only with keys "score" (0–10 integer) and "rationale" (2–3 concise sentences citing the most important cues; do not mention lack of video).
"""

USER_TEMPLATE = """
Evaluate ATTENTION-TO-VIDEO in the model’s reasoning. Score only the text inside <think>…</think>, not the final <answer> and not answer correctness.

You do not have access to the video. Judge whether the reasoning *claims* to use concrete visual evidence (what is seen, where it appears, how it changes) versus relying on generic patterns, options, or prior knowledge.

Use this rubric (0–10):
- 10: Dense, specific visual descriptions (entities, attributes, counts, on-screen text, spatial relations, temporal order) directly supporting the question.
- 7–9: Clearly video-based reasoning with multiple specific claims tied to what the video shows, even if phrased as “the video states…” or “the video shows…”.
- 4–6: Claims to recall or cite specific video facts (numbers, labels, or entities) but lacks spatial or visual detail.
- 1–3: Minimal visual content; generic “the video talks about…” or option-based logic.
- 0: No visual grounding; guessing, meta-reasoning only, or missing <think> block.

Edge handling: If <think> is missing/empty → score 0.

Input:
{input_text}

Response:
{response_text}

Reply ONLY with JSON like: {{\"score\": <0-10 int>, \"rationale\": \"...\"}}
"""

def build_messages(sample: UnifiedSample) -> List[Dict[str, str]]:
    user = USER_TEMPLATE.format(
        input_text=sample.input_text or "",
        response_text=sample.response_text or "",
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

def parse_llm_json(text: str) -> Tuple[Optional[int], str]:
    if not isinstance(text, str):
        return None, ""
    m = JSON_OBJ_RE.search(text)
    if not m:
        return None, ""
    try:
        data = json.loads(m.group(0))
        score = data.get('score')
        if isinstance(score, (int, float)):
            s = int(round(float(score)))
            s = max(0, min(10, s))
        else:
            s = None
        rationale = data.get('rationale') if isinstance(data.get('rationale'), str) else ""
        return s, rationale
    except Exception:
        return None, ""

def load_vllm(model: str, tp: int):
    global LLM, SamplingParams, AutoTokenizer
    from vllm import LLM as _LLM, SamplingParams as _SamplingParams
    from transformers import AutoTokenizer as _AutoTokenizer
    LLM = _LLM
    SamplingParams = _SamplingParams
    AutoTokenizer = _AutoTokenizer
    llm = LLM(model=model, tensor_parallel_size=max(1, int(tp)))
    tok = _AutoTokenizer.from_pretrained(model)
    return llm, tok

def batch_generate(llm, tok, messages_list: List[List[Dict[str, str]]], batch_size: int) -> List[str]:
    prompts = [
        tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for msgs in messages_list
    ]
    outs: List[str] = []
    sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=256)
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i+batch_size]
        results = llm.generate(chunk, sp)
        for res in results:
            text = res.outputs[0].text if res.outputs else ""
            outs.append(text)
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=-1, help='-1 => per-file size (mvbench => all files combined)')
    ap.add_argument('--overwrite', action='store_true', help='Re-run even if outputs already exist for a benchmark.')
    ap.add_argument('--dry_run', action='store_true', help='Print planned actions without running the LLM')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = find_sample_files(args.input_dir)
    if not files:
        print('No *samples*.jsonl files found in input_dir', file=sys.stderr)
        sys.exit(1)

    bench_to_files: Dict[str, List[str]] = defaultdict(list)
    for fp in files:
        bench = infer_bench_from_filename(os.path.basename(fp))
        bench_to_files[bench].append(fp)

    plan = {b: {"n_files": len(fps)} for b, fps in bench_to_files.items()}

    llm = tok = None
    per_bench_scores: Dict[str, List[int]] = defaultdict(list)
    writers: Dict[str, Any] = {}

    def get_writer(bench: str):
        if bench in writers:
            return writers[bench]
        path = os.path.join(args.output_dir, f'per_sample_{bench}.jsonl')
        fh = open(path, 'w', encoding='utf-8')
        writers[bench] = fh
        return fh

    try:
        for bench, fps in bench_to_files.items():
            out_path = os.path.join(args.output_dir, f'per_sample_{bench}.jsonl')
            plan.setdefault(bench, {})["output_per_sample"] = out_path
            if (not args.overwrite) and os.path.exists(out_path):
                plan[bench]["skipped"] = True
                if args.dry_run:
                    continue
                print(f"[skip] {bench}: {out_path} exists (use --overwrite to re-run)")
                continue
            raw_rows: List[Dict[str, Any]] = []
            # mvbench: concatenate all files
            for fp in fps:
                with open(fp, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        raw_rows.append(row)

            unified: List[UnifiedSample] = []
            # Use first file path as provenance indicator for all
            for src_fp, row in zip([fps[0]] * len(raw_rows), raw_rows):
                sample = parse_row_to_unified(bench, src_fp, row)
                if sample:
                    unified.append(sample)

            total_items = len(unified)
            plan.setdefault(bench, {}).update({"n_items": total_items})
            B = total_items if args.batch_size == -1 else max(1, args.batch_size)
            plan[bench]["batch_size"] = B
            plan[bench]["output_per_sample"] = os.path.join(args.output_dir, f'per_sample_{bench}.jsonl')

            if args.dry_run:
                continue

            if not unified:
                continue

            if llm is None:
                llm, tok = load_vllm(args.model, args.tp)

            messages_list = [build_messages(s) for s in unified]
            raw_outputs = batch_generate(llm, tok, messages_list, B)

            writer = get_writer(bench)
            for sample, prompt_msgs, raw in zip(unified, messages_list, raw_outputs):
                score, rationale = parse_llm_json(raw)
                if score is None:
                    fallback_msgs = prompt_msgs[:-1] + [{
                        "role": "user",
                        "content": prompt_msgs[-1]["content"] + "\n\nIMPORTANT: Reply with JSON only."
                    }]
                    raw_retry = batch_generate(llm, tok, [fallback_msgs], 1)[0]
                    raw = raw_retry or raw
                    score, rationale = parse_llm_json(raw_retry)
                if score is None:
                    score, rationale = 0, "Failed to parse JSON; defaulting to 0."

                per_bench_scores[bench].append(score)
                out_row = dict(sample.raw)
                out_row.update({
                    "__bench": sample.bench,
                    "__file": sample.file_path,
                    "__uid": sample.uid,
                    "__attention_to_video_score": score,
                    "__attention_to_video_rationale": rationale,
                    "__llm_raw_response": raw,
                    "__llm_prompt": prompt_msgs,
                })
                writer.write(json.dumps(out_row, ensure_ascii=False) + "\n")

        if args.dry_run:
            print("==== DRY RUN PLAN ====")
            for b, info in sorted(plan.items()):
                print(f"Benchmark: {b}")
                for k, v in info.items():
                    print(f"  - {k}: {v}")
            print("(No LLM calls were made.)")
            return

        summary: Dict[str, Any] = {"per_benchmark": {}, "overall": {}}
        all_scores: List[int] = []
        for bench, scores in per_bench_scores.items():
            if not scores:
                continue
            avg = sum(scores) / len(scores)
            dist = {
                "0-2":  sum(1 for s in scores if 0 <= s <= 2),
                "3-5":  sum(1 for s in scores if 3 <= s <= 5),
                "6-8":  sum(1 for s in scores if 6 <= s <= 8),
                "9-10": sum(1 for s in scores if 9 <= s <= 10),
            }
            summary["per_benchmark"][bench] = {
                "n": len(scores),
                "avg": round(avg, 3),
                "min": min(scores),
                "max": max(scores),
                "distribution": dist,
            }
            all_scores.extend(scores)
        if all_scores:
            overall = {
                "n": len(all_scores),
                "avg": round(sum(all_scores) / len(all_scores), 3),
                "min": min(all_scores),
                "max": max(all_scores),
            }
            summary["overall"] = overall

        with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf-8') as fsum:
            json.dump(summary, fsum, ensure_ascii=False, indent=2)

    finally:
        for fh in writers.values():
            try:
                fh.close()
            except Exception:
                pass

if __name__ == '__main__':
    main()
