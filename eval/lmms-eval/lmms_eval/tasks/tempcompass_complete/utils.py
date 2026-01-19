import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from loguru import logger as eval_logger
from lmms_eval.utils import extract_answer

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "tempcompass_complete.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def tempcompass_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["video_id"] + ".mp4"
    video_path = os.path.join(cache_dir, "videos", video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    elif os.path.exists(video_path.replace("mp4", "webm")):
        video_path = video_path.replace("mp4", "webm")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check {doc}")
    return [video_path]


def tempcompass_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    question = question + "\n" + option
    post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    full_prompt = question + "\n" + post_prompt
    return full_prompt


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    
    # Remove known prefixes
    for prefix in answer_prefixes:
        if s.startswith(prefix):
            s = s[len(prefix):].strip()

    # Case 1: Match (C), C), C. or C: at the beginning
    match = re.match(r"^\(?([A-E])\)?[:.]?", s)
    if match:
        return match.group(1)

    # Case 2: Look for a single A–E anywhere if text is short
    if len(s.split()) <= 10:
        match = re.search(r"\b([A-E])\b", s)
        if match:
            return match.group(1)

    # Case 3: Handle embedded ")" (e.g., "Answer) ...")
    if ")" in s:
        index = s.index(")")
        if index > 0 and s[index - 1].upper() in "ABCDE":
            return s[index - 1].upper()

    # Case 4: If nothing else, return original
    return s


def tempcompass_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case tempcompass score), value: metric value
    """
    pred = extract_answer(results[0])
    pred_ans = extract_characters_regex(pred)

    data_dict = {
        "uuid": doc["uuid"],
        "pred_answer": pred_ans,
        "answer": doc["answer"],
    }

    return {"tempcompass_perception_score": data_dict}


def tempcompass_mcq_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        Overall accuracy score
    """
    total_correct = sum(result["pred_answer"] == result["answer"] for result in results)
    total_answered = len(results)

    overall_score = 100 * total_correct / total_answered if total_answered > 0 else 0
    eval_logger.info(f"Overall Performance: {overall_score:.1f}%")

    return overall_score


def tempcompass_multi_binary_aggregate_results(results):
    grouped = defaultdict(list)
    for result in results:
        grouped[result["uuid"]].append(result)

    total_correct = 0
    total_answered = 0
    for qid, group in grouped.items():
        all_correct = all(g["pred_answer"] == g["answer"] for g in group)
        total_answered += 1
        if all_correct:
            total_correct += 1

    overall_score = 100 * total_correct / total_answered if total_answered > 0 else 0
    eval_logger.info(f"Overall Performance: {overall_score:.1f}%")

    return overall_score
