import os
import re

import torch
from openai import OpenAI
from src.train.reward_utils import *
from src.train.temporal_grounding_reward import calculate_grounding_reward

OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
_MODEL = os.environ.get("SERVED_MODEL", None)

# API timeout of 10sec to avoid NCCL errors because of large timeouts
_oai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE, timeout=10.0)


def accuracy_reward(completions, assistant, **kwargs):
    solutions = [a["content"] for a in assistant]
    contents = [completion[0]["content"] for completion in completions]

    rewards = []
    for content, sol in zip(contents, solutions):
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            question_type = infer_question_type(gt_ans)

            if question_type == "multiple choice":
                gt_letter = normalize_mcq(gt_ans)
                out_letter = normalize_mcq(output_ans)
                reward = 1.0 if out_letter == gt_letter else 0.0

            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    reward = 0.0
                else:
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    else:
                        reward = (
                            1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
                        )

            elif question_type == "OCR":
                error_rate = wer(gt_ans, output_ans)
                reward = max(0.0, min(1.0, 1 - error_rate))

            elif question_type == "free-form":
                score = compute_rouge_score(gt_ans, output_ans)
                reward = max(0.0, min(1.0, score))

            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    reward = 0.0
                else:
                    rel_diff = (abs(out_number - gt_number) + 1e-9) / (
                        abs(gt_number) + 1e-9
                    )
                    reward = max(0.0, min(1.0, 1 - rel_diff))

            else:
                reward = 0.0

        except Exception as e:
            print(f"Error in reward_fn for inferred question_type: {e}")
            reward = 0.0

        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [
        re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents
    ]
    rewards = [1.0 if match else 0.0 for match in matches]

    return rewards


def temporal_grounding_sentence_embedding_consistency_reward(completions, assistant, temporal_grounding, sentence_model, buffer_seconds, similarity_threshold, **kwargs):
    questions = [a[1]["content"][1]["text"] for a in kwargs["prompts"]]
    predictions = [completion[0]["content"] for completion in completions]
    temporal_grnd_list = [a for a in temporal_grounding]

    rewards = []
    for question, prediction, temporal_grnd in zip(questions, predictions, temporal_grnd_list):
        try:
            # Step 1: Calculate temporal grounding sentence embedding reward

            # Pass the sentence_model instance to calculate_reward
            with torch.no_grad():
                grounding_reward = calculate_grounding_reward(
                    prediction_text=prediction,
                    temporal_grnd=temporal_grnd,
                    sentence_model=sentence_model,
                    oai_client=_oai_client,
                    served_model=_MODEL,
                    buffer_seconds=buffer_seconds,
                    similarity_threshold=similarity_threshold,
                )

            # Step 2: Calculate consistency reward
            reasoninig = extract_reasoning(prediction)
            answer = extract_answer(prediction)
            llm_score_instructions = (
                "You are a meticulous auditor. Determine whether the reasoning (THINK) and the final answer (ANSWER) "
                "are logically consistent with each other for the given question. "
                "Ignore style, verbosity, or extra details; focus strictly on whether the conclusion in THINK matches "
                "and supports the final ANSWER for the same question."
            )
            user_msg = (
                "QUESTION:\n"
                f"{question.strip() if question else '(none)'}\n\n"
                "THINK (model's internal reasoning):\n"
                f"{reasoninig.strip() if reasoninig else '(missing)'}\n\n"
                "ANSWER (model's final answer):\n"
                f"{answer.strip() if answer else '(missing)'}\n\n"
                "TASK:\n"
                "1) Output ONLY one of the TRUE or FALSE on the first line.\n"
                "   - TRUE  => THINK and ANSWER are consistent and the same conclusion.\n"
                "   - FALSE => THINK contradicts or does not support the ANSWER (e.g., mismatched conclusion).\n"
                "2) On the next line(s), give a brief justification (1-3 sentences)."
            )
            output_text = chat(
                _oai_client, _MODEL, llm_score_instructions, user_msg
            )
            decision, reason = parse_decision(output_text)
            if decision is True:
                consistency_reward = 1
            else:
                consistency_reward = 0

            # Step 3: Total reward would be the multiplication of both rewards 
            # (for example, consider temporal grounding reward only if the prediction is consistent)
            reward = consistency_reward * grounding_reward
        
        except Exception as e:
            print(f"Error in temporal_grounding_sentence_embedding_consistency_reward: {e}")
            reward = 0.0
        
        rewards.append(reward)

    return rewards
