from typing import Dict, List, Optional


def build_answer_parsing_message(options: Optional[List[str]], content_text: str, is_mcq_task: bool = True) -> List[Dict[str, str]]:
    sys_prompt = (
        "You are a deterministic parsing agen.\n"
        "Task: Read ONLY the provided text and emit a SINGLE-LINE answer in the exact format requested.\n"
        "Hard rules (apply all):\n"
        "1) Do not infer or reason beyond the text. If the text lacks a valid answer, output exactly: UNKNOWN\n"
        "2) Output must contain no explanations, no extra words, no labels, no code fences, no quotes, no brackets.\n"
        "3) Strip leading/trailing whitespace. No trailing punctuation unless it is required by the format (e.g., a % sign).\n"
        "4) Normalize internal whitespace to single spaces.\n"
        "5) Treat case-insensitive tokens like 'option c', '(c)', '[c]', 'C)' as the letter C when MCQ is requested.\n"
    )

    if is_mcq_task:
        # MCQ-specific user prompt
        user_prompt = (
            (("Options:\n" + "\n".join([f"{chr(ord('A')+i)}. {opt}" for i, opt in enumerate(options)]) + "\n\n")
             if options else "")
            + "Text to parse (final answer snippet):\n"
            + (content_text.strip() if content_text else "")
            + "\n\n"
            + "MCQ output format:\n"
            + "- Return ONLY one capital letter A–Z on a single line.\n"
            + "- Do NOT include any other characters or spaces.\n"
        )
    else:
        # Open-form (numeric/text) user prompt
        user_prompt = (
            "Text to parse (final answer snippet):\n"
            + (content_text.strip() if content_text else "")
            + "\n\n"
            + "Open-form output format:\n"
            + "- If the correct answer is numeric, return ONLY the number (digits, optional decimal). "
            + "- If it's text, return ONLY the minimal text answer."
            + "- Output must be a single line with no extra characters.\n"
        )

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_thinking_extraction_message(options: Optional[List[str]], thinking: str, is_mcq_task: bool = True) -> List[Dict[str, str]]:
    sys_prompt = (
        "You are a strict extractor.\n"
        "Your ONLY job is to read the Reasoning text appended at the end of the user message and output a SINGLE LETTER "
        "corresponding to the option that the Reasoning explicitly concludes as the final answer.\n\n"
        "Rules (follow in order):\n"
        "1) Look ONLY at the Reasoning text. Ignore the options text, and any non-Reasoning content for decision-making.\n"
        "2) If the Reasoning contains an explicit final choice (e.g., 'Therefore, D', 'So the answer is C', 'I choose B', 'Answer: A', 'Option D is correct'), "
        "output that letter. If multiple explicit finals appear, output the LAST one.\n"
        "3) If there is no explicit 'final choice' sentence, but the Reasoning clearly labels a single option as correct using letter notation (e.g., 'D is correct'), "
        "output that letter. If multiple appear, output the LAST one.\n"
        "4) If the Reasoning states the correct option by its text (e.g., 'the correct answer is \"Girl in green clothing with braided hair\"'), "
        "match that text EXACTLY to the provided options and output the corresponding LETTER. If multiple such matches appear, output the LAST one.\n"
        "5) Do NOT judge correctness yourself. Do NOT infer from descriptions. Do NOT vote or reason beyond what the Reasoning states. "
        "If the Reasoning contradicts itself, prefer the LAST explicit conclusion it gives.\n"
        "6) Output only a single uppercase letter that exists in the provided options (A, B, C, ...). No punctuation, no words, no explanations.\n"
    )

    if is_mcq_task:
        user_prompt = (
            "Options:\n" + "\n".join([f"{chr(ord('A')+i)}. {opt}" for i, opt in enumerate(options)]) + "\n\n"
            + "Reasoning:\n" + (thinking.strip() if thinking else "") + "\n"
            + "\n\n"
            + "MCQ output format:\n"
            + "- Return ONLY one capital letter A–Z on a single line.\n"
            + "- Do NOT include any other characters or spaces.\n"
        )
    else:
        user_prompt = (
            "Reasoning:\n" + (thinking.strip() if thinking else "")
            + "\n\n"
            + "Open-form output format:\n"
            + "- If the correct answer is numeric, return ONLY the number (digits, optional decimal). "
            + "- If it's text, return ONLY the minimal text answer."
            + "- Output must be a single line with no extra characters.\n"
        )

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
