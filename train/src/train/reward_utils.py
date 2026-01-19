import re
from rouge_score import rouge_scorer
from typing import Tuple, Optional


def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_reasoning(text):
    pattern = r'<think>\s*(.*?)\s*</think>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def normalize_number(num_str):
    try:
        num_str = num_str.replace(',', '')
        return float(num_str)
    except Exception as e:
        print(f"Error converting '{num_str}' to float: {e}")
        return None

def wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    m = len(ref_words)
    n = len(hyp_words)
    d = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        d[i][0] = i
    for j in range(n+1):
        d[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return d[m][n] / max(1, m)

def compute_rouge_score(reference, hypothesis, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
    scores = scorer.score(reference, hypothesis)
    average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
    return average_fmeasure

def infer_question_type(answer):
    ans = answer.strip()

    # Rule: multiple choice (letter optionally followed by '.' or ')' and text)
    if re.match(r"^[A-Ea-e][\.\)]?\s*(.*)", ans):
        return "multiple choice"

    # Rule: numerical
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})*(\.\d+)?", ans) or re.fullmatch(r"[-+]?\d+(\.\d+)?", ans):
        if '.' in ans and len(ans.split('.')[-1]) > 3:
            return "regression"
        return "numerical"

    if len(ans.split()) <= 5 and re.fullmatch(r"[A-Za-z0-9\s\-:]+", ans):
        return "OCR"

    return "free-form"

def normalize_mcq(ans: str) -> str:
    """Extract just the option letter (A–E) if present."""
    match = re.match(r"([A-Ea-e])", ans.strip())
    if match:
        return match.group(1).upper()
    return ans.strip()


def parse_decision(text: str) -> Tuple[Optional[bool], str]:
    """
    Returns:
      (decision_bool, justification)
      decision_bool: True / False / None (if undecidable)
      justification: rest of the text (may be empty)
    """
    if not text:
        return None, ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None, ""
    first = lines[0].upper()
    if first.startswith("TRUE"):
        return True, "\n".join(lines[1:]).strip()
    if first.startswith("FALSE"):
        return False, "\n".join(lines[1:]).strip()
    # fallback: try to detect token anywhere in first line
    if "TRUE" in first and "FALSE" not in first:
        return True, "\n".join(lines[1:]).strip()
    if "FALSE" in first and "TRUE" not in first:
        return False, "\n".join(lines[1:]).strip()
    return None, "\n".join(lines[1:]).strip()


def chat(client, model, system, user):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        stream=False,
    )
    return (r.choices[0].message.content or "").strip()
