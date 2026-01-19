import ast
import json
import re

from sentence_transformers import SentenceTransformer, util
from src.train.reward_utils import chat
from typing import Optional, Tuple, Dict


def extract_answer(text):
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_reasoning(text):
    pattern = r"<think>\s*(.*?)\s*</think>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _to_seconds(ts: str) -> int:
    ts = ts.strip()
    parts = [int(p) for p in ts.split(":")]
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = parts
        return m * 60 + s
    else:
        raise ValueError(f"Unsupported timestamp format: {ts}")


def _normalize(ts: str) -> str:
    ts = ts.strip()
    parts = [int(p) for p in ts.split(":")]
    if len(parts) == 3:
        h, m, s = parts
        return f"{h:02d}:{m:02d}:{s:02d}"
    elif len(parts) == 2:
        m, s = parts
        return f"{m:02d}:{s:02d}"
    else:
        raise ValueError(f"Unsupported timestamp format: {ts}")


def extract_grounding_claims(raw_text: str | dict) -> list[dict]:
    """
    Extracts timestamped claims from a dict mapping timestamp(s) -> sentence.

    Supports:
      - Single timestamps: "MM:SS" or "HH:MM:SS"
      - Ranges: "MM:SS-MM:SS" or "HH:MM:SS-HH:MM:SS"

    Input may be a Python/JSON string or an actual dict.
    Output is sorted by (start) time ascending.
    """
    # 0) Parse input into a dict
    if isinstance(raw_text, dict):
        data = raw_text
    else:
        raw_text = raw_text.strip()
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            # Fallback for Python-literal style dicts (single quotes, etc.)
            try:
                data = ast.literal_eval(raw_text)
            except Exception:
                return []
        except Exception:
            return []

    claims = []
    for key, sentence in data.items():
        key = str(key).strip()
        sentence = (sentence or "").strip()
        if not sentence:
            continue

        if "-" in key:
            start_str_raw, end_str_raw = [p.strip() for p in key.split("-", 1)]
            start_sec = _to_seconds(start_str_raw)
            end_sec = _to_seconds(end_str_raw)
            start_str = _normalize(start_str_raw)
            end_str = _normalize(end_str_raw)

            claims.append(
                {
                    "timestamp_str": f"{start_str}-{end_str}",
                    "timestamp_sec": (start_sec + end_sec) / 2,  # Mid of star & end sec 
                    "start_timestamp_str": start_str,
                    "end_timestamp_str": end_str,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "sentence": sentence,
                    "is_range": True,
                }
            )
        else:
            ts_sec = _to_seconds(key)
            ts_norm = _normalize(key)
            claims.append(
                {
                    "timestamp_str": ts_norm,
                    "timestamp_sec": ts_sec,
                    "sentence": sentence,
                    "is_range": False,
                }
            )

    # Sort by start time
    claims.sort(key=lambda x: x["timestamp_sec"])
    return claims


def _ensure_interval_fields(claims):
    for c in claims:
        if not c.get("is_range", False):
            ts = int(c.get("timestamp_sec", 0))
            c["start_sec"] = ts
            c["end_sec"] = ts
            # Optional: keep normalized strings too if you rely on them elsewhere
            c.setdefault("start_timestamp_str", c.get("timestamp_str", ""))
            c.setdefault("end_timestamp_str", c.get("timestamp_str", ""))
    return claims


# ---- Helpers: temporal overlap with buffer ----
def _interval(c):
    # Prefer explicit start/end; fall back to timestamp_sec for point claims
    if "start_sec" in c and "end_sec" in c:
        s, e = int(c["start_sec"]), int(c["end_sec"])
    else:
        ts = int(c.get("timestamp_sec", 0))
        s, e = ts, ts
    if e < s:
        e = s
    return s, e


def _temporal_match(pc, gc, buf):
    ps, pe = _interval(pc)
    gs, ge = _interval(gc)
    # Expand intervals by buffer and check overlap
    ps_buf, pe_buf = ps - buf, pe + buf
    gs_buf, ge_buf = gs - buf, ge + buf
    return (ps_buf <= ge_buf) and (gs_buf <= pe_buf)


################## Patch # 1 ##################
_TIME_SEP = re.compile(r"\s*(?:-|–|—|to)\s*", flags=re.IGNORECASE)

def _to_seconds_loose(token: str) -> Optional[int]:
    if token is None:
        return None
    t = token.strip().strip("+.,").lower()
    if t in {"end", "eof"}:
        return None
    if ":" in t:
        parts = t.split(":")
        try:
            h = 0
            if len(parts) == 3:
                h = int(parts[0])
                m = int(parts[1])
                s = float(parts[2])
            elif len(parts) == 2:
                m = int(parts[0])
                s = float(parts[1])
            else:
                return None
            total = h * 3600 + m * 60 + s
            return int(round(total))
        except ValueError:
            return None
    try:
        return int(t)
    except ValueError:
        try:
            return int(round(float(t)))
        except ValueError:
            return None

def _seconds_to_key(sec: int) -> str:
    if sec >= 3600:
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        m = sec // 60
        s = sec % 60
        return f"{m:02d}:{s:02d}"

def _normalize_key_to_interval(key: str) -> Optional[Tuple[Optional[int], Optional[int]]]:
    if not isinstance(key, str):
        return None
    parts = _TIME_SEP.split(key.strip())
    if len(parts) == 1:
        start = _to_seconds_loose(parts[0])
        if start is None:
            return None
        return (start, start)
    elif len(parts) == 2:
        start = _to_seconds_loose(parts[0])
        end = _to_seconds_loose(parts[1])
        if start is None:
            return None
        return (start, end if end is not None else None)
    else:
        return None

def sanitize_timestamps_dict(d: Dict[str, str],
                             drop_open_ended: bool = True,
                             coerce_open_ended_by: Optional[int] = None) -> Dict[str, str]:
    out = {}
    for k, v in d.items():
        iv = _normalize_key_to_interval(k)
        if iv is None:
            continue
        start, end = iv
        if end is None:
            if drop_open_ended and coerce_open_ended_by is None:
                continue
            end = start + (coerce_open_ended_by or 0)
        if end < start:
            start, end = end, start
        ck = _seconds_to_key(start) if start == end else f"{_seconds_to_key(start)}-{_seconds_to_key(end)}"
        out[ck] = v
    return out

def _robust_json_from_text(text: str) -> dict:
    import json
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
    except Exception:
        return {}
    return {}
################## Patch # 1 ##################


def calculate_grounding_reward(
    prediction_text: str,
    temporal_grnd: dict,
    sentence_model: SentenceTransformer,
    oai_client,
    served_model: str,
    buffer_seconds: int = 2,
    similarity_threshold: float = 0.75,
) -> float:
    SYSTEM = """
You are a precise temporal information extraction assistant.

Goal:
Given ONLY the provided text, extract every timestamp mention and produce a compact JSON object mapping each timestamp (key) to ONE short sentence (value) describing what happens at that time according to the text.

Output format (STRICT):
- Return ONLY a JSON object. No prose, no code fences, no explanations.
- Keys: timestamps exactly as mentioned, normalized to MM:SS or HH:MM:SS with leading zeros.
  • Single time example: "00:42"
  • Range examples: "00:42-00:45", "01:45-02:01"
- Order keys by start time ascending.
- If the text uses phrases like "around the 16-second mark" or "at about 1:02", convert to "00:16" or "01:02".
- If a single continuous action is described across adjacent times (e.g., 00:07 and 00:08), you MAY consolidate into a range "00:07-00:08" with one concise sentence.
- If the same timestamp appears multiple times, merge into a single key and summarize succinctly.
- If no timestamps are present, return {}.

Content rules:
- Derive sentences ONLY from the given text. No hallucinations.
- Each value is a brief, plain-language clause (≤ 10 words), sentence case, ending with a period.
- Prefer action-focused wording (“Performer executes a cartwheel while holding the jump rope.”).
"""

    USER_PROMPT = """
Extract timestamped events from the following text and return ONLY the JSON object as specified.

STRICT KEYS FORMAT:
- Use ONLY zero-padded MM:SS or HH:MM:SS.
- For ranges, use a single hyphen: "MM:SS-HH:MM:SS".
- Do NOT output plain seconds without a colon (e.g., "6" or "020" is forbidden).
- Do NOT output words like "end", "EOF", "+", or any trailing symbols.
- If the text mentions decimal seconds (e.g., 01.77s or 3.5s), round to the nearest second BEFORE emitting.
- Normalize phrases like "around 16 seconds" to "00:16".
- Keys must be unique and ordered by start time ascending.
- If no timestamps are present, return {{}} exactly.

VALUES:
- Each value is ONE short clause (≤ 10 words), sentence case, action-focused, ends with a period.
- Derive strictly from the provided text (no hallucinations).

Return ONLY the JSON object. No prose, no code fences, no explanations.

The text:
{text}
"""

    # Step 1: Extract the {timestamp: sentence, ...} from prediction
    thinking_part = extract_reasoning(prediction_text)
    output_text = chat(
        client=oai_client,
        model=served_model,
        system=SYSTEM,
        user=USER_PROMPT.format(text=thinking_part),
    )
    prediction_grnd = _robust_json_from_text(output_text)
    
    # Step 1.5: Normalize/sanitize timestamp keys for both prediction and GT dicts (if dict-shaped)
    if isinstance(prediction_grnd, dict):
        prediction_grnd = sanitize_timestamps_dict(
            prediction_grnd,
            drop_open_ended=True,          # set to False + coerce_open_ended_by=buffer_seconds if you prefer coercion
            coerce_open_ended_by=None
        )

    if isinstance(temporal_grnd, dict):
        temporal_grnd = sanitize_timestamps_dict(
            temporal_grnd,
            drop_open_ended=True,
            coerce_open_ended_by=None
        )
    
    # Step 2: Match the temporal_grnd & prediction_grnd
    predicted_claims = extract_grounding_claims(prediction_grnd)
    gt_claims = extract_grounding_claims(temporal_grnd)

    if not predicted_claims or not gt_claims:
        return 0.0

    # ---- Step 3: Batch sentence embeddings ----
    pred_sentences = [c.get("sentence", "") for c in predicted_claims]
    gt_sentences = [c.get("sentence", "") for c in gt_claims]

    # Inject start/end for point claims so downstream code is uniform
    predicted_claims = _ensure_interval_fields(predicted_claims)
    gt_claims = _ensure_interval_fields(gt_claims)

    if not any(pred_sentences) or not any(gt_sentences):
        return 0.0

    # Encode sentences into embeddings using sentence_model
    pred_embeddings = sentence_model.encode(pred_sentences)
    gt_embeddings = sentence_model.encode(gt_sentences)

    # ---- Step 4: One-to-one best matching with similarity threshold ----
    match_count = 0
    used_gt = set()

    for i, p_claim in enumerate(predicted_claims):
        # Collect temporal candidates first
        temporal_candidates = []
        for j, g_claim in enumerate(gt_claims):
            if j in used_gt:
                continue
            if _temporal_match(p_claim, g_claim, buffer_seconds):
                temporal_candidates.append(j)

        if not temporal_candidates:
            continue

        # among candidates, compute similarity and pick the best above threshold
        p_emb = pred_embeddings[i]
        best_j = None
        best_sim = -1.0

        for j in temporal_candidates:
            sim = util.cos_sim(p_emb, gt_embeddings[j]).item()
            if sim >= similarity_threshold and sim > best_sim:
                best_sim = sim
                best_j = j

        if best_j is not None:
            used_gt.add(best_j)
            match_count += 1

    # ---- Step 5: Score ----
    score = match_count / max(1, len(predicted_claims))

    return float(score)
