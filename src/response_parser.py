"""
response_parser.py
------------------
Robustly extracts the structured prediction from an LLM reply.

Handles three cases:
  - zero-shot / few-shot: the reply is (mostly) a JSON object.
  - chain-of-thought: free-text reasoning followed by a line `FINAL: {...}`.
  - messy real-world output: JSON embedded somewhere in surrounding prose,
    optionally wrapped in ```json ... ``` fences.

Returns a dict with keys: risk, confidence, justification.
Falls back to a safe default (risk=UNKNOWN) rather than raising, so a single
malformed reply never crashes a batch run.
"""

import json
import re

_DEFAULT = {"risk": "UNKNOWN", "confidence": 0.0, "justification": ""}


def _try_json(blob: str):
    try:
        obj = json.loads(blob)
        if isinstance(obj, dict) and "risk" in obj:
            return obj
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def parse_response(text: str) -> dict:
    """Extract {risk, confidence, justification} from a raw LLM reply."""
    if not text:
        return dict(_DEFAULT)

    # 1. Prefer content after a CoT FINAL: marker, if present.
    marker = re.search(r"FINAL\s*:\s*(\{.*)", text, re.DOTALL | re.IGNORECASE)
    search_space = marker.group(1) if marker else text

    # 2. Strip code fences if the model wrapped its JSON.
    search_space = re.sub(r"```(?:json)?", "", search_space)

    # 3. Direct parse first.
    obj = _try_json(search_space.strip())
    if obj:
        return _normalise(obj)

    # 4. Otherwise grab the first balanced-looking {...} block and parse it.
    for match in re.finditer(r"\{[^{}]*\}", search_space, re.DOTALL):
        obj = _try_json(match.group(0))
        if obj:
            return _normalise(obj)

    return dict(_DEFAULT)


def _normalise(obj: dict) -> dict:
    out = dict(_DEFAULT)
    risk = str(obj.get("risk", "")).strip().upper()
    out["risk"] = risk if risk in {"HIGH", "LOW"} else "UNKNOWN"
    try:
        out["confidence"] = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        out["confidence"] = 0.0
    out["justification"] = str(obj.get("justification", "")).strip()
    return out


if __name__ == "__main__":
    # Quick self-test across the three output styles.
    samples = [
        '{"risk": "HIGH", "confidence": 0.9, "justification": "CHF, frequent admissions."}',
        '```json\n{"risk":"LOW","confidence":0.8,"justification":"Elective, healthy."}\n```',
        "Step 1... Step 2...\nFINAL: {\"risk\": \"HIGH\", \"confidence\": 0.85, \"justification\": \"COPD, unstable housing.\"}",
        "I think this is probably high risk but I'm not sure.",  # malformed -> UNKNOWN
    ]
    for s in samples:
        print(parse_response(s))
