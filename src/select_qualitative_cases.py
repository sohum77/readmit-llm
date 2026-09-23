import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

def load(strategy):
    with open(RESULTS_DIR / f"predictions_{strategy}.json") as f:
        return json.load(f)

def find_cases(preds, risk, correct, n=1):
    matches = []
    for p in preds:
        pred_label = 1 if p.get("risk") == "HIGH" else 0
        is_correct = (pred_label == p["true_label"])
        if p.get("risk") == risk and is_correct == correct:
            matches.append(p)
    return matches[:n]

strategies = ["zero_shot", "few_shot", "cot"]
for strat in strategies:
    preds = load(strat)
    print(f"\n{'='*70}\n{strat.upper()}\n{'='*70}")

    correct_high = find_cases(preds, "HIGH", True, n=1)
    correct_low = find_cases(preds, "LOW", True, n=1)
    miss_high = find_cases(preds, "HIGH", False, n=1)
    miss_low = find_cases(preds, "LOW", False, n=1)
    unknown = [p for p in preds if p.get("risk") == "UNKNOWN"][:1]

    for label, cases in [
        ("CORRECT (predicted HIGH, true=1)", correct_high),
        ("CORRECT (predicted LOW, true=0)", correct_low),
        ("MISS (predicted HIGH, true=0)", miss_high),
        ("MISS (predicted LOW, true=1)", miss_low),
        ("UNKNOWN (unparsed)", unknown),
    ]:
        print(f"\n--- {label} ---")
        for c in cases:
            print(f"note_id: {c['note_id']}, true_label: {c['true_label']}, risk: {c.get('risk')}, confidence: {c.get('confidence')}")
            print(f"justification: {c.get('justification')[:300]}")
