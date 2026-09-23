"""
compute_metrics.py
-------------------
Computes AUROC, F1, and Brier score from a saved predictions_<strategy>.json
file, using the same evaluation protocol as baseline.py so all four methods
are directly comparable.
"""

import argparse
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, classification_report

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def risk_to_proba(risk, confidence):
    if risk == "HIGH":
        return confidence
    elif risk == "LOW":
        return 1 - confidence
    else:
        return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=["zero_shot", "few_shot", "cot"])
    args = ap.parse_args()

    path = RESULTS_DIR / f"predictions_{args.strategy}.json"
    with open(path) as f:
        preds = json.load(f)

    print(f"Loaded {len(preds)} predictions from {path}")

    unknown_count = sum(1 for p in preds if p.get("risk") == "UNKNOWN")
    print(f"UNKNOWN (unparsed) predictions: {unknown_count} ({unknown_count/len(preds):.1%})")

    y_true = [p["true_label"] for p in preds]
    y_proba = [risk_to_proba(p.get("risk", "UNKNOWN"), p.get("confidence", 0.0)) for p in preds]
    y_pred = [1 if p.get("risk") == "HIGH" else 0 for p in preds]

    auroc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_proba)

    print(f"\n=== {args.strategy} Results ({len(preds)} predictions) ===")
    print(f"AUROC: {auroc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Brier score: {brier:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred))

    results = {
        "model": f"llm_{args.strategy}",
        "auroc": auroc,
        "f1": f1,
        "brier_score": brier,
        "n_predictions": len(preds),
        "n_unknown": unknown_count,
    }

    out_path = RESULTS_DIR / f"metrics_{args.strategy}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
