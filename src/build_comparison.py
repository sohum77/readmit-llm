import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

files = {
    "Logistic Regression (baseline)": "baseline_results.json",
    "Zero-shot LLM": "metrics_zero_shot.json",
    "Few-shot LLM": "metrics_few_shot.json",
    "Chain-of-thought LLM": "metrics_cot.json",
}

print(f"{'Method':<32}{'AUROC':<10}{'F1':<10}{'Brier':<10}{'Unparsed %':<12}")
print("-" * 74)
for label, fname in files.items():
    with open(RESULTS_DIR / fname) as f:
        d = json.load(f)
    unknown_pct = f"{d.get('n_unknown', 0) / d.get('n_predictions', 1) * 100:.1f}%" if "n_unknown" in d else "—"
    print(f"{label:<32}{d['auroc']:<10.3f}{d['f1']:<10.3f}{d['brier_score']:<10.3f}{unknown_pct:<12}")
