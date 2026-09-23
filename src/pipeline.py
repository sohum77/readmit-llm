"""
pipeline.py
-----------
End-to-end readmission-risk prediction pipeline.

Implements the three-module architecture from the research proposal:
  1. Data ingestion / preprocessing  (load_notes)
  2. Prompt generation               (build_prompt)
  3. LLM inference + parsing         (predict)

Run:  python src/pipeline.py                 (default: zero-shot)
      python src/pipeline.py --prompt few_shot
      python src/pipeline.py --prompt cot
"""

import argparse
import json
import re
from pathlib import Path

from llm_client import call_llm, PROVIDER, MODELS
from response_parser import parse_response
from load_real_data import load_sample

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "synthetic_notes.json"
PROMPTS_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"

# Available prompt strategies -> template file
TEMPLATES = {
    "zero_shot": PROMPTS_DIR / "zero_shot.txt",
    "few_shot": PROMPTS_DIR / "few_shot.txt",
    "cot": PROMPTS_DIR / "cot.txt",
}


# ---- Module 1: data ingestion & preprocessing -----------------------
def load_notes(path=DATA):
    with open(path) as f:
        notes = json.load(f)
    for n in notes:
        # basic cleaning: collapse whitespace
        n["text"] = re.sub(r"\s+", " ", n["text"]).strip()
    return notes


# ---- Module 2: prompt generation ------------------------------------
def build_prompt(note_text, template_path):
    template = Path(template_path).read_text()
    return template.replace("{note_text}", note_text)


# ---- Module 3: LLM inference & parsing ------------------------------
def predict(note, template_path):
    prompt = build_prompt(note["text"], template_path)
    raw = call_llm(prompt, temperature=0.0)
    parsed = parse_response(raw)          # robust: handles JSON, fences, CoT FINAL:
    parsed["note_id"] = note["note_id"]
    parsed["true_label"] = note["true_label"]
    return parsed


# ---- Orchestration --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prompt", choices=TEMPLATES.keys(), default="zero_shot",
        help="Prompt strategy to use (default: zero_shot)",
    )
    args = ap.parse_args()

    strategy = args.prompt
    template_path = TEMPLATES[strategy]
    results_path = RESULTS_DIR / f"predictions_{strategy}.json"

    print(f"Provider: {PROVIDER}  |  Model: {MODELS[PROVIDER]}  |  Strategy: {strategy}\n")

    notes = load_sample(n_total=200)
    results = []
    for note in notes:
        print(f"--- Note {note['note_id']} (true label = {note['true_label']}) ---")
        try:
            result = predict(note, template_path)
        except Exception as e:
            print(f"  ERROR: {e}")
            print("  (Is your provider running? For Ollama: `ollama serve` + "
                  "`ollama pull llama3.2`)\n")
            continue
        pred = 1 if str(result.get("risk", "")).upper() == "HIGH" else 0
        correct = "OK" if pred == note["true_label"] else "MISS"
        print(f"  Predicted risk : {result.get('risk')}  [{correct}]")
        print(f"  Confidence     : {result.get('confidence')}")
        print(f"  Justification  : {result.get('justification')}\n")
        results.append(result)

    if results:
        results_path.write_text(json.dumps(results, indent=2))
        acc = sum(
            (1 if str(r.get("risk", "")).upper() == "HIGH" else 0) == r["true_label"]
            for r in results
        ) / len(results)
        print(f"Saved {len(results)} predictions to {results_path}")
        print(f"Toy accuracy on synthetic set ({strategy}): {acc:.0%}")


if __name__ == "__main__":
    main()
