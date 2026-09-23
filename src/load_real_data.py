"""
load_real_data.py
------------------
Loads a stratified sample from the held-out test split
(data/processed/test_split.csv) in the {note_id, text, true_label}
format expected by pipeline.py. Sampling from the test split (not the
full dataset) ensures the LLM strategies are evaluated on exactly the
same held-out data as the logistic regression baseline.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "test_split.csv"


def load_sample(n_total=200, pos_frac=0.5, seed=42):
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["TEXT"])

    n_pos = int(n_total * pos_frac)
    n_neg = n_total - n_pos

    pos = df[df["READMISSION_30D"] == 1].sample(
        n=min(n_pos, (df["READMISSION_30D"] == 1).sum()), random_state=seed
    )
    neg = df[df["READMISSION_30D"] == 0].sample(n=n_neg, random_state=seed)

    sample = pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)

    notes = []
    for _, row in sample.iterrows():
        notes.append({
            "note_id": str(row["HADM_ID"]),
            "text": row["TEXT"],
            "true_label": int(row["READMISSION_30D"]),
        })

    print(f"Loaded sample: {len(notes)} notes, {sum(n['true_label'] for n in notes)} positive")
    return notes


if __name__ == "__main__":
    notes = load_sample(n_total=10)
    print(notes[0]["note_id"], notes[0]["true_label"], notes[0]["text"][:100])
