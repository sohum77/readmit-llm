import pandas as pd

PROCESSED_DIR = "data/processed"

# --- Load cohort labels and discharge notes ---
cohort = pd.read_csv(f"{PROCESSED_DIR}/cohort_labels.csv")
notes = pd.read_csv(f"{PROCESSED_DIR}/discharge_notes.csv")

print(f"Cohort: {cohort.shape}, Notes (before dedup): {notes.shape}")

# --- Deduplicate: some admissions have multiple discharge summary entries ---
# Keep the LAST note per HADM_ID (typically the final/corrected version,
# since addenda are usually charted after the original summary)
notes = notes.sort_values("CHARTDATE")
notes = notes.drop_duplicates(subset="HADM_ID", keep="last")

print(f"Notes (after dedup): {notes.shape}")

# --- Merge cohort labels with notes on HADM_ID ---
dataset = cohort.merge(notes[["HADM_ID", "TEXT"]], on="HADM_ID", how="inner")

print(f"Final dataset: {dataset.shape}")
print(f"Positive readmission rate in final dataset: {dataset['READMISSION_30D'].mean():.3f}")

# --- Save final dataset ---
dataset.to_csv(f"{PROCESSED_DIR}/final_dataset.csv", index=False)
print(f"Saved final dataset to {PROCESSED_DIR}/final_dataset.csv")
