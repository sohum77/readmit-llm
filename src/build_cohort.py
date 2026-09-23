import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# --- Step 1: Load admissions and patients ---
admissions = pd.read_csv(f"{RAW_DIR}/ADMISSIONS.csv", parse_dates=["ADMITTIME", "DISCHTIME", "DEATHTIME"])
patients = pd.read_csv(f"{RAW_DIR}/PATIENTS.csv")

print(f"Admissions: {admissions.shape}, Patients: {patients.shape}")

# --- Step 2: Exclude admissions where the patient died during that stay ---
admissions = admissions[admissions["DEATHTIME"].isna()].copy()

# --- Step 3: Exclude newborn admissions (birth records, not hospitalizations) ---
admissions = admissions[admissions["ADMISSION_TYPE"] != "NEWBORN"].copy()

print(f"Cohort size after death/newborn exclusions: {admissions.shape[0]}")

# --- Step 4: Sort by patient and admission time, compute next admission gap ---
admissions = admissions.sort_values(["SUBJECT_ID", "ADMITTIME"])
admissions["NEXT_ADMITTIME"] = admissions.groupby("SUBJECT_ID")["ADMITTIME"].shift(-1)
admissions["DAYS_TO_NEXT_ADMIT"] = (admissions["NEXT_ADMITTIME"] - admissions["DISCHTIME"]).dt.days

# --- Step 5: Label = 1 if readmitted within 30 days, else 0 ---
admissions["READMISSION_30D"] = (
    (admissions["DAYS_TO_NEXT_ADMIT"] >= 0) & (admissions["DAYS_TO_NEXT_ADMIT"] <= 30)
).astype(int)

print(f"Positive readmission rate: {admissions['READMISSION_30D'].mean():.3f}")

# --- Step 6: Save cohort (labels only, no notes yet) ---
cohort = admissions[["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "READMISSION_30D"]]
cohort.to_csv(f"{PROCESSED_DIR}/cohort_labels.csv", index=False)
print(f"Saved cohort labels to {PROCESSED_DIR}/cohort_labels.csv")

# --- Step 7: Stream NOTEEVENTS in chunks, keep only discharge summaries for our cohort ---
cohort_hadm_ids = set(cohort["HADM_ID"])
matched_notes = []

chunksize = 50000
for i, chunk in enumerate(pd.read_csv(f"{RAW_DIR}/NOTEEVENTS.csv", chunksize=chunksize, low_memory=False)):
    chunk = chunk[
        (chunk["CATEGORY"] == "Discharge summary") &
        (chunk["HADM_ID"].isin(cohort_hadm_ids))
    ]
    if not chunk.empty:
        matched_notes.append(chunk[["SUBJECT_ID", "HADM_ID", "CHARTDATE", "TEXT"]])
    if i % 5 == 0:
        print(f"Processed chunk {i} ({(i+1)*chunksize:,} rows scanned)")

notes_df = pd.concat(matched_notes, ignore_index=True)
notes_df.to_csv(f"{PROCESSED_DIR}/discharge_notes.csv", index=False)
print(f"Saved {notes_df.shape[0]} matched discharge notes to {PROCESSED_DIR}/discharge_notes.csv")