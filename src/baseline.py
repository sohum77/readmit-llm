import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, classification_report
import json

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"

# --- Load final dataset ---
df = pd.read_csv(f"{PROCESSED_DIR}/final_dataset.csv")
print(f"Dataset: {df.shape}, positive rate: {df['READMISSION_30D'].mean():.3f}")

df = df.dropna(subset=["TEXT"])
print(f"After dropping empty notes: {df.shape}")

# --- Stratified train/test split (locked-in protocol) ---
# Keep HADM_ID alongside X/y so the SAME held-out test set can be reused
# for the LLM prompting strategies later (fair, apples-to-apples comparison).
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["READMISSION_30D"], random_state=42
)
X_train, y_train = train_df["TEXT"], train_df["READMISSION_30D"]
X_test, y_test = test_df["TEXT"], test_df["READMISSION_30D"]

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print(f"Train positive rate: {y_train.mean():.3f}, Test positive rate: {y_test.mean():.3f}")

# --- Save the test split (with HADM_ID) so the LLM pipeline samples from
#     this exact held-out set, keeping all four methods comparable ---
test_df[["HADM_ID", "TEXT", "READMISSION_30D"]].to_csv(
    f"{PROCESSED_DIR}/test_split.csv", index=False
)
print(f"Saved held-out test split to {PROCESSED_DIR}/test_split.csv")

# --- TF-IDF vectorization ---
vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"TF-IDF matrix: {X_train_tfidf.shape}")

# --- Logistic regression with class_weight to handle imbalance ---
clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
clf.fit(X_train_tfidf, y_train)

# --- Predictions ---
y_pred = clf.predict(X_test_tfidf)
y_proba = clf.predict_proba(X_test_tfidf)[:, 1]

# --- Metrics (matching locked-in evaluation protocol) ---
auroc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)
brier = brier_score_loss(y_test, y_proba)

print("\n=== Logistic Regression Baseline Results ===")
print(f"AUROC: {auroc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Brier score: {brier:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# --- Save results ---
results = {
    "model": "logistic_regression_tfidf_baseline",
    "auroc": auroc,
    "f1": f1,
    "brier_score": brier,
    "n_train": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
    "test_positive_rate": float(y_test.mean())
}

with open(f"{RESULTS_DIR}/baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {RESULTS_DIR}/baseline_results.json")