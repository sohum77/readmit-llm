# LLM-Based 30-Day Hospital Readmission Prediction

MSc dissertation project (University of Strathclyde, MSc Advanced Computer Science with Data Science).
Predicting 30-day unplanned hospital readmission from unstructured clinical
discharge notes using prompt-engineered Large Language Models (Llama 3.2 3B via Ollama),
benchmarked against a classical TF-IDF + logistic regression baseline.

**Supervisor:** Dr. Dmitri Roussinov
**Dataset:** MIMIC-III (PhysioNet credentialed access, CITI training completed). All reported results use real MIMIC-III discharge notes. No patient data is included in this repository.

**Headline result:** on AUROC, the only metric that is comparable across the two evaluation sets, all three prompting strategies (zero-shot, few-shot, chain-of-thought) performed near chance (0.50-0.54) and well below the classical baseline (0.70).

> **Note on comparability:** the baseline was evaluated on the full held-out test set (about 6% readmissions), while the LLMs were evaluated on a balanced 200-note sample (100 readmitted, 100 not). Precision, recall, F1 and Brier score therefore **cannot be compared directly** between the baseline and the LLMs. AUROC is the fair comparison.

---

## Key takeaways

- A small locally hosted LLM (Llama 3.2 3B) with prompting alone did not beat a simple TF-IDF + logistic regression baseline at predicting readmission from discharge notes.
- Output reliability was a major failure mode: few-shot and chain-of-thought prompts produced unparseable output on 36-38% of notes.
- A classical baseline is a strong and necessary benchmark before reaching for an LLM.

## Status

- [x] Project scaffold and three-module pipeline (ingestion → prompt → inference)
- [x] Zero-shot, few-shot and chain-of-thought prompt strategies
- [x] CITI training and MIMIC-III access
- [x] Cohort built from MIMIC-III: 53,122 admissions → 47,463 labelled admissions (5.9% 30-day readmission rate)
- [x] Classical baseline (TF-IDF + logistic regression)
- [x] Quantitative evaluation (AUROC, F1, Brier score, precision, recall)
- [x] Qualitative case analysis
- [x] Dissertation submitted

## What this project demonstrates

- **Data engineering on real clinical data:** cohort construction and labelling from MIMIC-III.
- **Prompt engineering:** three strategies with structured JSON output and a robust response parser (clean JSON, chain-of-thought with a final answer line, and messy text with embedded or fenced JSON).
- **Local LLM deployment:** Llama 3.2 3B served with Ollama, so clinical text never leaves the machine (a data-governance choice).
- **Honest evaluation:** classical baseline, quantitative metrics, qualitative analysis, and disclosed limitations, including a comparability caveat between evaluation sets.

## Results

### Table 1: Performance across methods

| Method | AUROC | F1 (readmitted class) | Brier score | Unparsed rate | Evaluation set |
|---|---|---|---|---|---|
| Logistic regression (TF-IDF baseline) | 0.702 | 0.199 | 0.149 | n/a | 9,493 test admissions (~6% readmitted) |
| Zero-shot LLM | 0.501 | 0.574 | 0.327 | 11.5% | 200 notes, balanced |
| Few-shot LLM | 0.535 | 0.416 | 0.300 | 38.0% | 200 notes, balanced |
| Chain-of-thought LLM | 0.544 | 0.192 | 0.287 | 36.0% | 200 notes, balanced |

F1, precision, recall and Brier score are not directly comparable across the two evaluation sets (see the note above). Compare methods on AUROC.

### Table 2: Per-class precision, recall and F1

| Method | Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| Logistic regression | Not readmitted | 0.96 | 0.82 | 0.88 | 8,937 |
| | Readmitted | 0.13 | 0.44 | 0.20 | 556 |
| Zero-shot LLM | Not readmitted | 0.49 | 0.31 | 0.38 | 100 |
| | Readmitted | 0.50 | 0.68 | 0.57 | 100 |
| Few-shot LLM | Not readmitted | 0.48 | 0.59 | 0.53 | 100 |
| | Readmitted | 0.47 | 0.37 | 0.42 | 100 |
| Chain-of-thought LLM | Not readmitted | 0.50 | 0.87 | 0.63 | 100 |
| | Readmitted | 0.48 | 0.12 | 0.19 | 100 |

The high unparsed rates for few-shot and chain-of-thought prompts suggest that a 3B-parameter model struggles to follow a strict output format.

Aggregate metrics are in [`results/`](results/).

## Architecture

1. **Data ingestion / preprocessing:** `src/build_cohort.py`, `src/load_real_data.py`, `src/merge_dataset.py`
2. **Prompt generation:** `src/pipeline.py` and the templates in `prompts/`
3. **LLM inference and parsing:** `src/pipeline.py`, `src/llm_client.py`, `src/response_parser.py`
4. **Evaluation:** `src/baseline.py`, `src/compute_metrics.py`, `src/build_comparison.py`, `src/select_qualitative_cases.py`

## Data

MIMIC-III is **not included** and cannot be redistributed. Access requires CITI training and a signed
PhysioNet data use agreement: https://physionet.org/content/mimiciii/

All results reported here come from real MIMIC-III discharge notes. The three worked examples in `prompts/few_shot.txt` were written by the author and are not taken from MIMIC-III. A small set of synthetic notes was used only to develop and test the pipeline before database access was granted; it is not part of the evaluation and is not included.

## Running it

Requires MIMIC-III access, Python 3 and [Ollama](https://ollama.com/download).

```
pip install -r requirements.txt
ollama pull llama3.2
```

Then run the scripts under `src/` in the order given in the architecture list above (cohort building, LLM pipeline, baseline, metrics).

## Limitations

- The baseline and the LLMs were evaluated on different sets (full imbalanced test set versus a balanced 200-note sample), so only AUROC is directly comparable between them.
- The train/test split is at admission level, not patient level, so patients with multiple admissions (34.6% of admissions) can appear in both sets. This may inflate results and is disclosed rather than hidden.
- The LLM evaluation used a small 3B-parameter local model. Larger models may behave differently, and a high unparsed rate (up to 38%) affected the few-shot and chain-of-thought runs.
- Results come from a single dataset (MIMIC-III, one hospital system).

## Author

Sohum Patil, Glasgow. [LinkedIn](https://www.linkedin.com/in/sohum-patil/) | [GitHub](https://github.com/sohum77)

Licensed under MIT.
