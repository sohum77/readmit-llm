# LLM-Based 30-Day Hospital Readmission Prediction

MSc dissertation project (University of Strathclyde, MSc Advanced Computer Science with Data Science).
Predicting 30-day unplanned hospital readmission from unstructured clinical
discharge notes using prompt-engineered Large Language Models (Llama 3.2 3B via Ollama),
benchmarked against a classical TF-IDF + logistic regression baseline.

**Supervisor:** Dr. Dmitri Roussinov
**Dataset:** MIMIC-III (PhysioNet credentialed access, CITI training completed). No patient data is included in this repository.

**Headline result:** all three prompting strategies (zero-shot, few-shot, chain-of-thought) underperformed the classical baseline (AUROC 0.701).

---

## Status

- [x] Project scaffold and three-module pipeline (ingestion → prompt → inference)
- [x] Zero-shot, few-shot and chain-of-thought prompt strategies
- [x] CITI training and MIMIC-III access
- [x] Cohort built from MIMIC-III: 53,122 admissions → 47,463 labelled admissions (5.9% 30-day readmission rate)
- [x] Classical baseline (TF-IDF + logistic regression)
- [x] Quantitative evaluation (AUROC, precision, recall, F1)
- [x] Qualitative case analysis
- [x] Dissertation submitted

## What this project demonstrates

- **Data engineering on real clinical data:** cohort construction and labelling from MIMIC-III.
- **Prompt engineering:** three strategies with structured JSON output and a robust response parser (clean JSON, chain-of-thought with a final answer line, and messy text with embedded or fenced JSON).
- **Local LLM deployment:** Llama 3.2 3B served with Ollama, so clinical text never leaves the machine (a data-governance choice).
- **Honest evaluation:** classical baseline, quantitative metrics, qualitative analysis, and disclosed limitations.

## Results

| Method | AUROC | Precision | Recall | F1 |
|---|---|---|---|---|
| TF-IDF + logistic regression (baseline) | 0.701 | [fill] | [fill] | [fill] |
| Zero-shot Llama 3.2 3B | [fill] | [fill] | [fill] | [fill] |
| Few-shot Llama 3.2 3B | [fill] | [fill] | [fill] | [fill] |
| Chain-of-thought Llama 3.2 3B | [fill] | [fill] | [fill] | [fill] |

Aggregate metrics are in [`results/`](results/). LLM strategies were evaluated on a 200-note stratified sample.

## Architecture

1. **Data ingestion / preprocessing:** `src/build_cohort.py`, `src/load_real_data.py`, `src/merge_dataset.py`
2. **Prompt generation:** `src/pipeline.py:build_prompt` + `prompts/`
3. **LLM inference and parsing:** `src/pipeline.py:predict`, `src/llm_client.py`, `src/response_parser.py`
4. **Evaluation:** `src/baseline.py`, `src/compute_metrics.py`, `src/build_comparison.py`, `src/select_qualitative_cases.py`

## Data

MIMIC-III is **not included** and cannot be redistributed. Access requires CITI training and a signed
PhysioNet data use agreement: https://physionet.org/content/mimiciii/

The example notes in `prompts/few_shot.txt` are synthetic.

## Running it

Requires MIMIC-III access, Python 3.10+ and [Ollama](https://ollama.com/download).

```
pip install -r requirements.txt
ollama pull llama3.2
python src/pipeline.py
```

Cohort building, the baseline and metrics have their own scripts under `src/` (see the architecture list above).

## Limitations

- The train/test split is at admission level, not patient level, so patients with multiple admissions (34.6% of admissions) can appear in both sets. This may inflate results and is disclosed rather than hidden.
- The LLM evaluation used a 200-note stratified sample and a small 3B-parameter local model. Larger models may behave differently.
- Results come from a single dataset (MIMIC-III, one hospital system).

## Author

Sohum Patil, Glasgow. [LinkedIn](https://www.linkedin.com/in/sohum-patil/) | [GitHub](https://github.com/sohum77)

Licensed under MIT.
