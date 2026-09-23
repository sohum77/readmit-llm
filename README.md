# LLM-Based 30-Day Hospital Readmission Prediction

MSc dissertation project (University of Strathclyde).
Predicting 30-day unplanned hospital readmission from unstructured clinical
discharge notes using prompt-engineered Large Language Models, benchmarked
against a structured-data classical baseline.

**Supervisor:** Dr. Dmitri Roussinov
**Dataset (target):** MIMIC-III (PhysioNet credentialed access — application submitted)

---

## Status

- [x] Project scaffold & three-module pipeline (ingestion → prompt → inference)
- [x] Zero-shot prompt template
- [x] Synthetic discharge notes for development before MIMIC-III access
- [ ] CITI training / MIMIC-III access (in progress)
- [ ] Few-shot and chain-of-thought prompt strategies
- [ ] Classical baseline (logistic regression / TF-IDF)
- [ ] Quantitative evaluation (AUC, precision, recall, F1)
- [ ] Qualitative explanation analysis

## Architecture (per proposal)

1. **Data ingestion / preprocessing** — `src/pipeline.py:load_notes`
2. **Prompt generation** — `src/pipeline.py:build_prompt` + `prompts/`
3. **LLM inference & parsing** — `src/pipeline.py:predict` + `src/llm_client.py`

## Quick start (free, local — runs tonight)

1. Install [Ollama](https://ollama.com/download) (Windows installer).
2. Pull a model:
   ```
   ollama pull llama3.2
   ```
3. Run the pipeline:
   ```
   python src/pipeline.py
   ```

You should see a HIGH/LOW risk prediction + justification for each synthetic note.

## Switching to OpenAI for final experiments

Set two environment variables — no code changes:
```
set LLM_PROVIDER=openai
set OPENAI_API_KEY=sk-...
python src/pipeline.py
```
(`gpt-4o-mini` is the cheap default; edit `MODELS` in `src/llm_client.py`
to use `gpt-4o` for the final reported run.)

## Next steps

- Add few-shot + chain-of-thought templates under `prompts/`.
- Swap `data/synthetic_notes.json` for the real MIMIC-III discharge-note cohort
  once credentialed access is granted.
- Build the classical baseline and evaluation scripts (`scikit-learn`).
