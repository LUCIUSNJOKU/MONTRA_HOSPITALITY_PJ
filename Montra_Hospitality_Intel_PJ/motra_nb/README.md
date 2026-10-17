# Montra – Guest Experience Analytics & Sentiment Intelligence (Jupyter notebooks)

The whole project as a sequence of notebooks. Run them **in order**; each reads what the previous ones wrote.

| # | Notebook | What it does | Needs |
|---|---|---|---|
| 01 | `01_data_preparation_and_eda` | Join 4 tables, quality checks, EDA, text cleaning, de-duplication, guest-grouped split | – |
| 02 | `02_baseline_sentiment_models` | LogReg / SVM / Naive Bayes grid (96 MLflow runs), best configs, confusion matrices | – (≈10 min; set `FULL_GRID=False` for ≈3 min) |
| 03 | `03_label_quality_audit` | cleanlab audit: ~5% of labels contradict their text; raw vs corrected accuracy | – |
| 04 | `04_bert_finetuning` | DistilBERT/BERT fine-tuning, learning curves, comparison with baselines | torch, transformers, internet, GPU* |
| 05 | `05_error_analysis` | Error rate by tag (negation, contrast, ambiguous band, label noise), example errors | – |
| 06 | `06_topic_modelling` | Seeded topics for the 9 project categories vs NMF baseline, NPMI coherence | – |
| 07 | `07_bertopic` | Guided BERTopic, config sweep, mapping to categories | bertopic stack, internet* |
| 08 | `08_topic_sentiment` | Out-of-fold review sentiment, sentence sentiment, topic × sentiment, driver analysis | – |
| 09 | `09_interactive_dashboard` | Static snapshot + live ipywidgets/Plotly dashboard with filters | ipywidgets, plotly |

\* 04 and 07 detect missing dependencies and skip their heavy cells. They were **not executed in the authoring
sandbox** (no access to Hugging Face): expect small version-specific fixes on first run.

## Setup
```bash
pip install -r requirements.txt
# put booking_data.csv, guest_data.csv, hotel_data.csv, review_data.csv in data/raw/
jupyter lab notebooks/
```
Notebooks locate the project root automatically (they work from `notebooks/` or the project root).
MLflow UI: `mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db`

## Headless re-run of everything
```bash
cd notebooks
for nb in 0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$nb"; done
```

## Key results on the supplied data (test set = 4,923 reviews, split by guest)
* Best baselines: **~94–94.5%** accuracy on raw labels; every configuration plateaus there.
* cleanlab flags **5% of labels (1,717)** as likely wrong; they explain ~82% of test errors. With flagged rows removed from
  the test set accuracy is **~99.4%** — optimistic (a similar model does the flagging). Quote the raw-label figure as the headline
  and verify a sample of `outputs/label_issues.csv` by hand.
* Topics: seeded categories cover ~72% of sentences; the rest are general impressions. NPMI coherence is negative for all methods
  (synonym-style topics rarely share a sentence) – judge on interpretability and coverage as well.

## Data caveats surfaced in the notebooks
* No review mixes a positive topic with a negative one, and topics are mentioned at similar rates in positive and negative
  reviews (some sentences repeat 1,800+ times: the text looks template-generated), so topic "drivers" are weak on this data.
* Comfort has only ~350 mentions (flagged low volume). Sentence sentiment is weakly supervised from review labels.

## Outputs
`data/processed/` (clean splits) · `outputs/` (predictions, summaries, `label_issues.csv`, confusion matrices) ·
`outputs/topics/` (sentences, topics, seeds) · `outputs/dashboard/` (tables behind notebook 09) · `mlflow.db`
