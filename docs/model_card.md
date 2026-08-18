# Model Card: OSHA Construction Fatality Risk Classifier

## What this model predicts

A binary classifier that predicts `is_fatality` — whether an OSHA construction
incident record is associated with a fatality — from structured features of
the incident: occupation (top 15 + "Other"), industry (top 10 + "Other"),
whether it was a fall incident, fall distance in feet, narrative length, and
cyclical month-of-year. It does not use any text/NLP features from the
incident narrative itself, and it excludes `injury_type` and `fatality_cause`,
which would leak the answer (see Data leakage section below).

## Training data

- Source: `data/raw/OSHA_Acc-master/osha 4470 (...).xlsx`, cleaned by
  `src/load_data.py` into `data/processed/osha_clean.csv` (4,470 records),
  then transformed into `data/processed/features.csv` (4,463 records after
  dropping 7 rows with missing `occupation_primary`) by `src/build_features.py`.
- Records span 1984–2014.

**Scrape-selection caveat (carried forward from Day 2/3 EDA):** the dataset's
79%+ fatality rate is a property of which incidents the source site chose to
scrape full narratives for — it is **not** a general construction-industry
fatality rate. This model's target class balance, and therefore any
probability it outputs, reflects that selection bias, not the true incidence
of fatalities among all OSHA-recorded construction incidents. This caveat
applies to every metric and prediction below.

## Data leakage check

Confirmed `injury_type` and `fatality_cause` (the raw column used to derive
`is_fatality` in Day 1) are **not** present in `features.csv`'s 33 columns,
and are not used as model inputs.

## Models trained and evaluated

Two baseline classifiers, both with `class_weight="balanced"`, no
hyperparameter tuning, on an 80/20 stratified train/test split (3,570 /
893 rows, both preserving the ~79%/21% class balance):

| Metric | Logistic Regression | Random Forest | Naive baseline ("always fatal") |
|---|---|---|---|
| Accuracy | 0.66 | 0.76 | 0.79 |
| ROC-AUC | 0.7994 | 0.7921 | 0.5000 |
| Precision (non-fatal) | 0.37 | 0.43 | 0.00 |
| Recall (non-fatal) | 0.90 | 0.51 | 0.00 |
| F1 (non-fatal) | 0.52 | 0.47 | 0.00 |
| Precision (fatal) | 0.96 | 0.86 | 0.79 |
| Recall (fatal) | 0.60 | 0.82 | 1.00 |
| F1 (fatal) | 0.74 | 0.84 | 0.88 |

The naive baseline's higher raw accuracy than Logistic Regression is exactly
the imbalance trap this comparison is meant to surface: its ROC-AUC of 0.50
and 0.00 recall on the non-fatal class confirm it has learned nothing. Both
real models clear ROC-AUC ~0.79-0.80, well above that reference point.

### Best structured-only model (Day 4 baseline): Random Forest

Random Forest is kept as the best of the two structured-only models. It has
better accuracy, better F1 on both classes, and much more balanced recall
across the two classes (82% fatal / 51% non-fatal, vs. Logistic Regression's
60% fatal / 90% non-fatal), while its ROC-AUC (0.7921) is nearly identical to
Logistic Regression's (0.7994). Logistic Regression's very high non-fatal
recall comes at the cost of missing 40% of fatal cases, which is a worse
tradeoff for a safety-relevant use case than Random Forest's more even split.

SHAP analysis of the Random Forest (`docs/shap_summary.png`) shows
`is_fall_incident` and `fall_distance_ft` are by far the strongest drivers of
predicted fatality risk; `narrative_length` and month-of-year contribute
weaker, mixed signal.

This was the best model *before* text features were tried — see the
structured vs. text vs. combined comparison below for the current
recommendation.

## Structured vs. text vs. combined

Two more approaches were tried after the Day 4 baseline, using the same
80/20 stratified split (`random_state=42`) so results are directly
comparable:

- **Structured-only**: Random Forest (Day 4 baseline above).
- **Text-only** (`src/train_text_model.py`): Logistic Regression on TF-IDF
  of the incident narrative alone (`ngram_range=(1,2)`, `max_features=2000`,
  `min_df=5`, `stop_words='english'`) — no structured features.
- **Combined** (`src/train_combined_model.py`): Logistic Regression on
  TF-IDF features concatenated with all structured features.

| Metric | RF (structured) | LogReg (text-only) | LogReg (combined) |
|---|---|---|---|
| Accuracy | 0.7600 | 0.6969 | 0.7458 |
| ROC-AUC | 0.7921 | 0.6888 | **0.8540** |
| Non-fatal precision | 0.43 | 0.36 | 0.44 |
| Non-fatal recall | 0.51 | 0.56 | **0.78** |
| Non-fatal F1 | 0.47 | 0.44 | **0.56** |
| Fatal precision | 0.86 | 0.86 | **0.93** |
| Fatal recall | **0.82** | 0.73 | 0.74 |
| Fatal F1 | **0.84** | 0.79 | 0.82 |

(Note: the text-only run's split is drawn from `osha_clean.csv` directly —
894/4,470 test rows — while the structured and combined runs are drawn from
the occupation-filtered 4,463-row set — 893 test rows. A 7-row difference,
not large enough to affect the conclusions below, but noted for exact
reproducibility.)

**Text-only is the weakest of the three on nearly every metric.** This
matches Day 2's EDA finding that narrative vocabulary only weakly separates
fatal from non-fatal incidents, and the follow-up coefficient inspection,
which found several of its top features (specific calendar years, the word
"death" itself) are reporting-template artifacts rather than genuine causal
signal — not something to trust as a standalone risk model.

**Combining text and structured features gives the best ROC-AUC by a wide
margin** (0.854 vs. 0.79 structured-only vs. 0.69 text-only) and the best
non-fatal-class precision/recall/F1 of the three — the two feature sources
are complementary, not redundant. Its tradeoff is lower fatal-class recall
than the structured-only Random Forest (0.74 vs. 0.82).

### Final model recommendation: combined text + structured Logistic Regression

**Recommended model: `src/train_combined_model.py`.**

ROC-AUC — the most honest single number here given the class imbalance —
favors the combined model decisively (0.854 vs. Random Forest's 0.792), and
it also substantially improves the historically weaker non-fatal class
(recall 0.78 vs. 0.51). That said, this is a real tradeoff, not a free win:
the combined model misses more actual fatalities than Random Forest (26% vs.
18% of fatal cases go undetected). If the priority were specifically
"minimize missed fatalities" rather than overall discrimination, Random
Forest would still be the better pick. Choosing the combined model does not
resolve or improve any of the limitations below — it is two imperfect signal
sources combined, not a validated risk model.

## Known limitations

- **Small dataset**: 4,463 rows after cleaning, with only 893 in the test
  set (185 non-fatal) — metrics, especially for the minority class, carry
  meaningful sampling uncertainty.
- **Class imbalance**: ~79% fatal / 21% non-fatal in both splits. Mitigated
  with `class_weight="balanced"` but not with resampling or threshold tuning.
- **Non-representative fatality rate**: as above, the 79% rate reflects
  scrape selection, not real-world incidence — this model's output
  probabilities should not be read as calibrated real-world fatality risk.
- **Stale time window**: data ends in 2014; construction safety practices,
  regulations, and equipment have changed since, and the model has seen
  nothing from the last decade-plus.
- **No hyperparameter tuning**: none of the models (structured, text, or
  combined) have been tuned; all are untuned baselines.
- **Text features carry known artifacts**: the TF-IDF vocabulary includes
  reporting-template noise (calendar years, the negation-blind word "death"
  predicting *non-fatal*) rather than purely causal signal — see the
  structured vs. text vs. combined section above. The combined model
  inherits this weakness alongside its structured features.

## Intended use

**This is a portfolio/demonstration model built to showcase an end-to-end
EHS data science workflow (cleaning, EDA, feature engineering, baseline
modeling, and explainability). It has not been validated for, and must not
be used for, real workplace safety decisions, risk scoring of real
incidents, or any other production safety-critical application.**
