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

### Model chosen: Random Forest

Random Forest is kept as the primary model. It has better accuracy, better
F1 on both classes, and much more balanced recall across the two classes
(82% fatal / 51% non-fatal, vs. Logistic Regression's 60% fatal / 90%
non-fatal), while its ROC-AUC (0.7921) is nearly identical to Logistic
Regression's (0.7994). Logistic Regression's very high non-fatal recall
comes at the cost of missing 40% of fatal cases, which is a worse tradeoff
for a safety-relevant use case than Random Forest's more even split.

SHAP analysis of the Random Forest (`docs/shap_summary.png`) shows
`is_fall_incident` and `fall_distance_ft` are by far the strongest drivers of
predicted fatality risk; `narrative_length` and month-of-year contribute
weaker, mixed signal.

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
- **No hyperparameter tuning**: both models are untuned baselines.
- **Structured features only**: the incident narrative text itself is not
  used as a model input (Day 2 EDA found narrative word frequency to be only
  a weak fatality signal anyway).

## Intended use

**This is a portfolio/demonstration model built to showcase an end-to-end
EHS data science workflow (cleaning, EDA, feature engineering, baseline
modeling, and explainability). It has not been validated for, and must not
be used for, real workplace safety decisions, risk scoring of real
incidents, or any other production safety-critical application.**
