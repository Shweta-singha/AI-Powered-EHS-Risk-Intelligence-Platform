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
  `min_df=5`, custom analyzer — see Feature Cleaning Iteration History below)
  — no structured features.
- **Combined** (`src/train_combined_model.py`): Logistic Regression on
  TF-IDF features concatenated with all structured features.

| Metric | RF (structured) | LogReg (text-only) | LogReg (combined) |
|---|---|---|---|
| Accuracy | 0.7600 | 0.6812 | 0.7447 |
| ROC-AUC | 0.7921 | 0.6708 | **0.8454** |
| Non-fatal precision | 0.43 | 0.33 | 0.44 |
| Non-fatal recall | 0.51 | 0.51 | **0.78** |
| Non-fatal F1 | 0.47 | 0.40 | **0.56** |
| Fatal precision | 0.86 | 0.85 | **0.93** |
| Fatal recall | **0.82** | 0.73 | 0.73 |
| Fatal F1 | **0.84** | 0.78 | 0.82 |

(Note: the text-only run's split is drawn from `osha_clean.csv` directly —
894/4,470 test rows — while the structured and combined runs are drawn from
the occupation-filtered 4,463-row set — 893 test rows. A 7-row difference,
not large enough to affect the conclusions below, but noted for exact
reproducibility. These numbers are post-cleaning, final figures — see Feature
Cleaning Iteration History for how they got here from the original run.)

**Text-only is the weakest of the three on nearly every metric.** This
matches Day 2's EDA finding that narrative vocabulary only weakly separates
fatal from non-fatal incidents. A coefficient inspection additionally found
several of its original top features were reporting-template and
ID-numbering artifacts rather than genuine causal signal; most have since
been removed from the vectorizer (see Feature Cleaning Iteration History) —
text-only remains the weakest approach either way, and is not something to
trust as a standalone risk model.

**Combining text and structured features gives the best ROC-AUC by a wide
margin** (0.845 vs. 0.79 structured-only vs. 0.67 text-only) and the best
non-fatal-class precision/recall/F1 of the three — the two feature sources
are complementary, not redundant. Its tradeoff is lower fatal-class recall
than the structured-only Random Forest (0.73 vs. 0.82).

## Feature Cleaning Iteration History

The combined model's headline ROC-AUC was checked for artifact inflation in
three passes, each re-running both `src/train_text_model.py` and
`src/train_combined_model.py` on the same 80/20 split (`random_state=42`):

| Step | Combined ROC-AUC | What it found / fixed |
|---|---|---|
| 1. Original | 0.854 | Baseline TF-IDF (`stop_words='english'`, no other filtering). Coefficient inspection (prompted by a question about whether post-outcome forensic language like "coroner"/"autopsy" was leaking into the model) found the top FATAL-pushing features were dominated by calendar-year and month tokens (`2007`, `2009`, `october`, ...) — a scrape-selection/dataset-construction confound, not risk signal. Explicit forensic vocabulary was checked and ruled out: `coroner`, `pronounced dead`, `medical examiner` were all in-vocabulary but near-zero coefficient. |
| 2. Year/month filtered | 0.847 | Added calendar years (1900–2030) and month names to the TF-IDF stop-word list, removing the artifact identified in step 1. The small AUC drop (0.854→0.847) is expected: those tokens had been supplying real (if illegitimate) predictive lift. |
| 3. Bigram-fixed | **0.845** | A duplicate-word-bigram check found `employee employee` in 5.3% of narratives and `20 employee` in the top-20 list — both artifacts of sklearn's stopword-removal-then-bigram-formation order collapsing "Employee #1 ... Employee #2" into adjacent duplicate tokens. Fixed with a custom analyzer that drops any bigram where the two words are identical or one token is a bare number under 100. Verified against the full ~146k-term vocabulary, not just the top-20 view: zero such bigrams remain. |

**0.845 is the number to cite going forward** for the combined model's
ROC-AUC. The text-only model's ROC-AUC moved the same way across the same
three steps: 0.689 → 0.674 → 0.671.

Final top-20 TF-IDF coefficients, text-only model:

| Pushing toward FATAL | Coef | Pushing toward NON-FATAL | Coef |
|---|---|---|---|
| home | +1.976 | belt | −2.034 |
| column | +1.835 | employees | −2.006 |
| transported | +1.630 | death | −1.740 |
| employee | +1.626 | arm | −1.661 |
| working | +1.485 | flooring | −1.627 |
| exterior | +1.462 | safety belt | −1.499 |
| crew | +1.451 | air | −1.472 |
| employee fell | +1.366 | drill | −1.463 |
| landed | +1.349 | conductor | −1.450 |
| trench | +1.313 | cable | −1.382 |
| day | +1.311 | electric | −1.380 |
| performing | +1.302 | pm | −1.357 |
| iron | +1.263 | temporary | −1.324 |
| fell | +1.258 | glass | −1.296 |
| laborer | +1.244 | bed | −1.275 |
| walking | +1.236 | struck employee | −1.275 |
| general | +1.223 | right hand | −1.272 |
| fractured | +1.206 | wire | −1.231 |
| number | +1.189 | set | −1.218 |
| landing | +1.180 | john | −1.207 |

Final top-20 TF-IDF coefficients (text slice), combined model:

| Pushing toward FATAL | Coef | Pushing toward NON-FATAL | Coef |
|---|---|---|---|
| column | +1.705 | feet | −1.880 |
| employee | +1.511 | opening | −1.846 |
| struck | +1.365 | ca | −1.701 |
| home | +1.316 | fall | −1.607 |
| employee killed | +1.211 | air | −1.562 |
| burns | +1.190 | fell | −1.499 |
| test | +1.175 | employees | −1.488 |
| collapse | +1.161 | death | −1.436 |
| operating | +1.144 | belt | −1.410 |
| finger | +1.142 | burned | −1.319 |
| number | +1.130 | bed | −1.299 |
| hospitalized | +1.129 | conductor | −1.296 |
| nearby | +1.106 | arm | −1.291 |
| contacted | +1.081 | floor | −1.265 |
| sewer | +1.029 | digging | −1.256 |
| attached | +1.022 | injured | −1.205 |
| 200 | +1.021 | pm | −1.181 |
| spray | +1.020 | cable | −1.162 |
| propane | +1.013 | scaffold | −1.158 |
| electrocuted | +1.001 | flooring | −1.146 |

At this point, the surviving top features look more like genuine
injury-mechanism vocabulary (`electrocuted`, `hospitalized`, `collapse`,
`burns`, `employee killed`) than reporting or scraping artifacts, based on
this iteration history. **This has not been independently verified beyond
coefficient inspection** — no held-out audit of individual narratives was
done to confirm these words track real hazard exposure rather than some
other confound not yet identified. Two known residuals remain: `death`
still (counterintuitively) predicts non-fatal, and a standalone `200`
appears in the combined model (numeric-unigram filtering was out of scope
for this cleaning pass).

### Final model recommendation: combined text + structured Logistic Regression

**Recommended model: `src/train_combined_model.py`.**

ROC-AUC — the most honest single number here given the class imbalance —
favors the combined model decisively (0.845 vs. Random Forest's 0.792), and
it also substantially improves the historically weaker non-fatal class
(recall 0.78 vs. 0.51). That said, this is a real tradeoff, not a free win:
the combined model misses more actual fatalities than Random Forest (27% vs.
18% of fatal cases go undetected). If the priority were specifically
"minimize missed fatalities" rather than overall discrimination, Random
Forest would still be the better pick. Choosing the combined model does not
resolve or improve any of the limitations below — it is two imperfect signal
sources combined, not a validated risk model.

## Threshold Selection and Cost-Sensitive Framing

The combined model outputs a probability (`predict_proba`), not just a label
— the 0.5 threshold used elsewhere in this model card (and in `evaluate()`'s
default `y_pred`) is a default, not a considered choice. In a safety
context, the two error types are not equally costly:

- **False negative** (predicted non-fatal, actually fatal): the model misses
  the incident type most in need of attention. This is the costly error.
- **False positive** (predicted fatal, actually non-fatal): an over-cautious
  flag a safety officer reviews and dismisses — mildly wasteful, not
  dangerous.

That asymmetry favors optimizing for fatal-class recall over raw accuracy or
even fatal precision. `src/train_combined_model.py`'s `sweep_thresholds()`
re-scores the same test set's already-computed probabilities at five
thresholds, without retraining:

| Threshold | Fatal precision | Fatal recall | Fatal F1 | Non-fatal precision | Non-fatal recall |
|---|---|---|---|---|---|
| 0.30 | 0.8527 | 0.9322 | 0.8907 | 0.5966 | 0.3838 |
| 0.40 | 0.8953 | 0.8333 | 0.8632 | 0.4957 | 0.6270 |
| 0.50 (default) | 0.9286 | 0.7345 | 0.8202 | 0.4354 | 0.7838 |
| 0.60 | 0.9601 | 0.6116 | 0.7472 | 0.3778 | 0.9027 |
| 0.70 | 0.9769 | 0.5367 | 0.6928 | 0.3492 | 0.9514 |

**Recommended threshold: 0.30.** Lowering the threshold from the default 0.5
to 0.30 raises fatal recall from 73.5% to 93.2% — catching many more fatal
incidents the default threshold would have missed — while fatal precision
only drops from 92.9% to 85.3%. It's also the threshold with the best fatal
F1 in this sweep (0.8907 vs. 0.8202 at 0.5), so this isn't a pure
recall-for-precision trade: 0.30 is close to a strict improvement over the
default on the class that matters most here. The cost lands on the
non-fatal class — its recall drops from 78.4% to 38.4%, meaning a safety
officer reviews more false alarms. Given the cost asymmetry above (a missed
fatal is costly; an extra review is not), that trade is worth taking.

This is a recommendation about which threshold to apply to this model's
existing probabilities, not a claim that those probabilities are themselves
well-calibrated — see below.

### Probability calibration

`check_calibration()` bins the test set's predicted probabilities and
compares each bin's mean predicted probability to the actual observed fatal
rate within that bin:

| Mean predicted probability | Observed fatal rate |
|---|---|
| 0.0728 | 0.0000 |
| 0.1562 | 0.2222 |
| 0.2512 | 0.4884 |
| 0.3510 | 0.6087 |
| 0.4488 | 0.7071 |
| 0.5461 | 0.7982 |
| 0.6508 | 0.8548 |
| 0.7451 | 0.8871 |
| 0.8515 | 0.9600 |
| 0.9946 | 1.0000 |

The model is systematically **underconfident** across nearly the whole
range: at a ~25% predicted probability, the actual observed fatal rate in
that bin is 48.8%; at a ~55% prediction, observed is 79.8%. Only at the
extremes (near 0 and near 1) do predicted and observed align closely. This
is a plausible side effect of `class_weight="balanced"`, which reweights the
loss as if the two classes were equally frequent — deflating predicted fatal
probabilities relative to this dataset's actual ~79% fatal base rate, even
as it improves classification metrics on the minority (non-fatal) class.
`sklearn.calibration.CalibratedClassifierCV` supports post-hoc recalibration
via `method='sigmoid'` (Platt scaling) or `method='isotonic'` (isotonic
regression) — not implemented here, since it needs a full
retrain-and-compare pass out of scope for this update. Combined with the
scrape-selection caveat documented above, these probabilities should be read
as a relative ranking signal, not as calibrated real-world fatality
likelihoods, until that recalibration work is done.

## Known limitations

- **Small dataset**: 4,463 rows after cleaning, with only 893 in the test
  set (185 non-fatal) — metrics, especially for the minority class, carry
  meaningful sampling uncertainty.
- **Class imbalance**: ~79% fatal / 21% non-fatal in both splits. Mitigated
  with `class_weight="balanced"` and, as of the Threshold Selection and
  Cost-Sensitive Framing section above, threshold tuning (0.30 selected over
  the 0.5 default) — resampling has not been tried.
- **Non-representative fatality rate**: as above, the 79% rate reflects
  scrape selection, not real-world incidence — this model's output
  probabilities should not be read as calibrated real-world fatality risk.
- **Stale time window**: data ends in 2014; construction safety practices,
  regulations, and equipment have changed since, and the model has seen
  nothing from the last decade-plus.
- **No hyperparameter tuning**: none of the models (structured, text, or
  combined) have been tuned; all are untuned baselines.
- **Text features carry residual artifacts**: calendar-year, month, and
  ID-numbering artifacts (e.g. `employee employee`, `20 employee`) were
  identified and removed — see Feature Cleaning Iteration History above.
  What remains still includes the negation-blind word "death" predicting
  *non-fatal*, and a standalone numeric token (`200`) that wasn't in scope
  for the bigram fix. The combined model inherits whatever residual noise
  is left in the TF-IDF vocabulary alongside its structured features.

## Intended use

**This is a portfolio/demonstration model built to showcase an end-to-end
EHS data science workflow (cleaning, EDA, feature engineering, baseline
modeling, and explainability). It has not been validated for, and must not
be used for, real workplace safety decisions, risk scoring of real
incidents, or any other production safety-critical application.**
