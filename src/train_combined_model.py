import sys

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append("src")
from build_features import build_feature_frame, load_and_filter

TEST_SIZE = 0.2
RANDOM_STATE = 42  # same as train_model.py / train_text_model.py, for consistent splits

# Same calendar-year/month stop-word exclusion as train_text_model.py -- see the
# comment there for why (scrape-selection confound, not risk signal).
MONTH_NAMES = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}
YEAR_TOKENS = {str(y) for y in range(1900, 2031)}
CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS | MONTH_NAMES | YEAR_TOKENS)


def build_custom_analyzer():
    """Same artifact-dropping analyzer as train_text_model.py -- see the
    comment there for why (ID-numbering artifacts, not risk signal)."""
    base_vectorizer = TfidfVectorizer(stop_words=CUSTOM_STOP_WORDS, ngram_range=(1, 2))
    base_analyzer = base_vectorizer.build_analyzer()

    def analyzer(doc):
        tokens = []
        for token in base_analyzer(doc):
            words = token.split(" ")
            if len(words) == 2:
                a, b = words
                if a == b:
                    continue
                if any(w.isdigit() and int(w) < 100 for w in words):
                    continue
            tokens.append(token)
        return tokens

    return analyzer

# Reference numbers from earlier steps, for direct comparison.
DAY4_RF_METRICS = {
    "accuracy": 0.76,
    "roc_auc": 0.7921,
    "non_fatal_precision": 0.43,
    "non_fatal_recall": 0.51,
    "non_fatal_f1": 0.47,
    "fatal_precision": 0.86,
    "fatal_recall": 0.82,
    "fatal_f1": 0.84,
}
TEXT_ONLY_METRICS = {
    "accuracy": 0.6969,
    "roc_auc": 0.6888,
    "non_fatal_precision": 0.3571,
    "non_fatal_recall": 0.5615,
    "non_fatal_f1": 0.4366,
    "fatal_precision": 0.8633,
    "fatal_recall": 0.7327,
    "fatal_f1": 0.7927,
}


def evaluate(name, y_test, y_pred, y_score):
    report = classification_report(
        y_test, y_pred, target_names=["non-fatal (False)", "fatal (True)"],
        zero_division=0, output_dict=True,
    )
    auc = roc_auc_score(y_test, y_score)

    print(f"\n--- {name} ---")
    print("Confusion matrix (rows=actual, cols=predicted, labels=[False, True]):")
    print(confusion_matrix(y_test, y_pred, labels=[False, True]))
    print("Precision/recall/F1 for both classes:")
    print(classification_report(y_test, y_pred, target_names=["non-fatal (False)", "fatal (True)"], zero_division=0))
    print(f"ROC-AUC: {auc:.4f}")

    return {
        "accuracy": report["accuracy"],
        "roc_auc": auc,
        "non_fatal_precision": report["non-fatal (False)"]["precision"],
        "non_fatal_recall": report["non-fatal (False)"]["recall"],
        "non_fatal_f1": report["non-fatal (False)"]["f1-score"],
        "fatal_precision": report["fatal (True)"]["precision"],
        "fatal_recall": report["fatal (True)"]["recall"],
        "fatal_f1": report["fatal (True)"]["f1-score"],
    }


def print_comparison(combined_metrics):
    labels = {
        "accuracy": "Accuracy",
        "roc_auc": "ROC-AUC",
        "non_fatal_precision": "Non-fatal precision",
        "non_fatal_recall": "Non-fatal recall",
        "non_fatal_f1": "Non-fatal F1",
        "fatal_precision": "Fatal precision",
        "fatal_recall": "Fatal recall",
        "fatal_f1": "Fatal F1",
    }
    print("\n--- RF (structured) vs. LogReg (TF-IDF text-only) vs. LogReg (combined) ---")
    header = f"{'Metric':<22}{'RF structured':>16}{'LogReg text':>16}{'LogReg combined':>18}"
    print(header)
    print("-" * len(header))
    for key, label in labels.items():
        print(
            f"{label:<22}{DAY4_RF_METRICS[key]:>16.4f}"
            f"{TEXT_ONLY_METRICS[key]:>16.4f}{combined_metrics[key]:>18.4f}"
        )

    combined_best = sum(
        combined_metrics[k] >= max(DAY4_RF_METRICS[k], TEXT_ONLY_METRICS[k]) for k in labels
    )
    print(
        f"\nCombined model matches or beats both single-source models on "
        f"{combined_best}/{len(labels)} metrics."
    )


def print_top_text_features(vectorizer, log_reg, n=20):
    """Top TF-IDF coefficients within the combined model (text slice only --
    the structured-feature coefficients aren't comparable on the same scale
    since those inputs aren't TF-IDF-normalized)."""
    feature_names = vectorizer.get_feature_names_out()
    n_text = len(feature_names)
    coefs = log_reg.coef_[0][:n_text]
    order = np.argsort(coefs)

    print(f"\n--- Top {n} TF-IDF features (within combined model) pushing toward FATAL ---")
    for idx in order[::-1][:n]:
        print(f"{feature_names[idx]:<25} {coefs[idx]:+.4f}")

    print(f"\n--- Top {n} TF-IDF features (within combined model) pushing toward NON-FATAL ---")
    for idx in order[:n]:
        print(f"{feature_names[idx]:<25} {coefs[idx]:+.4f}")


def main():
    # Structured features and narrative text are both derived from the same
    # filtered dataframe (same row order, same index), so they can be
    # concatenated positionally without a separate join step.
    df = load_and_filter()
    structured = build_feature_frame(df)

    narrative = df["narrative"]
    y = structured["is_fatality"]
    X_structured = structured.drop(columns=["is_fatality"])

    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    X_struct_train = X_structured.iloc[idx_train]
    X_struct_test = X_structured.iloc[idx_test]
    narrative_train = narrative.iloc[idx_train]
    narrative_test = narrative.iloc[idx_test]
    y_train = y.iloc[idx_train]
    y_test = y.iloc[idx_test]

    vectorizer = TfidfVectorizer(
        analyzer=build_custom_analyzer(), max_features=2000, min_df=5
    )
    X_text_train = vectorizer.fit_transform(narrative_train)
    X_text_test = vectorizer.transform(narrative_test)

    X_train = hstack([X_text_train, csr_matrix(X_struct_train.values.astype(float))]).tocsr()
    X_test = hstack([X_text_test, csr_matrix(X_struct_test.values.astype(float))]).tocsr()

    print(f"Combined train: {X_train.shape}, test: {X_test.shape}")
    print(f"  ({X_text_train.shape[1]} TF-IDF features + {X_struct_train.shape[1]} structured features)")

    # max_iter raised vs. the other scripts: unscaled structured features (e.g.
    # narrative_length in the hundreds) mixed with sparse TF-IDF slows lbfgs convergence.
    log_reg = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)
    log_reg.fit(X_train, y_train)
    print("Fit LogisticRegression on combined text + structured features (class_weight='balanced')")

    y_pred = log_reg.predict(X_test)
    y_score = log_reg.predict_proba(X_test)[:, 1]
    combined_metrics = evaluate("Combined text + structured Logistic Regression", y_test, y_pred, y_score)

    print_comparison(combined_metrics)

    print_top_text_features(vectorizer, log_reg, n=20)


if __name__ == "__main__":
    main()
