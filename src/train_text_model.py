import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

IN_PATH = "data/processed/osha_clean.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42  # same as train_model.py, for a consistent train/test split

# Day 4 Random Forest numbers (structured features only), for direct comparison.
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


def print_comparison(text_metrics):
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
    print("\n--- Day 4 Random Forest (structured) vs. TF-IDF Logistic Regression (text-only) ---")
    header = f"{'Metric':<22}{'RF (structured)':>18}{'LogReg (TF-IDF)':>18}"
    print(header)
    print("-" * len(header))
    for key, label in labels.items():
        print(f"{label:<22}{DAY4_RF_METRICS[key]:>18.4f}{text_metrics[key]:>18.4f}")


def print_top_features(vectorizer, log_reg, n=20):
    feature_names = vectorizer.get_feature_names_out()
    coefs = log_reg.coef_[0]
    order = np.argsort(coefs)

    print(f"\n--- Top {n} TF-IDF features pushing toward FATAL ---")
    for idx in order[::-1][:n]:
        print(f"{feature_names[idx]:<25} {coefs[idx]:+.4f}")

    print(f"\n--- Top {n} TF-IDF features pushing toward NON-FATAL ---")
    for idx in order[:n]:
        print(f"{feature_names[idx]:<25} {coefs[idx]:+.4f}")


def main():
    df = pd.read_csv(IN_PATH)

    X_text = df["narrative"]
    y = df["is_fatality"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), max_features=2000, stop_words="english", min_df=5
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    log_reg.fit(X_train, y_train)
    print("Fit LogisticRegression on TF-IDF narrative features (class_weight='balanced')")

    y_pred = log_reg.predict(X_test)
    y_score = log_reg.predict_proba(X_test)[:, 1]
    text_metrics = evaluate("TF-IDF Logistic Regression (text-only)", y_test, y_pred, y_score)

    print_comparison(text_metrics)

    print_top_features(vectorizer, log_reg, n=20)


if __name__ == "__main__":
    main()
