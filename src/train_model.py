import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

IN_PATH = "data/processed/features.csv"
SHAP_PLOT_PATH = "docs/shap_summary.png"
MODEL_PATH = "models/day4_risk_rf.joblib"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def evaluate(name, y_test, y_pred, y_score):
    print(f"\n--- {name} ---")
    print("Confusion matrix (rows=actual, cols=predicted, labels=[False, True]):")
    print(confusion_matrix(y_test, y_pred, labels=[False, True]))
    print("Precision/recall/F1 for both classes:")
    print(classification_report(y_test, y_pred, target_names=["non-fatal (False)", "fatal (True)"], zero_division=0))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_score):.4f}")


def main():
    df = pd.read_csv(IN_PATH)

    X = df.drop(columns=["is_fatality"])
    y = df["is_fatality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print("Train class balance:")
    print(y_train.value_counts(normalize=True).rename("proportion"))
    print("Test class balance:")
    print(y_test.value_counts(normalize=True).rename("proportion"))

    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    log_reg.fit(X_train, y_train)
    print("Fit LogisticRegression (class_weight='balanced')")

    rand_forest = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    rand_forest.fit(X_train, y_train)
    print("Fit RandomForestClassifier (class_weight='balanced')")

    # Random Forest is the model chosen below (better balanced recall, see the
    # SHAP comment) -- it's the one downstream tools/agents should load, so it's
    # saved along with the exact training column order (one-hot dummy columns
    # depend on which occupations/industries were in the training data's top-N).
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": rand_forest, "feature_columns": X_train.columns.tolist()}, MODEL_PATH)
    print(f"Saved Random Forest model -> {MODEL_PATH}")

    log_reg_pred = log_reg.predict(X_test)
    log_reg_score = log_reg.predict_proba(X_test)[:, 1]
    evaluate("Logistic Regression", y_test, log_reg_pred, log_reg_score)

    rand_forest_pred = rand_forest.predict(X_test)
    rand_forest_score = rand_forest.predict_proba(X_test)[:, 1]
    evaluate("Random Forest", y_test, rand_forest_pred, rand_forest_score)

    # Reference point: a model that always predicts the majority class ("fatal")
    # regardless of input. If a real model can't beat this on the minority class,
    # it isn't learning anything -- it's just exploiting the imbalance.
    naive_pred = np.full_like(y_test, fill_value=True, dtype=bool)
    naive_score = np.ones(len(y_test))
    evaluate("Naive baseline (always predict 'fatal')", y_test, naive_pred, naive_score)

    # Random Forest chosen over Logistic Regression: better accuracy/F1 and far
    # more balanced recall across both classes, with near-identical ROC-AUC.
    explainer = shap.TreeExplainer(rand_forest)
    shap_values = explainer.shap_values(X_test)
    # For a binary RandomForestClassifier, shap_values is [class_0, class_1];
    # index 1 = contribution toward predicting "fatal" (is_fatality=True).
    shap_values_fatal = shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]

    os.makedirs(os.path.dirname(SHAP_PLOT_PATH), exist_ok=True)
    shap.summary_plot(shap_values_fatal, X_test, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot -> {SHAP_PLOT_PATH}")


if __name__ == "__main__":
    main()
