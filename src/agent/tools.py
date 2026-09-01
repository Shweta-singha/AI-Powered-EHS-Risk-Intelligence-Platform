import re
import sys
from pathlib import Path

import joblib
import pandas as pd
from langchain_core.tools import tool

SRC_DIR = Path(__file__).resolve().parent.parent  # .../src
REPO_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from test_retrieval import search as _search_chunks  # noqa: E402

DAY4_MODEL_PATH = REPO_ROOT / "models" / "day4_risk_rf.joblib"
DAY5_MODEL_PATH = REPO_ROOT / "models" / "day5_text_logreg.joblib"

_day4 = None
_day5 = None


def _load_day4():
    global _day4
    if _day4 is None:
        _day4 = joblib.load(DAY4_MODEL_PATH)
    return _day4


def _load_day5():
    global _day5
    if _day5 is None:
        _day5 = joblib.load(DAY5_MODEL_PATH)
    return _day5


# Substring hits are enough here (not word-boundary matches) so stems like
# "elevat" also catch "elevated"/"elevation" -- a miss just means
# is_fall_incident/fall_distance_ft default to "no fall detected", not a
# hard failure of the whole extraction.
FALL_KEYWORDS = ("fell", "fall", "falling", "ladder", "roof", "scaffold", "elevat")
FALL_DISTANCE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:foot|feet|ft\.?)\b", re.IGNORECASE)

# Words too generic to use as occupation/industry match signals on their own
# -- dropped before matching a training category's label against the text.
GENERIC_CATEGORY_WORDS = {
    "and", "or", "not", "other", "elsewhere", "classified", "work", "works",
    "trade", "trades", "general", "special", "line", "lines", "reported",
}

# Diagnostic run against the Day 5 test set found a median of ~37 non-zero
# TF-IDF terms per row (min 3, max 168). Below ~15-20, a couple of unusually
# high-weight but topic-agnostic tokens (e.g. "home", "day", "employee" were
# found to be the #1/#4/#11 highest-coefficient words in the whole 2000-word
# vocabulary) can dominate the logit on their own -- see the "dizzy after
# lunch" case, which activated only 6 terms and got 0.759 almost entirely
# from those three words, not genuine risk content.
FALLBACK_MIN_ACTIVATED_TERMS = 15


def _category_keywords(label: str) -> list[str]:
    words = re.findall(r"[a-z]+", label.lower())
    return [w.rstrip("s") for w in words if w not in GENERIC_CATEGORY_WORDS and len(w) > 3]


def _match_category(description_lower: str, feature_columns: list[str], prefix: str) -> tuple[str, bool]:
    """Matches free text against the one-hot categories the Day 4 model was
    trained on (e.g. "occupation_Electricians" -> keyword "electrician").
    Returns (category_label, matched); matched=False means no category beat
    the generic "Other"/"not reported" bucket, which is a legitimate model
    input but carries no real signal -- exactly what predict_risk's
    confidence check below cares about."""
    best_label, best_score = "Other", 0
    for col in feature_columns:
        if not col.startswith(prefix):
            continue
        label = col[len(prefix):]
        if label == "Other" or "not reported" in label.lower():
            continue
        score = sum(
            1 for kw in _category_keywords(label)
            if re.search(rf"\b{re.escape(kw)}\w*\b", description_lower)
        )
        if score > best_score:
            best_label, best_score = label, score
    return best_label, best_score > 0


def _extract_structured_features(description: str, feature_columns: list[str]) -> tuple[dict, bool]:
    """Builds a one-row feature dict matching the Day 4 model's training
    columns from free text, and reports whether the mapping is confident
    enough to trust for a prediction."""
    description_lower = description.lower()

    is_fall_incident = any(kw in description_lower for kw in FALL_KEYWORDS)

    distance_match = FALL_DISTANCE_RE.search(description)
    fall_distance_ft = float(distance_match.group(1)) if distance_match else 0.0

    occupation_label, occupation_matched = _match_category(description_lower, feature_columns, "occupation_")
    industry_label, industry_matched = _match_category(description_lower, feature_columns, "industry_")

    row = {col: False for col in feature_columns if col.startswith(("occupation_", "industry_"))}
    row[f"occupation_{occupation_label}"] = True
    row[f"industry_{industry_label}"] = True

    row["is_fall_incident"] = is_fall_incident
    row["fall_distance_ft"] = fall_distance_ft
    # load_data.py defines narrative_length as a character count (SUMMARY.str.len()),
    # not a word count -- matched here so this row is on the same scale the
    # model was trained on.
    row["narrative_length"] = len(description)
    # No date is present in a free-text description; 0/0 is the same neutral
    # midpoint build_features.py uses for rows with an unparseable month.
    row["month_sin"] = 0.0
    row["month_cos"] = 0.0

    # Trust the structured path only if occupation or industry was pinned to
    # a real category rather than the catch-all "Other" bucket -- an
    # all-"Other" row looks nearly the same for every unmatched description,
    # and isn't worth preferring over a model that actually reads the text.
    confident = occupation_matched or industry_matched

    return row, confident


@tool
def predict_risk(description: str) -> dict:
    """Estimate the probability that a described construction-site incident is fatal.

    Tries to map the free-text description onto the Day 4 structured-feature
    Random Forest model (occupation, industry, fall involvement/distance,
    narrative length). If the description doesn't give enough signal to
    confidently pick an occupation or industry category, falls back to the
    Day 5 TF-IDF text-only Logistic Regression model instead. The result
    always states which model produced the estimate; for the Day 5 fallback
    it also reports how many vocabulary terms activated and flags
    low_confidence=True with a plain-language caveat when too few did,
    since a handful of activated terms lets a few high-weight-but-generic
    words dominate the score (see FALLBACK_MIN_ACTIVATED_TERMS above).
    """
    day4 = _load_day4()
    feature_columns = day4["feature_columns"]

    row, confident = _extract_structured_features(description, feature_columns)

    if confident:
        X = pd.DataFrame([row], columns=feature_columns)
        probability = float(day4["model"].predict_proba(X)[0, 1])
        return {
            "probability": round(probability, 3),
            "model_used": "Day 4 structured Random Forest",
            "low_confidence": False,
        }

    day5 = _load_day5()
    X_text = day5["vectorizer"].transform([description])
    probability = float(day5["model"].predict_proba(X_text)[0, 1])
    activated_terms = int((X_text != 0).sum())
    low_confidence = activated_terms < FALLBACK_MIN_ACTIVATED_TERMS

    result = {
        "probability": round(probability, 3),
        "model_used": (
            "Day 5 TF-IDF text-only Logistic Regression "
            "(fallback: no confident occupation/industry match in the description)"
        ),
        "activated_vocab_terms": activated_terms,
        "low_confidence": low_confidence,
    }
    if low_confidence:
        result["caveat"] = (
            f"Only {activated_terms} vocabulary term(s) matched this description "
            f"(typical inputs activate ~37) -- input is too short/sparse for a "
            f"reliable prediction from this model; treat this score with caution."
        )
    return result


@tool
def retrieve_guidance(query: str) -> str:
    """Retrieve the top 3 most relevant OSHA compliance-guidance chunks for a query.

    Wraps the same retrieval logic as test_retrieval.py (build_rag_index.py's
    Chroma collection, overfetch + boilerplate-penalty re-ranking) so generic
    legal-preamble/footer text doesn't crowd out topic-specific guidance.
    """
    results = _search_chunks(query, top_k=3)
    lines = []
    for rank, result in enumerate(results, start=1):
        flag = " [boilerplate]" if result["is_boilerplate"] else ""
        lines.append(
            f"{rank}. source={result['source']} chunk={result['chunk_index']} "
            f"similarity={result['similarity']:.4f}{flag}\n   {result['text'][:400]}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 80)
    print("Manual test: predict_risk")
    print("=" * 80)
    print(predict_risk.invoke({
        "description": "An electrician fell 15 feet from a ladder while working on wiring in an unfinished building."
    }))
    print()
    print(predict_risk.invoke({
        "description": "Employee reported feeling dizzy after lunch and went home early for the day."
    }))
    print()

    print()
    print("=" * 80)
    print("Manual test: retrieve_guidance")
    print("=" * 80)
    print(retrieve_guidance.invoke({"query": "electrical hazard protection for construction workers"}))
