import numpy as np
import pandas as pd

IN_PATH = "data/processed/osha_clean.csv"
OUT_PATH = "data/processed/features.csv"

TOP_OCCUPATIONS = 15
TOP_INDUSTRIES = 10


def top_n_or_other(series: pd.Series, n: int) -> pd.Series:
    top = series.value_counts().head(n).index
    return series.where(series.isin(top), "Other")


def load_and_filter(path=IN_PATH):
    df = pd.read_csv(path)

    # occupation_primary is missing for only 7/4470 rows (0.16%) -- small enough to drop
    # rather than invent an "Unknown" category for it.
    return df.dropna(subset=["occupation_primary"]).reset_index(drop=True)


def build_feature_frame(df):
    occupation_grouped = top_n_or_other(df["occupation_primary"], TOP_OCCUPATIONS)
    industry_grouped = top_n_or_other(df["industry_name"], TOP_INDUSTRIES)

    occupation_dummies = pd.get_dummies(occupation_grouped, prefix="occupation")
    industry_dummies = pd.get_dummies(industry_grouped, prefix="industry")

    is_fall_incident = df["is_fall_incident"].astype(bool)
    # 0 ft is a real value for non-fall incidents' filler and would otherwise be
    # indistinguishable from "fell 0 ft" -- is_fall_incident disambiguates it.
    fall_distance_ft = df["fall_distance_ft"].fillna(0)

    narrative_length = df["narrative_length"]

    # ~21% of rows have no parseable incident_date/month. There's no reliable value
    # to impute a missing month with, so sin/cos are set to 0 (a neutral midpoint,
    # not a real month) for those rows rather than dropping them.
    month = df["month"]
    month_sin = np.sin(2 * np.pi * month / 12).fillna(0)
    month_cos = np.cos(2 * np.pi * month / 12).fillna(0)

    target = df["is_fatality"].astype(bool)

    features = pd.concat(
        [
            occupation_dummies,
            industry_dummies,
            is_fall_incident.rename("is_fall_incident"),
            fall_distance_ft.rename("fall_distance_ft"),
            narrative_length.rename("narrative_length"),
            month_sin.rename("month_sin"),
            month_cos.rename("month_cos"),
            target.rename("is_fatality"),
        ],
        axis=1,
    )

    return features


def main():
    df = load_and_filter()
    features = build_feature_frame(df)

    features.to_csv(OUT_PATH, index=False)

    print(f"Saved -> {OUT_PATH}")
    print(f"Shape: {features.shape}")
    print(f"Columns: {features.columns.tolist()}")
    print("Class balance (is_fatality):")
    print(features["is_fatality"].value_counts(normalize=True).rename("proportion"))


if __name__ == "__main__":
    main()
