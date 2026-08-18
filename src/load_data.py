import re
import sys
import pandas as pd

sys.path.append("src")
from sic_codes import sic_to_name

RAW_PATH = "data/raw/OSHA_Acc-master/osha 4470 (with additional metadata scraped from narrative site).xlsx"
OUT_PATH = "data/processed/osha_clean.csv"

MONTH_DATE_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(\d{1,2})\s*,?\s*(\d{4})"
)

def extract_date(text: str):
    """Pull a date like 'August 27, 2013' out of the narrative text."""
    if not isinstance(text, str):
        return pd.NaT
    m = MONTH_DATE_RE.search(text)
    if not m:
        return pd.NaT
    month, day, year = m.groups()
    year_int = int(year)
    # fix scraping typos like 2912 -> 2012 (source only covers 1984-present)
    if year_int > 2026 or year_int < 1970:
        digits = str(year_int)
        year = str(int("20" + digits[2:])) if len(digits) == 4 else year
    try:
        return pd.to_datetime(f"{month} {day} {year}", format="%B %d %Y")
    except ValueError:
        return pd.NaT

def clean_diagnosis(val):
    if not isinstance(val, str):
        return "Unknown"
    parts = [p.strip() for p in val.split(",") if p.strip()]
    return parts[0] if parts else "Unknown"

def clean_fatcause(val):
    if not isinstance(val, str):
        return None
    v = val.strip()
    return None if v in ("[]", "", "nan") else v

def clean_cause(val):
    if not isinstance(val, str):
        return None
    v = val.strip()
    return None if v in ("[]", "", "nan") else v

def main():
    df = pd.read_excel(RAW_PATH)
    print(f"Loaded {len(df)} rows")

    df["incident_date"] = df["SUMMARY"].apply(extract_date)

    # null out any dates that are still impossible after the fix
    still_bad = (df["incident_date"].dt.year > 2026) | (df["incident_date"].dt.year < 1970)
    df.loc[still_bad, "incident_date"] = pd.NaT

    df["year"] = df["incident_date"].dt.year
    df["month"] = df["incident_date"].dt.month
    df["month_name"] = df["incident_date"].dt.month_name()

    df["industry_name"] = df["industry"].apply(sic_to_name)
    df["diagnosis_clean"] = df["diagnosis"].apply(clean_diagnosis)
    df["occupation_clean"] = df["occupation"].astype(str).str.replace(r"^,+", "", regex=True).str.strip()
    df["occupation_clean"] = df["occupation_clean"].replace("", "Not reported")

    df["is_fall_incident"] = df["fall dist"] > 0
    df["fall_distance_ft"] = df["fall dist"].where(df["fall dist"] > 0)

    df["fatcause_clean"] = df["fatcause"].apply(clean_fatcause)
    df["is_fatality"] = df["fatcause_clean"].notna()

    df["cause_clean"] = df["cause"].apply(clean_cause)

    df["narrative_length"] = df["SUMMARY"].astype(str).str.len()

    keep_cols = [
        "id", "title", "SUMMARY", "narrative_length",
        "industry", "industry_name",
        "occupation_clean", "1st occupation",
        "diagnosis_clean", "cause", "cause_clean", "fatcause_clean", "is_fatality",
        "is_fall_incident", "fall_distance_ft",
        "incident_date", "year", "month", "month_name",
    ]
    clean_df = df[keep_cols].rename(columns={
        "SUMMARY": "narrative",
        "occupation_clean": "occupation",
        "diagnosis_clean": "injury_type",
        "1st occupation": "occupation_primary",
        "fatcause_clean": "fatality_cause",
    })

    clean_df.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")
    print(f"Shape: {clean_df.shape}")
    print(f"Fatality rate: {clean_df['is_fatality'].mean()*100:.1f}%")

if __name__ == "__main__":
    main()