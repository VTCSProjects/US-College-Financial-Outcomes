
import pandas as pd
from pathlib import Path

# 1. Load the full dataset
input_path = Path(
    "data/raw/Most-Recent-Cohorts-Institution.csv"
)

df = pd.read_csv(input_path, low_memory=False)

print("ORIGINAL DATASET")
print("Institutions:", len(df))
print("Columns:", len(df.columns))

# 2. Select variables relevant to our project
columns = {
    "UNITID": "college_id",
    "INSTNM": "college_name",
    "CITY": "city",
    "STABBR": "state",
    "CONTROL": "institution_control",
    "PREDDEG": "predominant_degree",
    "HIGHDEG": "highest_degree",
    "UGDS": "undergraduate_enrollment",
    "ADM_RATE": "acceptance_rate",
    "TUITIONFEE_IN": "tuition_in_state",
    "TUITIONFEE_OUT": "tuition_out_state",
    "COSTT4_A": "annual_cost",
    "NPT4_PUB": "net_price_public",
    "NPT4_PRIV": "net_price_private",
    "DEBT_MDN": "median_debt",
    "GRAD_DEBT_MDN": "graduate_debt",
    "MD_EARN_WNE_P6": "earnings_6yr",
    "MD_EARN_WNE_P10": "earnings_10yr",
    "C150_4": "graduation_rate",
    "RET_FT4": "retention_rate",
    "PCTPELL": "pell_grant_rate",
    "PCTFLOAN": "federal_loan_rate",
}

missing_columns = [
    col for col in columns if col not in df.columns
]
if missing_columns:
    raise ValueError(
        f"Missing expected columns: {missing_columns}"
    )

clean = df[list(columns)].rename(columns=columns).copy()

# 3. Replace suppressed and unavailable values
clean = clean.replace({
    "PS": pd.NA,
    "PrivacySuppressed": pd.NA,
    "NULL": pd.NA
})

# 4. Convert numeric variables
text_columns = {
    "college_name", "city", "state"
}

numeric_columns = [
    col for col in clean.columns
    if col not in text_columns
]

for col in numeric_columns:
    clean[col] = pd.to_numeric(
        clean[col], errors="coerce"
    )

# 5. Combine public and private net price
clean["net_price"] = clean[
    "net_price_public"
].combine_first(clean["net_price_private"])

# 6. Label institution types
control_mapping = {
    1: "Public",
    2: "Private nonprofit",
    3: "Private for-profit"
}

clean["institution_type"] = clean[
    "institution_control"
].map(control_mapping)

# 7. Perform quality checks
print("\nDATA QUALITY SUMMARY")
print("Total institutions:", len(clean))
print("Duplicate IDs:",
      clean["college_id"].duplicated().sum())

print("\nAVAILABLE VALUES")
print(clean.notna().sum().sort_values())

# 8. Save cleaned data without deleting rows
output_path = Path(
    "data/processed/college_clean.csv"
)
output_path.parent.mkdir(
    parents=True, exist_ok=True
)

clean.to_csv(output_path, index=False)

print("\nSUCCESS!")
print("Clean dataset saved:", output_path)
print("Final shape:", clean.shape)
