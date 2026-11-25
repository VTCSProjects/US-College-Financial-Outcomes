
from pathlib import Path
import pandas as pd

# Locate the College Scorecard CSV
files = list(Path("data/raw").glob("*.csv"))

if not files:
    raise FileNotFoundError("No CSV found in data/raw")

file = next(
    (f for f in files if "Institution" in f.name),
    files[0]
)

print("Loading:", file.name)

df = pd.read_csv(file, low_memory=False)

print("\n===== DATASET SUMMARY =====")
print("Total institutions:", len(df))
print("Total columns:", len(df.columns))
print("Duplicate college IDs:", df["UNITID"].duplicated().sum())

print("\n===== GEOGRAPHIC COVERAGE =====")
print("Number of states/territories:", df["STABBR"].nunique())
print(df["STABBR"].value_counts().head(10))

print("\n===== KEY VARIABLES =====")
columns = [
    "INSTNM",
    "TUITIONFEE_IN",
    "TUITIONFEE_OUT",
    "DEBT_MDN",
    "GRAD_DEBT_MDN",
    "MD_EARN_WNE_P10",
    "C150_4"
]

for col in columns:
    values = df[col].replace("PS", pd.NA)
    print(f"{col}: {values.notna().sum()} available")

print("\n===== SAMPLE COLLEGES =====")
print(df[["INSTNM", "STABBR"]].head(10).to_string(index=False))

print("\nINSPECTION COMPLETE")
