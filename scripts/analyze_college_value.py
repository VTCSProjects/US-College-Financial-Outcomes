
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/processed/college_clean.csv")

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

print("=" * 55)
print("STUDENT DEBT & COLLEGE VALUE ANALYSIS")
print("=" * 55)

# 1. Student debt statistics
print("\nMEDIAN STUDENT DEBT STATISTICS")
print(df["median_debt"].describe())

print("\nMEDIAN GRADUATE DEBT STATISTICS")
print(df["graduate_debt"].describe())

# 2. Debt by institution type
print("\nDEBT BY INSTITUTION TYPE")
print(
    df.groupby("institution_type")[
        ["median_debt", "graduate_debt"]
    ].median().round(2)
)

# 3. Earnings relative to net price
value = df.dropna(
    subset=[
        "net_price",
        "earnings_10yr",
        "graduate_debt"
    ]
).copy()

# Exclude invalid or nonpositive denominators
value = value[
    (value["net_price"] > 0)
    & (value["earnings_10yr"] > 0)
    & (value["graduate_debt"] >= 0)
]

# Earnings relative to annual net price
value["earnings_to_price"] = (
    value["earnings_10yr"] / value["net_price"]
)

# Median graduate debt relative to annual earnings
value["debt_to_earnings"] = (
    value["graduate_debt"] / value["earnings_10yr"]
)

# 4. Restrict rankings to institutions with
# at least 1,000 undergraduate students
ranked = value[
    value["undergraduate_enrollment"] >= 1000
].copy()

print("\nINSTITUTIONS WITH COMPLETE VALUE DATA:", len(value))
print("ELIGIBLE FOR RANKINGS:", len(ranked))

print("\nTOP 20: EARNINGS RELATIVE TO NET PRICE")
print(
    ranked.sort_values(
        "earnings_to_price", ascending=False
    )[
        [
            "college_name",
            "state",
            "institution_type",
            "net_price",
            "earnings_10yr",
            "earnings_to_price"
        ]
    ].head(20).to_string(index=False)
)

print("\nTOP 20: LOWEST DEBT RELATIVE TO EARNINGS")
print(
    ranked.sort_values("debt_to_earnings")[
        [
            "college_name",
            "state",
            "graduate_debt",
            "earnings_10yr",
            "debt_to_earnings"
        ]
    ].head(20).to_string(index=False)
)

ranked.to_csv(
    output_dir / "college_value_analysis.csv",
    index=False
)

print("\nSUCCESS!")
print("Saved: outputs/college_value_analysis.csv")
