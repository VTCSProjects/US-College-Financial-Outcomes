
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

df = pd.read_csv("data/processed/college_clean.csv")

output_dir = Path("outputs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("COLLEGE COST & CAREER OUTCOMES ANALYSIS")
print("=" * 55)

# 1. Institution type distribution
print("\nINSTITUTIONS BY TYPE")
print(df["institution_type"].value_counts())

# 2. Tuition statistics
print("\nTUITION STATISTICS")
print(df["tuition_in_state"].describe())

# 3. Earnings statistics
print("\n10-YEAR EARNINGS STATISTICS")
print(df["earnings_10yr"].describe())

# 4. Median earnings by institution type
print("\nMEDIAN EARNINGS BY INSTITUTION TYPE")
print(
    df.groupby("institution_type")["earnings_10yr"]
    .median()
    .sort_values(ascending=False)
)

# 5. Correlation between tuition and earnings
subset = df.dropna(
    subset=["tuition_in_state", "earnings_10yr"]
)

correlation = subset[
    "tuition_in_state"
].corr(subset["earnings_10yr"])

print("\nTUITION VS EARNINGS")
print("Valid institutions:", len(subset))
print("Pearson correlation:", round(correlation, 3))

# 6. Scatter plot
plt.figure(figsize=(11, 7))

sns.scatterplot(
    data=subset,
    x="tuition_in_state",
    y="earnings_10yr",
    hue="institution_type",
    alpha=0.55
)

plt.title("College Tuition vs Earnings 10 Years After Entry")
plt.xlabel("Annual In-State or Listed Tuition ($)")
plt.ylabel("Median Annual Earnings ($)")
plt.tight_layout()

plt.savefig(
    output_dir / "tuition_vs_earnings.png",
    dpi=300
)
plt.close()

# 7. Compare median earnings by type
earnings_by_type = (
    df.groupby("institution_type")["earnings_10yr"]
    .median()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=earnings_by_type.index,
    y=earnings_by_type.values
)

plt.title("Median Earnings by Institution Type")
plt.xlabel("Institution Type")
plt.ylabel("Median Annual Earnings ($)")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    output_dir / "earnings_by_type.png",
    dpi=300
)
plt.close()

print("\nSUCCESS!")
print("Charts saved to:", output_dir)
