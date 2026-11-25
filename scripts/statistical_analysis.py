
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

df = pd.read_csv("data/processed/college_clean.csv")

output_dir = Path("outputs/statistics")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("COLLEGE EARNINGS: STATISTICAL ANALYSIS")
print("=" * 60)

# Focus on predominantly bachelor's-degree institutions
df = df[df["predominant_degree"] == 3].copy()

print("\nBachelor's-predominant institutions:", len(df))

# Select relevant predictors
columns = [
    "earnings_10yr",
    "net_price",
    "graduation_rate",
    "acceptance_rate",
    "pell_grant_rate",
    "undergraduate_enrollment",
    "institution_type"
]

df = df[columns].copy()

# Remove impossible values
df.loc[df["earnings_10yr"] <= 0, "earnings_10yr"] = np.nan
df.loc[df["net_price"] < 0, "net_price"] = np.nan

for col in [
    "graduation_rate",
    "acceptance_rate",
    "pell_grant_rate"
]:
    df.loc[~df[col].between(0, 1), col] = np.nan

df.loc[
    df["undergraduate_enrollment"] <= 0,
    "undergraduate_enrollment"
] = np.nan

# Complete-case data for a fair model comparison
model_df = df.dropna(subset=columns).copy()

# Express net price in thousands of dollars
model_df["net_price_1000"] = model_df["net_price"] / 1000

# Log transform enrollment to reduce skew
model_df["log_enrollment"] = np.log(
    model_df["undergraduate_enrollment"]
)

print("\nValid institutions for both models:", len(model_df))

if len(model_df) < 30:
    raise ValueError(
        "Insufficient complete records for regression."
    )

# Model 1: Simple linear regression
model1 = smf.ols(
    "earnings_10yr ~ net_price_1000",
    data=model_df
).fit(cov_type="HC3")

print("\nMODEL 1: NET PRICE ONLY")
print("R-squared:", round(model1.rsquared, 4))
print(model1.summary())

# Model 2: Multiple linear regression
model2 = smf.ols(
    """
    earnings_10yr ~
    net_price_1000 +
    graduation_rate +
    acceptance_rate +
    pell_grant_rate +
    log_enrollment +
    C(institution_type)
    """,
    data=model_df
).fit(cov_type="HC3")

print("\nMODEL 2: ADJUSTED REGRESSION")
print("R-squared:", round(model2.rsquared, 4))
print("Adjusted R-squared:", round(model2.rsquared_adj, 4))

print(model2.summary())

# Compare models
comparison = pd.DataFrame({
    "model": ["Net price only", "Adjusted model"],
    "observations": [model1.nobs, model2.nobs],
    "r_squared": [model1.rsquared, model2.rsquared],
    "adjusted_r_squared": [
        model1.rsquared_adj,
        model2.rsquared_adj
    ]
})

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

comparison.to_csv(
    output_dir / "regression_comparison.csv",
    index=False
)

with open(
    output_dir / "regression_results.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write("MODEL 1\n")
    f.write(model1.summary().as_text())
    f.write("\n\nMODEL 2\n")
    f.write(model2.summary().as_text())

print("\nSUCCESS!")
print("Results saved to:", output_dir)
