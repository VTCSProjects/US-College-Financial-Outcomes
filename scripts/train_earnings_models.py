
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    KFold
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==================================================
# 1. LOAD DATA
# ==================================================

df = pd.read_csv(
    "data/processed/college_clean.csv"
)

output_dir = Path("outputs/machine_learning")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("COLLEGE EARNINGS PREDICTION")
print("=" * 60)

# Focus on bachelor's-predominant institutions
df = df[df["predominant_degree"] == 3].copy()

features = [
    "net_price",
    "graduation_rate",
    "acceptance_rate",
    "pell_grant_rate",
    "undergraduate_enrollment",
    "institution_type",
    "state"
]

target = "earnings_10yr"

# ==================================================
# 2. DATA VALIDATION
# ==================================================

df.loc[df[target] <= 0, target] = np.nan
df.loc[df["net_price"] < 0, "net_price"] = np.nan

df.loc[
    df["undergraduate_enrollment"] <= 0,
    "undergraduate_enrollment"
] = np.nan

for col in [
    "graduation_rate",
    "acceptance_rate",
    "pell_grant_rate"
]:
    df.loc[
        ~df[col].between(0, 1), col
    ] = np.nan

df = df.dropna(subset=[target]).copy()

numeric_features = [
    "net_price",
    "graduation_rate",
    "acceptance_rate",
    "pell_grant_rate",
    "undergraduate_enrollment"
]

categorical_features = [
    "institution_type",
    "state"
]

X = df[features]
y = df[target]

print("\nInstitutions with known earnings:", len(df))

# ==================================================
# 3. TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training institutions:", len(X_train))
print("Testing institutions:", len(X_test))

# ==================================================
# 4. PREPROCESSING
# ==================================================

numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_transformer,
        numeric_features
    ),
    (
        "categorical",
        categorical_transformer,
        categorical_features
    )
])

# ==================================================
# 5. MACHINE LEARNING MODELS
# ==================================================

models = {
    "Baseline": DummyRegressor(
        strategy="mean"
    ),

    "Ridge Regression": Ridge(
        alpha=10.0
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=10.0,
        random_state=42
    )
}

# ==================================================
# 6. FIVE-FOLD CROSS-VALIDATION
# ==================================================

print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for name, estimator in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", estimator)
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "r2": "r2"
        },
        n_jobs=1
    )

    fold_maes = -scores["test_mae"]
    fold_r2s = scores["test_r2"]

    cv_results.append({
        "model": name,
        "mean_mae": round(fold_maes.mean(), 2),
        "std_mae": round(fold_maes.std(), 2),
        "mean_r2": round(fold_r2s.mean(), 4),
        "std_r2": round(fold_r2s.std(), 4)
    })

cv_df = pd.DataFrame(cv_results)
cv_df = cv_df.sort_values("mean_mae")

print("\nCROSS-VALIDATION RESULTS")
print(cv_df.to_string(index=False))

cv_df.to_csv(
    output_dir / "cross_validation.csv",
    index=False
)

# ==================================================
# 7. FINAL HELD-OUT TEST EVALUATION
# ==================================================

print("\n" + "=" * 60)
print("FINAL TEST PERFORMANCE")
print("=" * 60)

results = []
fitted_models = {}

for name, estimator in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", estimator)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test, predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(
        y_test, predictions
    )

    results.append({
        "model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4)
    })

    fitted_models[name] = pipeline

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("MAE")

print("\nMODEL PERFORMANCE ON HELD-OUT DATA")
print(results_df.to_string(index=False))

results_df.to_csv(
    output_dir / "model_comparison.csv",
    index=False
)

# ==================================================
# 8. SAVE TRAINED MODELS
# ==================================================

for name, pipeline in fitted_models.items():

    filename = (
        name.lower().replace(" ", "_") + ".joblib"
    )

    joblib.dump(
        pipeline,
        output_dir / filename
    )

# ==================================================
# 9. SAVE RANDOM FOREST PREDICTIONS
# ==================================================

rf_model = fitted_models["Random Forest"]

rf_predictions = rf_model.predict(X_test)

prediction_results = pd.DataFrame({
    "actual_earnings": y_test.to_numpy(),
    "predicted_earnings": rf_predictions
})

prediction_results["absolute_error"] = (
    prediction_results["actual_earnings"]
    - prediction_results["predicted_earnings"]
).abs()

prediction_results.to_csv(
    output_dir / "random_forest_predictions.csv",
    index=False
)

# ==================================================
# 10. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("FINAL MODEL SUMMARY")
print("=" * 60)

best_cv_model = cv_df.iloc[0]
best_test_model = results_df.iloc[0]

print("\nBest cross-validation model:")
print(best_cv_model["model"])

print("Cross-validation MAE:",
      best_cv_model["mean_mae"])

print("Cross-validation R2:",
      best_cv_model["mean_r2"])

print("\nBest held-out test model:")
print(best_test_model["model"])

print("Test MAE:", best_test_model["MAE"])
print("Test RMSE:", best_test_model["RMSE"])
print("Test R2:", best_test_model["R2"])

print("\nSUCCESS!")
print("All results saved to:", output_dir)
