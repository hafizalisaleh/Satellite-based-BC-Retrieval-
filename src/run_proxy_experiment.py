import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INPUT_FILE = "data/processed/modis_merra2_aligned.csv"
FEATURE_FILE = "data/processed/proxy_bc_features.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_FILE)

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

df = df.sort_values("datetime").reset_index(drop=True)

print("=" * 60)
print("DATASET")
print("=" * 60)
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")


# =========================================================
# 2. IDENTIFY MODIS AOD COLUMNS
# =========================================================

def find_column(candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


aod047 = find_column([
    "AOD047",
    "aod047",
    "AOD_047",
    "aod_047",
    "Optical_Depth_047"
])

aod055 = find_column([
    "AOD055",
    "aod055",
    "AOD_055",
    "aod_055",
    "Optical_Depth_055"
])

if aod047 is None or aod055 is None:
    raise ValueError(
        "MODIS AOD columns could not be identified.\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


# =========================================================
# 3. DEFINE FEATURES
# =========================================================

features = [
    aod047,
    aod055,
    "T2M",
    "PS",
    "PBLTOP",
    "U10M",
    "V10M",
    "QV2M",
    "WIND_SPEED",
    "TOTEXTTAU",
]

missing_columns = [
    column for column in features + ["BCCMASS"]
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )


# =========================================================
# 4. FEATURE DATASET
# =========================================================

model_df = df[
    ["datetime", "BCCMASS"] + features
].copy()

# Temporal features
model_df["month"] = model_df["datetime"].dt.month
model_df["hour"] = model_df["datetime"].dt.hour

features += ["month", "hour"]


# =========================================================
# 5. MISSING VALUE CHECK
# =========================================================

missing_report = pd.DataFrame({
    "column": model_df.columns,
    "missing_count": model_df.isna().sum().values,
})

missing_report["missing_percentage"] = (
    missing_report["missing_count"]
    / len(model_df)
    * 100
)

missing_report.to_csv(
    f"{OUTPUT_DIR}/missing_values.csv",
    index=False
)

print("\nMissing values:")
print(missing_report.to_string(index=False))


# =========================================================
# 6. CLEAN DATA
# =========================================================

before = len(model_df)

model_df = model_df.dropna(
    subset=["BCCMASS"] + features
).reset_index(drop=True)

after = len(model_df)

print(f"\nRows before cleaning: {before}")
print(f"Rows after cleaning:  {after}")
print(f"Rows removed:         {before - after}")


# =========================================================
# 7. SAVE FINAL FEATURE DATASET
# =========================================================

model_df.to_csv(
    FEATURE_FILE,
    index=False
)

print(f"\nFeature dataset saved to:")
print(FEATURE_FILE)


# =========================================================
# 8. CORRELATION ANALYSIS
# =========================================================

correlation_columns = features + ["BCCMASS"]

correlation = model_df[
    correlation_columns
].corr()

correlation.to_csv(
    f"{OUTPUT_DIR}/feature_correlation.csv"
)


# =========================================================
# 9. CORRELATION HEATMAP
# =========================================================

plt.figure(figsize=(11, 8))

plt.imshow(
    correlation,
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=60,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/correlation_heatmap.png",
    dpi=200
)

plt.close()


# =========================================================
# 10. AOD VS PROXY BC
# =========================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    model_df[aod047],
    model_df["BCCMASS"],
    alpha=0.7
)

plt.xlabel("MODIS AOD 0.47 µm")
plt.ylabel("MERRA-2 BCCMASS")
plt.title("MODIS AOD vs Proxy Black Carbon")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/aod_vs_bc.png",
    dpi=200
)

plt.close()


# =========================================================
# 11. PROXY BC DISTRIBUTION
# =========================================================

plt.figure(figsize=(8, 5))

plt.hist(
    model_df["BCCMASS"],
    bins=20,
    edgecolor="black"
)

plt.xlabel("MERRA-2 BCCMASS")
plt.ylabel("Number of observations")
plt.title("Distribution of Proxy Black Carbon")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/target_distribution.png",
    dpi=200
)

plt.close()


# =========================================================
# 12. TRAIN / TEST SPLIT
# =========================================================

split_index = int(len(model_df) * 0.80)

X = model_df[features]
y = model_df["BCCMASS"]

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# =========================================================
# 13. MODELS
# =========================================================

models = {

    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.04,
        max_depth=2,
        min_samples_leaf=3,
        random_state=42
    )
}


# =========================================================
# 14. TRAIN + EVALUATE
# =========================================================

results = []

predictions_df = pd.DataFrame({
    "datetime": model_df.iloc[split_index:]["datetime"].values,
    "actual_BCCMASS": y_test.values
})


for name, model in models.items():

    print("\n" + "-" * 50)
    print(name)
    print("-" * 50)

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(f"MAE  : {mae:.6e}")
    print(f"RMSE : {rmse:.6e}")
    print(f"R²   : {r2:.4f}")

    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    safe_name = name.lower().replace(" ", "_")

    predictions_df[
        f"{safe_name}_prediction"
    ] = predictions

    # Actual vs predicted
    plt.figure(figsize=(7, 6))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.75
    )

    minimum = min(
        y_test.min(),
        predictions.min()
    )

    maximum = max(
        y_test.max(),
        predictions.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel("Actual Proxy BC")
    plt.ylabel("Predicted Proxy BC")
    plt.title(
        f"{name}: Actual vs Predicted"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/{safe_name}_actual_vs_predicted.png",
        dpi=200
    )

    plt.close()


# =========================================================
# 15. SAVE METRICS
# =========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    f"{OUTPUT_DIR}/model_comparison.csv",
    index=False
)

predictions_df.to_csv(
    f"{OUTPUT_DIR}/test_predictions.csv",
    index=False
)


# =========================================================
# 16. MODEL COMPARISON GRAPH
# =========================================================

x = np.arange(len(results_df))
width = 0.25

plt.figure(figsize=(9, 5))

plt.bar(
    x - width,
    results_df["R2"],
    width,
    label="R²"
)

plt.bar(
    x,
    results_df["RMSE"],
    width,
    label="RMSE"
)

plt.bar(
    x + width,
    results_df["MAE"],
    width,
    label="MAE"
)

plt.xticks(
    x,
    results_df["model"]
)

plt.ylabel("Metric Value")
plt.title("Model Performance Comparison")
plt.legend()

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/model_comparison.png",
    dpi=200
)

plt.close()


# =========================================================
# 17. FEATURE IMPORTANCE
# =========================================================

for name in [
    "Random Forest",
    "Gradient Boosting"
]:

    model = models[name]

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values(
        "importance",
        ascending=False
    )

    safe_name = name.lower().replace(" ", "_")

    importance.to_csv(
        f"{OUTPUT_DIR}/{safe_name}_feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        importance["feature"],
        importance["importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(
        f"{name} Feature Importance"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/{safe_name}_feature_importance.png",
        dpi=200
    )

    plt.close()


# =========================================================
# 18. FINAL SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)

print("\nModel comparison:")
print(results_df.to_string(index=False))

print("\nGenerated outputs:")
for file in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {file}")

print("\nImportant:")
print(
    "BCCMASS is being used only as a temporary proxy target. "
    "These results are not final ground-level BC retrieval results."
)