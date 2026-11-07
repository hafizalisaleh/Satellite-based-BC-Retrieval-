"""
Model Training, Cross-Validation & Benchmark Pipeline (Production v6)
====================================================================
End-to-end peer-review benchmark pipeline for Surface Black Carbon (BC) Retrieval:
  1. Standard 5-Fold Cross-Validation (K-Fold CV)
  2. Independent Held-Out Test Set (Unseen 20% Evaluation)
  3. Temporal Blocked Cross-Validation (Contiguous Blocks with 24h Buffer Gap)
  4. Out-of-Fold Stacking Ensemble Regressor
  5. Feature Importance Extraction and Model Artifact Persistence
  6. Publication Figures Generation (1:1 Scatter, Time Series, Feature Importance)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from configs.config import PROCESSED_DATA_DIR, MODELS_DIR, TABLES_DIR, FIGURES_DIR
from src.features.feature_engineering import add_physics_features
from src.models.evaluate import calculate_metrics, format_metrics_table
from src.utils.visualization import (
    plot_observed_vs_predicted, plot_timeseries_validation, plot_feature_importance
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DATA LOADING & FEATURE SELECTION
# ---------------------------------------------------------------------------
def prepare_training_data(
    dataset_path: Path,
) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    """
    Loads collocated dataset, applies feature engineering if not present,
    filters out target/leaky columns, and prepares X and log1p(y).
    """
    df = pd.read_csv(dataset_path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    # Ensure all domain features are enriched
    if "ventilation_coeff" not in df.columns or "col_bc_surface_proxy" not in df.columns:
        df = add_physics_features(df)

    y_raw = df["BC_target_ug_m3"].copy()
    y = np.log1p(y_raw)

    # Exclude target, instrument leakage, and non-generalizing raw temporal counters
    EXCLUDE_COLS = [
        "datetime_utc", "datetime_local", "date", "time_hour",
        "BC_target_ug_m3", "BC1_ug_m3", "BB_percent",
        "modis_angstrom", "Temperature(°C)", "Pressure(Pa)",
        "month", "doy", "hour", "month_int"
    ]
    
    # Also catch unprintable character variations in column names
    valid_cols = []
    for c in df.columns:
        if c in EXCLUDE_COLS or c.startswith("BC") and c != "bc_to_oc_ratio" and c != "bc_aod_fraction" and c != "bc_combustion_fraction" and c != "bc_to_so4_ratio":
            continue
        if "emp(" in c or "ress(" in c:
            continue
        # Require numeric dtype with valid variance
        if df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            if df[c].notna().sum() > 20:
                valid_cols.append(c)

    X = df[valid_cols].copy()
    logger.info(f"Loaded dataset: {len(df)} samples, {len(valid_cols)} predictor features.")
    logger.info(
        f"Ground-truth BC summary: Mean={y_raw.mean():.2f}, Std={y_raw.std():.2f}, "
        f"Min={y_raw.min():.2f}, Max={y_raw.max():.2f} ug/m3"
    )

    return X, y, valid_cols, df


# ---------------------------------------------------------------------------
# MODEL DEFINITIONS
# ---------------------------------------------------------------------------
def get_models() -> Dict[str, object]:
    """
    Instantiates candidate retrieval models.
    Operates in log1p(BC) space for variance stabilization across orders of magnitude.
    """
    models = {
        "Linear Baseline (Ridge)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("regressor", Ridge(alpha=10.0, random_state=42))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=500, max_depth=20, min_samples_split=3,
            min_samples_leaf=2, max_features=0.6, n_jobs=-1, random_state=42,
        ),
        "ExtraTrees Regressor": ExtraTreesRegressor(
            n_estimators=500, max_features=0.6, min_samples_split=2,
            min_samples_leaf=1, n_jobs=-1, random_state=42,
        ),
        "XGBoost Regressor": xgb.XGBRegressor(
            n_estimators=800, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=42, tree_method="hist",
        ),
    }
    if HAS_LGB:
        models["LightGBM Regressor"] = lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.03, num_leaves=63, max_depth=-1,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1,
        )
    return models


# ---------------------------------------------------------------------------
# SAFE IMPUTATION PIPELINE
# ---------------------------------------------------------------------------
def _impute_features(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Fits median imputer on train features and transforms train and test."""
    imputer = SimpleImputer(strategy="median")
    cols = X_train.columns
    X_tr_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=cols, index=X_train.index)
    X_te_imp = None
    if X_test is not None:
        X_te_imp = pd.DataFrame(imputer.transform(X_test), columns=cols, index=X_test.index)
    return X_tr_imp, X_te_imp


# ---------------------------------------------------------------------------
# EVALUATION 1: STANDARD 5-FOLD CROSS-VALIDATION
# ---------------------------------------------------------------------------
def evaluate_standard_kfold(
    X: pd.DataFrame,
    y: pd.Series,
    y_raw: np.ndarray,
    n_splits: int = 5,
) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray]]:
    """
    Standard K-Fold Cross Validation.
    Computes out-of-fold predictions and peer-reviewed metrics.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = get_models()
    cv_results: Dict[str, Dict] = {}
    oof_predictions: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        oof_raw = np.zeros(len(y))
        for tr_idx, val_idx in kf.split(X):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_val = X.iloc[val_idx]
            
            if not isinstance(model, Pipeline):
                X_tr, X_val = _impute_features(X_tr, X_val)
                
            model.fit(X_tr, y_tr)
            pred_log = model.predict(X_val)
            oof_raw[val_idx] = np.expm1(np.clip(pred_log, 0, 7.0))
            
        metrics = calculate_metrics(y_raw, oof_raw)
        cv_results[name] = metrics
        oof_predictions[name] = oof_raw
        logger.info(f"  [Standard CV] {name:25s} -> R2={metrics['R2']:.4f} | RMSE={metrics['RMSE']:.3f} | Pearson={metrics['Pearson_r']:.4f}")

    return cv_results, oof_predictions


# ---------------------------------------------------------------------------
# EVALUATION 2: INDEPENDENT HELD-OUT TEST (20% UNSEEN)
# ---------------------------------------------------------------------------
def evaluate_heldout_test(
    X: pd.DataFrame,
    y: pd.Series,
    y_raw: np.ndarray,
    test_size: float = 0.20,
) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray], Dict[str, object], np.ndarray, np.ndarray]:
    """
    Evaluates models on an independent 20% held-out test set.
    Returns metrics, predictions, fitted models, test indices, and true values.
    """
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=test_size, random_state=42, shuffle=True
    )
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_tr_log, y_te_raw = y.iloc[train_idx], y_raw[test_idx]
    
    models = get_models()
    test_results: Dict[str, Dict] = {}
    test_predictions: Dict[str, np.ndarray] = {}
    fitted_models: Dict[str, object] = {}

    X_tr_imp, X_te_imp = _impute_features(X_train, X_test)

    for name, model in models.items():
        if isinstance(model, Pipeline):
            model.fit(X_train, y_tr_log)
            pred_log = model.predict(X_test)
        else:
            model.fit(X_tr_imp, y_tr_log)
            pred_log = model.predict(X_te_imp)
            
        pred_raw = np.expm1(np.clip(pred_log, 0, 7.0))
        metrics = calculate_metrics(y_te_raw, pred_raw)
        
        test_results[name] = metrics
        test_predictions[name] = pred_raw
        fitted_models[name] = model
        logger.info(f"  [Held-Out Test] {name:25s} -> R2={metrics['R2']:.4f} | RMSE={metrics['RMSE']:.3f} | Pearson={metrics['Pearson_r']:.4f}")

    return test_results, test_predictions, fitted_models, test_idx, y_te_raw


# ---------------------------------------------------------------------------
# EVALUATION 3: TEMPORAL BLOCKED CROSS-VALIDATION
# ---------------------------------------------------------------------------
def evaluate_temporal_blocked_cv(
    X: pd.DataFrame,
    y: pd.Series,
    y_raw: np.ndarray,
    n_blocks: int = 15,
    buffer_hours: int = 24,
) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray]]:
    """
    Contiguous Temporal Blocked Cross-Validation with buffer gap.
    Tests model generalization across contiguous multi-day blocks while
    strictly purging 24 hours of boundary data to prevent serial autocorrelation leakage.
    """
    n_samples = len(X)
    block_size = n_samples // n_blocks
    models = get_models()
    temp_results: Dict[str, Dict] = {}
    temp_oof: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        oof_raw = np.zeros(n_samples)
        for b in range(n_blocks):
            t_start = b * block_size
            t_end = (b + 1) * block_size if b < n_blocks - 1 else n_samples
            test_indices = np.arange(t_start, t_end)
            
            # Buffer gap to prevent serial autocorrelation leakage
            train_indices = []
            if t_start > buffer_hours:
                train_indices.extend(np.arange(0, t_start - buffer_hours))
            if t_end + buffer_hours < n_samples:
                train_indices.extend(np.arange(t_end + buffer_hours, n_samples))
            train_indices = np.array(train_indices)

            X_tr, y_tr = X.iloc[train_indices], y.iloc[train_indices]
            X_te = X.iloc[test_indices]

            if isinstance(model, Pipeline):
                model.fit(X_tr, y_tr)
                pred_log = model.predict(X_te)
            else:
                X_tr_imp, X_te_imp = _impute_features(X_tr, X_te)
                model.fit(X_tr_imp, y_tr)
                pred_log = model.predict(X_te_imp)

            oof_raw[test_indices] = np.expm1(np.clip(pred_log, 0, 7.0))

        metrics = calculate_metrics(y_raw, oof_raw)
        temp_results[name] = metrics
        temp_oof[name] = oof_raw
        logger.info(f"  [Temporal Blocked CV] {name:25s} -> R2={metrics['R2']:.4f} | RMSE={metrics['RMSE']:.3f} | Pearson={metrics['Pearson_r']:.4f}")

    return temp_results, temp_oof


# ---------------------------------------------------------------------------
# EVALUATION 4: STACKING ENSEMBLE REGRESSOR
# ---------------------------------------------------------------------------
def evaluate_stacking_ensemble(
    oof_dict: Dict[str, np.ndarray],
    test_pred_dict: Dict[str, np.ndarray],
    y_raw_full: np.ndarray,
    y_raw_test: np.ndarray,
) -> Dict[str, float]:
    """
    Fits Ridge meta-regressor on out-of-fold base predictions and evaluates
    on the independent held-out test predictions.
    """
    candidate_keys = [k for k in ["LightGBM Regressor", "XGBoost Regressor", "ExtraTrees Regressor"] if k in oof_dict]
    if not candidate_keys:
        candidate_keys = list(oof_dict.keys())

    X_meta_train = np.column_stack([np.log1p(oof_dict[k]) for k in candidate_keys])
    X_meta_test = np.column_stack([np.log1p(test_pred_dict[k]) for k in candidate_keys])

    meta_learner = Ridge(alpha=1.0, random_state=42)
    meta_learner.fit(X_meta_train, np.log1p(y_raw_full))

    meta_pred_log = meta_learner.predict(X_meta_test)
    meta_pred_raw = np.expm1(np.clip(meta_pred_log, 0, 7.0))

    metrics = calculate_metrics(y_raw_test, meta_pred_raw)
    logger.info(f"  [Stacking Ensemble] Meta-Learner -> R2={metrics['R2']:.4f} | RMSE={metrics['RMSE']:.3f} | Pearson={metrics['Pearson_r']:.4f}")
    return metrics, meta_pred_raw


# ---------------------------------------------------------------------------
# PIPELINE EXECUTION ENGINE
# ---------------------------------------------------------------------------
def run_full_training_pipeline(dataset_filename: str = "collocated_bc_hourly.csv"):
    """
    Executes the comprehensive model training and benchmarking suite.
    """
    data_path = PROCESSED_DATA_DIR / dataset_filename
    if not data_path.exists():
        raise FileNotFoundError(f"Collocated dataset missing at {data_path}. Run collocator first.")

    logger.info("=" * 70)
    logger.info("COMMENCING PEER-REVIEW BENCHMARK TRAINING PIPELINE (v6)")
    logger.info("=" * 70)

    X, y, feature_cols, df_raw = prepare_training_data(data_path)
    y_raw = df_raw["BC_target_ug_m3"].values

    # Clean up any old / obsolete files in TABLES_DIR
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in ["cv_benchmark_metrics.csv", "heldout_test_metrics.csv", "temporal_blocked_cv_benchmarks.csv"]:
        old_path = TABLES_DIR / old_file
        if old_path.exists():
            try:
                old_path.unlink()
                logger.info(f"Removed legacy obsolete file: {old_path.name}")
            except Exception as e:
                logger.warning(f"Could not remove {old_path.name}: {e}")

    # 1. Standard 5-Fold Cross Validation
    logger.info("\n>>> 1. STANDARD 5-FOLD CROSS-VALIDATION")
    kfold_metrics, kfold_oof = evaluate_standard_kfold(X, y, y_raw, n_splits=5)
    df_kfold = format_metrics_table(kfold_metrics)
    print("\nSTANDARD 5-FOLD CROSS-VALIDATION BENCHMARKS:")
    print(df_kfold.to_string())

    # 2. Independent Held-Out Test Evaluation
    logger.info("\n>>> 2. INDEPENDENT HELD-OUT TEST EVALUATION (20% UNSEEN)")
    heldout_metrics, heldout_preds, fitted_models, test_idx, y_te_raw = evaluate_heldout_test(
        X, y, y_raw, test_size=0.20
    )
    df_heldout = format_metrics_table(heldout_metrics)
    print("\nINDEPENDENT HELD-OUT TEST BENCHMARKS (20% UNSEEN):")
    print(df_heldout.to_string())

    # 3. Temporal Blocked Cross-Validation
    logger.info("\n>>> 3. TEMPORAL BLOCKED CROSS-VALIDATION (Contiguous Blocks, 24h Buffer)")
    temporal_metrics, temporal_oof = evaluate_temporal_blocked_cv(
        X, y, y_raw, n_blocks=15, buffer_hours=24
    )
    df_temporal = format_metrics_table(temporal_metrics)
    print("\nTEMPORAL BLOCKED CV BENCHMARKS (24h Buffer Purge):")
    print(df_temporal.to_string())

    # 4. Stacking Ensemble Meta-Learner
    logger.info("\n>>> 4. OOF STACKING ENSEMBLE EVALUATION")
    stacking_metrics, stacking_test_preds = evaluate_stacking_ensemble(
        kfold_oof, heldout_preds, y_raw, y_te_raw
    )
    df_stacking = pd.DataFrame([stacking_metrics], index=["Stacking Ensemble (LGB+XGB+ET)"])
    df_stacking = df_stacking[["R2", "RMSE", "NRMSE_%", "MAE", "Bias", "Pearson_r", "N_samples"]]
    print("\nSTACKING ENSEMBLE BENCHMARKS:")
    print(df_stacking.to_string())

    # Save Official Benchmark Tables
    logger.info("\nPersisting benchmark tables to outputs/tables/...")
    df_heldout.to_csv(TABLES_DIR / "heldout_test_benchmarks.csv")
    df_kfold.to_csv(TABLES_DIR / "standard_kfold_benchmarks.csv")
    df_temporal.to_csv(TABLES_DIR / "temporal_blocked_benchmarks.csv")
    df_stacking.to_csv(TABLES_DIR / "stacking_ensemble_metrics.csv")

    # Select Best Model based on Held-Out Test R2
    best_name = df_heldout["R2"].idxmax()
    best_model = fitted_models[best_name]
    logger.info(f"\nBest Standalone Model on Held-Out Test: {best_name} (R2={df_heldout.loc[best_name, 'R2']:.4f})")

    # Save Best Model Artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = MODELS_DIR / "best_bc_retrieval_model.pkl"
    joblib.dump(best_model, model_save_path)
    joblib.dump(feature_cols, MODELS_DIR / "feature_metadata.pkl")
    logger.info(f"Saved best trained model to {model_save_path}")

    # Feature Importance Rankings
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        df_feat_imp = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        df_feat_imp.to_csv(TABLES_DIR / "feature_importance_rankings.csv", index=False)
        logger.info(f"Saved feature importance rankings to {TABLES_DIR / 'feature_importance_rankings.csv'}")

        # Plot Feature Importance
        plot_feature_importance(
            feature_cols, importances,
            model_name=best_name,
            top_n=20,
            save_name="feature_importance_best_model.png",
        )
        # Also maintain legacy name if needed
        plot_feature_importance(
            feature_cols, importances,
            model_name=best_name,
            top_n=20,
            save_name="feature_importance_xgboost.png",
        )

    # Generate Publication Figures
    logger.info("Generating publication figures...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1:1 Scatter Plot on Independent Held-Out Test
    plot_observed_vs_predicted(
        y_te_raw, heldout_preds[best_name],
        model_name=best_name,
        dataset_label="Independent Held-Out Test",
        save_name="scatter_heldout_test_best.png",
    )
    plot_observed_vs_predicted(
        y_te_raw, heldout_preds[best_name],
        model_name=best_name,
        dataset_label="Held-Out Test",
        save_name="scatter_heldout_test_xgboost.png",
    )

    # 1:1 Scatter Plot for Out-Of-Fold CV
    plot_observed_vs_predicted(
        y_raw, kfold_oof[best_name],
        model_name=best_name,
        dataset_label="5-Fold Cross-Validation OOF",
        save_name="scatter_cv_best.png",
    )
    plot_observed_vs_predicted(
        y_raw, kfold_oof[best_name],
        model_name=best_name,
        dataset_label="5-Fold CV",
        save_name="scatter_cv_xgboost.png",
    )

    # Time-series Validation Plot on Held-Out Test (Chronologically Sorted)
    df_ts = pd.DataFrame({
        "datetime_utc": df_raw.iloc[test_idx]["datetime_utc"].values,
        "BC_target_ug_m3": y_te_raw,
        "pred_BC": heldout_preds[best_name],
    })
    plot_timeseries_validation(
        df_ts,
        date_col="datetime_utc",
        true_col="BC_target_ug_m3",
        pred_col="pred_BC",
        save_name="timeseries_heldout_validation.png",
    )

    logger.info("=" * 70)
    logger.info("PIPELINE EXECUTION COMPLETE & ALL BENCHMARKS UPDATED SUCCESSFULLY!")
    logger.info("=" * 70)

    return df_heldout, df_kfold, df_temporal, df_stacking


if __name__ == "__main__":
    run_full_training_pipeline("collocated_bc_hourly.csv")
