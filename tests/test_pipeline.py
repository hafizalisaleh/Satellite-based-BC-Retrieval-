"""
Automated Pipeline Verification & Unit Tests
============================================
Tests data preprocessing, physics-informed feature engineering,
model inference, evaluation metrics, and artifact persistence.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR
from src.features.feature_engineering import add_physics_features
from src.models.evaluate import calculate_metrics


class TestBlackCarbonPipeline(unittest.TestCase):

    def test_metrics_calculation(self):
        """Test accuracy of evaluation metrics computation."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([11.0, 19.0, 31.0, 38.0, 52.0])
        
        metrics = calculate_metrics(y_true, y_pred)
        self.assertIn("R2", metrics)
        self.assertIn("RMSE", metrics)
        self.assertIn("MAE", metrics)
        self.assertIn("Bias", metrics)
        self.assertIn("Pearson_r", metrics)
        self.assertGreater(metrics["R2"], 0.95)
        self.assertGreater(metrics["Pearson_r"], 0.95)

    def test_physics_feature_engineering(self):
        """Test calculation of atmospheric physics formulas."""
        dummy_df = pd.DataFrame({
            "merra2_pblh": [100.0, 500.0, 1000.0],
            "merra2_surface_pressure_hpa": [1013.0, 1010.0, 1005.0],
            "merra2_wind_speed": [1.5, 3.0, 5.0],
            "merra2_rh": [60.0, 75.0, 90.0],
            "merra2_bc_smass_ug_m3": [5.0, 15.0, 25.0],
            "merra2_oc_smass_ug_m3": [10.0, 30.0, 50.0],
            "merra2_so4_smass_ug_m3": [8.0, 20.0, 35.0],
            "merra2_dust_smass_ug_m3": [2.0, 5.0, 10.0],
            "datetime_utc": pd.date_range("2025-11-01 00:00", periods=3, freq="h"),
        })

        enriched = add_physics_features(dummy_df)
        self.assertIn("ventilation_coeff", enriched.columns)
        self.assertIn("pblh_meters", enriched.columns)
        self.assertIn("bc_to_oc_ratio", enriched.columns)
        self.assertIn("bc_to_so4_ratio", enriched.columns)
        self.assertIn("stagnation_factor", enriched.columns)
        self.assertIn("month_sin", enriched.columns)
        self.assertIn("hour_cos", enriched.columns)
        
        # Verify ventilation coefficient is positive
        self.assertTrue((enriched["ventilation_coeff"] > 0).all())

    def test_model_artifact_loading_and_inference(self):
        """Test loading of trained best model and feature metadata."""
        model_path = MODELS_DIR / "best_bc_retrieval_model.pkl"
        feat_path = MODELS_DIR / "feature_metadata.pkl"
        
        self.assertTrue(model_path.exists(), f"Model artifact missing at {model_path}")
        self.assertTrue(feat_path.exists(), f"Feature metadata missing at {feat_path}")
        
        model = joblib.load(model_path)
        feature_cols = joblib.load(feat_path)
        
        self.assertIsInstance(feature_cols, list)
        self.assertGreater(len(feature_cols), 5)
        
        # Test synthetic inference
        synthetic_sample = pd.DataFrame(
            np.ones((1, len(feature_cols))),
            columns=feature_cols
        )
        pred = model.predict(synthetic_sample)
        self.assertEqual(len(pred), 1)
        self.assertTrue(np.isfinite(pred[0]))

    def test_benchmark_tables_exist(self):
        """Verify that all output tables exist and are non-empty."""
        for tbl in ["heldout_test_benchmarks.csv", "standard_kfold_benchmarks.csv", "temporal_blocked_benchmarks.csv"]:
            p = TABLES_DIR / tbl
            self.assertTrue(p.exists(), f"Table {tbl} does not exist in {TABLES_DIR}")
            df = pd.read_csv(p)
            self.assertGreater(len(df), 0)


if __name__ == "__main__":
    unittest.main()

