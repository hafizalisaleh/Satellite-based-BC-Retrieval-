"""
Model Evaluation Metrics & Validation Engine
============================================
Computes peer-review publication metrics: R2, RMSE, NRMSE, MAE, Mean Bias, Pearson r.
Generates evaluation comparison tables and 1:1 scatter validation plots.
"""

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard atmospheric remote sensing evaluation metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Filter NaNs if any
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    
    if len(y_t) == 0:
        return {"r2": 0.0, "rmse": 0.0, "nrmse": 0.0, "mae": 0.0, "bias": 0.0, "pearson_r": 0.0}
        
    r2 = r2_score(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mean_val = np.mean(y_t)
    nrmse = rmse / (mean_val + 1e-8)
    mae = mean_absolute_error(y_t, y_p)
    bias = np.mean(y_p - y_t)
    
    if np.std(y_t) > 0 and np.std(y_p) > 0:
        pearson_r = np.corrcoef(y_t, y_p)[0, 1]
    else:
        pearson_r = 0.0
        
    return {
        "R2": round(float(r2), 4),
        "RMSE": round(float(rmse), 4),
        "NRMSE_%": round(float(nrmse * 100.0), 2),
        "MAE": round(float(mae), 4),
        "Bias": round(float(bias), 4),
        "Pearson_r": round(float(pearson_r), 4),
        "N_samples": int(len(y_t)),
    }


def format_metrics_table(results_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Converts model evaluation dictionary into a clean benchmark table.
    """
    df = pd.DataFrame(results_dict).T
    df = df[["R2", "RMSE", "NRMSE_%", "MAE", "Bias", "Pearson_r", "N_samples"]]
    return df
