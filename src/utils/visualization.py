"""
Publication-Grade Visualizations for Black Carbon ML Retrieval
==============================================================
Generates:
1. 1:1 Observed vs Predicted Scatter with Density & Regression Line
2. Time Series Validation on Held-Out Unseen Test Periods
3. Feature Importance Bar Plots
4. Spatial Surface BC Gridded Hotspot Heatmaps for Pilot Area
"""

import sys
from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.config import FIGURES_DIR, SITE_COORDS, PILOT_BBOX

# Styling preferences for peer-reviewed figures
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0


def plot_observed_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "XGBoost",
    dataset_label: str = "Unseen Test Period",
    save_name: str = "scatter_observed_vs_predicted.png"
):
    """
    Creates high-impact density scatter plot with 1:1 line and metric annotations.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t, y_p = y_true[mask], y_pred[mask]
    
    fig, ax = plt.subplots(figsize=(7, 6.5), dpi=300)
    
    # Calculate point density
    try:
        xy = np.vstack([y_t, y_p])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x_plot, y_plot, z_plot = y_t[idx], y_p[idx], z[idx]
        scatter = ax.scatter(x_plot, y_plot, c=z_plot, s=25, cmap="Spectral_r", alpha=0.85, edgecolors="none")
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Data Density (KDE)", fontsize=11, fontweight="bold")
    except Exception:
        ax.scatter(y_t, y_p, c="#1f77b4", s=25, alpha=0.6, edgecolors="none")
        
    # Axis limits
    max_val = max(float(np.percentile(y_t, 99.5)), float(np.percentile(y_p, 99.5))) * 1.15
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    
    # 1:1 Line
    ax.plot([0, max_val], [0, max_val], "k--", lw=1.5, label="1:1 Reference Line")
    
    # Linear Fit Line
    slope, intercept = np.polyfit(y_t, y_p, 1)
    fit_x = np.linspace(0, max_val, 100)
    fit_y = slope * fit_x + intercept
    ax.plot(fit_x, fit_y, color="#d62728", lw=2.0, label=f"Fit Line: $y = {slope:.2f}x {intercept:+.2f}$")
    
    # Metrics calculation
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    r2 = r2_score(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)
    bias = np.mean(y_p - y_t)
    r = np.corrcoef(y_t, y_p)[0, 1]
    
    # Metric annotation box
    metrics_text = (
        f"$\\mathbf{{{model_name}}}$ ({dataset_label})\n"
        f"$R^2 = {r2:.3f}$\n"
        f"Pearson $r = {r:.3f}$\n"
        f"RMSE = {rmse:.2f} $\\mu g/m^3$\n"
        f"MAE = {mae:.2f} $\\mu g/m^3$\n"
        f"Bias = {bias:+.2f} $\\mu g/m^3$\n"
        f"$N = {len(y_t):,}$"
    )
    ax.text(
        0.05, 0.95, metrics_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#cccccc", alpha=0.92)
    )
    
    ax.set_xlabel("Ground Aethalometer BC [$\\mu g/m^3$]", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Estimated BC ({model_name}) [$\\mu g/m^3$]", fontsize=12, fontweight="bold")
    ax.set_title(f"Model Validation: Ground vs Satellite-Reanalysis BC Retrieval", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    out_file = FIGURES_DIR / save_name
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def plot_timeseries_validation(
    df_preds: pd.DataFrame,
    date_col: str = "datetime_utc",
    true_col: str = "BC_target_ug_m3",
    pred_col: str = "pred_XGBoost Regressor",
    save_name: str = "timeseries_validation.png"
):
    """
    Plots continuous time series comparing ground measurements and model estimates.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = df_preds.sort_values(date_col).copy()
    
    fig, ax = plt.subplots(figsize=(13, 5), dpi=300)
    
    ax.plot(df[date_col], df[true_col], color="#2b2b2b", lw=1.6, label="Aethalometer Ground Truth (Observed)", alpha=0.85)
    ax.plot(df[date_col], df[pred_col], color="#e6550d", lw=1.8, label="Machine Learning Retrieval (Estimated)", alpha=0.9)
    
    ax.fill_between(df[date_col], df[true_col], df[pred_col], color="#fdbb84", alpha=0.35, label="Residual Discrepancy")
    
    ax.set_xlabel("Time (UTC)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Surface Black Carbon [$\\mu g/m^3$]", fontsize=11, fontweight="bold")
    ax.set_title("Temporal Validation on Unseen Test Horizon: Lahore Station", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.92)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    out_file = FIGURES_DIR / save_name
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str = "XGBoost",
    top_n: int = 15,
    save_name: str = "feature_importance.png"
):
    """
    Horizontal bar plot of top atmospheric and satellite predictors.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df_imp = df_imp.sort_values("Importance", ascending=True).tail(top_n)
    
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    bars = ax.barh(df_imp["Feature"], df_imp["Importance"], color="#3182bd", edgecolor="#08519c", height=0.65)
    
    # Value annotations on bars
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005 * max(df_imp["Importance"]), bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", ha="left", fontsize=9, color="#252525")
                
    ax.set_xlabel("Relative Feature Importance (Gain / MDI)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top Predictors Driving Ground BC Retrieval ({model_name})", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    out_file = FIGURES_DIR / save_name
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file
