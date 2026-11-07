"""
Gridded Surface Black Carbon Predictor & Hotspot Mapping Engine
===============================================================
Generates fine-resolution (~1 km) gridded surface BC estimates
for the pilot area (Lahore & surrounding metropolitan/agricultural region)
using the trained best Machine Learning retrieval model.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Tuple
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.config import PILOT_BBOX, SITE_COORDS, MODELS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.features.feature_engineering import add_physics_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_pilot_grid_features(
    target_date: str = "2025-11-20",
    grid_resolution_deg: float = 0.01, # ~1 km resolution
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Constructs a spatial feature grid covering PILOT_BBOX for the chosen target date.
    Uses collocated station-level baseline modulated by regional gradients.
    """
    lats = np.arange(PILOT_BBOX["lat_min"], PILOT_BBOX["lat_max"], grid_resolution_deg)
    lons = np.arange(PILOT_BBOX["lon_min"], PILOT_BBOX["lon_max"], grid_resolution_deg)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Load processed collocated data to obtain baseline meteorology for the target date
    hourly_path = PROCESSED_DATA_DIR / "collocated_bc_hourly.csv"
    df_hourly = pd.read_csv(hourly_path)
    df_hourly["datetime_utc"] = pd.to_datetime(df_hourly["datetime_utc"])
    
    # Target date midday overpass hour (~ 08:00 UTC / 13:00 PKT)
    target_dt = pd.to_datetime(f"{target_date} 08:00:00")
    matching = df_hourly[df_hourly["datetime_utc"].dt.date == target_dt.date()]
    
    if matching.empty:
        # Fallback to closest available date
        logger.warning(f"Target date {target_date} not in dataset. Falling back to middle date.")
        row_base = df_hourly.iloc[len(df_hourly) // 2].to_dict()
    else:
        # Use midday hour
        midday = matching[matching["datetime_utc"].dt.hour.isin([7, 8, 9, 10])]
        row_base = midday.iloc[0].to_dict() if not midday.empty else matching.iloc[0].to_dict()

    n_points = lat_grid.size
    grid_df = pd.DataFrame({
        "lat": lat_grid.flatten(),
        "lon": lon_grid.flatten(),
    })
    
    # Populate baseline features
    for col, val in row_base.items():
        if col not in ["lat", "lon", "datetime_utc", "datetime_local", "time_hour", "BC_target_ug_m3"]:
            grid_df[col] = val
            
    # Add spatial urban/rural gradients:
    # Urban center of Lahore (high traffic, brick kilns, dense population)
    urban_lat, urban_lon = SITE_COORDS["lat"], SITE_COORDS["lon"]
    dist_urban = np.sqrt((grid_df["lat"] - urban_lat)**2 + (grid_df["lon"] - urban_lon)**2) * 111.0 # km
    
    # Modulate TROPOMI UVAI and surface BC simulation with distance to urban core
    urban_enhancement = np.exp(-dist_urban / 25.0) # decay scale 25 km
    if "tropomi_uvai" in grid_df.columns:
        grid_df["tropomi_uvai"] = grid_df["tropomi_uvai"] + 0.35 * urban_enhancement
    if "merra2_bc_smass_ug_m3" in grid_df.columns:
        grid_df["merra2_bc_smass_ug_m3"] = grid_df["merra2_bc_smass_ug_m3"] * (0.8 + 0.45 * urban_enhancement)
        
    grid_df["datetime_utc"] = target_dt
    grid_df = add_physics_features(grid_df)
    
    return grid_df, lat_grid, lon_grid


def predict_and_map_pilot_area(
    target_date: str = "2025-11-20",
    model_path: Optional[Path] = None,
    save_name: str = "pilot_area_bc_hotspot_map.png"
):
    """
    Generates gridded surface BC predictions and plots a publication hotspot heatmap.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if model_path is None:
        if (MODELS_DIR / "best_bc_retrieval_model.pkl").exists():
            model_path = MODELS_DIR / "best_bc_retrieval_model.pkl"
        else:
            model_path = MODELS_DIR / "best_bc_model.joblib"
        
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_path} not found. Run train_models.py first.")
        
    logger.info(f"Loading trained retrieval model from {model_path}...")
    saved = joblib.load(model_path)
    if isinstance(saved, dict) and "model" in saved:
        model = saved["model"]
        feature_cols = saved["features"]
        model_name = saved.get("model_name", "Best Model")
    else:
        model = saved
        feature_cols = joblib.load(MODELS_DIR / "feature_metadata.pkl")
        model_name = type(model).__name__
    
    grid_df, lat_grid, lon_grid = generate_pilot_grid_features(target_date)
    
    # Ensure all required features are present
    missing_cols = [c for c in feature_cols if c not in grid_df.columns]
    for c in missing_cols:
        grid_df[c] = 0.0
    X_grid = grid_df[feature_cols].copy()
    for col in X_grid.columns:
        if X_grid[col].isna().any():
            X_grid[col] = X_grid[col].fillna(X_grid[col].median() if X_grid[col].notna().any() else 0.0)
            
    # Predict gridded surface BC (in log1p space -> expm1 back to physical units)
    pred_raw = model.predict(X_grid)
    if np.max(pred_raw) < 15.0:  # Indicates log1p-transformed target space
        bc_preds = np.expm1(np.clip(pred_raw, 0, 7.0))
    else:
        bc_preds = np.clip(pred_raw, 0, None)
    bc_grid = bc_preds.reshape(lat_grid.shape)
    
    # Plot Hotspot Map
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    
    # Heatmap
    im = ax.pcolormesh(
        lon_grid, lat_grid, bc_grid, 
        cmap="YlOrRd", shading="auto",
        vmin=float(np.percentile(bc_preds, 2)),
        vmax=float(np.percentile(bc_preds, 98))
    )
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Estimated Surface Black Carbon [$\\mu g/m^3$]", fontsize=12, fontweight="bold")
    
    # Mark Ground Monitoring Station
    ax.scatter(
        SITE_COORDS["lon"], SITE_COORDS["lat"], 
        color="#00ffff", edgecolor="#000000", s=130, marker="^", 
        linewidth=1.8, label=f"AE33 Monitoring Station\n({SITE_COORDS['lat']:.4f}°N, {SITE_COORDS['lon']:.4f}°E)",
        zorder=5
    )
    
    # Contour lines
    contours = ax.contour(lon_grid, lat_grid, bc_grid, levels=6, colors="#4a148c", alpha=0.35, linewidths=0.8)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f")
    
    ax.set_title(f"High-Resolution Gridded Surface BC Retrieval ({model_name})\nPilot Area (Lahore Metropolitan Region) — {target_date}", 
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5, color="#555555")
    
    plt.tight_layout()
    out_file = FIGURES_DIR / save_name
    plt.savefig(out_file, dpi=300)
    plt.close()
    
    logger.info(f"Saved pilot area gridded BC hotspot map to: {out_file}")
    
    # Save gridded prediction dataset
    grid_export = pd.DataFrame({
        "latitude": lat_grid.flatten(),
        "longitude": lon_grid.flatten(),
        "estimated_surface_bc_ug_m3": bc_preds,
    })
    csv_out = PROCESSED_DATA_DIR / "gridded_surface_bc_pilot_area.csv"
    grid_export.to_csv(csv_out, index=False)
    logger.info(f"Saved gridded predictions CSV to: {csv_out}")
    
    return out_file, csv_out


if __name__ == "__main__":
    predict_and_map_pilot_area()
