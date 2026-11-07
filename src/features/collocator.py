"""
Spatiotemporal Collocator & Dataset Builder
===========================================
Aligns ground-based Aethalometer BC with Sentinel-5P TROPOMI UVAI,
MODIS MAIAC AOD, and MERRA-2 Reanalysis meteorology/aerosol fields.
Generates both:
1. Hourly Collocated Dataset (Thousands of points for deep model learning)
2. Daily Satellite-Overpass Collocated Dataset (Specific to satellite daytime retrieval)
"""

import sys
import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from src.features.feature_engineering import add_physics_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_collocated_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads all interim extracted files, matches them in time and space,
    applies feature engineering, and exports machine-learning-ready CSVs.
    """
    ae_path = INTERIM_DATA_DIR / "aethalometer_hourly.csv"
    merra_path = INTERIM_DATA_DIR / "merra2_station_hourly.csv"
    trop_path = INTERIM_DATA_DIR / "tropomi_station_overpass.csv"
    modis_path = INTERIM_DATA_DIR / "modis_station_overpass.csv"
    
    if not ae_path.exists():
        raise FileNotFoundError(f"Missing {ae_path}. Run aethalometer_loader.py first.")
    if not merra_path.exists():
        raise FileNotFoundError(f"Missing {merra_path}. Run merra2_extractor.py first.")
        
    logger.info("Loading interim files for collocation...")
    df_ae = pd.read_csv(ae_path)
    df_merra = pd.read_csv(merra_path)
    
    df_ae["datetime_utc"] = pd.to_datetime(df_ae["datetime_utc"])
    df_merra["datetime_utc"] = pd.to_datetime(df_merra["datetime_utc"])
    
    # 1. Base Hourly Merge: Ground AE33 + MERRA-2
    # Round or match nearest hour
    df_ae["time_hour"] = df_ae["datetime_utc"].dt.floor("h")
    df_merra["time_hour"] = df_merra["datetime_utc"].dt.floor("h")
    
    hourly = pd.merge(df_ae, df_merra.drop(columns=["datetime_utc"]), on="time_hour", how="inner")
    hourly["date"] = hourly["datetime_utc"].dt.date
    
    logger.info(f"Hourly Ground + MERRA-2 matched shape: {hourly.shape}")
    
    # 2. Add Daily TROPOMI UVAI if available
    if trop_path.exists():
        df_trop = pd.read_csv(trop_path)
        df_trop["date"] = pd.to_datetime(df_trop["date"]).dt.date
        trop_cols = ["date", "tropomi_uvai", "tropomi_uvai_10km_mean", "tropomi_distance_km"]
        trop_cols = [c for c in trop_cols if c in df_trop.columns]
        hourly = pd.merge(hourly, df_trop[trop_cols], on="date", how="left")
        logger.info(f"Merged TROPOMI UVAI into hourly dataset. Valid UVAI count: {hourly['tropomi_uvai'].notna().sum()}")
    else:
        logger.warning("TROPOMI overpass file not found. TROPOMI features omitted.")

    # 3. Add Daily MODIS MAIAC AOD if available
    if modis_path.exists():
        df_modis = pd.read_csv(modis_path)
        df_modis["date"] = pd.to_datetime(df_modis["date"]).dt.date
        modis_cols = ["date", "modis_aod_550", "modis_angstrom", "modis_column_wv"]
        modis_cols = [c for c in modis_cols if c in df_modis.columns]
        hourly = pd.merge(hourly, df_modis[modis_cols], on="date", how="left")
        logger.info(f"Merged MODIS MAIAC AOD into hourly dataset. Valid MODIS count: {hourly['modis_aod_550'].notna().sum()}")
    else:
        logger.warning("MODIS overpass file not found. MODIS features omitted.")

    # 4. Apply Feature Engineering
    hourly = add_physics_features(hourly)
    
    # Clean target
    hourly = hourly.dropna(subset=["BC_target_ug_m3"])
    hourly = hourly[hourly["BC_target_ug_m3"] > 0].copy()
    
    # 5. Build Satellite Overpass Dataset (Daytime satellite window: 07:00 to 10:00 UTC / 12:00 to 15:00 PKT)
    overpass_mask = hourly["datetime_utc"].dt.hour.isin([7, 8, 9, 10])
    df_overpass_hours = hourly[overpass_mask].copy()
    
    # Aggregate to daily overpass mean
    overpass_daily = df_overpass_hours.groupby("date").agg({
        col: "mean" for col in df_overpass_hours.columns 
        if col not in ["date", "datetime_utc", "datetime_local", "time_hour"]
    }).reset_index()
    overpass_daily["datetime_utc"] = pd.to_datetime(overpass_daily["date"]) + pd.Timedelta(hours=8, minutes=30)
    
    # Re-apply cyclical temporal features on overpass daily
    overpass_daily = add_physics_features(overpass_daily)
    
    # Save processed files
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    hourly_out = PROCESSED_DATA_DIR / "collocated_bc_hourly.csv"
    overpass_out = PROCESSED_DATA_DIR / "collocated_bc_overpass.csv"
    
    hourly.to_csv(hourly_out, index=False)
    overpass_daily.to_csv(overpass_out, index=False)
    
    logger.info(f"Saved Final Hourly Collocated Dataset: {hourly_out} (Shape: {hourly.shape})")
    logger.info(f"Saved Final Overpass Collocated Dataset: {overpass_out} (Shape: {overpass_daily.shape})")
    
    return hourly, overpass_daily


if __name__ == "__main__":
    hourly, overpass = build_collocated_datasets()
    print("\nHourly Target (BC_target_ug_m3) Stats:")
    print(hourly["BC_target_ug_m3"].describe())
    print("\nOverpass Target (BC_target_ug_m3) Stats:")
    print(overpass["BC_target_ug_m3"].describe())
