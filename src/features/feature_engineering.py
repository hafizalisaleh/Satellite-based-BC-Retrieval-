"""
Physics-Informed Feature Engineering for Surface Black Carbon Retrieval
========================================================================
Implements atmospheric physics & chemistry transformations:
- Vertical column-to-surface conversion (AOD / PBLH, UVAI / PBLH, Column BC / PBLH)
- Atmospheric ventilation coefficient (PBLH * Wind Speed) & Stagnation index
- Hygroscopic and secondary aerosol indicators & Dew Point Depression
- Combustion and aerosol chemistry ratios (BC/OC, BC/SO4, BC AOD fraction)
- Multi-scale persistence, lag, and rolling statistics
- Seasonal and diurnal cyclical embeddings
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies comprehensive domain-specific atmospheric physics, chemistry,
    and multi-scale temporal transformations to collocated dataset.
    """
    df = df.copy()
    
    # 1. Boundary Layer Dynamics & Vertical Compression (PBLH in meters)
    if "merra2_pblh" in df.columns:
        if "merra2_surface_pressure_hpa" in df.columns:
            pblh_pa = df["merra2_surface_pressure_hpa"] * 100.0 - df["merra2_pblh"]
            df["pblh_meters"] = np.maximum(pblh_pa / (1.2 * 9.81), 25.0)
        else:
            df["pblh_meters"] = 500.0
            
        df["pblh_inverse"] = 1000.0 / (df["pblh_meters"] + 10.0)
        
        if "modis_aod_550" in df.columns:
            df["modis_aod_pblh_ratio"] = df["modis_aod_550"] / (df["pblh_meters"] + 10.0) * 1000.0
            
        if "merra2_total_aod" in df.columns:
            df["merra2_aod_pblh_ratio"] = df["merra2_total_aod"] / (df["pblh_meters"] + 10.0) * 1000.0
            
        if "tropomi_uvai" in df.columns:
            uvai_pos = np.maximum(df["tropomi_uvai"], 0.0)
            df["uvai_pblh_ratio"] = uvai_pos / (df["pblh_meters"] + 10.0) * 1000.0

    # 2. Atmospheric Ventilation & Stagnation (Micrometeorology)
    if "merra2_wind_speed" in df.columns:
        df["stagnation_factor"] = 1.0 / (df["merra2_wind_speed"] + 0.5)
        
        if "pblh_meters" in df.columns:
            # Ventilation coefficient (m^2/s) = PBLH * Wind Speed
            df["ventilation_coeff"] = df["pblh_meters"] * np.maximum(df["merra2_wind_speed"], 0.1)
            df["log_ventilation_coeff"] = np.log1p(df["ventilation_coeff"])
            df["inv_ventilation_coeff"] = 1000.0 / (df["ventilation_coeff"] + 10.0)
        
        if "merra2_rh" in df.columns:
            df["fog_smog_index"] = (df["merra2_rh"] / 100.0) * df["stagnation_factor"]

    # 3. Columnar Mass to Surface Concentration Proxies
    if "pblh_meters" in df.columns:
        if "merra2_bc_cmass_mg_m2" in df.columns:
            # Column mass in mg/m2 converted to ug/m3 via PBLH
            df["col_bc_surface_proxy"] = (df["merra2_bc_cmass_mg_m2"] * 1000.0) / (df["pblh_meters"] + 10.0)
        if "merra2_bc_aod" in df.columns:
            df["col_bc_ext_surface_proxy"] = (df["merra2_bc_aod"] * 1000.0) / (df["pblh_meters"] + 10.0)

    # 4. Absorbing Aerosol & Satellite Interactions
    if "tropomi_uvai" in df.columns and "merra2_total_aod" in df.columns:
        df["uvai_x_merra2_aod"] = df["tropomi_uvai"] * df["merra2_total_aod"]
    if "tropomi_uvai" in df.columns and "modis_aod_550" in df.columns:
        df["uvai_x_modis_aod"] = df["tropomi_uvai"] * df["modis_aod_550"]
    if "tropomi_uvai" in df.columns and "merra2_bc_smass_ug_m3" in df.columns:
        df["uvai_x_merra2_bc"] = df["tropomi_uvai"] * df["merra2_bc_smass_ug_m3"]

    # 5. Aerosol Chemistry Ratios & Log Transforms
    for col in ["merra2_bc_smass_ug_m3", "merra2_oc_smass_ug_m3", "merra2_so4_smass_ug_m3", 
                "merra2_dust_smass_ug_m3", "merra2_total_aod", "merra2_bc_aod"]:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(np.maximum(df[col], 0.0))

    if "merra2_bc_smass_ug_m3" in df.columns and "merra2_oc_smass_ug_m3" in df.columns:
        df["bc_to_oc_ratio"] = df["merra2_bc_smass_ug_m3"] / (df["merra2_oc_smass_ug_m3"] + 1e-4)
        df["bc_combustion_fraction"] = df["merra2_bc_smass_ug_m3"] / (df["merra2_bc_smass_ug_m3"] + df["merra2_oc_smass_ug_m3"] + 1e-5)
        
    if "merra2_bc_aod" in df.columns and "merra2_total_aod" in df.columns:
        df["bc_aod_fraction"] = df["merra2_bc_aod"] / (df["merra2_total_aod"] + 1e-6)

    if "merra2_so4_smass_ug_m3" in df.columns and "merra2_bc_smass_ug_m3" in df.columns:
        df["bc_to_so4_ratio"] = df["merra2_bc_smass_ug_m3"] / (df["merra2_so4_smass_ug_m3"] + 1e-5)

    if "merra2_temp_c" in df.columns and "merra2_dew_c" in df.columns:
        df["dew_point_depression"] = df["merra2_temp_c"] - df["merra2_dew_c"]

    # 6. Atmospheric Persistence & Multi-Scale Dynamics
    if "merra2_bc_smass_ug_m3" in df.columns:
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f"merra2_bc_lag{lag}"] = df["merra2_bc_smass_ug_m3"].shift(lag)
            if f"merra2_bc_smass_ug_m3_log" in df.columns:
                df[f"merra2_bc_log_lag{lag}"] = df["merra2_bc_smass_ug_m3_log"].shift(lag)
            if "merra2_wind_speed" in df.columns:
                df[f"wind_speed_lag{lag}"] = df["merra2_wind_speed"].shift(lag)
            if "pblh_meters" in df.columns:
                df[f"pblh_lag{lag}"] = df["pblh_meters"].shift(lag)

        for window in [3, 6, 12, 24]:
            df[f"merra2_bc_roll_mean_{window}"] = df["merra2_bc_smass_ug_m3"].rolling(window, min_periods=1).mean()
            df[f"merra2_bc_roll_std_{window}"] = df["merra2_bc_smass_ug_m3"].rolling(window, min_periods=1).std()
            df[f"merra2_bc_roll_max_{window}"] = df["merra2_bc_smass_ug_m3"].rolling(window, min_periods=1).max()
            df[f"merra2_bc_roll_min_{window}"] = df["merra2_bc_smass_ug_m3"].rolling(window, min_periods=1).min()
            if "pblh_meters" in df.columns:
                df[f"pblh_roll_mean_{window}"] = df["pblh_meters"].rolling(window, min_periods=1).mean()
            if "merra2_wind_speed" in df.columns:
                df[f"wind_roll_mean_{window}"] = df["merra2_wind_speed"].rolling(window, min_periods=1).mean()

        # Rates of change (finite differences)
        df["merra2_bc_diff1h"] = df["merra2_bc_smass_ug_m3"].diff(1)
        df["merra2_bc_diff3h"] = df["merra2_bc_smass_ug_m3"].diff(3)
        if "pblh_meters" in df.columns:
            df["pblh_diff1h"] = df["pblh_meters"].diff(1)
        if "merra2_temp_c" in df.columns:
            df["temp_diff1h"] = df["merra2_temp_c"].diff(1)

    # 7. Temporal Cyclical & Calendar Features
    if "datetime_utc" in df.columns:
        dt = pd.to_datetime(df["datetime_utc"])
        
        # Day of Year harmonic
        doy = dt.dt.dayofyear
        df["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
        df["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
        
        # Hour of Day harmonic
        hour = dt.dt.hour
        df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        
        # Month harmonic
        month = dt.dt.month
        df["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        df["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
        
        # Weekend & Season
        dow = dt.dt.dayofweek
        df["is_weekend"] = (dow >= 5).astype(int)
        df["is_smog_season"] = dt.dt.month.isin([10, 11, 12, 1]).astype(int)
        
    return df
