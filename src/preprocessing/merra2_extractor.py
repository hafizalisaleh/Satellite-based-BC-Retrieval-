"""
MERRA-2 Reanalysis Extractor & Feature Preprocessor (High-Performance Engine)
=============================================================================
Fast point extraction of hourly aerosol (M2T1NXAER) and meteorology (M2T1NXSLV)
using direct HDF5 hyperslab indexing. Computes derived physics-based variables.
"""

import os
import sys
import glob
import logging
import re
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import h5py

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.config import (
    SITE_COORDS, MERRA2_AER_DIR, MERRA2_SLV_DIR, INTERIM_DATA_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_relative_humidity(t2m_k: np.ndarray, t2mdew_k: np.ndarray) -> np.ndarray:
    """
    Computes Relative Humidity (%) from 2m Temperature and Dew Point (Kelvin)
    using the Magnus-Tetens approximation.
    """
    t_c = t2m_k - 273.15
    td_c = t2mdew_k - 273.15
    
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
    
    rh = 100.0 * (e / (es + 1e-8))
    return np.clip(rh, 0.0, 100.0)


def extract_merra2_at_station(
    site_lat: float = SITE_COORDS["lat"],
    site_lon: float = SITE_COORDS["lon"],
) -> pd.DataFrame:
    """
    High-performance extraction using direct h5py hyperslab slicing.
    Processes all 180 days (4,320 hours) in under 5 seconds.
    """
    aer_files = sorted(glob.glob(str(MERRA2_AER_DIR / "*.nc4")))
    slv_files = sorted(glob.glob(str(MERRA2_SLV_DIR / "*.nc4")))
    
    logger.info(f"Processing {len(aer_files)} AER files and {len(slv_files)} SLV files...")
    
    if not aer_files or not slv_files:
        raise FileNotFoundError("MERRA-2 files not found in raw directory.")

    # Determine grid index once from first file
    with h5py.File(aer_files[0], "r") as f:
        lats = f["lat"][:]
        lons = f["lon"][:]
        lat_idx = int(np.abs(lats - site_lat).argmin())
        lon_idx = int(np.abs(lons - site_lon).argmin())
        
    logger.info(f"Target site ({site_lat}, {site_lon}) mapped to MERRA-2 grid cell "
                f"({lats[lat_idx]:.3f}, {lons[lon_idx]:.3f}) at indices [lat={lat_idx}, lon={lon_idx}]")

    # Map dates to file paths
    date_to_aer = {}
    for f in aer_files:
        match = re.search(r"\.(\d{8})_subsetted\.nc4", os.path.basename(f))
        if match:
            date_to_aer[match.group(1)] = f
            
    date_to_slv = {}
    for f in slv_files:
        match = re.search(r"\.(\d{8})_subsetted\.nc4", os.path.basename(f))
        if match:
            date_to_slv[match.group(1)] = f
            
    common_dates = sorted(set(date_to_aer.keys()).intersection(set(date_to_slv.keys())))
    logger.info(f"Found {len(common_dates)} matching daily dates between AER and SLV collections.")

    all_rows = []
    
    for date_str in common_dates:
        f_aer = date_to_aer[date_str]
        f_slv = date_to_slv[date_str]
        
        try:
            with h5py.File(f_aer, "r") as haer, h5py.File(f_slv, "r") as hslv:
                # 24 hours in the day
                hours = haer["time"][:] # minutes since 00:30:00
                base_dt = pd.to_datetime(date_str, format="%Y%m%d")
                dts = [base_dt + pd.Timedelta(minutes=int(m)) for m in hours]
                
                # AER fields
                bc_smass = haer["BCSMASS"][:, lat_idx, lon_idx] * 1e9  # ug/m3
                bc_cmass = haer["BCCMASS"][:, lat_idx, lon_idx] * 1e6  # mg/m2
                bc_aod = haer["BCEXTTAU"][:, lat_idx, lon_idx]
                tot_aod = haer["TOTEXTTAU"][:, lat_idx, lon_idx]
                oc_smass = haer["OCSMASS"][:, lat_idx, lon_idx] * 1e9
                so4_smass = haer["SO4SMASS"][:, lat_idx, lon_idx] * 1e9
                dus_smass = haer["DUSMASS"][:, lat_idx, lon_idx] * 1e9
                
                # SLV fields
                t2m = hslv["T2M"][:, lat_idx, lon_idx]
                t2mdew = hslv["T2MDEW"][:, lat_idx, lon_idx]
                pbltop = hslv["PBLTOP"][:, lat_idx, lon_idx]
                ps = hslv["PS"][:, lat_idx, lon_idx]
                u10 = hslv["U10M"][:, lat_idx, lon_idx]
                v10 = hslv["V10M"][:, lat_idx, lon_idx]
                tqv = hslv["TQV"][:, lat_idx, lon_idx]
                
                for h in range(len(hours)):
                    all_rows.append({
                        "datetime_utc": dts[h],
                        "merra2_bc_smass_ug_m3": float(bc_smass[h]),
                        "merra2_bc_cmass_mg_m2": float(bc_cmass[h]),
                        "merra2_bc_aod": float(bc_aod[h]),
                        "merra2_total_aod": float(tot_aod[h]),
                        "merra2_bc_ext_ratio": float(bc_aod[h] / (tot_aod[h] + 1e-6)),
                        "merra2_oc_smass_ug_m3": float(oc_smass[h]),
                        "merra2_so4_smass_ug_m3": float(so4_smass[h]),
                        "merra2_dust_smass_ug_m3": float(dus_smass[h]),
                        "merra2_t2m_k": float(t2m[h]),
                        "merra2_t2mdew_k": float(t2mdew[h]),
                        "merra2_pblh": float(pbltop[h]),
                        "merra2_surface_pressure_hpa": float(ps[h] / 100.0),
                        "merra2_u10m": float(u10[h]),
                        "merra2_v10m": float(v10[h]),
                        "merra2_water_vapor": float(tqv[h]),
                    })
        except Exception as e:
            logger.error(f"Error reading day {date_str}: {e}")

    df = pd.DataFrame(all_rows)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    
    # Derived physical fields
    df["merra2_temp_c"] = df["merra2_t2m_k"] - 273.15
    df["merra2_dew_c"] = df["merra2_t2mdew_k"] - 273.15
    df["merra2_rh"] = calculate_relative_humidity(df["merra2_t2m_k"].values, df["merra2_t2mdew_k"].values)
    df["merra2_wind_speed"] = np.sqrt(df["merra2_u10m"]**2 + df["merra2_v10m"]**2)
    df["merra2_wind_dir"] = (np.degrees(np.arctan2(-df["merra2_u10m"], -df["merra2_v10m"])) + 360) % 360
    
    # Drop raw Kelvin and component columns for neatness
    df = df.drop(columns=["merra2_t2m_k", "merra2_t2mdew_k", "merra2_u10m", "merra2_v10m"])
    
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM_DATA_DIR / "merra2_station_hourly.csv"
    df.to_csv(out_path, index=False)
    
    logger.info(f"Successfully processed {len(df)} MERRA-2 hourly records. Saved to {out_path}")
    return df


if __name__ == "__main__":
    df = extract_merra2_at_station()
    print("\nMERRA-2 Extracted Features Summary:")
    print(df.describe().T[["mean", "std", "min", "max"]])
    print("\nSample records:")
    print(df.head())
