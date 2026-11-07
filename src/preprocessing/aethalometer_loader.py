"""
Aethalometer (Magee AE33) Ground Truth Loader & Preprocessor
============================================================
Parses daily AE33 dat files, performs QA screening, converts units (ng/m3 -> ug/m3),
and generates hourly and daily aggregated ground-level Black Carbon benchmarks.
"""

import os
import sys
import glob
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Optional, Tuple
import pandas as pd
import numpy as np

from configs.config import AETHALOMETER_DIRS, INTERIM_DATA_DIR, COLLOCATION_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_single_ae33_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    Parses a single Magee AE33 .dat file.
    Skips instrument header lines, cleans semicolon artifacts, and extracts BC channels.
    """
    try:
        # AE33 header starts after metadata lines (row index 5 is the header line)
        df = pd.read_csv(file_path, sep=r"\s+", skiprows=5, header=0, low_memory=False)
        
        # Strip trailing semicolons from column names
        df.columns = [c.rstrip(";") for c in df.columns]
        
        # Verify required columns exist
        date_col = [c for c in df.columns if "Date" in c]
        time_col = [c for c in df.columns if "Time(" in c]
        
        if not date_col or not time_col or "BC6" not in df.columns:
            logger.warning(f"File {file_path} missing required columns. Skipped.")
            return None
        
        date_c, time_c = date_col[0], time_col[0]
        
        # Filter valid measurements (Status == 0)
        if "Status" in df.columns:
            df["Status"] = pd.to_numeric(df["Status"], errors="coerce")
            df = df[df["Status"] == 0].copy()
        
        if df.empty:
            return None
            
        # Parse datetime (local time PKT UTC+5)
        dt_str = df[date_c].astype(str) + " " + df[time_c].astype(str)
        df["datetime_local"] = pd.to_datetime(dt_str, format="%Y/%m/%d %H:%M:%S", errors="coerce")
        df = df.dropna(subset=["datetime_local"])
        
        # Compute UTC datetime
        tz_offset = pd.Timedelta(hours=COLLOCATION_CONFIG["timezone_offset_hours"])
        df["datetime_utc"] = df["datetime_local"] - tz_offset
        
        # Convert BC channels from ng/m3 to ug/m3
        # BC6 (880 nm) is standard eBC; BC1 (370 nm) is UVPM / Brown Carbon proxy
        for col in ["BC1", "BC2", "BC3", "BC4", "BC5", "BC6", "BC7"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # Remove unphysical negative numbers or extreme artifacts
                df.loc[df[col] < 0, col] = np.nan
                df[f"{col}_ug_m3"] = df[col] / 1000.0
                
        # Primary target
        df["BC_target_ug_m3"] = df["BC6_ug_m3"]
        
        # Biomass burning %
        if "BB(%)" in df.columns:
            df["BB_percent"] = pd.to_numeric(df["BB(%)"], errors="coerce")
        else:
            df["BB_percent"] = np.nan
            
        # Meteorology / Diagnostic from AE33 internal sensors
        for diag in ["Temperature(°C)", "Pressure(Pa)", "Flow1", "Flow2"]:
            if diag in df.columns:
                df[diag] = pd.to_numeric(df[diag], errors="coerce")

        keep_cols = [
            "datetime_local", "datetime_utc", "BC_target_ug_m3", "BC1_ug_m3", 
            "BB_percent", "Temperature(°C)", "Pressure(Pa)"
        ]
        keep_cols = [c for c in keep_cols if c in df.columns]
        
        return df[keep_cols]
        
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        return None


def load_all_aethalometer_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scans all AE33 measurement files, aggregates them, and computes:
    1. Hourly averaged time series (with local and UTC timestamps)
    2. Daily averaged time series
    """
    all_files = []
    for d in AETHALOMETER_DIRS:
        if os.path.exists(d):
            # Only match measurement files (starts with AE33_AE33), not log files
            files = glob.glob(os.path.join(d, "AE33_AE33*.dat"))
            all_files.extend(files)
        else:
            logger.warning(f"Directory {d} does not exist.")
            
    all_files = sorted(set(all_files))
    logger.info(f"Found {len(all_files)} AE33 measurement files to process.")
    
    records = []
    for f in all_files:
        df_file = parse_single_ae33_file(f)
        if df_file is not None and not df_file.empty:
            records.append(df_file)
            
    if not records:
        raise ValueError("No valid Aethalometer data could be loaded!")
        
    full_df = pd.concat(records, ignore_index=True)
    full_df = full_df.sort_values("datetime_utc").drop_duplicates(subset=["datetime_utc"])
    
    logger.info(f"Loaded {len(full_df)} 1-minute ground BC records from "
                f"{full_df['datetime_utc'].min()} to {full_df['datetime_utc'].max()}")
                
    # Hourly Resampling (indexed by UTC)
    full_df = full_df.set_index("datetime_utc")
    hourly_df = full_df.resample("1h").agg({
        "BC_target_ug_m3": "mean",
        "BC1_ug_m3": "mean",
        "BB_percent": "mean",
        "Temperature(°C)": "mean",
        "Pressure(Pa)": "mean",
    }).reset_index()
    
    # Re-derive local time for convenience
    tz_offset = pd.Timedelta(hours=COLLOCATION_CONFIG["timezone_offset_hours"])
    hourly_df["datetime_local"] = hourly_df["datetime_utc"] + tz_offset
    
    # Daily Resampling
    daily_df = hourly_df.set_index("datetime_utc").resample("1D").agg({
        "BC_target_ug_m3": "mean",
        "BC1_ug_m3": "mean",
        "BB_percent": "mean",
    }).reset_index()
    daily_df["date"] = daily_df["datetime_utc"].dt.date
    
    # Save interim files
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    hourly_path = INTERIM_DATA_DIR / "aethalometer_hourly.csv"
    daily_path = INTERIM_DATA_DIR / "aethalometer_daily.csv"
    
    hourly_df.to_csv(hourly_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    
    logger.info(f"Saved hourly ground BC dataset to: {hourly_path} (Shape: {hourly_df.shape})")
    logger.info(f"Saved daily ground BC dataset to: {daily_path} (Shape: {daily_df.shape})")
    
    return hourly_df, daily_df


if __name__ == "__main__":
    hourly, daily = load_all_aethalometer_data()
    print("\nHourly Summary Statistics:")
    print(hourly["BC_target_ug_m3"].describe())
    print("\nFirst 5 rows:")
    print(hourly.head())
