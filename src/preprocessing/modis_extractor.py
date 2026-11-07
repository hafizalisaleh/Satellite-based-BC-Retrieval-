"""
MODIS MAIAC (MCD19A2) AOD Extractor
===================================
Extracts high-resolution (1 km) daily aerosol optical depth (AOD 550nm, 470nm),
Angstrom Exponent, and Column Water Vapor from MAIAC tile h24v05 over the monitoring site.
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
from pyhdf.SD import SD, SDC

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.config import SITE_COORDS, MODIS_DIR, INTERIM_DATA_DIR, MODIS_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def latlon_to_sinusoidal_pixel(
    lat_deg: float, 
    lon_deg: float,
    x_ul: float = 6671703.118,
    y_ul: float = 4447802.078667,
    x_lr: float = 7783653.637667,
    y_lr: float = 3335851.559,
    nrows: int = 1200,
    ncols: int = 1200,
    R: float = 6371007.181,
) -> tuple[int, int]:
    """
    Converts WGS-84 Latitude and Longitude to MODIS Sinusoidal Tile pixel (row, col).
    """
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    
    x = R * lam * np.cos(phi)
    y = R * phi
    
    dx = (x_lr - x_ul) / float(ncols)
    dy = (y_ul - y_lr) / float(nrows)
    
    col = int((x - x_ul) / dx)
    row = int((y_ul - y) / dy)
    
    return row, col


def extract_single_modis_hdf(
    file_path: str,
    row: int,
    col: int,
    window: int = MODIS_CONFIG["window_size"],
) -> Optional[Dict]:
    """
    Extracts MAIAC AOD and Angstrom exponent from a single MCD19A2 HDF file.
    """
    try:
        # Extract Date from filename e.g. MCD19A2.A2026001.h24v05...
        match = re.search(r"\.A(\d{4})(\d{3})\.", os.path.basename(file_path))
        if not match:
            return None
            
        year = int(match.group(1))
        doy = int(match.group(2))
        date = pd.to_datetime(f"{year}-{doy}", format="%Y-%j").date()
        
        hdf = SD(file_path, SDC.READ)
        sds_names = list(hdf.datasets().keys())
        
        if "Optical_Depth_055" not in sds_names:
            hdf.end()
            return None
            
        sds_aod550 = hdf.select("Optical_Depth_055")
        aod550_data = sds_aod550.get()
        scale_aod = sds_aod550.attributes().get("scale_factor", 0.001)
        fill_aod = sds_aod550.attributes().get("_FillValue", -28672)
        
        # Optional datasets
        angstrom_data = hdf.select("AngstromExp_470-780").get() if "AngstromExp_470-780" in sds_names else None
        cwv_data = hdf.select("Column_WV").get() if "Column_WV" in sds_names else None
        
        r_start = max(0, row - window // 2)
        r_end = min(1200, row + window // 2 + 1)
        c_start = max(0, col - window // 2)
        c_end = min(1200, col + window // 2 + 1)
        
        # Check all available orbits in file
        num_orbits = aod550_data.shape[0] if aod550_data.ndim == 3 else 1
        best_orbit_aod = None
        best_orbit_ang = None
        best_orbit_cwv = None
        
        for orb in range(num_orbits):
            sub_aod = aod550_data[orb, r_start:r_end, c_start:c_end] if aod550_data.ndim == 3 else aod550_data[r_start:r_end, c_start:c_end]
            valid = sub_aod[sub_aod != fill_aod]
            
            if len(valid) > 0:
                mean_val = float(valid.mean()) * scale_aod
                if 0.0 <= mean_val <= MODIS_CONFIG["max_aod"]:
                    best_orbit_aod = mean_val
                    
                    if angstrom_data is not None:
                        sub_ang = angstrom_data[orb, r_start:r_end, c_start:c_end] if angstrom_data.ndim == 3 else angstrom_data[r_start:r_end, c_start:c_end]
                        valid_ang = sub_ang[sub_ang != fill_aod]
                        if len(valid_ang) > 0:
                            best_orbit_ang = float(valid_ang.mean()) * 0.001
                            
                    if cwv_data is not None:
                        sub_cwv = cwv_data[orb, r_start:r_end, c_start:c_end] if cwv_data.ndim == 3 else cwv_data[r_start:r_end, c_start:c_end]
                        valid_cwv = sub_cwv[sub_cwv != fill_aod]
                        if len(valid_cwv) > 0:
                            best_orbit_cwv = float(valid_cwv.mean()) * 0.001
                    break
                    
        hdf.end()
        
        if best_orbit_aod is not None:
            return {
                "date": date,
                "modis_aod_550": best_orbit_aod,
                "modis_angstrom": best_orbit_ang if best_orbit_ang is not None else np.nan,
                "modis_column_wv": best_orbit_cwv if best_orbit_cwv is not None else np.nan,
            }
        return None
        
    except Exception as e:
        logger.debug(f"Error reading MODIS file {file_path}: {e}")
        return None


def extract_all_modis(
    modis_dir: Path = MODIS_DIR,
    site_lat: float = SITE_COORDS["lat"],
    site_lon: float = SITE_COORDS["lon"],
) -> pd.DataFrame:
    """
    Extracts MODIS MAIAC AOD for all available HDF files over the monitoring station.
    """
    row, col = latlon_to_sinusoidal_pixel(site_lat, site_lon)
    logger.info(f"Target site Sinusoidal pixel index in tile h24v05: row={row}, col={col}")
    
    files = sorted(glob.glob(str(modis_dir / "*.hdf")))
    logger.info(f"Found {len(files)} MODIS MCD19A2 files to extract.")
    
    records = []
    for f in files:
        rec = extract_single_modis_hdf(f, row, col)
        if rec is not None:
            records.append(rec)
            
    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM_DATA_DIR / "modis_station_overpass.csv"
    df.to_csv(out_path, index=False)
    
    logger.info(f"Extracted {len(df)} valid MODIS AOD days. Saved to {out_path}")
    return df


if __name__ == "__main__":
    df = extract_all_modis()
    if not df.empty:
        print("\nMODIS MAIAC Summary:")
        print(df.describe())
        print("\nSample records:")
        print(df.head())
    else:
        print("No valid MODIS records extracted.")
