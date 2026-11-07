"""
Sentinel-5P TROPOMI UV Aerosol Index Extractor
==============================================
Extracts daily overpass UVAI (354/388 nm, 340/380 nm) from S5P L2 AER_AI products
for the monitoring station coordinates with QA cloud screening.
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

from configs.config import SITE_COORDS, TROPOMI_DIR, INTERIM_DATA_DIR, TROPOMI_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def haversine_distance_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Computes great-circle distance in kilometers between a point and a grid of points.
    """
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0)**2 + 
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0)**2)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c


def extract_single_tropomi_file(
    file_path: str,
    site_lat: float = SITE_COORDS["lat"],
    site_lon: float = SITE_COORDS["lon"],
    max_dist_km: float = TROPOMI_CONFIG["max_distance_km"],
    qa_thresh: float = TROPOMI_CONFIG["qa_threshold"],
) -> Optional[Dict]:
    """
    Extracts UVAI and QA from a single TROPOMI S5P NetCDF file.
    """
    try:
        with h5py.File(file_path, "r") as h:
            prod = h["PRODUCT"]
            lats = prod["latitude"][0]  # shape: (scanline, ground_pixel)
            lons = prod["longitude"][0]
            
            # Quick bounding box filter to avoid full haversine if far away
            deg_margin = max_dist_km / 111.0 + 0.5
            if not ((lats.min() <= site_lat <= lats.max()) and 
                    (lons.min() <= site_lon <= lons.max())):
                # Swath does not cover the point
                return None
                
            dist_km = haversine_distance_km(site_lat, site_lon, lats, lons)
            min_idx = np.unravel_index(np.nanargmin(dist_km), dist_km.shape)
            min_dist = dist_km[min_idx]
            
            if min_dist > max_dist_km:
                return None
                
            qa = float(prod["qa_value"][0][min_idx])
            ai_354 = float(prod["aerosol_index_354_388"][0][min_idx])
            ai_340 = float(prod["aerosol_index_340_380"][0][min_idx])
            ai_335 = float(prod["aerosol_index_335_367"][0][min_idx])
            
            # Check QA threshold
            if qa < qa_thresh or np.isnan(ai_354):
                return None
                
            # Extract overpass timestamp
            # Format in filename: S5P_OFFL_L2__AER_AI_YYYYMMDDTHHMMSS_...
            match = re.search(r"AER_AI_(\d{8}T\d{6})", os.path.basename(file_path))
            if match:
                dt_utc = pd.to_datetime(match.group(1), format="%Y%m%dT%H%M%S")
            else:
                dt_utc = pd.to_datetime(prod["time_utc"][0][min_idx[0]][0].decode("utf-8"))

            # Also compute local radius mean (within 10 km) for spatial robustness
            nearby_mask = (dist_km <= 10.0) & (prod["qa_value"][0] >= qa_thresh)
            ai_nearby = prod["aerosol_index_354_388"][0][nearby_mask]
            valid_nearby = ai_nearby[~np.isnan(ai_nearby)]
            ai_mean_10km = float(np.mean(valid_nearby)) if len(valid_nearby) > 0 else ai_354

            return {
                "datetime_utc": dt_utc,
                "date": dt_utc.date(),
                "tropomi_uvai": ai_354,
                "tropomi_uvai_10km_mean": ai_mean_10km,
                "tropomi_uvai_340": ai_340,
                "tropomi_uvai_335": ai_335,
                "tropomi_qa": qa,
                "tropomi_distance_km": min_dist,
                "tropomi_pixel_lat": float(lats[min_idx]),
                "tropomi_pixel_lon": float(lons[min_idx]),
            }
            
    except Exception as e:
        logger.debug(f"Error extracting {file_path}: {e}")
        return None


def extract_all_tropomi(
    tropomi_dir: Path = TROPOMI_DIR,
) -> pd.DataFrame:
    """
    Iterates over all TROPOMI files and creates a daily overpass timeseries.
    """
    files = sorted(glob.glob(str(tropomi_dir / "*.nc")))
    logger.info(f"Scanning {len(files)} TROPOMI files for site collocation...")
    
    records = []
    for f in files:
        rec = extract_single_tropomi_file(f)
        if rec is not None:
            records.append(rec)
            
    if not records:
        raise ValueError("No valid TROPOMI overpass observations found near site!")
        
    df = pd.DataFrame(records)
    # If multiple orbits in one day, keep the one closest to station
    df = df.sort_values(["datetime_utc", "tropomi_distance_km"])
    # Group by date and pick best observation
    df_daily = df.groupby("date").first().reset_index()
    df_daily["datetime_utc"] = pd.to_datetime(df_daily["datetime_utc"])
    
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM_DATA_DIR / "tropomi_station_overpass.csv"
    df_daily.to_csv(out_path, index=False)
    
    logger.info(f"Extracted {len(df_daily)} valid TROPOMI overpass days. Saved to {out_path}")
    return df_daily


if __name__ == "__main__":
    df = extract_all_tropomi()
    print("\nTROPOMI UVAI Summary Statistics:")
    print(df[["tropomi_uvai", "tropomi_uvai_10km_mean", "tropomi_distance_km"]].describe())
    print("\nSample records:")
    print(df.head())
