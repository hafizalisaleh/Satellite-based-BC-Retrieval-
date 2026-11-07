"""
Project Configuration for Satellite-Based Black Carbon (BC) Retrieval
=====================================================================
Site: Lahore (Latitude: 31.479214, Longitude: 74.264876)
Study Window: September 2025 – February 2026
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Monitoring Site Coordinates (Ground-truth station in Lahore)
SITE_COORDS = {
    "name": "Lahore_Station",
    "lat": 31.479214,
    "lon": 74.264876,
    "elevation_m": 217.0,
}

# Regional Pilot Area for Gridded BC Estimation (Lahore Metropolitan & Surrounding)
# Defined roughly as [lat_min, lat_max, lon_min, lon_max] (~ 100km x 100km region)
PILOT_BBOX = {
    "lat_min": 31.0,
    "lat_max": 32.0,
    "lon_min": 73.8,
    "lon_max": 74.8,
}

# Raw Data Paths
AETHALOMETER_DIRS = [
    RAW_DATA_DIR / "athalometer" / "2025  BC Data" / "2025  BC Data",
    RAW_DATA_DIR / "athalometer" / "2026" / "2026",
]

MERRA2_AER_DIR = RAW_DATA_DIR / "merra_2" / "bce7e626-94c5-4376-84d8-883a94999a7b"
MERRA2_SLV_DIR = RAW_DATA_DIR / "merra_2" / "e7aea662-2cd4-4aae-8555-9818b6f164ff"
TROPOMI_DIR = RAW_DATA_DIR / "tropomi_2026"
MODIS_DIR = RAW_DATA_DIR / "modis_2026"

# Aethalometer AE33 Settings
AE33_CONFIG = {
    "target_channel": "BC6",  # 880 nm (Standard Equivalent Black Carbon - eBC)
    "biomass_channel": "BC1", # 370 nm (Brown Carbon / UVPM)
    "target_unit": "ug/m3",   # Input is in ng/m3 -> divide by 1000
    "valid_status": [0],      # 0 = normal measurement
}

# TROPOMI Settings
TROPOMI_CONFIG = {
    "variable": "aerosol_index_354_388",
    "qa_threshold": 50,       # qa_value >= 50 (or 0.5 depending on scale)
    "max_distance_km": 15.0,  # Max spatial search radius from site
}

# MODIS MAIAC Settings
MODIS_CONFIG = {
    "tile": "h24v05",
    "aod_variable": "Optical_Depth_055",
    "angstrom_variable": "AngstromExp_470-780",
    "scale_factor": 0.001,
    "window_size": 3,         # 3x3 pixel window (~3km x 3km) around site
    "max_aod": 5.0,
}

# Collocation Settings
COLLOCATION_CONFIG = {
    "time_window_minutes": 60, # +/- 60 min window around satellite overpass
    "timezone_offset_hours": 5.0, # Pakistan Standard Time (PKT = UTC+5)
}
