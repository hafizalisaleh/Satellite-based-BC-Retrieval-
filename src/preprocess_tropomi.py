from pathlib import Path
import numpy as np
import pandas as pd
from netCDF4 import Dataset

RAW_DIR = Path("data/raw/tropomi")
OUT_FILE = Path("data/processed/tropomi_site.csv")

SITE_LAT = 31.479213798934566
SITE_LON = 74.26487633895766

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def find_variable(group, name):
    if name in group.variables:
        return group.variables[name]

    for sub_name, sub_group in group.groups.items():
        result = find_variable(sub_group, name)
        if result is not None:
            return result

    return None


def find_group(group, name):
    if name in group.groups:
        return group.groups[name]

    for sub_name, sub_group in group.groups.items():
        result = find_group(sub_group, name)
        if result is not None:
            return result

    return None


records = []

for nc_file in sorted(RAW_DIR.glob("*.nc")):
    print(f"Processing: {nc_file.name}")

    with Dataset(nc_file, "r") as ds:
        lat_var = find_variable(ds, "latitude")
        lon_var = find_variable(ds, "longitude")
        ai_340_380_var = find_variable(ds, "aerosol_index_340_380")
        ai_354_388_var = find_variable(ds, "aerosol_index_354_388")
        qa_var = find_variable(ds, "qa_value")

        if lat_var is None or lon_var is None:
            raise RuntimeError(f"Latitude/longitude not found in {nc_file.name}")

        if ai_340_380_var is None and ai_354_388_var is None:
            raise RuntimeError(f"Aerosol Index variables not found in {nc_file.name}")

        lat = np.asarray(lat_var[:])
        lon = np.asarray(lon_var[:])

        # TROPOMI latitude/longitude can be 2-D or 3-D with a time/orbit dimension.
        lat = np.squeeze(lat)
        lon = np.squeeze(lon)

        valid_geo = np.isfinite(lat) & np.isfinite(lon)

        if not np.any(valid_geo):
            print("  No valid geolocation pixels found.")
            continue

        distance = np.full(lat.shape, np.inf, dtype=np.float64)
        distance[valid_geo] = (
            (lat[valid_geo] - SITE_LAT) ** 2
            + (lon[valid_geo] - SITE_LON) ** 2
        )

        flat_index = np.argmin(distance)
        pixel_index = np.unravel_index(flat_index, lat.shape)

        nearest_lat = float(lat[pixel_index])
        nearest_lon = float(lon[pixel_index])

        def get_value(var):
            if var is None:
                return np.nan

            arr = np.asarray(var[:])
            arr = np.squeeze(arr)

            try:
                value = arr[pixel_index]
            except IndexError:
                return np.nan

            value = np.asarray(value).item()

            if np.ma.is_masked(value):
                return np.nan

            try:
                value = float(value)
            except (TypeError, ValueError):
                return np.nan

            if not np.isfinite(value):
                return np.nan

            return value

        ai_340_380 = get_value(ai_340_380_var)
        ai_354_388 = get_value(ai_354_388_var)
        qa_value = get_value(qa_var)

        # Extract sensing start time from filename.
        parts = nc_file.name.split("_")
        start_timestamp = None

        for part in parts:
            if "T" in part and len(part) >= 15:
                try:
                    start_timestamp = pd.to_datetime(part[:15], format="%Y%m%dT%H%M%S")
                    break
                except Exception:
                    pass

        records.append({
            "datetime": start_timestamp,
            "file": nc_file.name,
            "site_lat": SITE_LAT,
            "site_lon": SITE_LON,
            "pixel_lat": nearest_lat,
            "pixel_lon": nearest_lon,
            "aerosol_index_340_380": ai_340_380,
            "aerosol_index_354_388": ai_354_388,
            "qa_value": qa_value,
            "distance_deg": float(np.sqrt(distance[pixel_index]))
        })

result = pd.DataFrame(records)

if not result.empty:
    result = result.sort_values("datetime").reset_index(drop=True)

result.to_csv(OUT_FILE, index=False)

print("\nTROPOMI preprocessing complete.")
print(f"Input files: {len(list(RAW_DIR.glob('*.nc')))}")
print(f"Output records: {len(result)}")
print(f"Output: {OUT_FILE}")

if not result.empty:
    print("\nResult:")
    print(result.to_string(index=False))
