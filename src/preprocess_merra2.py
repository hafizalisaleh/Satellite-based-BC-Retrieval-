from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

AEROSOL_DIR = REPO_ROOT / "data" / "raw" / "merra2_aerosol"
METEO_DIR = REPO_ROOT / "data" / "raw" / "merra2_meteorology"

OUTPUT_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SITE_LAT = 31.4792138
SITE_LON = 74.2648763


# -------------------------------------------------------------------
# MERRA-2 aerosol
# -------------------------------------------------------------------

def process_aerosol():
    files = sorted(AEROSOL_DIR.glob("*.nc4"))

    if not files:
        raise FileNotFoundError("No MERRA-2 aerosol files found.")

    frames = []

    for file in files:
        print(f"Processing aerosol: {file.name}")

        with xr.open_dataset(file) as ds:
            site = ds.sel(
                lat=SITE_LAT,
                lon=SITE_LON,
                method="nearest"
            )

            df = site[
                [
                    "BCCMASS",
                    "BCSMASS",
                    "BCEXTTAU",
                    "TOTEXTTAU",
                ]
            ].to_dataframe().reset_index()

            frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    result = result.sort_values("time")
    result = result.drop_duplicates(subset=["time"])

    return result


# -------------------------------------------------------------------
# MERRA-2 meteorology
# -------------------------------------------------------------------

def process_meteorology():
    files = sorted(METEO_DIR.glob("*.nc4"))

    if not files:
        raise FileNotFoundError("No MERRA-2 meteorology files found.")

    frames = []

    for file in files:
        print(f"Processing meteorology: {file.name}")

        with xr.open_dataset(file) as ds:
            site = ds.sel(
                lat=SITE_LAT,
                lon=SITE_LON,
                method="nearest"
            )

            df = site[
                [
                    "T2M",
                    "PS",
                    "PBLTOP",
                    "V10M",
                    "U10M",
                    "QV2M",
                ]
            ].to_dataframe().reset_index()

            frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    result = result.sort_values("time")
    result = result.drop_duplicates(subset=["time"])

    # Calculate 10 m wind speed from its components.
    result["WIND_SPEED"] = np.sqrt(
        result["U10M"] ** 2 +
        result["V10M"] ** 2
    )

    return result


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    print("Starting MERRA-2 preprocessing...\n")

    aerosol = process_aerosol()
    meteorology = process_meteorology()

    aerosol_output = OUTPUT_DIR / "merra2_aerosol_site.csv"
    meteorology_output = OUTPUT_DIR / "merra2_meteorology_site.csv"

    aerosol.to_csv(aerosol_output, index=False)
    meteorology.to_csv(meteorology_output, index=False)

    print("\nPreprocessing complete.")
    print(f"Aerosol observations: {len(aerosol)}")
    print(f"Meteorology observations: {len(meteorology)}")

    print(f"\nSaved:")
    print(aerosol_output)
    print(meteorology_output)


if __name__ == "__main__":
    main()