from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from pyhdf.SD import SD, SDC


# --------------------------------------------------
# Paths
# --------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = REPO_ROOT / "data" / "raw" / "modis" / "MCD19A2"
OUTPUT_FILE = REPO_ROOT / "data" / "processed" / "modis_site.csv"


# --------------------------------------------------
# MODIS site / grid information
# --------------------------------------------------

SITE_LAT = 31.479213798934566
SITE_LON = 74.26487633895766

# MCD19A2 h24v05 site pixel
PIXEL_X = 400
PIXEL_Y = 1022

# MCD19A2 AOD scale factor
AOD_SCALE = 0.001

# Fill value
AOD_FILL = -28672


# --------------------------------------------------
# QA decoding
# --------------------------------------------------

def decode_qa(qa):
    """
    Decode MCD19A2 AOD_QA bit fields.
    """

    qa = int(qa)

    cloud_mask = qa & 0b111
    land_water_snow_mask = (qa >> 3) & 0b11
    adjacency_mask = (qa >> 5) & 0b111
    aod_quality = (qa >> 8) & 0b1111
    glint_mask = (qa >> 12) & 0b1
    aerosol_model = (qa >> 13) & 0b11

    return {
        "cloud_mask": cloud_mask,
        "land_water_snow_mask": land_water_snow_mask,
        "adjacency_mask": adjacency_mask,
        "aod_quality": aod_quality,
        "glint_mask": glint_mask,
        "aerosol_model": aerosol_model,
    }


# --------------------------------------------------
# Process one HDF file
# --------------------------------------------------

def process_file(file_path):
    hdf = SD(str(file_path), SDC.READ)

    aod047 = hdf.select("Optical_Depth_047")
    aod055 = hdf.select("Optical_Depth_055")
    qa_data = hdf.select("AOD_QA")

    # Read all 4 orbit layers at the site pixel
    raw047 = np.asarray(
        aod047[:, PIXEL_Y, PIXEL_X]
    ).astype(np.int32)

    raw055 = np.asarray(
        aod055[:, PIXEL_Y, PIXEL_X]
    ).astype(np.int32)

    qa = np.asarray(
        qa_data[:, PIXEL_Y, PIXEL_X]
    ).astype(np.int32)

    # Get orbit timestamps
    orbit_timestamps = hdf.attributes()["Orbit_time_stamp"].split()

    hdf.end()

    rows = []

    # Date from filename, e.g. A2025244 -> 2025 day 244
    filename = file_path.name
    date_code = filename.split(".")[1]  # A2025244

    year = int(date_code[1:5])
    day_of_year = int(date_code[5:])

    base_date = datetime.strptime(
        f"{year} {day_of_year}",
        "%Y %j"
    ).date()

    for orbit_idx in range(len(raw047)):

        qa_info = decode_qa(qa[orbit_idx])

        # Convert raw AOD to physical value
        if raw047[orbit_idx] == AOD_FILL:
            aod047_value = np.nan
        else:
            aod047_value = raw047[orbit_idx] * AOD_SCALE

        if raw055[orbit_idx] == AOD_FILL:
            aod055_value = np.nan
        else:
            aod055_value = raw055[orbit_idx] * AOD_SCALE

        # AOD is considered usable only when:
        # 1. AOD is not fill
        # 2. cloud mask = 001 (clear)
        # 3. AOD quality = 0000 (best quality)
        # 4. glint is not detected
        valid_best_quality = (
            not np.isnan(aod047_value)
            and not np.isnan(aod055_value)
            and qa_info["cloud_mask"] == 1
            and qa_info["aod_quality"] == 0
            and qa_info["glint_mask"] == 0
        )

        # Parse orbit timestamp
        timestamp_raw = orbit_timestamps[orbit_idx]

        # Example: 20252440330T
        timestamp_clean = timestamp_raw.rstrip("TA")

        orbit_time = datetime.strptime(
            timestamp_clean,
            "%Y%j%H%M"
        )

        rows.append(
            {
                "date": base_date,
                "orbit": orbit_idx + 1,
                "orbit_time": orbit_time,
                "aod_047": aod047_value,
                "aod_055": aod055_value,
                "qa": qa[orbit_idx],
                "cloud_mask": qa_info["cloud_mask"],
                "aod_quality": qa_info["aod_quality"],
                "glint_mask": qa_info["glint_mask"],
                "aerosol_model": qa_info["aerosol_model"],
                "valid_best_quality": valid_best_quality,
            }
        )

    return rows


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    files = sorted(INPUT_DIR.glob("*.hdf"))

    if not files:
        raise FileNotFoundError(
            f"No MODIS HDF files found in: {INPUT_DIR}"
        )

    print(f"Found {len(files)} MODIS HDF files.")

    all_rows = []

    for i, file_path in enumerate(files, start=1):

        try:
            rows = process_file(file_path)
            all_rows.extend(rows)

            print(
                f"[{i}/{len(files)}] "
                f"{file_path.name} -> {len(rows)} orbit records"
            )

        except Exception as e:
            print(
                f"[ERROR] {file_path.name}: {e}"
            )

    df = pd.DataFrame(all_rows)

    # Sort chronologically
    df = df.sort_values(
        ["date", "orbit_time", "orbit"]
    ).reset_index(drop=True)

    # Add site information
    df["site_lat"] = SITE_LAT
    df["site_lon"] = SITE_LON
    df["pixel_x"] = PIXEL_X
    df["pixel_y"] = PIXEL_Y

    # Save
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("MODIS PREPROCESSING SUMMARY")
    print("=" * 60)

    print(f"Input files:          {len(files)}")
    print(f"Orbit records:        {len(df)}")
    print(f"Output file:          {OUTPUT_FILE}")

    print(
        f"Valid best-quality:   "
        f"{df['valid_best_quality'].sum()}"
    )

    print(
        f"Invalid/missing:      "
        f"{(~df['valid_best_quality']).sum()}"
    )

    print(
        f"Date range:           "
        f"{df['date'].min()} to {df['date'].max()}"
    )

    print("\nValid AOD statistics:")

    valid = df[df["valid_best_quality"]]

    if len(valid) > 0:
        print(
            valid[
                ["aod_047", "aod_055"]
            ].describe()
        )
    else:
        print("No best-quality observations found.")

    print("\nQA distribution:")
    print(
        df["aod_quality"]
        .value_counts()
        .sort_index()
    )

    print("\nCloud-mask distribution:")
    print(
        df["cloud_mask"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()