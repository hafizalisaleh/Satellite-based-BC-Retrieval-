from pathlib import Path
from pyhdf.SD import SD, SDC
import pandas as pd
import re

DATA_DIR = Path("data/raw/modis/MCD19A2")
CSV_FILE = Path("data/processed/modis_site.csv")

files = sorted(DATA_DIR.glob("*.hdf"))

print("=" * 60)
print("MODIS DATA VALIDATION")
print("=" * 60)

print(f"\nHDF files found: {len(files)}")

dimension_issues = []
timestamp_issues = []
orbit_distribution = {}

for i, f in enumerate(files, 1):
    hdf = SD(str(f), SDC.READ)

    aod047 = hdf.select("Optical_Depth_047").get()
    aod055 = hdf.select("Optical_Depth_055").get()
    qa = hdf.select("AOD_QA").get()

    n047 = len(aod047)
    n055 = len(aod055)
    nqa = len(qa)

    orbit_distribution[n047] = orbit_distribution.get(n047, 0) + 1

    if not (n047 == n055 == nqa):
        dimension_issues.append(
            (f.name, n047, n055, nqa)
        )

    attrs = hdf.attributes()
    metadata = attrs.get("Orbit_time_stamp", "")

    timestamps = re.findall(
        r"\d{10}[A-Z]",
        str(metadata)
    )

    if len(timestamps) != n047:
        timestamp_issues.append(
            (f.name, n047, len(timestamps), timestamps)
        )

    hdf.end()

print("\n--- Orbit distribution ---")
for n in sorted(orbit_distribution):
    print(f"{n} orbit(s): {orbit_distribution[n]} file(s)")

print(f"\nTotal orbit records expected: "
      f"{sum(k*v for k,v in orbit_distribution.items())}")

print("\n--- AOD / QA dimension check ---")
print(f"Files with dimension mismatch: {len(dimension_issues)}")

if dimension_issues:
    for x in dimension_issues:
        print(x)
else:
    print("PASS - AOD047, AOD055 and QA dimensions match in all files.")

print("\n--- Timestamp check ---")
print(f"Files with timestamp mismatch: {len(timestamp_issues)}")

if timestamp_issues:
    for x in timestamp_issues:
        print(x[0], "AOD layers:", x[1], "timestamps:", x[2])
else:
    print("PASS - Orbit timestamps match the number of AOD layers.")

print("\n--- Processed CSV check ---")

if CSV_FILE.exists():
    df = pd.read_csv(CSV_FILE)

    print(f"CSV rows: {len(df)}")
    print(f"CSV columns: {len(df.columns)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Valid best-quality records: {df['valid_best_quality'].sum()}")
    print(f"Missing AOD047: {df['aod_047'].isna().sum()}")
    print(f"Missing AOD055: {df['aod_055'].isna().sum()}")

    print("\nCSV columns:")
    print(", ".join(df.columns))

else:
    print("ERROR - processed MODIS CSV not found.")

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
