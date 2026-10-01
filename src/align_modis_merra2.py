from pathlib import Path
import pandas as pd

MODIS_FILE = Path("data/processed/modis_site.csv")
AEROSOL_FILE = Path("data/processed/merra2_aerosol_site.csv")
MET_FILE = Path("data/processed/merra2_meteorology_site.csv")
OUT_FILE = Path("data/processed/modis_merra2_aligned.csv")

print("=" * 70)
print("MODIS + MERRA-2 TEMPORAL ALIGNMENT")
print("=" * 70)

# --------------------------------------------------
# Load data
# --------------------------------------------------
modis = pd.read_csv(MODIS_FILE)
aerosol = pd.read_csv(AEROSOL_FILE)
met = pd.read_csv(MET_FILE)

print("\nLoaded:")
print(f"MODIS records:               {len(modis)}")
print(f"MERRA-2 aerosol records:     {len(aerosol)}")
print(f"MERRA-2 meteorology records: {len(met)}")

# --------------------------------------------------
# Parse timestamps
# --------------------------------------------------
modis["datetime"] = pd.to_datetime(
    modis["orbit_time"],
    errors="raise"
)

aerosol["time"] = pd.to_datetime(
    aerosol["time"],
    errors="raise"
)

met["time"] = pd.to_datetime(
    met["time"],
    errors="raise"
)

# Rename MERRA timestamps BEFORE merging
aerosol = aerosol.rename(columns={"time": "aerosol_time"})
met = met.rename(columns={"time": "met_time"})

# --------------------------------------------------
# Keep only best-quality MODIS
# --------------------------------------------------
modis_valid = (
    modis[modis["valid_best_quality"] == True]
    .copy()
    .sort_values("datetime")
)

print(f"\nBest-quality MODIS records: {len(modis_valid)}")

# --------------------------------------------------
# Sort MERRA data
# --------------------------------------------------
aerosol = aerosol.sort_values("aerosol_time")
met = met.sort_values("met_time")

# --------------------------------------------------
# Match nearest MERRA aerosol within 1 hour
# --------------------------------------------------
aligned = pd.merge_asof(
    modis_valid,
    aerosol,
    left_on="datetime",
    right_on="aerosol_time",
    direction="nearest",
    tolerance=pd.Timedelta("1h")
)

# --------------------------------------------------
# Match nearest MERRA meteorology within 1 hour
# --------------------------------------------------
aligned = pd.merge_asof(
    aligned.sort_values("datetime"),
    met,
    left_on="datetime",
    right_on="met_time",
    direction="nearest",
    tolerance=pd.Timedelta("1h")
)

# --------------------------------------------------
# Calculate time differences
# --------------------------------------------------
aligned["aerosol_time_diff_min"] = (
    aligned["datetime"] - aligned["aerosol_time"]
).abs().dt.total_seconds() / 60

aligned["met_time_diff_min"] = (
    aligned["datetime"] - aligned["met_time"]
).abs().dt.total_seconds() / 60

# --------------------------------------------------
# Save
# --------------------------------------------------
aligned.to_csv(OUT_FILE, index=False)

# --------------------------------------------------
# Validation
# --------------------------------------------------
print("\n" + "=" * 70)
print("ALIGNMENT RESULTS")
print("=" * 70)

print(f"\nOutput file: {OUT_FILE}")
print(f"Output records: {len(aligned)}")

aero_matched = aligned["aerosol_time"].notna().sum()
met_matched = aligned["met_time"].notna().sum()

print(f"MERRA aerosol matched:      {aero_matched}/{len(aligned)}")
print(f"MERRA meteorology matched:  {met_matched}/{len(aligned)}")

print("\nAerosol time difference (minutes):")
print(
    aligned["aerosol_time_diff_min"]
    .describe()
    .to_string()
)

print("\nMeteorology time difference (minutes):")
print(
    aligned["met_time_diff_min"]
    .describe()
    .to_string()
)

print("\nMissing values after alignment:")
print(
    aligned.isna()
    .sum()
    .sort_values(ascending=False)
    .head(15)
    .to_string()
)

print("\nFirst 5 aligned records:")
print(
    aligned[
        [
            "datetime",
            "aerosol_time",
            "met_time",
            "aerosol_time_diff_min",
            "met_time_diff_min"
        ]
    ]
    .head()
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("ALIGNMENT COMPLETE")
print("=" * 70)
