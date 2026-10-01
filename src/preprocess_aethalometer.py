from pathlib import Path
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "2025  BC Data"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "aethalometer_2025.csv"


# AE33 measurement columns
COLUMNS = [
    "date",
    "time",
    "timebase",
    "RefCh1", "Sen1Ch1", "Sen2Ch1",
    "RefCh2", "Sen1Ch2", "Sen2Ch2",
    "RefCh3", "Sen1Ch3", "Sen2Ch3",
    "RefCh4", "Sen1Ch4", "Sen2Ch4",
    "RefCh5", "Sen1Ch5", "Sen2Ch5",
    "RefCh6", "Sen1Ch6", "Sen2Ch6",
    "RefCh7", "Sen1Ch7", "Sen2Ch7",
    "Flow1", "Flow2", "FlowC",
    "Pressure",
    "Temperature",
    "BB",
    "ContTemp",
    "SupplyTemp",
    "Status",
    "ContStatus",
    "DetectStatus",
    "LedStatus",
    "ValveStatus",
    "LedTemp",
    "BC11", "BC12", "BC1",
    "BC21", "BC22", "BC2",
    "BC31", "BC32", "BC3",
    "BC41", "BC42", "BC4",
    "BC51", "BC52", "BC5",
    "BC61", "BC62", "BC6",
    "BC71", "BC72", "BC7",
    "K1", "K2", "K3", "K4", "K5", "K6", "K7",
    "TapeAdvCount",
]


def read_aethalometer_file(file_path):
    """Read one AE33 measurement file and return a clean DataFrame."""

    # Header/data structure:
    # line 1-4: instrument metadata
    # line 5: blank
    # line 6: column header
    # line 7: blank
    # line 8 onward: measurements

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        skiprows=7,
        header=None,
        names=COLUMNS,
        engine="python",
    )

    # Remove completely empty rows
    df = df.dropna(how="all")

    return df


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    files = sorted(
        INPUT_DIR.glob("AE33_AE33-*.dat")
    )

    print(f"Found {len(files)} Aethalometer measurement files.")

    if not files:
        raise FileNotFoundError("No AE33 measurement files found.")

    all_data = []

    for file_path in files:
        print(f"Reading {file_path.name}")

        df = read_aethalometer_file(file_path)

        # Add source file for traceability
        df["source_file"] = file_path.name

        all_data.append(df)

    data = pd.concat(all_data, ignore_index=True)

    print(f"\nTotal raw rows: {len(data):,}")

    # Create datetime
    data["datetime"] = pd.to_datetime(
        data["date"].astype(str) + " " + data["time"].astype(str),
        errors="coerce",
    )

    # Convert BC6 to numeric
    data["BC6"] = pd.to_numeric(data["BC6"], errors="coerce")

    # Convert selected instrument/QC fields to numeric
    numeric_columns = [
        "timebase",
        "Flow1",
        "Flow2",
        "FlowC",
        "Pressure",
        "Temperature",
        "BB",
        "ContTemp",
        "SupplyTemp",
        "Status",
        "ContStatus",
        "LedTemp",
        "BC6",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    # Keep observations with a valid timestamp and BC6
    data = data.dropna(subset=["datetime", "BC6"])

    # Remove startup / non-measurement zero concentrations.
    # Actual BC6 measurements are positive in the measurement period.
    data = data[data["BC6"] > 0].copy()

    # Sort chronologically
    data = data.sort_values("datetime").reset_index(drop=True)

    # Keep the most useful fields for the processed dataset.
    # BC6 is the 880 nm BC concentration.
    output_columns = [
        "datetime",
        "BC6",
        "Flow1",
        "Flow2",
        "FlowC",
        "Pressure",
        "Temperature",
        "BB",
        "ContTemp",
        "SupplyTemp",
        "Status",
        "ContStatus",
        "LedTemp",
        "source_file",
    ]

    data = data[output_columns]

    # Remove duplicate timestamps if any
    before_duplicates = len(data)

    data = data.drop_duplicates(
        subset=["datetime"],
        keep="first",
    ).reset_index(drop=True)

    duplicates_removed = before_duplicates - len(data)

    # Save
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # Summary
    print("\n" + "=" * 60)
    print("AETHALOMETER PREPROCESSING SUMMARY")
    print("=" * 60)

    print(f"Files processed:       {len(files):,}")
    print(f"Raw rows:              {sum(len(df) for df in all_data):,}")
    print(f"Valid BC6 rows:        {len(data):,}")
    print(f"Duplicates removed:    {duplicates_removed:,}")

    print(
        f"Date range:            "
        f"{data['datetime'].min()} → {data['datetime'].max()}"
    )

    print("\nBC6 statistics:")
    print(data["BC6"].describe())

    print(f"\nOutput:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()