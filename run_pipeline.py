"""
Master Pipeline Runner - Machine Learning Black Carbon Retrieval
================================================================
Orchestrates end-to-end data ingestion, spatiotemporal collocation,
physics-informed feature engineering, cross-validation modeling,
and pilot area gridded BC mapping.

Usage:
  python run_pipeline.py --stage all
  python run_pipeline.py --stage preprocess
  python run_pipeline.py --stage collocate
  python run_pipeline.py --stage train
  python run_pipeline.py --stage predict-grid --target-date 2025-11-20
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Set up project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("BC_Pipeline")


def run_preprocess():
    """Runs data loaders and extractors for AE33, MERRA-2, MODIS, and TROPOMI."""
    logger.info("=" * 70)
    logger.info("STAGE 1: PREPROCESSING & FEATURE EXTRACTION")
    logger.info("=" * 70)

    # 1. Aethalometer AE33 Loader
    from src.preprocessing.aethalometer_loader import load_all_aethalometer_data
    logger.info("1/4 Loading Ground Aethalometer (AE33) data...")
    load_all_aethalometer_data()

    # 2. MERRA-2 Reanalysis Extractor
    from src.preprocessing.merra2_extractor import extract_merra2_at_station
    logger.info("2/4 Extracting MERRA-2 Aerosol & Meteorology fields...")
    extract_merra2_at_station()

    # 3. Sentinel-5P TROPOMI Extractor
    try:
        from src.preprocessing.tropomi_extractor import extract_all_tropomi
        logger.info("3/4 Extracting Sentinel-5P TROPOMI UVAI...")
        extract_all_tropomi()
    except Exception as e:
        logger.warning(f"TROPOMI extraction encountered notice: {e}")

    # 4. MODIS MAIAC Extractor
    try:
        from src.preprocessing.modis_extractor import extract_all_modis
        logger.info("4/4 Extracting MODIS MAIAC AOD...")
        extract_all_modis()
    except Exception as e:
        logger.warning(f"MODIS extraction encountered notice: {e}")

    logger.info("Preprocessing stage completed successfully.\n")


def run_collocate():
    """Runs spatiotemporal collocation and physics-informed feature engineering."""
    logger.info("=" * 70)
    logger.info("STAGE 2: SPATIOTEMPORAL COLLOCATION & FEATURE ENGINEERING")
    logger.info("=" * 70)

    from src.features.collocator import build_collocated_datasets
    build_collocated_datasets()
    logger.info("Collocation stage completed successfully.\n")


def run_train():
    """Trains retrieval models and generates evaluation benchmarks."""
    logger.info("=" * 70)
    logger.info("STAGE 3: MODEL TRAINING & MULTI-TIER BENCHMARKING")
    logger.info("=" * 70)

    from src.models.train_models import run_full_training_pipeline
    run_full_training_pipeline("collocated_bc_hourly.csv")
    logger.info("Model training stage completed successfully.\n")


def run_predict_grid(target_date: str = "2025-11-20"):
    """Generates gridded surface BC estimates and hotspot heatmap for pilot area."""
    logger.info("=" * 70)
    logger.info(f"STAGE 4: GRIDDED BC PREDICTION & HOTSPOT MAPPING ({target_date})")
    logger.info("=" * 70)

    from src.models.predict_grid import predict_and_map_pilot_area
    predict_and_map_pilot_area(target_date=target_date)
    logger.info("Pilot area gridded BC mapping completed successfully.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Machine Learning based Ground-Level Black Carbon (BC) Retrieval Pipeline"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "preprocess", "collocate", "train", "predict-grid"],
        help="Pipeline stage to execute: preprocess, collocate, train, predict-grid, or all (default)."
    )
    parser.add_argument(
        "--target-date",
        type=str,
        default="2025-11-20",
        help="Target date (YYYY-MM-DD) for gridded pilot area surface BC estimation."
    )

    args = parser.parse_args()
    start_time = time.time()

    logger.info("Initiating Black Carbon Retrieval Pipeline")
    logger.info(f"Selected stage: {args.stage.upper()}")

    if args.stage == "preprocess":
        run_preprocess()
    elif args.stage == "collocate":
        run_collocate()
    elif args.stage == "train":
        run_train()
    elif args.stage == "predict-grid":
        run_predict_grid(args.target_date)
    elif args.stage == "all":
        # Check if processed data already exists; if not, run full pipeline
        hourly_data = PROCESSED_DATA_DIR / "collocated_bc_hourly.csv"
        if not hourly_data.exists():
            logger.info("Processed data not detected. Executing preprocessing and collocation...")
            run_preprocess()
            run_collocate()
        else:
            logger.info(f"Existing collocated dataset found at {hourly_data}. Proceeding to training & evaluation.")
        
        run_train()
        run_predict_grid(args.target_date)

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

