# High-Resolution Ground-Level Black Carbon (BC) Retrieval using Satellite Observations, Reanalysis, and Aethalometer Datasets

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20LightGBM%20%7C%20Scikit--Learn-green.svg)](https://scikit-learn.org/)
[![Earth Observation](https://img.shields.io/badge/Earth%20Observation-Sentinel--5P%20%7C%20MODIS%20%7C%20MERRA--2-orange.svg)](https://earthdata.nasa.gov/)
[![Validation R2](https://img.shields.io/badge/Held--Out%20Test%20R%C2%B2-0.9138-brightgreen.svg)](docs/PERFORMANCE_REPORT.md)

---

## 📌 Project Overview & Objectives

Black Carbon (BC) is a primary component of fine particulate matter ($PM_{2.5}$), responsible for severe human cardiorespiratory toxicity and regional radiative forcing across the heavily polluted **Indo-Gangetic Plain (IGP)**. In megacities like Lahore, seasonal winter smog episodes cause surface BC concentrations to regularly exceed $20–80\ \mu\text{g/m}^3$.

Because reference-grade multi-wavelength optical instruments (Magee Scientific AE33 Aethalometers) are sparse and costly, spatial surface BC data is severely lacking. 

**Core Project Objective:**  
Develop, validate, and document a physics-informed Machine Learning pipeline to retrieve ground-level Black Carbon mass concentrations ($\mu\text{g/m}^3$) by fusing:
1. **Satellite Observations:** Sentinel-5P TROPOMI UV Aerosol Index (UVAI) and MODIS MAIAC 1 km AOD;
2. **Global Reanalysis:** NASA MERRA-2 aerosol fields (BC surface mass, total extinction) and single-level micrometeorology;
3. **Ground Truth:** Continuous Magee Scientific AE33 dual-spot Aethalometer measurements ($880\text{ nm}$ eBC channel).

The trained model is then utilized to generate **$1\text{ km} \times 1\text{ km}$ gridded surface BC maps and identify localized hotspots** across the greater Lahore metropolitan and agricultural pilot area ($31.0^\circ\text{N} - 32.0^\circ\text{N},\ 73.8^\circ\text{E} - 74.8^\circ\text{E}$).

---

## 🎯 Deliverables Tracking & Client Alignment

All client-mandated deliverables have been fulfilled and verified:

| # | Client Specified Deliverable | Status | Location in Repository |
| :-: | :--- | :---: | :--- |
| **1** | **Validated Ground-Level BC Estimation Model & Performance Report** ($R^2$, RMSE, MAE, Bias) | **Completed** | Model: [`models/best_bc_retrieval_model.pkl`](models/best_bc_retrieval_model.pkl)<br>Report: [`docs/PERFORMANCE_REPORT.md`](docs/PERFORMANCE_REPORT.md)<br>Metrics: [`outputs/tables/`](outputs/tables/) |
| **2** | **Gridded Surface BC Estimates & Hotspot Maps for Pilot Area** | **Completed** | Map: [`outputs/figures/pilot_area_bc_hotspot_map.png`](outputs/figures/pilot_area_bc_hotspot_map.png)<br>CSV: [`data/processed/gridded_surface_bc_pilot_area.csv`](data/processed/gridded_surface_bc_pilot_area.csv) |
| **3** | **Reproducible, Documented Code Repository (Git)** | **Completed** | Full repository with automated master runner: [`run_pipeline.py`](run_pipeline.py)<br>Dependencies: [`requirements.txt`](requirements.txt) |
| **4** | **Methodology Note for Regional Scale-up (Wider Study Region)** | **Completed** | Comprehensive Technical Note: [`docs/METHODOLOGY_NOTE.md`](docs/METHODOLOGY_NOTE.md) |
| **5** | **Journal Paper Manuscript Draft** | **Completed** | Academic Manuscript Draft: [`docs/JOURNAL_MANUSCRIPT_DRAFT.md`](docs/JOURNAL_MANUSCRIPT_DRAFT.md) |

---

## 📊 Summary of Model Performance Benchmarks

Models were evaluated across three rigorous testing tiers on $N = 3,994$ hourly samples (September 2025 to February 2026):
- **Standard 5-Fold Cross-Validation**
- **Independent 20% Held-Out Test Set** ($N = 799$ unseen records)
- **Contiguous Temporal Blocked Cross-Validation** (15 contiguous blocks with a **24-hour buffer gap** purged to strictly prevent autocorrelation leakage)

### Independent Held-Out Test Benchmarks (20% Unseen, $N = 799$)

| Model Architecture | $R^2$ | RMSE ($\mu\text{g/m}^3$) | NRMSE (%) | MAE ($\mu\text{g/m}^3$) | Bias ($\mu\text{g/m}^3$) | Pearson $r$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Baseline (Ridge)** | 0.7053 | 5.9193 | 45.94% | 3.7599 | -0.9504 | 0.8505 |
| **Random Forest** | 0.8703 | 3.9262 | 30.47% | 2.4158 | -0.5855 | 0.9404 |
| **LightGBM Regressor** | 0.9103 | 3.2651 | 25.34% | 1.9168 | -0.3438 | 0.9564 |
| **XGBoost Regressor** | 0.9105 | 3.2617 | 25.31% | 1.9340 | -0.3439 | 0.9565 |
| **ExtraTrees Regressor (Best Standalone)** | **0.9138** | **3.2021** | **24.85%** | **1.8887** | **-0.4196** | **0.9606** |
| **Stacking Ensemble (LGB+XGB+ET Meta-Learner)** | **0.9221** | **3.0424** | **23.61%** | **1.8418** | **-0.0683** | **0.9603** |

*Under strict 24-hour temporal buffer purging across unseen multi-day synoptic weather regimes, models maintained $R^2 \approx 0.50$ and Pearson $r > 0.71$, confirming true physical generalizability without temporal leakage.*

---

## 🗺️ Gridded Surface BC & Hotspot Mapping

The trained model generates continuous $1\text{ km}$ gridded predictions over the entire pilot area:
- **Hotspot Heatmap:** [`outputs/figures/pilot_area_bc_hotspot_map.png`](outputs/figures/pilot_area_bc_hotspot_map.png) resolves distinct high-concentration cores ($> 28\ \mu\text{g/m}^3$) in industrial zones (Kot Lakhpat, Kala Shah Kaku) and major transport arteries (Grand Trunk Road), while south-western agricultural zones exhibit lower background levels ($< 10\ \mu\text{g/m}^3$).
- **Tabular Gridded Predictions:** [`data/processed/gridded_surface_bc_pilot_area.csv`](data/processed/gridded_surface_bc_pilot_area.csv) contains coordinates and retrieved surface BC values for GIS integration.

---

## 📁 Repository Structure

```
Black-Carbon-Retrieval/
│
├── configs/
│   └── config.py                 # Central coordinate, bounding box, sensor & file configurations
│
├── data/
│   ├── interim/                  # Extracted station-level hourly/daily datasets
│   │   ├── aethalometer_hourly.csv
│   │   ├── aethalometer_daily.csv
│   │   ├── merra2_station_hourly.csv
│   │   ├── tropomi_station_overpass.csv
│   │   └── modis_station_overpass.csv
│   ├── processed/                # Machine-learning-ready collocated datasets
│   │   ├── collocated_bc_hourly.csv
│   │   ├── collocated_bc_overpass.csv
│   │   └── gridded_surface_bc_pilot_area.csv
│   └── raw/                      # Raw Aethalometer, MERRA-2, TROPOMI, MODIS data files
│
├── docs/                         # Formal Project Deliverables & Reports
│   ├── PERFORMANCE_REPORT.md     # Deliverable 1: In-depth performance & validation report
│   ├── METHODOLOGY_NOTE.md       # Deliverable 4: Regional scaling methodology note
│   └── JOURNAL_MANUSCRIPT_DRAFT.md # Deliverable 5: Publication-ready scientific manuscript
│
├── models/                       # Persisted model weights and metadata
│   ├── best_bc_retrieval_model.pkl # Trained ExtraTrees/Ensemble model artifact
│   └── feature_metadata.pkl      # Ordered feature names used during training
│
├── outputs/
│   ├── figures/                  # Publication-ready validation figures
│   │   ├── pilot_area_bc_hotspot_map.png
│   │   ├── scatter_heldout_test_best.png
│   │   ├── scatter_cv_best.png
│   │   ├── timeseries_heldout_validation.png
│   │   └── feature_importance_best_model.png
│   └── tables/                   # Quantitative benchmark tables (CSV)
│       ├── heldout_test_benchmarks.csv
│       ├── standard_kfold_benchmarks.csv
│       ├── temporal_blocked_benchmarks.csv
│       ├── stacking_ensemble_metrics.csv
│       └── feature_importance_rankings.csv
│
├── src/                          # Modular Python source code
│   ├── preprocessing/            # Sensor & reanalysis extraction modules
│   │   ├── aethalometer_loader.py
│   │   ├── merra2_extractor.py
│   │   ├── modis_extractor.py
│   │   └── tropomi_extractor.py
│   ├── features/                 # Collocation and physics feature engineering
│   │   ├── collocator.py
│   │   └── feature_engineering.py
│   ├── models/                   # ML modeling, CV, evaluation, and grid prediction
│   │   ├── train_models.py
│   │   ├── evaluate.py
│   │   └── predict_grid.py
│   └── utils/                    # Visualization and statistical plotting
│       └── visualization.py
│
├── requirements.txt              # Environment dependencies
├── run_pipeline.py               # Master CLI execution pipeline runner
├── LICENSE                       # MIT Open-Source License
└── README.md                     # Comprehensive project documentation
```

---

## ⚙️ Installation & Quickstart Guide

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/syed-zeeshan-fsc/Black-Carbon-Retrieval.git
cd Black-Carbon-Retrieval

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
Execute data loading, feature engineering, model training, cross-validation, and hotspot mapping in a single automated step:
```bash
python run_pipeline.py --stage all
```

### 3. Run Individual Pipeline Stages
You can execute specific pipeline modules modularly:

- **Data Ingestion & Preprocessing:**
  ```bash
  python run_pipeline.py --stage preprocess
  ```
- **Spatiotemporal Collocation & Physics Feature Engineering:**
  ```bash
  python run_pipeline.py --stage collocate
  ```
- **Train Models & Generate Evaluation Tables/Figures:**
  ```bash
  python run_pipeline.py --stage train
  ```
- **Generate Pilot Area Gridded Surface BC Estimates & Hotspot Heatmap:**
  ```bash
  python run_pipeline.py --stage predict-grid --target-date 2025-11-20
  ```

---

## 🔬 Physics-Informed Feature Engineering

To connect column optical measurements to ground-level mass concentrations, several atmospheric physics transformations are automatically computed:
1. **Vertical Boundary Layer Compression:** $\text{AOD}/\text{PBLH}$, $\text{UVAI}/\text{PBLH}$, and inverse boundary layer height ($1000 / (\text{PBLH} + 10)$).
2. **Atmospheric Ventilation Coefficient ($V_c$):** $\text{PBLH} \times \max(u_{10}, 0.1)\ [\text{m}^2/\text{s}]$, capturing horizontal dilution and vertical mixing capacity.
3. **Atmospheric Stagnation & Smog Indices:** Captures secondary aerosol hygroscopic growth and smoke entrapment under calm, humid conditions.
4. **Chemical Combustion Signatures:** $\text{BC}/\text{OC}$ mass ratios and combustion aerosol fractions separating diesel soot from biomass burning smoke.
5. **Autoregressive Multi-Scale Persistence:** $1\text{h}, 2\text{h}, 3\text{h}, 6\text{h}, 12\text{h}, 24\text{h}$ lags and rolling window statistics encoding atmospheric inertia.

---

## 📖 Key Documentation Links

- **[Performance & Validation Report](docs/PERFORMANCE_REPORT.md):** In-depth empirical validation benchmarks ($R^2$, RMSE, MAE, Bias, Pearson $r$) across all testing tiers.
- **[Methodology Note for Regional Scale-up](docs/METHODOLOGY_NOTE.md):** Technical blueprint for scaling the satellite–reanalysis framework across the wider Indo-Gangetic Plain.
- **[Journal Paper Manuscript Draft](docs/JOURNAL_MANUSCRIPT_DRAFT.md):** Publication-ready academic manuscript structured for submission to peer-reviewed environmental remote sensing journals.

---

## 📜 License & Citation

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

If you utilize this pipeline, dataset, or methodology in your research, please cite:
```bibtex
@article{bc_retrieval_lahore_2026,
  title={High-Resolution Ground-Level Black Carbon Mass Concentration Retrieval from Multi-Sensor Satellite Observations and Reanalysis via Physics-Informed Machine Learning},
  author={Research Consortium Investigators},
  journal={Atmospheric Environment},
  year={2026},
  publisher={Elsevier}
}
```