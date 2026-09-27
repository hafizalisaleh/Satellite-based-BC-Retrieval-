# Satellite-Based Black Carbon Retrieval

<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Research-blue" />
  <img src="https://img.shields.io/badge/Remote%20Sensing-Satellite%20Data-6f42c1" />
  <img src="https://img.shields.io/badge/SUPARCO-FSC%20Internship-orange" />
</p>

## Overview

A machine learning-based research project for estimating **ground-level Black Carbon (BC) concentrations** from satellite observations, atmospheric reanalysis, meteorological data, and ground-based measurements.

The study focuses on a pilot location in **Lahore, Pakistan**.

## Data Sources

| Source                  | Data Used                            |
| ----------------------- | ------------------------------------ |
| **MODIS MCD19A2**       | Aerosol Optical Depth (AOD)          |
| **Sentinel-5P TROPOMI** | Aerosol Index                        |
| **MERRA-2**             | Aerosol and meteorological variables |
| **Aethalometer**        | Ground-level BC measurements         |

## Research Pipeline

```text
Satellite Observations
        ↓
Data Processing & Quality Filtering
        ↓
MERRA-2 Atmospheric + Meteorological Data
        ↓
Feature Engineering
        ↓
Ground-Based BC Measurements
        ↓
Machine Learning Model
        ↓
Ground-Level BC Estimation
```

## Objective

The project aims to investigate how satellite-derived and atmospheric features can be used to model **near-surface Black Carbon concentrations**, with potential applications in spatial and temporal air-pollution analysis.

## Repository

```text
data/          Raw and processed datasets
src/           Data preprocessing
notebooks/     Analysis and experiments
models/        Trained models
outputs/       Results and visualizations
```

## Research Areas

**Machine Learning · Remote Sensing · Atmospheric Data · Geospatial Analysis · Air Pollution**

### Project Context

FAST School of Computing

