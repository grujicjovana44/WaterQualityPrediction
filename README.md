# Water Quality Prediction (WQI) - Model Comparison

This project predicts the **Water Quality Index (WQI)** of the Brisbane River using sensor measurements combined with rainfall data, and compares five different modeling approaches — from classical linear regression to a custom Transformer built with PyTorch.

## Overview

Water quality is affected by multiple interacting environmental factors (dissolved oxygen, turbidity, pH, conductivity, chlorophyll levels) and external conditions like rainfall. This project builds an end-to-end pipeline that:

1. Loads and merges water quality sensor data with daily rainfall records
2. Cleans the data (missing value interpolation, outlier treatment)
3. Computes a Water Quality Index (WQI) as a weighted combination of key water parameters
4. Trains and compares several regression models to predict WQI from the underlying sensor readings and time-based features
5. Evaluates and visualizes model performance

## Dataset

- **`data/brisbane_water_quality.csv`** — time-stamped water quality sensor readings (pH, dissolved oxygen, turbidity, specific conductance, chlorophyll, salinity, etc.)
- **`data/RainfallData.csv`** — daily rainfall measurements, merged with the water quality data by date

## Pipeline

**1. Data loading & merging**
Water quality readings are timestamped and merged with daily rainfall totals. Non-informative or redundant columns (e.g. quality flags, salinity, dissolved oxygen in % saturation) are dropped.

**2. Missing value handling**
Missing values are interpolated using time-based interpolation, followed by forward/backward fill for any remaining gaps.

**3. WQI calculation**
A custom function computes the Water Quality Index as a weighted sum of sub-indices for pH, dissolved oxygen, turbidity, conductivity, and chlorophyll, each scaled against ideal and standard reference values.

**4. Outlier treatment**
Turbidity and chlorophyll are log-transformed to reduce skew, then all key attributes are clipped at ±3 standard deviations (Z-score clipping) to control for extreme outliers.

**5. Feature engineering**
Time-based features (year, month, day of year, hour, day of week) are added, and the WQI serves as the prediction target.

**6. Train/test split & scaling**
Data is split chronologically (80/20) to respect the time-series nature of the data, then scaled using MinMax scaling.

**7. Model training & tuning**
Six models are trained and tuned via grid search with `TimeSeriesSplit` cross-validation:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor
- MLP Neural Network (scikit-learn)
- **Transformer Regressor** — a custom PyTorch sequence model with learned positional embeddings and a multi-head self-attention encoder, trained on sliding windows of past readings to predict WQI

**8. Evaluation**
All models are evaluated on the held-out test set using MAE, MSE, RMSE, and R². Results are compared visually via bar charts, and predictions are plotted against actual WQI values over time. Feature importance is extracted for the tree-based models (Random Forest, XGBoost).

## Requirements

```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
xgboost
torch
```

## How to Run

1. Place `brisbane_water_quality.csv` and `RainfallData.csv` inside a `data/` folder next to the notebook.
2. Open and run `Evaluation.ipynb` sequentially — each section is organized as an independent step in the pipeline (loading, cleaning, feature engineering, model training, evaluation).

## Notes

- **The WQI–feature relationship is intentional.** The WQI target is calculated directly from a subset of the same sensor readings (pH, dissolved oxygen, turbidity, specific conductance, chlorophyll) that are also used as model inputs. This is a deliberate design choice: the goal of this project is for models to learn and approximate the relationship between the WQI and its underlying water quality parameters, not to perform independent multi-step-ahead forecasting from unrelated signals. This explains the very high R² scores achieved by some models (e.g. MLP, XGBoost) — they are effectively learning to reconstruct a known weighted formula from its inputs, which is a valid and useful task in its own right (e.g. for imputing WQI when only raw sensor readings are available, or for understanding which parameters drive the index most strongly).
- Time-series cross-validation (`TimeSeriesSplit`) is used throughout to avoid leaking future information into training folds.

## Project Structure

```
WaterQualityPrediction/
├── data/
│   ├── brisbane_water_quality.csv
│   └── RainfallData.csv
└── Evaluation.ipynb
```
