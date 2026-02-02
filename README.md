# Fleet Demand Forecasting (Prophet)

Forecast daily fleet demand using public transportation trip data and time-series modeling (Prophet).  
This project demonstrates an end-to-end analytics workflow: data acquisition → aggregation → forecasting → evaluation.

## Problem
Fleet and operations teams need accurate short-term demand forecasts to support capacity planning, scheduling, and budgeting.

## Data
Public NYC transportation trip records (Yellow Taxi). We aggregate trips into a daily time series (`ds`, `y`).

## Method
- Aggregate daily trip counts
- Train/validate a Prophet forecasting model
- Evaluate with MAE/MAPE
- Produce forecast plots and artifacts for reporting

## Project Structure
- `src/`: pipeline code (download, build time series, train/evaluate model)
- `notebooks/`: EDA and modeling exploration
- `reports/figures/`: exported plots

## How to run (coming next)
Scripts and a reproducible run command will be added in the next commits.
