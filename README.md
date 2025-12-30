# TIS3 CapstoneProject

**Members:**
- Ortner Karim
- Schlosser Lorenz 
- Welt Jonas 

## Overview
This repository contains the capstone project for the Time Series Forecasting (TIS3) course at Hagenberg University of Applied Sciences. The project, developed by **VoltVision** (Lorenz, Karim, and Jonas), focuses on Austrian wholesale electricity price forecasting for 2026 (in specific January).

## Project Goal
Predict monthly wholesale electricity prices for Austria throughout 2026 to help utilities, energy companies, and policymakers make informed procurement, hedging, and budgeting decisions in the face of volatile energy markets.

## Key Results
Best Model: LGBMRegressor with lag features
Performance Metric: WAPE (Weighted Absolute Percentage Error) = 0.1348
MAE: 13.47 EUR/MWh | RMSE: 18.48 EUR/MWh
Forecast Period: Full year 2026 with daily granularity

## Repo Structure:
```
TIS3_CapstoneProject/
├── code/
│   ├── capstone.ipynb              # Main analysis and forecast generation
│   ├── modelselection.ipynb        # Model comparison and selection
│   └── utils/
│       ├── constants.py            # Configuration and constants
│       ├── preprocessing.py        # Data loading and splitting
│       ├── baseline.py             # Baseline model implementations
│       ├── statistical.py          # Statistical forecasting models
│       ├── machine_learning.py     # ML model implementations (LGBM, RF, etc.)
│       ├── deep_learning.py        # Deep learning models (NHITS, KAN, RNN, LSTM)
│       ├── forecast_utils.py       # Feature engineering and forecast utilities
│       ├── metric_utils.py         # Metrics calculation and model ranking
│       ├── metrics.py              # Error metric computations
│       ├── visualizations.py       # Plotting and visualization functions
│       └── visualization_utils.py  # Helper functions for visualizations
├── data/
│   ├── eu_electricity_daily.csv    # Raw daily electricity prices (2015+)
│   └── eu_electricity_monthly.csv  # Aggregated monthly data
├── results/
│   ├── preprocessed_data/          # Train/val/test splits (CSV)
│   ├── forecast_data/              # Model predictions (CSV)
│   ├── metrics/                    # Performance metrics by model family
│   └── visualizations/
│       └── forecasts/              # Forecast plots and residual analysis
├── docu/
│   └── documentation.md            # Comprehensive project documentation
├── handin/
│   └── handin.txt                  # Submission checklist
└── README.md                       # This file
```

## Use Cases:

### Energy Utilities & Traders
- Optimize electricity procurement strategies and hedging decisions
- Plan procurement timing to minimize costs during predicted price peaks
- Reduce exposure to volatile wholesale prices through forward-looking visibility

### Energy-Intensive Industries
- Schedule energy-intensive operations (manufacturing, mining, data centers) during predicted low-price periods
- Improve budget forecasting and reduce operational cost uncertainty
- Optimize production planning around electricity cost cycles

### Policy Makers & Regulators
- Assess market stability and inform subsidy policies
- Design market interventions based on predicted price trends
- Monitor energy market dynamics and plan capacity investments

### Financial Institutions
- Price electricity derivatives and futures contracts
- Manage energy portfolio risk and hedging strategies
- Develop investment strategies around renewable vs. conventional energy

### Energy Planners & Consultants
- Provide clients with forward-looking price intelligence
- Support business case analysis for energy projects
- Benchmark forecasting accuracy across different approaches


## Documentation

For technical details and the hole documentation about this Project -> see [`documentation.md`](docu/documentation.md).
