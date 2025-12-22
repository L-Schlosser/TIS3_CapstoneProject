# Capstone Project 

## Team Name: VoltVision
## Dataset: european_wholesale_electricity_price_data_daily

### Business Problem:
#### Problem:
> Wholesale electricity prices in Austria fluctuate significantly due to seasonal demand, renewable generation variability, and market dynamics. Uncertainty in future prices makes procurement, trading, and operational planning difficult for utilities, industries, and policymakers.

#### Value of predicted variables
> The model predicts monthly electricity prices using historical prices, additional features, like lag features (previous months) or a holiday feature, rolling averages, seasonal indicators, and potentially correlated variables like demand or neighboring countries’ prices.

> Predicted prices with confidence intervals allow stakeholders to anticipate cost fluctuations and plan ahead.

#### Desicions/Actions based on the prediction
- **Utilities**: Optimize electricity procurement and trading strategies to reduce costs.
- **Industries**: Schedule energy-intensive operations during low-price periods.
- **Policymakers**: Adjust subsidies or implement market interventions to stabilize costs.
- **Quantified impact**: Even a 5–10% improvement in price prediction accuracy can reduce operational and procurement costs by millions of euros annually, while confidence intervals help assess financial risk under uncertainty.

### Data engineering and preparation:
> The raw dataset contains daily wholesale electricity prices for multiple European countries.  
We filter the data to **Austria (ISO3 = AUT)** and standardize the schema for time-series modeling by renaming columns to `ds` (date) and `y` (target variable).

To align with the forecasting objective and reduce daily volatility, daily prices are **aggregated to monthly averages**.  
A **time-based split** is applied to avoid data leakage:
- **Training set:** data before January 2025  
- **Validation set:** January 2025 – December 2025  
- **Test set:** reserved for future forecasting (January 2026)

All preprocessed datasets are stored as CSV files to ensure reproducibility and efficient iteration.


### Exploratory analysis
> Exploratory Data Analysis (EDA) is performed to understand price behavior and guide modeling decisions.

Key findings include:
- Clear **seasonal patterns**, with higher prices during winter months
- A **long-term upward trend**, especially after 2021
- Periods of **high volatility and extreme price spikes**
- Strong **autocorrelation**, indicating predictive power of past prices

Time-series plots, rolling averages, and distribution visualizations are used to detect trends, seasonality, and anomalies.

### Feature engineering
> Feature engineering focuses on capturing temporal dependencies and seasonal effects.

Engineered features include:
- **Lag features** based on historical monthly prices
- **holiday feature** 
- **Rolling statistics** such as 3- and 6-month moving averages
- **Seasonal indicators** including month and quarter

These features enable models to capture both short-term dynamics and long-term trends.

### Model training
> Multiple forecasting approaches are explored:

- **ARIMA / SARIMA** to model trend and seasonality
- **Prophet** for robust seasonality modeling and uncertainty estimation
- **Machine learning models** using lag-based features to capture non-linear patterns

Models are trained using a time-aware approach to preserve temporal structure.

### Evaluation and selection
> Models are evaluated using **time-series cross-validation**, ensuring chronological consistency.

Evaluation metrics include:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

Visual diagnostics such as predicted vs. actual plots, residual analysis, and prediction intervals are used to assess performance.

The final model is selected based on forecast accuracy, stability, and the quality of uncertainty estimates.

### Final prediction
> The selected model generates a **monthly electricity price forecast for Austria**, with a focus on **January 2026**.
