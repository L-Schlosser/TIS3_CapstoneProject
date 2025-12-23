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
> The raw dataset (`eu_electricity_daily.csv`) contains daily wholesale electricity prices for multiple European countries from 2015 onwards.

**Data Filtering and Schema Standardization:**
- Filter data for **Austria (ISO3 Code = "AUT")** using Polars for efficient processing
- Standardize schema for time-series modeling:
  - `Country` → `unique_id` (identifier for time series)
  - `Date` → `ds` (datetime column)
  - `Price (EUR/MWhe)` → `y` (target variable)
- Date parsing with format `%Y-%m-%d` and sorting by date

**Frequency Variants:**
Two separate datasets are prepared:
1. **Daily data:** Original daily frequency preserved for capturing short-term patterns
2. **Monthly data:** Daily prices aggregated to monthly frequency using mean values to reduce volatility and focus on longer-term trends

**Time-Based Split (avoiding data leakage):**
- **Training set:** All data before January 1, 2025 (~10 years of historical data)
- **Validation set:** January 1, 2025 – December 31, 2025 (365 days / 12 months)
- **Test set:** Empty DataFrame, reserved for future real-world validation (January 2026 onwards)

**Storage and Reproducibility:**
- All preprocessed datasets stored as CSV files in `results/preprocessed_data/`
- Naming convention: `{frequency}_{split}.csv` (e.g., `daily_train.csv`, `monthly_val.csv`)
- Functions support `use_existing` parameter to load cached preprocessed data or regenerate from source
- This approach ensures reproducibility and enables efficient iteration during model development


### Exploratory analysis
> Exploratory Data Analysis (EDA) is performed to understand price behavior and guide modeling decisions.

Key findings include:
- Clear **seasonal patterns**, with higher prices during winter months (December-February)
- A **long-term upward trend**, especially after 2021, driven by energy market volatility
- Periods of **high volatility and extreme price spikes** caused by supply disruptions and demand shocks
- Strong **autocorrelation** at multiple lags, indicating significant predictive power of historical prices
- **Weekly seasonality** in daily data (7-day cycles) and **yearly seasonality** in monthly data (12-month cycles)

Visualization techniques employed:
- Time-series plots showing raw prices and rolling averages to identify trends
- Seasonal decomposition to separate trend, seasonal, and residual components
- Distribution analysis (histograms, box plots) to detect outliers and understand price ranges
- Autocorrelation and Partial Autocorrelation Function (ACF/PACF) plots to determine optimal lag features

### Feature engineering
> Feature engineering focuses on capturing temporal dependencies and seasonal effects. Different feature sets are engineered for daily and monthly frequencies.

**Daily Data Features:**
- **Lag features:** Previous day (lag1), 1 week ago (lag7), 4 weeks ago (lag28), and 1 year ago (lag365)
- **Rolling statistics:** 7-day rolling mean (weekly trend) and 30-day rolling mean (monthly trend)
- **Holiday indicator:** Binary feature marking public holidays to account for reduced consumption
- **Date features:** Day of week, month, and quarter for temporal patterns
- **Lag-based rolling statistics:** 3-day rolling standard deviation on lag1 (short-term volatility), 3-day rolling mean on lag7 (weekly smoothing), 14-day rolling mean on lag28 (monthly smoothing)

**Monthly Data Features:**
- **Lag features:** Last month (lag1), last quarter (lag3), half-year (lag6), and last year (lag12)
- **Rolling statistics:** 3-month rolling mean (quarterly trend) and 12-month rolling mean (yearly trend)
- **Holiday aggregation:** Monthly aggregated holiday counts
- **Date features:** Month and quarter indicators
- **Lag-based rolling statistics:** Similar transformations applied at monthly scale

These features enable models to capture both short-term dynamics (daily/weekly patterns) and long-term trends (seasonal/yearly patterns).

### Model training
> Multiple forecasting approaches are explored to establish a comprehensive comparison across different forecasting paradigms:

**Baseline Models:**
- **Seasonal Naive:** Uses the value from the same season in the previous cycle as the forecast. Selected for capturing seasonal patterns without any statistical assumptions.
- **Random Walk with Drift (RWD):** Extends the naive forecast by adding a linear trend component. Useful for datasets with clear directional trends.
- **Historic Average:** Computes the mean of all historical observations. Provides a stable reference point for evaluating more complex models.
- **Structural (Ensemble):** Average of Seasonal Naive and RWD. Combines seasonal and trend components for a more robust baseline.

*Why these models?* Baseline models provide a performance floor that any sophisticated model must beat to be considered valuable. They are simple, interpretable, and computationally cheap.

**Statistical Models:**
- **Simple Exponential Smoothing (SES):** Applies exponentially decreasing weights to past observations (alpha=0.5). Best for data without trend or seasonality.
- **Holt's Linear Trend Method:** Extends SES by adding a trend component. Captures both level and linear trends in the time series.
- **Holt-Winters:** Adds seasonal component to Holt's method using additive error type. Handles data with trend and seasonality simultaneously.
- **AutoETS:** Automatically selects the best Error-Trend-Seasonality configuration. Provides robust forecasts by choosing optimal model structure.

*Why these models?* Statistical models offer a balance between simplicity and sophistication. They explicitly model trend and seasonality components, are computationally efficient, and provide interpretable parameters that can reveal insights about the underlying data patterns.

**Machine Learning Models:**
- **Linear Regression:** Establishes a baseline for ML approaches. Fast, interpretable, and works well with proper feature engineering.
- **Huber Regressor:** Robust regression that is less sensitive to outliers (epsilon=1.35, alpha=1e-3, max_iter=1000). Ideal for electricity price data which contains extreme spikes.
- **Random Forest Regressor:** Ensemble of decision trees (600 estimators, max_depth=20, min_samples_leaf=5, max_features='sqrt'). Captures non-linear relationships and feature interactions without overfitting.
- **LGBMRegressor:** Gradient boosting with GBDT (learning_rate=0.1, 600 estimators for daily, 20 for monthly). Provides state-of-the-art performance for tabular data with efficient training.

*Model configurations:*
- **Standard version:** Uses only date features (dayofweek, month, quarter) without lag features
- **Lag-enhanced version:** Incorporates multiple lag features (1, 7, 28 for both frequencies), date features (dayofweek, month, quarter), and rolling statistics:
  - lag1 with 3-period rolling standard deviation (short-term volatility)
  - lag7 with 3-period rolling mean (weekly trend)
  - lag28 with 14-period rolling mean (monthly smoothing)
- **Holiday features:** Integrated for daily and monthly models to account for consumption patterns on public holidays

*Why these models?* ML models excel at learning complex non-linear patterns from features. They can incorporate external variables, lag features, and engineered features naturally. The ensemble methods (RF, LGBM) are particularly robust and often achieve superior performance on structured data.

**Deep Learning Models:**
- **NHITS:** Neural Hierarchical Interpolation for Time Series. Captures multi-scale temporal patterns through hierarchical structure with n_freq_downsample=[2, 1, 1].
- **KAN:** Kolmogorov-Arnold Network for time series. Novel architecture designed to learn complex temporal relationships.
- **RNN:** Recurrent Neural Network with 2 encoder layers, 128 hidden units for encoder and decoder.
- **LSTM:** Long Short-Term Memory network with 2 encoder layers, 128 hidden units for encoder and decoder, optimized for capturing long-term dependencies.

*Configuration:*
- **Daily models:** input_size=180 (half year) for standard version, input_size=60 for lag-enhanced version, forecast horizon h=365
- **Monthly models:** input_size=36 (3 years) for standard version, input_size=24 (2 years) for lag-enhanced version, forecast horizon h=12
- **Loss function:** MAE (Mean Absolute Error) for all models
- **Scalers:** Robust scaler to handle outliers
- **Training:** max_steps=300-500 depending on model complexity
- **Lag-enhanced versions:** Include historical exogenous features:
  - Daily: is_holiday, lag1, lag7, lag28, lag365, rolling_mean_7, rolling_mean_30
  - Monthly: is_holiday, lag1, lag3, lag6, lag12, rolling_mean_3, rolling_mean_12

*Why these models?* Deep learning models can automatically learn hierarchical representations and complex temporal patterns from raw data. They are particularly effective for long-term forecasting and can handle multiple seasonalities. While computationally expensive, they often achieve state-of-the-art performance on complex time series.

Models are trained using a time-aware approach to preserve temporal structure, with separate training for daily and monthly frequencies.

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
