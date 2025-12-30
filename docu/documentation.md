# Capstone Project 

## Team Name: VoltVision
## Dataset: european_wholesale_electricity_price_data_daily

### Business Problem:
#### Problem:
> Wholesale electricity prices in Austria are highly volatile due to seasonal demand patterns, renewable generation variability, and changing market conditions. This volatility makes it difficult for utilities, energy-intensive companies, and planners to anticipate future costs, leading to suboptimal procurement timing, budgeting inaccuracies, and increased exposure to price spikes.

#### Value of predicted variables
> The model provides monthly wholesale electricity price forecasts based on historical price trends, additional features, like lag features (previous months) or a holiday feature, rolling averages seasonal patterns. These forecasts give stakeholders forward-looking price visibility.

> While the forecast does not quantify uncertainty, it serves as a directional planning tool that helps organizations move from reactive decision-making to proactive cost management. The core value is to reduce electricity cost volatility and planning risk by turning uncertain wholesale prices into actionable procurement, hedging and budgeting decisions.

#### Desicions/Actions based on the prediction
- **Utilities**: Optimize electricity procurement and trading strategies to reduce costs.
- **Industries**: Schedule energy-intensive operations during low-price periods.
- **Policymakers**: Adjust subsidies or implement market interventions to stabilize costs.
- **Quantified impact**: Even a 5–10% improvement in price prediction accuracy can reduce operational and procurement costs by millions of euros annually.

### Data engineering and preparation:
> The raw dataset (`eu_electricity_daily.csv`) contains daily wholesale electricity prices for multiple European countries from 2015 onwards.

**Data Filtering and Schema Standardization:**
- Filter data for **Austria (ISO3 Code = "AUT")** using Polars for efficient processing
- Standardize schema for time-series modeling:
  - `Country` → `unique_id` (identifier for time series)
  - `Date` → `ds` (datetime column)
  - `Price (EUR/MWhe)` → `y` (target variable)
- Date parsing with format `%Y-%m-%d` and sorting by date

**Frequency Variants:**while
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
- **Visualizations** help to understand the patterns of the distribution in the data. For example the Heatmap shows us that in 2022 in August there was a high peak of electricity prices due to energy crisis. This then rises the average price per month across all years in August, which we can see in the visualization above.

Visualization techniques employed:
- Time-series plot showing raw daily prices
- Seasonal summaries: monthly average by month and yearly average by year
- Distribution analysis (histograms, box plots) to detect outliers and understand price ranges
- Month–year heatmap highlighting seasonal patterns and extreme spikes

Implementation notes (EDA):
- The notebook generates the time-series plot, distribution charts (histogram and box), monthly and yearly average bar charts, and the month–year heatmap.

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

How metrics are calculated:
- For monthly models: metrics are computed directly on the validation split, comparing each model’s forecast column to `y`.
- For daily models: `ds` is converted to month start and forecasts are averaged per month (`unique_id`, `ds`) to align horizons with monthly metrics; then MAE, RMSE, and MAPE are computed per model column.
- Columns used for scoring exclude identifier and date fields; only forecast columns are evaluated. Rows with missing values are dropped before scoring.
- Results are sorted by MAPE (ascending) and persisted as CSVs under `results/metrics/` with the naming pattern `{family}_{frequency}_{split}_metrics.csv`. A `use_existing` flag allows reloading cached metrics for reproducibility.

Why these metrics:
- MAE: Interpretable in EUR/MWh, representing average absolute deviation. Suitable for business reporting and robust to extreme spikes compared to squared errors.
- RMSE: Penalizes large errors more strongly, highlighting models that avoid big misses. Useful when risk of large deviations carries higher operational costs.
- MAPE: Scale-free percentage error, enabling comparison across periods and facilitating stakeholder communication. Note: care is taken with low `y` values as MAPE can inflate when actuals are near zero.

Visual diagnostics such as predicted vs. actual plots and residual analysis are used to assess performance.

The final model is selected based on WAPE (Weighted Absolute Percentage Error), which accounts for the magnitude of actual values and provides a robust measure of forecast accuracy across different price levels.

### Model Selection and Performance Analysis
> After comprehensive evaluation across all model families, the **best-performing models** are identified based on WAPE scores on the validation set (January–December 2025).

**Selection Process:**
- **Overall ranking:** Models are sorted by WAPE across both daily and monthly frequencies to identify the top performers regardless of frequency or family.
- **Best per frequency:** Separate rankings for daily and monthly models allow frequency-specific model selection.
- **Best per family:** The top model from each family (baseline, statistical, machine learning, deep learning) is identified to compare approaches.

**Visualization of Model Performance:**

1. **Best 3 Models Overall:** A time-series plot comparing the three best-performing models against actual prices from 2023 onwards. This visualization uses monthly frequency to show clean trend comparisons and extends into the forecast horizon (2026). The plot demonstrates how the top models track actual price movements, handle volatility periods, and project future trends.

2. **Best Model by Frequency:** Separate plots for the best daily and monthly models show forecast performance at different granularities:
   - **Daily forecasts:** Capture short-term fluctuations and weekly patterns, useful for operational scheduling.
   - **Monthly forecasts:** Focus on longer-term trends and seasonal patterns, ideal for strategic planning and budgeting.

3. **Best Model per Family:** A comparative visualization showing the top performer from each modeling approach (baseline, statistical, ML, DL). This helps understand the relative strengths of different methodologies and validates whether sophisticated approaches outperform simpler baselines.

4. **Residual Analysis:** An actual vs. predicted scatter plot for the best model, showing:
   - How closely predictions align with actual values (points near the diagonal line indicate accurate forecasts)
   - Systematic biases (points consistently above or below the diagonal)
   - Variance patterns (spread of points indicates prediction uncertainty)
   - Performance across different price ranges

**Why the Best Model Was Selected:**

The chosen model, "LGBMRegressor_Lag", achieves the lowest WAPE score, indicating superior accuracy in predicting electricity prices while accounting for the magnitude of actual values. WAPE is preferred over MAPE because:
- It weights errors by actual values, preventing disproportionate impact from low-price periods
- It provides more stable performance assessment across different market conditions
- It aligns better with business objectives where absolute forecast error matters more at higher price levels

The selected model demonstrates:
- **Consistent accuracy:** Low error metrics (MAE, RMSE, MAPE, WAPE) across the validation period
- **Trend capture:** Successfully tracks the upward price trend observed since 2021
- **Seasonality handling:** Captures monthly and yearly seasonal patterns effectively
- **Volatility management:** Reasonable predictions even during high-volatility periods

### Final Prediction
> The selected model, "LGBMRegressor_Lag" generates a **monthly electricity price forecast for Austria** for the entire year 2026.

**January 2026 Forecast:**
The model provides daily price predictions throughout January 2026, which are visualized and aggregated to show expected price levels during the first month of the forecast horizon. This near-term forecast has higher confidence as it relies on recent historical patterns and shorter-term dependencies.

**Full Year 2026 Forecast:**
Monthly aggregated forecasts for the entire year 2026 reveal expected seasonal patterns and long-term trends. The forecast helps stakeholders anticipate:
- **Seasonal price variations:** Higher prices expected during winter months (January, February, December) due to increased heating demand
- **Summer trends:** Relatively lower prices during spring and summer months when demand moderates

**Practical Applications:**
- **Procurement planning:** Utilities can schedule bulk electricity purchases during forecasted low-price periods
- **Hedging strategies:** Energy traders can use forecasts to inform futures contracts and risk management
- **Operational scheduling:** Industrial consumers can plan energy-intensive operations during anticipated low-price windows
- **Budget planning:** Organizations can develop more accurate electricity cost budgets for 2026 based on monthly forecasts

The forecasts represent point estimates based on historical patterns and current trends. While uncertainty naturally increases for longer horizons, the model provides actionable insights for proactive energy management throughout 2026.



