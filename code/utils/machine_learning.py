import polars as pl
import pandas as pd
from typing import Tuple

if __package__:
    from .constants import FREQ_DAILY, HORIZON_DAILY, FREQ_MONTHLY, HORIZON_MONTHLY
    from .preprocessing import load_daily_data, load_monthly_data
    from .forecast_utils import load_existing_forecasts, write_existing_forecasts, merge_datasets_on_forecast
else:
    from constants import FREQ_DAILY, HORIZON_DAILY, FREQ_MONTHLY, HORIZON_MONTHLY
    from preprocessing import load_daily_data, load_monthly_data
    from forecast_utils import load_existing_forecasts, write_existing_forecasts, merge_datasets_on_forecast

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

def run_machine_learning_forecast_daily(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    use_existing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run machine learning forecasting methods on the provided data."""
    if use_existing:
        return load_existing_forecasts(val, test, "ml_daily")

    mlf_daily = MLForecast(
        models=[
            LinearRegression(),
            HuberRegressor(epsilon=1.35, alpha=1e-3),
            RandomForestRegressor(n_estimators=400, max_depth=20, min_samples_leaf=5, max_features='sqrt', random_state=42),
            LGBMRegressor(num_leaves=63, learning_rate=0.05, n_estimators=500, min_child_samples=50, subsample=0.9, colsample_bytree=0.9, random_state=42),
        ],
        lags=[1, 7, 28],
        date_features=['dayofweek', 'month', 'quarter'],
        freq=FREQ_DAILY,
        n_jobs=1
    )

    mlf_daily.fit(df=train, static_features=[])
    ml_daily_val = mlf_daily.forecast(h=HORIZON_DAILY)

    mlf_daily.fit(df=pd.concat([train, val]), static_features=[])
    ml_daily_test = mlf_daily.forecast(h=HORIZON_DAILY)

    write_existing_forecasts(ml_daily_val, ml_daily_test, "ml_daily")
    return merge_datasets_on_forecast(val, test, ml_daily_val, ml_daily_test)

def run_statistical_forecast_monthly(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    use_existing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run statistical forecasting methods on the provided monthly data."""
    if use_existing:
        return load_existing_forecasts(val, test, "stat_monthly")

    sf_monthly = StatsForecast(
        models=[
            SimpleExponentialSmoothing(alpha=0.5),
            Holt(season_length=HORIZON_MONTHLY, error_type="A"),
            HoltWinters(season_length=HORIZON_MONTHLY, error_type="A"),
            AutoRegressive(lags=HORIZON_MONTHLY)
        ],
        freq=FREQ_MONTHLY,
        n_jobs=1
    )

    stat_monthly_val = sf_monthly.forecast(df=train, h=HORIZON_MONTHLY)
    stat_monthly_test = sf_monthly.forecast(df=pd.concat([train, val]), h=HORIZON_MONTHLY)

    write_existing_forecasts(stat_monthly_val, stat_monthly_test, "stat_monthly")
    return merge_datasets_on_forecast(val, test, stat_monthly_val, stat_monthly_test)
        
if __name__ == "__main__":
    train, val, test = load_daily_data(use_existing=True)
    run_machine_learning_forecast_daily(train, val, test, use_existing=False)
    run_machine_learning_forecast_daily(train, val, test, use_existing=True)
    train_m, val_m, test_m = load_monthly_data(use_existing=True)
    run_statistical_forecast_monthly(train_m, val_m, test_m, use_existing=False)
    run_statistical_forecast_monthly(train_m, val_m, test_m, use_existing=True)


