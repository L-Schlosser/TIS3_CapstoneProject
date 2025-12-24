import polars as pl
import pandas as pd
from typing import Tuple

if __package__:
    from .constants import RANDOM_SEED, FREQ_DAILY, HORIZON_DAILY, FREQ_MONTHLY, HORIZON_MONTHLY, DATE_RANGE_TRAIN, DATE_RANGE_VAL, DATE_RANGE_VAL_EXTENDED, DATE_RANGE_TEST
    from .preprocessing import load_daily_data, load_monthly_data
    from .forecast_utils import load_existing_forecasts, write_existing_forecasts, merge_datasets_on_forecast, merge_holidays
else:
    from constants import RANDOM_SEED, FREQ_DAILY, HORIZON_DAILY, FREQ_MONTHLY, HORIZON_MONTHLY, DATE_RANGE_TRAIN, DATE_RANGE_VAL, DATE_RANGE_VAL_EXTENDED, DATE_RANGE_TEST
    from preprocessing import load_daily_data, load_monthly_data
    from forecast_utils import load_existing_forecasts, write_existing_forecasts, merge_datasets_on_forecast, merge_holidays

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

def _run_normal_mlforecast(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    freq: int,
    horizon: int,
    n_lgbm_estimators: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run MLForecast without lag features."""
    ml_forecast = MLForecast(
        models=[
            LinearRegression(),
            HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=1000),
            RandomForestRegressor(n_estimators=600, max_depth=20, min_samples_leaf=5, max_features='sqrt', random_state=RANDOM_SEED),
            LGBMRegressor(boosting_type="gbdt", learning_rate=0.1, n_estimators=n_lgbm_estimators, random_state=RANDOM_SEED),
        ],
        lags=[],
        date_features=['dayofweek', 'month', 'quarter'],
        freq=freq
    )

    ml_forecast.fit(df=train, static_features=[])
    ml_daily_val = ml_forecast.predict(h=horizon)

    ml_forecast.fit(df=pd.concat([train, val]), static_features=[])
    ml_daily_test = ml_forecast.predict(h=horizon)
    return ml_daily_val, ml_daily_test

def _run_lag_mlforecast(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    freq: int,
    horizon: int,
    n_lgbm_estimators: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run MLForecast with lag features."""
    train_lag = merge_holidays(train, freq, DATE_RANGE_TRAIN)
    val_lag = merge_holidays(val, freq, DATE_RANGE_VAL)
    val_lag_ext = merge_holidays(val, freq, DATE_RANGE_VAL_EXTENDED)
    test_lag = merge_holidays(test, freq, DATE_RANGE_TEST)

    ml_forecast_lag = MLForecast(
        models=[
            LinearRegression(),
            HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=1000),
            RandomForestRegressor(n_estimators=600, max_depth=20, min_samples_leaf=5, max_features='sqrt', random_state=RANDOM_SEED),
            LGBMRegressor(boosting_type="gbdt", learning_rate=0.1, n_estimators=n_lgbm_estimators, random_state=RANDOM_SEED),
        ],
        lags=[1, 7, 28],
        date_features=['dayofweek', 'month', 'quarter'],
        lag_transforms = {
            # Short-term volatility
            1: [RollingStd(window_size=3)],
            # Weekly trend
            7: [RollingMean(window_size=3)],
            # Monthly smoothing
            28: [RollingMean(window_size=14)],
        },
        freq=freq
    )

    ml_forecast_lag.fit(df=train_lag, static_features=[])
    ml_daily_val_lag = ml_forecast_lag.predict(h=horizon, X_df=val_lag_ext)

    ml_forecast_lag.fit(df=pd.concat([train_lag, val_lag]), static_features=[])
    ml_daily_test_lag = ml_forecast_lag.predict(h=horizon, X_df=test_lag)

    rename_dict = {col: f"{col}_Lag" for col in ml_daily_val_lag.columns if col not in ['unique_id', 'ds']}
    ml_daily_val_lag = ml_daily_val_lag.rename(columns=rename_dict)
    ml_daily_test_lag = ml_daily_test_lag.rename(columns=rename_dict)
    return ml_daily_val_lag, ml_daily_test_lag

def run_machine_learning_forecast_daily(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    use_existing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run machine learning forecasting methods on the provided data."""
    if use_existing:
        return load_existing_forecasts(val, test, "ml_daily")

    ml_daily_val, ml_daily_test = _run_normal_mlforecast(train, val, test, FREQ_DAILY, HORIZON_DAILY, 600)
    ml_daily_val_lag, ml_daily_test_lag = _run_lag_mlforecast(train, val, test, FREQ_DAILY, HORIZON_DAILY, 600)

    ml_daily_val_all = ml_daily_val_lag.merge(ml_daily_val, on=['unique_id','ds'], how='left')
    ml_daily_test_all = ml_daily_test_lag.merge(ml_daily_test, on=['unique_id','ds'], how='left')

    write_existing_forecasts(ml_daily_val_all, ml_daily_test_all, "ml_daily")
    return merge_datasets_on_forecast(val, test, ml_daily_val_all, ml_daily_test_all)

def run_machine_learning_forecast_monthly(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    use_existing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run machine learning forecasting methods on the provided data - monthly"""
    if use_existing:
        return load_existing_forecasts(val, test, "ml_monthly")

    ml_monthly_val, ml_monthly_test = _run_normal_mlforecast(train, val, test, FREQ_MONTHLY, HORIZON_MONTHLY, 100)
    ml_monthly_val_lag, ml_monthly_test_lag = _run_lag_mlforecast(train, val, test, FREQ_MONTHLY, HORIZON_MONTHLY, 100)

    ml_monthly_val_all = ml_monthly_val_lag.merge(ml_monthly_val, on=['unique_id','ds'], how='left')
    ml_monthly_test_all = ml_monthly_test_lag.merge(ml_monthly_test, on=['unique_id','ds'], how='left')

    write_existing_forecasts(ml_monthly_val_all, ml_monthly_test_all, "ml_monthly")
    return merge_datasets_on_forecast(val, test, ml_monthly_val_all, ml_monthly_test_all)

if __name__ == "__main__":
    train, val, test = load_daily_data(use_existing=True)
    run_machine_learning_forecast_daily(train, val, test, use_existing=False)
    run_machine_learning_forecast_daily(train, val, test, use_existing=True)
    train_m, val_m, test_m = load_monthly_data(use_existing=True)
    run_machine_learning_forecast_monthly(train_m, val_m, test_m, use_existing=False)
    run_machine_learning_forecast_monthly(train_m, val_m, test_m, use_existing=True)