import polars as pl
import pandas as pd
from typing import Tuple

if __package__:
    from .constants import RANDOM_SEED, FREQ_DAILY, HORIZON_DAILY, FREQ_MONTHLY, HORIZON_MONTHLY
    from .preprocessing import load_daily_data, load_monthly_data
    from .forecast_utils import load_existing_forecasts, write_existing_forecasts, merge_datasets_on_forecast, merge_holidays_daily, merge_holidays_monthly
else:
    from constants import RANDOM_SEED, FREQ_DAILY, HORIZON_DAILY, FREQ_MONTHLY, HORIZON_MONTHLY
    from preprocessing import load_daily_data, load_monthly_data
    from forecast_utils import load_existing_forecasts, write_existing_forecasts, merge_datasets_on_forecast, merge_holidays_daily, merge_holidays_monthly

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

def _fill_missing_data(df, end_date):
    import pandas as pd

    last_mean = df.tail(10)["y"].mean()

    full_dates = pd.date_range(start=df["ds"].min(), end=pd.to_datetime(end_date), freq="D")
    df_full = df.set_index("ds").reindex(full_dates).rename_axis("ds").reset_index()

    df_full["y"] = df_full["y"].fillna(last_mean)
    df_full["unique_id"] = "Austria"
    return df_full

def _run_normal_mlforecast(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    freq: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mlf_daily = MLForecast(
        models=[
            LinearRegression(),
            HuberRegressor(epsilon=1.35, alpha=1e-3),
            RandomForestRegressor(n_estimators=400, max_depth=20, min_samples_leaf=5, max_features='sqrt', random_state=RANDOM_SEED),
            LGBMRegressor(objective="regression", boosting_type="gbdt", learning_rate=0.05, n_estimators=600, num_leaves=31, max_depth=-1, min_child_samples=50, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.2, reg_lambda=0.2, random_state=RANDOM_SEED),
        ],
        lags=[],
        date_features=['dayofweek', 'month', 'quarter'],
        freq=freq
    )

    mlf_daily.fit(df=train, static_features=[])
    ml_daily_val = mlf_daily.predict(h=HORIZON_DAILY)

    mlf_daily.fit(df=pd.concat([train, val]), static_features=[])
    ml_daily_test = mlf_daily.predict(h=HORIZON_DAILY)
    return ml_daily_val, ml_daily_test

def _run_lag_mlforecast(
    train_lag: pd.DataFrame,
    val_lag: pd.DataFrame,
    test_lag: pd.DataFrame,
    freq: int,
    horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mlf_daily_lag = MLForecast(
        models=[
            LinearRegression(),
            HuberRegressor(epsilon=1.35, alpha=1e-3),
            RandomForestRegressor(n_estimators=400, max_depth=20, min_samples_leaf=5, max_features='sqrt', random_state=RANDOM_SEED),
            LGBMRegressor(objective="regression", boosting_type="gbdt", learning_rate=0.1, n_estimators=300, num_leaves=31, max_depth=6, min_child_samples=10, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0.3, random_state=RANDOM_SEED),
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

    mlf_daily_lag.fit(df=train_lag, static_features=[])
    ml_daily_val_lag = mlf_daily_lag.predict(h=horizon, X_df=val_lag)

    mlf_daily_lag.fit(df=pd.concat([train_lag, val_lag]), static_features=[])
    ml_daily_test_lag = mlf_daily_lag.predict(h=horizon, X_df=test_lag)

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

    ml_daily_val, ml_daily_test = _run_normal_mlforecast(train, val, test, FREQ_DAILY)
   
    val = _fill_missing_data(val, "2025-12-31")
    test = pd.DataFrame({"ds": pd.date_range(start="2026-01-01", end="2026-12-31", freq=FREQ_DAILY),
                         "unique_id": "Austria"})

    train_lag = merge_holidays_daily(train)
    val_lag = merge_holidays_daily(val)
    test_lag = merge_holidays_daily(test)

    ml_daily_val_lag, ml_daily_test_lag = _run_lag_mlforecast(train_lag, val_lag, test_lag, FREQ_DAILY, HORIZON_DAILY)

    ml_daily_val_all = ml_daily_val_lag.merge(ml_daily_val, on=['unique_id','ds'], how='left')
    ml_daily_test_all = ml_daily_test_lag.merge(ml_daily_test, on=['unique_id','ds'], how='left')


    write_existing_forecasts(ml_daily_val_all, ml_daily_test_all, "ml_daily")
    return merge_datasets_on_forecast(val, test, ml_daily_val_all, ml_daily_test_all)


def run_machine_learning_forecast_monthly(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    train_daily: pd.DataFrame,
    val_daily: pd.DataFrame,
    test_daily: pd.DataFrame,
    use_existing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run machine learning forecasting methods on the provided data - monthly"""
    if use_existing:
        return load_existing_forecasts(val, test, "ml_monthly")

    ml_monthly_val, ml_monthly_test = _run_normal_mlforecast(train, val, test, FREQ_MONTHLY)


    test = pd.DataFrame({"ds": pd.date_range(start="2026-01-01", end="2026-12-31", freq=FREQ_MONTHLY),
                         "unique_id": "Austria"})
    test_daily = pd.DataFrame({"ds": pd.date_range(start="2026-01-01", end="2026-12-31", freq=FREQ_DAILY),
                         "unique_id": "Austria"})
    train_lag = merge_holidays_monthly(train, train_daily)
    val_lag = merge_holidays_monthly(val, val_daily)
    test_lag = merge_holidays_monthly(test, test_daily)

    ml_monthly_val_lag, ml_monthly_test_lag = _run_lag_mlforecast(train_lag, val_lag, test_lag, FREQ_MONTHLY, HORIZON_MONTHLY)

    ml_monthly_val_all = ml_monthly_val_lag.merge(ml_monthly_val, on=['unique_id','ds'], how='left')
    ml_monthly_test_all = ml_monthly_test_lag.merge(ml_monthly_test, on=['unique_id','ds'], how='left')

    write_existing_forecasts(ml_monthly_val_all, ml_monthly_test_all, "ml_monthly")
    return merge_datasets_on_forecast(val, test, ml_monthly_val_all, ml_monthly_test_all)


        
if __name__ == "__main__":
    train, val, test = load_daily_data(use_existing=True)
    run_machine_learning_forecast_daily(train, val, test, use_existing=False)
    run_machine_learning_forecast_daily(train, val, test, use_existing=True)
    train_m, val_m, test_m = load_monthly_data(use_existing=True)
    run_machine_learning_forecast_monthly(train_m, val_m, test_m, train, val, test, use_existing=False)
    run_machine_learning_forecast_monthly(train_m, val_m, test_m, train, val, test, use_existing=True)


