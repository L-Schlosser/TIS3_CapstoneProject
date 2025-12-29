import os
import datetime
import numpy as np
import pandas as pd
import holidays
from typing import Tuple
if __package__:
    from .constants import FORECAST_DATA_DIR, FREQ_DAILY, FREQ_MONTHLY
else:
    from constants import FORECAST_DATA_DIR, FREQ_DAILY, FREQ_MONTHLY

def load_existing_forecasts(val, test, prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load existing forecasts from CSV files."""
    val_forecast = pd.read_csv(os.path.join(FORECAST_DATA_DIR, f"{prefix}_val.csv"), parse_dates=["ds"])
    test_forecast = pd.read_csv(os.path.join(FORECAST_DATA_DIR, f"{prefix}_test.csv"), parse_dates=["ds"])
    return merge_datasets_on_forecast(val, test, val_forecast, test_forecast)

def write_existing_forecasts(val_forecast: pd.DataFrame, test_forecast: pd.DataFrame, prefix: str) -> None:
    """Write forecasts to CSV files."""
    val_forecast.to_csv(os.path.join(FORECAST_DATA_DIR, f"{prefix}_val.csv"), index=False)
    test_forecast.to_csv(os.path.join(FORECAST_DATA_DIR, f"{prefix}_test.csv"), index=False)

def merge_datasets_on_forecast(val: pd.DataFrame, test: pd.DataFrame, val_forecase: pd.DataFrame, test_forecast: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge original datasets with forecasted values."""
    val_forecase = val_forecase.merge(val, on=["unique_id", "ds"], how="left")
    test_forecast = test_forecast.merge(test, on=["unique_id", "ds"], how="left")
    return val_forecase, test_forecast

def _merge_holidays_daily(df: pd.DataFrame, date_range: tuple) -> pd.DataFrame:
    """Merge holiday information into daily dataframe"""
    unique_dates = pd.DataFrame({"ds": pd.date_range(start=date_range[0], end=date_range[1], freq="D"), "unique_id": "Austria"})
    austrian_holidays = holidays.Austria(years=range(unique_dates["ds"].min().year, unique_dates["ds"].max().year + 1))
    df = unique_dates.merge(df, on=["unique_id", "ds"], how="left")
    holiday_ts = pd.to_datetime(list(austrian_holidays.keys()))
    df["is_holiday"] = df["ds"].isin(holiday_ts).astype("int8")
    return df

def _merge_holidays_monthly(df: pd.DataFrame, date_range: tuple) -> pd.DataFrame:
    """Merge holiday information into monthly dataframe"""
    daily_helper = _merge_holidays_daily(df, date_range)
    daily_helper["year_month"] = daily_helper["ds"].dt.to_period("M")

    monthly_holidays = (
        daily_helper
        .groupby(["unique_id", "year_month"], as_index=False)["is_holiday"]
        .sum()
        .rename(columns={"is_holiday": "count_holiday"})
    )

    monthly_holidays["ds"] = monthly_holidays["year_month"].dt.to_timestamp()
    monthly_holidays = monthly_holidays.drop(columns=["year_month"])

    df = monthly_holidays.merge(df, on=["unique_id", "ds"], how="left")
    return df

def merge_holidays(df: pd.DataFrame, freq: str, date_range: tuple) -> pd.DataFrame:
    """Merge holiday information into dataframe based on frequency"""
    if freq == FREQ_DAILY:
        daily_result = _merge_holidays_daily(df, date_range)
        daily_result["y"] = daily_result["y"].infer_objects(copy=False).fillna(-1)
        return daily_result
    elif freq == FREQ_MONTHLY:
        monthly_result = _merge_holidays_monthly(df, date_range)
        monthly_result["y"] = monthly_result["y"].infer_objects(copy=False).fillna(-1)
        return monthly_result
    else:
        raise ValueError(f"Unsupported frequency: {freq}")

def _create_dl_lag_daily(df: pd.DataFrame, date_range: tuple) -> pd.DataFrame:
    df = merge_holidays(df, FREQ_DAILY, date_range)

    # 6 useful lag features
    grouped = df.groupby('unique_id')['y']

    df['rolling_mean_3_lag1'] = grouped.transform(
        lambda x: x.shift(1).rolling(window=3).mean()
    ).fillna(0)   # short-term trend

    df['rolling_mean_3_lag7'] = grouped.transform(
        lambda x: x.shift(7).rolling(window=3).mean()
    ).fillna(0)   # weekly-smoothed trend

    df['rolling_mean_14_lag28'] = grouped.transform(
        lambda x: x.shift(28).rolling(window=14).mean()
    ).fillna(0)

    return df

def _create_dl_lag_monthly(df: pd.DataFrame, date_range: tuple) -> pd.DataFrame:
    df = merge_holidays(df, FREQ_MONTHLY, date_range)

    # 6 useful lag features
    grouped = df.groupby('unique_id')['y']

    df['rolling_mean_3_lag6'] = grouped.transform(
        lambda x: x.shift(6).rolling(window=3).mean()
    ).fillna(0)

    df['rolling_mean_12_lag24'] = grouped.transform(
        lambda x: x.shift(24).rolling(window=12).mean()
    ).fillna(0)
    return df

def create_deep_learning_lag(
    df: pd.DataFrame,
    freq: str,
    date_range: tuple) -> pd.DataFrame:
    """Create lag features for deep learning models based on frequency"""
    if freq == FREQ_DAILY:
        return _create_dl_lag_daily(df, date_range)
    elif freq == FREQ_MONTHLY:
        return _create_dl_lag_monthly(df, date_range)
    else:
        raise ValueError(f"Unsupported frequency: {freq}")