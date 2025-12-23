import os
import datetime
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

def _merge_holidays_daily(df: pd.DataFrame, years: list):
    """Merge holiday information into daily dataframe"""
    austrian_holidays = holidays.Austria(years=years)
    unique_dates = pd.DataFrame({'ds': pd.date_range(start=f'{min(years)}-01-01', end=f'{max(years)}-12-31', freq='D')})
    df = unique_dates.merge(df, on='ds', how='left')
    holiday_ts = pd.to_datetime(list(austrian_holidays.keys()))
    df['is_holiday'] = df['ds'].isin(holiday_ts).astype('int8')
    return df

def _merge_holidays_monthly(df: pd.DataFrame, years: list):
    """Merge holiday information into monthly dataframe"""
    daily_helper = _merge_holidays_daily(df, years)
    daily_helper['year_month'] = daily_helper['ds'].dt.to_period('M')

    monthly_holidays = (
        daily_helper
        .groupby(['unique_id', 'year_month'], as_index=False)['is_holiday']
        .sum()
        .rename(columns={'is_holiday': 'count_holiday'})
    )

    monthly_holidays['ds'] = monthly_holidays['year_month'].dt.to_timestamp()
    monthly_holidays = monthly_holidays.drop(columns=['year_month'])

    df = df.merge(monthly_holidays, on=['unique_id', 'ds'], how='left')
    df['count_holiday'] = df['count_holiday'].fillna(0).astype('int16')
    return df

def merge_holidays(df: pd.DataFrame, freq: str, years: list = range(2015, 2026)) -> pd.DataFrame:
    """Merge holiday information into dataframe based on frequency"""
    if freq == FREQ_DAILY:
        return _merge_holidays_daily(df, years)
    elif freq == FREQ_MONTHLY:
        return _merge_holidays_monthly(df, years)
    else:
        raise ValueError(f"Unsupported frequency: {freq}")