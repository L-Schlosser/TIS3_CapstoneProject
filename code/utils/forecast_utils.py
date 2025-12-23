import os
import pandas as pd
import holidays
from typing import Tuple
if __package__:
    from .constants import FORECAST_DATA_DIR
else:
    from constants import FORECAST_DATA_DIR

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
    val_forecase = val_forecase.merge(val, on=["unique_id", "ds"], how="left")
    test_forecast = test_forecast.merge(test, on=["unique_id", "ds"], how="left")
    return val_forecase, test_forecast


def merge_holidays_daily(df):
    """Merge holiday information into daily dataframe"""
    ##check if dates are missing
    aut_holidays = holidays.Austria(years=range(2015, 2026))
    df['is_holiday'] = df['ds'].isin(aut_holidays.keys()).astype('int8')
    return df

def merge_holidays_monthly(df, df_daily):
    """Merge holiday information into monthly dataframe"""
    if 'is_holiday' not in df_daily.columns:
        df_daily = merge_holidays_daily(df_daily)

    daily_helper = df_daily[['unique_id', 'ds', 'is_holiday']].copy()
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


def _fill_missing_data(df, end_date):
    import pandas as pd

    last_mean = df.tail(10)["y"].mean()

    full_dates = pd.date_range(start=df["ds"].min(), end=pd.to_datetime(end_date), freq="D")
    df_full = df.set_index("ds").reindex(full_dates).rename_axis("ds").reset_index()

    df_full["y"] = df_full["y"].fillna(last_mean)
    df_full["unique_id"] = "Austria"
    return df_full


    val = _fill_missing_data(val, "2025-12-31")
    test = pd.DataFrame({"ds": pd.date_range(start="2026-01-01", end="2026-12-31", freq=FREQ_DAILY),
                         "unique_id": "Austria"})
    
        test = pd.DataFrame({"ds": pd.date_range(start="2026-01-01", end="2026-12-31", freq=FREQ_MONTHLY),
                         "unique_id": "Austria"})
    test_daily = pd.DataFrame({"ds": pd.date_range(start="2026-01-01", end="2026-12-31", freq=FREQ_DAILY),
                         "unique_id": "Austria"})