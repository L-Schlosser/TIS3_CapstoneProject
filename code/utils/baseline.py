import os
import polars as pl
import pandas as pd
from typing import Tuple

if __name__ == "__main__":
    from constants import FREQ_DAILY, HORIZON_DAILY, FREQ_MONTHLY, HORIZON_MONTHLY
    from preprocessing import load_daily_data, load_monthly_data
else:
    from .constants import FREQ_DAILY, HORIZON_DAILY, FREQ_MONTHLY, HORIZON_MONTHLY
    from .preprocessing import load_daily_data, load_monthly_data

from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, HistoricAverage, RandomWalkWithDrift

RESULTS_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../results/forecast_data")

def _load_existing_forecasts(val, test, prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load existing forecasts from CSV files."""
    val_forecast = pd.read_csv(os.path.join(RESULTS_DATA_DIR, f"{prefix}_val.csv"), parse_dates=["ds"])
    test_forecast = pd.read_csv(os.path.join(RESULTS_DATA_DIR, f"{prefix}_test.csv"), parse_dates=["ds"])
    return _merge_datasets_on_forecast(val, test, val_forecast, test_forecast)

def _write_existing_forecasts(val_forecast: pd.DataFrame, test_forecast: pd.DataFrame, prefix: str) -> None:
    """Write forecasts to CSV files."""
    val_forecast.to_csv(os.path.join(RESULTS_DATA_DIR, f"{prefix}_val.csv"), index=False)
    test_forecast.to_csv(os.path.join(RESULTS_DATA_DIR, f"{prefix}_test.csv"), index=False)

def _merge_datasets_on_forecast(val: pd.DataFrame, test: pd.DataFrame, val_forecase: pd.DataFrame, test_forecast: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    val_forecase = val_forecase.merge(val, on=["unique_id", "ds"], how="left")
    test_forecast = test_forecast.merge(test, on=["unique_id", "ds"], how="left")
    return val_forecase, test_forecast

def run_baseline_forecast_daily(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    use_existing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run statistical forecasting methods on the provided data."""
    sf_daily = StatsForecast(
        models=[
            SeasonalNaive(season_length=HORIZON_DAILY),
            RandomWalkWithDrift(),
            HistoricAverage(),
        ],
        freq=FREQ_DAILY,
        n_jobs=1
    )
    if use_existing:
        return _load_existing_forecasts(val, test, "base_daily")

    base_daily_val = sf_daily.forecast(df=train, h=HORIZON_DAILY)
    base_daily_test = sf_daily.forecast(df=pd.concat([train, val]), h=HORIZON_DAILY)

    for forecast in [base_daily_val, base_daily_test]:
        forecast["Structural"] = (forecast["SeasonalNaive"] + forecast["RWD"]) / 2

    _write_existing_forecasts(base_daily_val, base_daily_test, "base_daily")
    return _merge_datasets_on_forecast(val, test, base_daily_val, base_daily_test)

def run_statistical_forecast_monthly(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    use_existing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run statistical forecasting methods on the provided monthly data."""
    sf_monthly = StatsForecast(
        models=[
            SeasonalNaive(season_length=HORIZON_MONTHLY),
            RandomWalkWithDrift(),
            HistoricAverage(),
        ],
        freq=FREQ_MONTHLY,
        n_jobs=1
    )
    if use_existing:
        return _load_existing_forecasts(val, test, "base_monthly")

    base_monthly_val = sf_monthly.forecast(df=train, h=HORIZON_MONTHLY)
    base_monthly_test = sf_monthly.forecast(df=pd.concat([train, val]), h=HORIZON_MONTHLY)

    for forecast in [base_monthly_val, base_monthly_test]:
        forecast["Structural"] = (forecast["SeasonalNaive"] + forecast["RWD"]) / 2

    _write_existing_forecasts(base_monthly_val, base_monthly_test, "base_monthly")
    return _merge_datasets_on_forecast(val, test, base_monthly_val, base_monthly_test)
        
if __name__ == "__main__":
    train, val, test = load_daily_data(use_existing=True)
    run_baseline_forecast_daily(train, val, test, use_existing=False)
    run_baseline_forecast_daily(train, val, test, use_existing=True)
    train_m, val_m, test_m = load_monthly_data(use_existing=True)
    run_statistical_forecast_monthly(train_m, val_m, test_m, use_existing=False)
    run_statistical_forecast_monthly(train_m, val_m, test_m, use_existing=True)