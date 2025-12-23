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

from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, HistoricAverage, RandomWalkWithDrift


def run_baseline_forecast_daily(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    use_existing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run baseline forecasting methods on the provided data."""
    if use_existing:
        return load_existing_forecasts(val, test, "base_daily")
    
    sf_daily = StatsForecast(
        models=[
            SeasonalNaive(season_length=7), # weekly seasonality
            RandomWalkWithDrift(),
            HistoricAverage(),
        ],
        freq=FREQ_DAILY,
        n_jobs=1
    )

    base_daily_val = sf_daily.forecast(df=train, h=HORIZON_DAILY)
    base_daily_test = sf_daily.forecast(df=pd.concat([train, val]), h=HORIZON_DAILY)

    for forecast in [base_daily_val, base_daily_test]:
        forecast["Structural"] = (forecast["SeasonalNaive"] + forecast["RWD"]) / 2

    write_existing_forecasts(base_daily_val, base_daily_test, "base_daily")
    return merge_datasets_on_forecast(val, test, base_daily_val, base_daily_test)

def run_baseline_forecast_monthly(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    use_existing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run baseline forecasting methods on the provided monthly data."""
    if use_existing:
        return load_existing_forecasts(val, test, "base_monthly")
    
    sf_monthly = StatsForecast(
        models=[
            SeasonalNaive(season_length=12), # yearly seasonality
            RandomWalkWithDrift(),
            HistoricAverage(),
        ],
        freq=FREQ_MONTHLY,
        n_jobs=1
    )
    
    base_monthly_val = sf_monthly.forecast(df=train, h=HORIZON_MONTHLY)
    base_monthly_test = sf_monthly.forecast(df=pd.concat([train, val]), h=HORIZON_MONTHLY)

    for forecast in [base_monthly_val, base_monthly_test]:
        forecast["Structural"] = (forecast["SeasonalNaive"] + forecast["RWD"]) / 2

    write_existing_forecasts(base_monthly_val, base_monthly_test, "base_monthly")
    return merge_datasets_on_forecast(val, test, base_monthly_val, base_monthly_test)
        
if __name__ == "__main__":
    train, val, test = load_daily_data(use_existing=True)
    run_baseline_forecast_daily(train, val, test, use_existing=False)
    run_baseline_forecast_daily(train, val, test, use_existing=True)
    train_m, val_m, test_m = load_monthly_data(use_existing=True)
    run_baseline_forecast_monthly(train_m, val_m, test_m, use_existing=False)
    run_baseline_forecast_monthly(train_m, val_m, test_m, use_existing=True)