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
from statsforecast.models import SimpleExponentialSmoothing, Holt, HoltWinters, AutoRegressive

def run_statistical_forecast_daily(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    use_existing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run statistical forecasting methods on the provided data."""
    if use_existing:
        return load_existing_forecasts(val, test, "stat_daily")

    sf_daily = StatsForecast(
        models=[
            SimpleExponentialSmoothing(alpha=0.5),
            Holt(season_length=HORIZON_DAILY, error_type="A"),
            HoltWinters(season_length=HORIZON_DAILY, error_type="A"),
            AutoRegressive(lags=HORIZON_DAILY)
        ],
        freq=FREQ_DAILY,
        n_jobs=1
    )

    stat_daily_val = sf_daily.forecast(df=train, h=HORIZON_DAILY)
    stat_daily_test = sf_daily.forecast(df=pd.concat([train, val]), h=HORIZON_DAILY)

    write_existing_forecasts(stat_daily_val, stat_daily_test, "stat_daily")
    return merge_datasets_on_forecast(val, test, stat_daily_val, stat_daily_test)

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
    run_statistical_forecast_daily(train, val, test, use_existing=False)
    run_statistical_forecast_daily(train, val, test, use_existing=True)
    train_m, val_m, test_m = load_monthly_data(use_existing=True)
    run_statistical_forecast_monthly(train_m, val_m, test_m, use_existing=False)
    run_statistical_forecast_monthly(train_m, val_m, test_m, use_existing=True)


