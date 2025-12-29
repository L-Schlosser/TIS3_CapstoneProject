import polars as pl
import pandas as pd
from typing import Tuple

if __package__:
    from .constants import FREQ_DAILY, HORIZON_DAILY, FREQ_MONTHLY, HORIZON_MONTHLY, INPUT_SIZE_DAILY, INPUT_SIZE_MONTHLY, INPUT_SIZE_DAILY_LAGS, INPUT_SIZE_MONTHLY_LAGS, DATE_RANGE_TRAIN, DATE_RANGE_VAL, DATE_RANGE_VAL_EXTENDED, DATE_RANGE_TEST
    from .preprocessing import load_daily_data, load_monthly_data
    from .forecast_utils import load_existing_forecasts, write_existing_forecasts, merge_datasets_on_forecast, create_deep_learning_lag
else:
    from constants import FREQ_DAILY, HORIZON_DAILY, FREQ_MONTHLY, HORIZON_MONTHLY, INPUT_SIZE_DAILY, INPUT_SIZE_MONTHLY, INPUT_SIZE_DAILY_LAGS, INPUT_SIZE_MONTHLY_LAGS, DATE_RANGE_TRAIN, DATE_RANGE_VAL, DATE_RANGE_VAL_EXTENDED, DATE_RANGE_TEST
    from preprocessing import load_daily_data, load_monthly_data
    from forecast_utils import load_existing_forecasts, write_existing_forecasts, merge_datasets_on_forecast, create_deep_learning_lag

from neuralforecast import NeuralForecast
from neuralforecast.models import KAN, NHITS, RNN, LSTM
from neuralforecast.losses.pytorch import MAE



def _run_normal_dlforecast(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    freq: int,
    horizon: int,
    input_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:

    print(int(horizon // 2))
    dl_forecast = NeuralForecast(
        models=[
            NHITS(
                h=horizon,
                input_size=input_size,
                n_freq_downsample=[2, 1, 1],
                scaler_type='robust',
                max_steps=500,
                inference_windows_batch_size=8,
                learning_rate=1e-3,
                early_stop_patience_steps=10
            ),
            KAN(
                h=horizon,
                input_size=input_size,
                loss=MAE(),
                scaler_type='robust',
                learning_rate=1e-3,
                max_steps=500,
                early_stop_patience_steps=10
            ),
            RNN(
                h=horizon,
                input_size=input_size,
                inference_input_size=input_size,
                loss=MAE(),
                scaler_type='robust',
                encoder_n_layers=1,
                encoder_hidden_size=64,
                decoder_hidden_size=64,
                decoder_layers=1,
                max_steps=500,
                early_stop_patience_steps=10
            ),
            LSTM(
                input_size=input_size,
                h=horizon,
                loss=MAE(),
                scaler_type='robust',
                encoder_n_layers=1,
                encoder_hidden_size=64,
                decoder_hidden_size=64,
                decoder_layers=1,
                max_steps=500,
                early_stop_patience_steps=10
            ),
        ],
        freq=freq,
    )

    dl_forecast.fit(df=train, val_size=horizon)
    dl_val = dl_forecast.predict(h=horizon)

    dl_forecast.fit(df=pd.concat([train, val]), val_size=horizon)
    dl_test = dl_forecast.predict(h=horizon)
    return dl_val, dl_test

def _run_lag_dlforecast(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    freq: int,
    horizon: int,
    input_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_lag = create_deep_learning_lag(train, freq, DATE_RANGE_TRAIN)
    val_lag = create_deep_learning_lag(val, freq, DATE_RANGE_VAL)
    lags_columns = [column for column in train_lag.columns if column not in ['unique_id', 'ds', 'y']]

    dl_forecast_lag = NeuralForecast(
        models=[
            NHITS(
                h=horizon,
                input_size=input_size,
                n_freq_downsample=[2, 1, 1],
                scaler_type='robust',
                max_steps=500,
                inference_windows_batch_size=8,
                learning_rate=1e-3,
                early_stop_patience_steps=10,
                hist_exog_list=lags_columns
            ),
            KAN(
                h=horizon,
                input_size=input_size,
                loss=MAE(),
                scaler_type='robust',
                learning_rate=1e-3,
                max_steps=500,
                early_stop_patience_steps=10,
                hist_exog_list=lags_columns
            ),
            RNN(
                h=horizon,
                input_size=input_size,
                inference_input_size=input_size,
                loss=MAE(),
                scaler_type='robust',
                encoder_n_layers=1,
                encoder_hidden_size=64,
                decoder_hidden_size=64,
                decoder_layers=1,
                max_steps=500,
                early_stop_patience_steps=10,        
                hist_exog_list=lags_columns
            ),
            LSTM(
                input_size=input_size,
                h=horizon,
                loss=MAE(),
                scaler_type='robust',
                encoder_n_layers=1,
                encoder_hidden_size=64,
                decoder_hidden_size=64,
                decoder_layers=1,
                max_steps=500,
                early_stop_patience_steps=10,
                hist_exog_list=lags_columns
            ),
        ],
        freq=freq,
    )

    dl_forecast_lag.fit(df=train_lag, val_size=horizon)
    dl_val_lag = dl_forecast_lag.predict(h=horizon)
    
    dl_forecast_lag.fit(df=pd.concat([train_lag, val_lag]), val_size=horizon)
    dl_test_lag = dl_forecast_lag.predict(h=horizon)

    rename_dict = {col: f"{col}_Lag" for col in dl_val_lag.columns if col not in ['unique_id', 'ds']}
    dl_val_lag = dl_val_lag.rename(columns=rename_dict)
    dl_test_lag = dl_test_lag.rename(columns=rename_dict)
    return dl_val_lag, dl_test_lag

def run_deep_learning_forecast_daily(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    use_existing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run deep learning forecasting methods on the provided data."""
    if use_existing:
        return load_existing_forecasts(val, test, "dl_daily")

    dl_daily_val, dl_daily_test = _run_normal_dlforecast(train, val, test, FREQ_DAILY, HORIZON_DAILY, INPUT_SIZE_DAILY)
    dl_daily_val_lag, dl_daily_test_lag = _run_lag_dlforecast(train, val, test, FREQ_DAILY, HORIZON_DAILY, INPUT_SIZE_DAILY_LAGS)
    
    dl_daily_val_all = dl_daily_val_lag.merge(dl_daily_val, on=['unique_id','ds'], how='left')
    dl_daily_test_all = dl_daily_test_lag.merge(dl_daily_test, on=['unique_id','ds'], how='left')

    write_existing_forecasts(dl_daily_val_all, dl_daily_test_all, "dl_daily")
    return merge_datasets_on_forecast(val, test, dl_daily_val_all, dl_daily_test_all)


def run_deep_learning_forecast_monthly(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    use_existing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run deep learning forecasting methods on the provided data."""
    if use_existing:
        return load_existing_forecasts(val, test, "dl_monthly")

    dl_monthly_val, dl_monthly_test = _run_normal_dlforecast(train, val, test, FREQ_MONTHLY, HORIZON_MONTHLY, INPUT_SIZE_MONTHLY)
    dl_monthly_val_lag, dl_monthly_test_lag = _run_lag_dlforecast(train, val, test, FREQ_MONTHLY, HORIZON_MONTHLY, INPUT_SIZE_MONTHLY_LAGS)

    dl_monthly_val_all = dl_monthly_val_lag.merge(dl_monthly_val, on=['unique_id','ds'], how='left')
    dl_monthly_test_all = dl_monthly_test_lag.merge(dl_monthly_test, on=['unique_id','ds'], how='left')

    write_existing_forecasts(dl_monthly_val_all, dl_monthly_test_all, "dl_monthly")
    return merge_datasets_on_forecast(val, test, dl_monthly_val_all, dl_monthly_test_all)

if __name__ == "__main__":
    train, val, test = load_daily_data(use_existing=True)
    run_deep_learning_forecast_daily(train, val, test, use_existing=False)
    run_deep_learning_forecast_daily(train, val, test, use_existing=True)
    train_m, val_m, test_m = load_monthly_data(use_existing=True)
    run_deep_learning_forecast_monthly(train_m, val_m, test_m, use_existing=False)
    run_deep_learning_forecast_monthly(train_m, val_m, test_m, use_existing=True)