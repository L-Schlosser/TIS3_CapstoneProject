import polars as pl
import pandas as pd
from typing import Tuple


if __package__:
    from .constants import FREQ_DAILY, HORIZON_DAILY, FREQ_MONTHLY, HORIZON_MONTHLY, INPUT_SIZE_DAILY, INPUT_SIZE_MONTHLY, H_VAL_DAILY, H_VAL_MONTHLY
    from .preprocessing import load_daily_data, load_monthly_data
    from .forecast_utils import load_existing_forecasts, write_existing_forecasts, merge_datasets_on_forecast, merge_holidays_daily, merge_holidays_monthly
else:
    from constants import FREQ_DAILY, HORIZON_DAILY, FREQ_MONTHLY, HORIZON_MONTHLY, INPUT_SIZE_DAILY, INPUT_SIZE_MONTHLY, H_VAL_DAILY, H_VAL_MONTHLY
    from preprocessing import load_daily_data, load_monthly_data
    from forecast_utils import load_existing_forecasts, write_existing_forecasts, merge_datasets_on_forecast, merge_holidays_daily, merge_holidays_monthly

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from neuralforecast import NeuralForecast
from neuralforecast.models import KAN, NHITS, RNN, LSTM
from neuralforecast.losses.pytorch import MAE

def _create_lag_daily(df):
    df = merge_holidays_daily(df, FREQ_DAILY)

    # 6 useful lag features
    grouped = df.groupby('unique_id')['y']

    df['lag1'] = grouped.transform(lambda x: x.shift(1)).fillna(0)         # yesterday
    df['lag7'] = grouped.transform(lambda x: x.shift(7)).fillna(0)         # 1 week ago
    df['lag28'] = grouped.transform(lambda x: x.shift(28)).fillna(0)         # 4 weeks ago
    df['lag365'] = grouped.transform(lambda x: x.shift(365)).fillna(0)         # 1 year ago
    df['rolling_mean_7'] = grouped.transform(lambda x: x.shift(1).rolling(7).mean()).fillna(0)   # weekly trend
    df['rolling_mean_30'] = grouped.transform(lambda x: x.shift(1).rolling(30).mean()).fillna(0)  # monthly trend

    return df

## CHANGE HERE
def _create_lag_monthly(df):
    df = df.copy()
    df = merge_holidays_monthly(df, FREQ_MONTHLY)

    # 6 useful lag features
    grouped = df.groupby('unique_id')['y']

    df['lag1'] = grouped.transform(lambda x: x.shift(1)).fillna(0)          # last month
    df['lag3'] = grouped.transform(lambda x: x.shift(3)).fillna(0)          # last quarter
    df['lag6'] = grouped.transform(lambda x: x.shift(6)).fillna(0)          # half-year
    df['lag12'] = grouped.transform(lambda x: x.shift(12)).fillna(0)        # last year

    df['rolling_mean_3'] = grouped.transform(lambda x: x.shift(1).rolling(3).mean()).fillna(0)
    # quarterly trend

    df['rolling_mean_12'] = grouped.transform(lambda x: x.shift(1).rolling(12).mean()).fillna(0)
    # yearly trend

    return df



def run_deep_learning_forecast_daily(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    use_existing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run deep learning forecasting methods on the provided data."""
    if use_existing:
        return load_existing_forecasts(val, test, "ml_daily")


## normal Version
    dlf_daily = NeuralForecast(
        models=[
            NHITS(
                h=H_VAL_DAILY,
                input_size=INPUT_SIZE_DAILY,
                n_freq_downsample=[2, 1, 1],
                scaler_type='robust',
                max_steps=200,
                inference_windows_batch_size=1,
                learning_rate=1e-3,
            ),
            KAN(
                h=H_VAL_DAILY,
                input_size=INPUT_SIZE_DAILY,
                loss=MAE(),
                scaler_type='robust',
                learning_rate=1e-3,
                max_steps=500,
            ),
            RNN(
                h=H_VAL_DAILY,
                input_size=INPUT_SIZE_DAILY,
                inference_input_size=INPUT_SIZE_DAILY,
                loss=MAE(),
                scaler_type='robust',
                encoder_n_layers=2,
                encoder_hidden_size=128,
                decoder_hidden_size=128,
                decoder_layers=2,
                max_steps=200,
            ),
            LSTM(
                input_size=INPUT_SIZE_DAILY,
                h=H_VAL_DAILY,
                max_steps=500,
                loss=MAE(),
                scaler_type='robust',
                encoder_n_layers=2,
                encoder_hidden_size=128,
                decoder_hidden_size=128,
                decoder_layers=2,
            ),
        ],
        freq=FREQ_DAILY,
    )

    dlf_daily.fit(df=train)
    dl_daily_val = dlf_daily.predict(h=HORIZON_DAILY)

    dlf_daily.fit(df=pd.concat([train, val]))
    dl_daily_test = dlf_daily.predict(h=HORIZON_DAILY)

### LAGS Version

    train_lag = _create_lag_daily(train)
    val_lag = _create_lag_daily(val)
    lags_columns = ['is_holiday', 'lag1', 'lag7', 'lag28', 'lag365', 'rolling_mean_7', 'rolling_mean_30']

    dlf_daily_lag = NeuralForecast(
        models=[
            NHITS(
                h=H_VAL_DAILY,
                input_size=INPUT_SIZE_DAILY,
                n_freq_downsample=[2, 1, 1],
                scaler_type='robust',
                max_steps=200,
                inference_windows_batch_size=1,
                learning_rate=1e-3,
                hist_exog_list=lags_columns
            ),
            
            KAN(
                h=H_VAL_DAILY,
                input_size=INPUT_SIZE_DAILY,
                loss=MAE(),
                scaler_type='robust',
                learning_rate=1e-3,
                max_steps=500,
                hist_exog_list=lags_columns
            ),
            RNN(
                h=H_VAL_DAILY,
                input_size=INPUT_SIZE_DAILY,
                inference_input_size=INPUT_SIZE_DAILY,
                loss=MAE(),
                scaler_type='robust',
                encoder_n_layers=2,
                encoder_hidden_size=128,
                decoder_hidden_size=128,
                decoder_layers=2,
                max_steps=200,
                hist_exog_list=lags_columns
            ),
            LSTM(
                input_size=INPUT_SIZE_DAILY,
                h=H_VAL_DAILY,
                max_steps=500,
                loss=MAE(),
                scaler_type='robust',
                encoder_n_layers=2,
                encoder_hidden_size=128,
                decoder_hidden_size=128,
                decoder_layers=2,
                hist_exog_list=lags_columns
            ),
        ],
        freq=FREQ_DAILY,
    )

    dlf_daily_lag.fit(df=train_lag)
    dl_daily_val_lag = dlf_daily_lag.predict(h=HORIZON_DAILY)
    dl_daily_val_lag = dl_daily_val_lag.rename(columns={
        "NHITS": "NHITS_Lag",
        "KAN": "KAN_Lag",
        "RNN": "RNN_Lag",
        "LSTM": "LSTM_Lag"
    })
    
    dlf_daily_lag.fit(df=pd.concat([train_lag, val_lag]))
    dl_daily_test_lag = dlf_daily_lag.predict(h=HORIZON_DAILY)
    dl_daily_test_lag = dl_daily_test_lag.rename(columns={
        "NHITS": "NHITS_Lag",
        "KAN": "KAN_Lag",
        "RNN": "RNN_Lag",
        "LSTM": "LSTM_Lag"
    })
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
        return load_existing_forecasts(val, test, "ml_monthly")

## normal Version
    dlf_monthly = NeuralForecast(
        models=[
            NHITS(
                h=H_VAL_MONTHLY,
                input_size=INPUT_SIZE_MONTHLY,
                n_freq_downsample=[2, 1, 1],
                scaler_type='robust',
                max_steps=200,
                inference_windows_batch_size=1,
                learning_rate=1e-3,
            ),
            KAN(
                h=H_VAL_MONTHLY,
                input_size=INPUT_SIZE_MONTHLY,
                loss=MAE(),
                scaler_type='robust',
                learning_rate=1e-3,
                max_steps=500,
            ),
            RNN(
                h=H_VAL_MONTHLY,
                input_size=INPUT_SIZE_MONTHLY,
                inference_input_size=INPUT_SIZE_MONTHLY,
                loss=MAE(),
                scaler_type='robust',
                encoder_n_layers=2,
                encoder_hidden_size=128,
                decoder_hidden_size=128,
                decoder_layers=2,
                max_steps=200,
            ),
            LSTM(
                input_size=INPUT_SIZE_MONTHLY,
                h=H_VAL_MONTHLY,
                max_steps=500,
                loss=MAE(),
                scaler_type='robust',
                encoder_n_layers=2,
                encoder_hidden_size=128,
                decoder_hidden_size=128,
                decoder_layers=2,
            ),
        ],
        freq=FREQ_MONTHLY,
    )

    dlf_monthly.fit(df=train)
    dl_monthly_val = dlf_monthly.predict(h=HORIZON_MONTHLY)

    dlf_monthly.fit(df=pd.concat([train, val]))
    dl_monthly_test = dlf_monthly.predict(h=HORIZON_MONTHLY)

    
### LAGS Version

    train_lag = _create_lag_monthly(train)
    val_lag = _create_lag_monthly(val)
    lags_columns = ['count_holiday', 'lag1', 'lag3', 'lag6', 'lag12', 'rolling_mean_3', 'rolling_mean_12']

    dlf_monthly_lag = NeuralForecast(
        models=[
            NHITS(
                h=H_VAL_MONTHLY,
                input_size=INPUT_SIZE_MONTHLY,
                n_freq_downsample=[2, 1, 1],
                scaler_type='robust',
                max_steps=200,
                inference_windows_batch_size=1,
                learning_rate=1e-3,
                hist_exog_list=lags_columns
            ),
            
            KAN(
                h=H_VAL_MONTHLY,
                input_size=INPUT_SIZE_MONTHLY,
                loss=MAE(),
                scaler_type='robust',
                learning_rate=1e-3,
                max_steps=500,
                hist_exog_list=lags_columns
            ),
            RNN(
                h=H_VAL_MONTHLY,
                input_size=INPUT_SIZE_MONTHLY,
                inference_input_size=INPUT_SIZE_MONTHLY,
                loss=MAE(),
                scaler_type='robust',
                encoder_n_layers=2,
                encoder_hidden_size=128,
                decoder_hidden_size=128,
                decoder_layers=2,
                max_steps=200,
                hist_exog_list=lags_columns
            ),
            LSTM(
                input_size=INPUT_SIZE_MONTHLY,
                h=H_VAL_MONTHLY,
                max_steps=500,
                loss=MAE(),
                scaler_type='robust',
                encoder_n_layers=2,
                encoder_hidden_size=128,
                decoder_hidden_size=128,
                decoder_layers=2,
                hist_exog_list=lags_columns
            ),
        ],
        freq=FREQ_MONTHLY,
    )

    dlf_monthly_lag.fit(df=train_lag)
    dl_monthly_val_lag = dlf_monthly_lag.predict(h=HORIZON_MONTHLY)
    dl_monthly_val_lag = dl_monthly_val_lag.rename(columns={
        "NHITS": "NHITS_Lag",
        "KAN": "KAN_Lag",
        "RNN": "RNN_Lag",
        "LSTM": "LSTM_Lag"
    })
    
    dlf_monthly_lag.fit(df=pd.concat([train_lag, val_lag]))
    dl_monthly_test_lag = dlf_monthly_lag.predict(h=HORIZON_MONTHLY)
    dl_monthly_test_lag = dl_monthly_test_lag.rename(columns={
        "NHITS": "NHITS_Lag",
        "KAN": "KAN_Lag",
        "RNN": "RNN_Lag",
        "LSTM": "LSTM_Lag"
    })
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


