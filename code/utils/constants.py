import os

RANDOM_SEED = 42

HORIZON_DAILY = 365 # days
HORIZON_MONTHLY = 12 # months
FREQ_DAILY = "D" # daily
FREQ_MONTHLY = "MS" # month start
H_VAL_DAILY = 365
H_VAL_MONTHLY = 12
INPUT_SIZE_DAILY = 180  #half a year
INPUT_SIZE_DAILY_LAGS = 60
INPUT_SIZE_MONTHLY = H_VAL_MONTHLY * 3
INPUT_SIZE_MONTHLY_LAGS = H_VAL_MONTHLY * 2


YEAR_RANGE_TRAIN = range(2015, 2024)
YEAR_RANGE_VAL = range(2024, 2025)
YEAR_RANGE_TEST = range(2025, 2026)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
PREPROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../results/preprocessed_data")
FORECAST_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../results/forecast_data")