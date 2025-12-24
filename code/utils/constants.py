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


DATE_RANGE_TRAIN = ("01.01.2015", "31.12.2024")
DATE_RANGE_VAL = ("01.01.2025", "22.12.2025")
DATE_RANGE_VAL_EXTENDED = ("01.01.2025", "31.12.2025")
DATE_RANGE_TEST = ("23.12.2025", "22.12.2026")

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
PREPROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../results/preprocessed_data")
FORECAST_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../results/forecast_data")