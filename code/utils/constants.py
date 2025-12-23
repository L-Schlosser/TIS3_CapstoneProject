import os

RANDOM_SEED = 42

HORIZON_DAILY = 365 # days
HORIZON_MONTHLY = 12 # months
FREQ_DAILY = "D" # daily
FREQ_MONTHLY = "MS" # month start
H_VAL_DAILY = 365
H_VAL_MONTHLY = 12
INPUT_SIZE_DAILY = H_VAL_DAILY * 9
INPUT_SIZE_MONTHLY = H_VAL_MONTHLY * 9

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
PREPROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../results/preprocessed_data")
FORECAST_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../results/forecast_data")