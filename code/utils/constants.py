import os

HORIZON_DAILY = 365 # days
HORIZON_MONTHLY = 12 # months
FREQ_DAILY = "D" # daily
FREQ_MONTHLY = "MS" # month start

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
PREPROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../results/preprocessed_data")
FORECAST_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../results/forecast_data")