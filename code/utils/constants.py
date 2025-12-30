import os

RANDOM_SEED = 42

HORIZON_DAILY = 365 # days
HORIZON_MONTHLY = 12 # months
FREQ_DAILY = "D" # daily
FREQ_MONTHLY = "MS" # month start
INPUT_SIZE_DAILY = 182  # half a year
INPUT_SIZE_DAILY_LAGS = 60
INPUT_SIZE_MONTHLY = HORIZON_MONTHLY * 3
INPUT_SIZE_MONTHLY_LAGS = HORIZON_MONTHLY * 2

FAMILY_BASELINE = "base"
FAMILY_STATISTICAL = "stat"
FAMILY_MACHINE_LEARNING = "ml"
FAMILY_DEEP_LEARNING = "dl"

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"

DATE_RANGE_TRAIN = ("01.01.2015", "31.12.2024")
DATE_RANGE_VAL = ("01.01.2025", "22.12.2025")
DATE_RANGE_VAL_EXTENDED = ("01.01.2025", "31.12.2025")
DATE_RANGE_TEST = ("23.12.2025", "22.12.2026")

DEFAULT_START_VAL_DATASET = "2015-01-01"
DEFAULT_START_VAL_VIS = "2023-01-01"
DEFAULT_END_VAL_VIS = "2025-12-31"
FUTURE_END_VAL_VIS = "2026-12-31"


DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
PREPROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../results/preprocessed_data")
FORECAST_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../results/forecast_data")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "../../results/metrics")
VISUALIZATION_DIR = os.path.join(os.path.dirname(__file__), "../../results/visualizations")