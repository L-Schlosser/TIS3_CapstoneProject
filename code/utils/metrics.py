import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

if __package__:
    from .constants import FREQ_DAILY, FREQ_MONTHLY, METRICS_DIR, FAMILY_BASELINE, FAMILY_STATISTICAL, FAMILY_MACHINE_LEARNING, FAMILY_DEEP_LEARNING, SPLIT_VAL
    from .preprocessing import load_daily_data, load_monthly_data
    from .baseline import run_baseline_forecast_daily, run_baseline_forecast_monthly
    from .statistical import run_statistical_forecast_daily, run_statistical_forecast_monthly
    from .machine_learning import run_machine_learning_forecast_daily, run_machine_learning_forecast_monthly
    from .deep_learning import run_deep_learning_forecast_daily, run_deep_learning_forecast_monthly
else:
    from constants import FREQ_DAILY, FREQ_MONTHLY, METRICS_DIR, FAMILY_BASELINE, FAMILY_STATISTICAL, FAMILY_MACHINE_LEARNING, FAMILY_DEEP_LEARNING, SPLIT_VAL
    from preprocessing import load_daily_data, load_monthly_data
    from baseline import run_baseline_forecast_daily, run_baseline_forecast_monthly
    from statistical import run_statistical_forecast_daily, run_statistical_forecast_monthly
    from machine_learning import run_machine_learning_forecast_daily, run_machine_learning_forecast_monthly
    from deep_learning import run_deep_learning_forecast_daily, run_deep_learning_forecast_monthly


def _calculate_metric_monthly(prediction_df: pd.DataFrame, family: str, frequency: str, split: str) -> pd.DataFrame:
    """Calculate evaluation metrics for monthly frequency."""
    metrics = {
        "Model": [],
        "Family": [],
        "Frequency": [],
        "Split": [],
        "MAE": [],
        "RMSE": [],
        "MAPE": [],
    }

    prediction_df = prediction_df.dropna()

    for col in prediction_df.columns :
        if col in ['unique_id', 'ds', 'y']:
            continue

        metrics["Model"].append(col)
        metrics["Family"].append(family)
        metrics["Frequency"].append(frequency)
        metrics["Split"].append(split)
        metrics["MAE"].append(mean_absolute_error(prediction_df['y'], prediction_df[col]))
        metrics["RMSE"].append(root_mean_squared_error(prediction_df['y'], prediction_df[col]))
        metrics["MAPE"].append(mean_absolute_percentage_error(prediction_df['y'], prediction_df[col]))

    return pd.DataFrame(metrics).sort_values(by=["MAPE"])

def calculate_metrics(prediction_df: pd.DataFrame, family: str, frequency: str, split: str, use_existing: bool = True) -> pd.DataFrame:
    """Calculate evaluation metrics for the predictions."""
    if frequency != FREQ_DAILY and frequency != FREQ_MONTHLY:
        raise ValueError(f"Unsupported frequency: {frequency}")
    
    frequency_str = "daily" if frequency == FREQ_DAILY else "monthly"
    path = os.path.join(METRICS_DIR, f"{family}_{frequency_str}_{split}_metrics.csv")

    if use_existing:
        return pd.read_csv(path)

    if frequency == FREQ_DAILY:
        prediction_df['ds'] = pd.to_datetime(prediction_df['ds']).dt.to_period('M').dt.to_timestamp()
        columns = [c for c in prediction_df.columns if c not in ['unique_id', 'ds']]
        prediction_df = prediction_df.groupby(['unique_id', 'ds'])[columns].mean().reset_index()

    metrics = _calculate_metric_monthly(prediction_df, family, frequency, split)
    metrics.to_csv(path)
    return metrics

if __name__ == "__main__":
    train, val, test = load_daily_data(use_existing=True)
    train_m, val_m, test_m = load_monthly_data(use_existing=True)

    val_base, test_base = run_baseline_forecast_daily(train, val, test, use_existing=True)
    val_stat, test_stat = run_statistical_forecast_daily(train, val, test, use_existing=True)
    val_ml, test_ml = run_machine_learning_forecast_daily(train, val, test, use_existing=True)
    val_dl, test_dl = run_deep_learning_forecast_daily(train, val, test, use_existing=True)

    calculate_metrics(val_base, FAMILY_BASELINE, FREQ_DAILY, SPLIT_VAL, use_existing=False)
    calculate_metrics(val_stat, FAMILY_STATISTICAL, FREQ_DAILY, SPLIT_VAL, use_existing=False)
    calculate_metrics(val_ml, FAMILY_MACHINE_LEARNING, FREQ_DAILY, SPLIT_VAL, use_existing=False)
    calculate_metrics(val_dl, FAMILY_DEEP_LEARNING, FREQ_DAILY, SPLIT_VAL, use_existing=False)

    val_m_base, test_m_base = run_baseline_forecast_monthly(train_m, val_m, test_m, use_existing=True)
    val_m_stat, test_m_stat = run_statistical_forecast_monthly(train_m, val_m, test_m, use_existing=True)
    val_m_ml, test_m_ml = run_machine_learning_forecast_monthly(train_m, val_m, test_m, use_existing=True)
    val_m_dl, test_m_dl = run_deep_learning_forecast_monthly(train_m, val_m, test_m, use_existing=True)

    calculate_metrics(val_m_base, FAMILY_BASELINE, FREQ_MONTHLY, SPLIT_VAL, use_existing=False)
    calculate_metrics(val_m_stat, FAMILY_STATISTICAL, FREQ_MONTHLY, SPLIT_VAL, use_existing=False)
    calculate_metrics(val_m_ml, FAMILY_MACHINE_LEARNING, FREQ_MONTHLY, SPLIT_VAL, use_existing=False)
    calculate_metrics(val_m_dl, FAMILY_DEEP_LEARNING, FREQ_MONTHLY, SPLIT_VAL, use_existing=False)