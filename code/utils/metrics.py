import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

if __package__:
    from .constants import FREQ_DAILY, FREQ_MONTHLY, METRICS_DIR, FAMILY_BASELINE
    from .preprocessing import load_daily_data, load_monthly_data
    from .baseline import run_baseline_forecast_daily, run_baseline_forecast_monthly
else:
    from constants import FREQ_DAILY, FREQ_MONTHLY, METRICS_DIR, FAMILY_BASELINE
    from preprocessing import load_daily_data, load_monthly_data
    from baseline import run_baseline_forecast_daily, run_baseline_forecast_monthly

def _calculate_metric_monthly(prediction_df: pd.DataFrame, family: str, frequency: str) -> pd.DataFrame:
    """Calculate evaluation metrics for monthly frequency."""
    metrics = {
        "Model": [],
        "Family": [],
        "Frequency": [],
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
        metrics["MAE"].append(mean_absolute_error(prediction_df['y'], prediction_df[col]))
        metrics["RMSE"].append(root_mean_squared_error(prediction_df['y'], prediction_df[col]))
        metrics["MAPE"].append(mean_absolute_percentage_error(prediction_df['y'], prediction_df[col]))

    return pd.DataFrame(metrics).sort_values(by=["MAPE"])

def calculate_metrics(prediction_df: pd.DataFrame, family: str, frequency: str, use_existing: bool) -> pd.DataFrame:
    """Calculate evaluation metrics for the predictions."""
    if frequency != FREQ_DAILY and frequency != FREQ_MONTHLY:
        raise ValueError(f"Unsupported frequency: {frequency}")
    
    frequency_str = "daily" if frequency == FREQ_DAILY else "monthly"
    path = os.path.join(METRICS_DIR, f"{family}_{frequency_str}_metrics.csv")

    if use_existing:
        return pd.read_csv(path)

    if frequency == FREQ_DAILY:
        prediction_df['ds'] = pd.to_datetime(prediction_df['ds']).dt.to_period('M').dt.to_timestamp()
        columns = [c for c in prediction_df.columns if c not in ['unique_id', 'ds']]
        prediction_df = prediction_df.groupby(['unique_id', 'ds'])[columns].mean().reset_index()

    metrics = _calculate_metric_monthly(prediction_df, family, frequency)
    metrics.to_csv(path)
    return metrics


if __name__ == "__main__":
    train, val, test = load_daily_data(use_existing=True)
    val_fc, test_fc = run_baseline_forecast_daily(train, val, test, use_existing=True)
    train_m, val_m, test_m = load_monthly_data(use_existing=True)
    val_m_fc, test_m_fc = run_baseline_forecast_monthly(train_m, val_m, test_m, use_existing=True)

    calculate_metrics(val_fc, FAMILY_BASELINE, FREQ_DAILY, use_existing=False)
    calculate_metrics(val_fc, FAMILY_BASELINE, FREQ_DAILY, use_existing=True)
    calculate_metrics(val_m_fc, FAMILY_BASELINE, FREQ_MONTHLY, use_existing=False)
    calculate_metrics(val_m_fc, FAMILY_BASELINE, FREQ_MONTHLY, use_existing=True)