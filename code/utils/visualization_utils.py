import pandas as pd

if __package__:
    from .constants import FREQ_DAILY, FREQ_MONTHLY
    from .preprocessing import load_daily_data, load_monthly_data
else:
    from constants import FREQ_DAILY, FREQ_MONTHLY
    from preprocessing import load_daily_data, load_monthly_data

def daily_to_monthly(prediction_df):
    prediction_df['ds'] = pd.to_datetime(prediction_df['ds']).dt.to_period('M').dt.to_timestamp()
    columns = [c for c in prediction_df.columns if c not in ['unique_id', 'ds']]
    prediction_df = prediction_df.groupby(['unique_id', 'ds'])[columns].mean().reset_index()
    return prediction_df

def get_ground_truth(frequency: str, start_val: str, end_val: str) -> pd.DataFrame:
    """Load ground truth data based on frequency and split."""
    if frequency == FREQ_DAILY:
        data_train, data_val, data_test = load_daily_data(use_existing=True)
    elif frequency == FREQ_MONTHLY:
        data_train, data_val, data_test = load_monthly_data(use_existing=True)
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")
    
    data = pd.concat([data_train, data_val], axis=0, ignore_index=True)

    return data[(data['ds'] >= start_val) & (data['ds'] <= end_val)]