import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd

if __package__:
    from .constants import DATE_RANGE_VAL, FREQ_DAILY, HORIZON_DAILY, FREQ_MONTHLY, DEFAULT_START_VAL_VIS, DEFAULT_END_VAL_VIS, FUTURE_END_VAL_VIS, DATE_RANGE_VAL_EXTENDED, DATE_RANGE_TEST
    from .preprocessing import load_daily_data, load_monthly_data
    from .forecast_utils import load_existing_forecasts, write_existing_forecasts, merge_datasets_on_forecast, merge_holidays
else:
    from constants import DATE_RANGE_VAL, FREQ_DAILY, HORIZON_DAILY, FREQ_MONTHLY, DEFAULT_START_VAL_VIS, DEFAULT_END_VAL_VIS, FUTURE_END_VAL_VIS, DATE_RANGE_VAL_EXTENDED, DATE_RANGE_TEST
    from preprocessing import load_daily_data, load_monthly_data
    from forecast_utils import load_existing_forecasts, write_existing_forecasts, merge_datasets_on_forecast, merge_holidays



#CODE FOR VISUALIZATIONS:

## Visualizations for:
# Best model
# Best model / category
# Best 3 models
# Daily / monthly compare
# Error of best model

def _to_month(prediction_df):
    prediction_df['ds'] = pd.to_datetime(prediction_df['ds']).dt.to_period('M').dt.to_timestamp()
    columns = [c for c in prediction_df.columns if c not in ['unique_id', 'ds']]
    prediction_df = prediction_df.groupby(['unique_id', 'ds'])[columns].mean().reset_index()
    return prediction_df


def _get_ground_truth(frequency: str, start_val: str, end_val: str) -> pd.DataFrame:
    """Load ground truth data based on frequency and split."""
    if frequency == FREQ_DAILY:
        data_train, data_val, data_test = load_daily_data(use_existing=True)
    elif frequency == FREQ_MONTHLY:
        data_train, data_val, data_test = load_monthly_data(use_existing=True)
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")
    
    data = pd.concat([data_train, data_val], axis=0, ignore_index=True)

    return data[(data['ds'] >= start_val) & (data['ds'] <= end_val)]
    
def plot_forecasts(
    models: dict,
    name: str,
    frequency: str = FREQ_MONTHLY,
    start_val: str = DEFAULT_START_VAL_VIS,
    end_val: str = FUTURE_END_VAL_VIS,
    gt_label: str = "Ground Truth",
    figsize=(14, 5)
):
    plt.figure(figsize=figsize)
    gt = _get_ground_truth(frequency, start_val, end_val)

    plt.plot(
        gt["ds"],
        gt["y"],
        label=gt_label,
        color="#1D3557",
        linewidth=2
    )

    # === Forecasts ===
    palette = sns.color_palette("pastel", n_colors=len(models))

    for i, model in enumerate(models.values()):
        if(frequency == FREQ_MONTHLY and model['frequency'] == FREQ_DAILY):
            df = _to_month(model["data"].copy())
        else:
            df = model["data"].copy()
        model_name = model["name"]

        # Optionally restrict future horizon
        df = df[(df["ds"] >= start_val) & (df["ds"] < end_val)]

        # Identify forecast column (only one expected)
        forecast_col = [
            c for c in df.columns if c not in ["unique_id", "ds", "y"]
        ]

        if len(forecast_col) != 1:
            raise ValueError(
                f"Expected exactly one forecast column in {model_name}, "
                f"found {forecast_col}"
            )

        forecast_col = forecast_col[0]

        plt.plot(
            df["ds"],
            df[forecast_col],
            label=f"{i+1}: {model_name}",
            color=palette[i],
            linewidth=2
        )

        # Forecast start marker
        plt.axvline(
            df["ds"].min(),
            color=palette[i],
            linestyle=":",
            alpha=0.4
        )

    # === Formatting ===
    plt.title("Forecast vs Ground Truth", fontsize=16)
    plt.xlabel("Datum", fontsize=12)
    plt.ylabel("Preis", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    plt.savefig(f'../results/visualizations/forecasts/forecast_{name}.png', dpi=300, bbox_inches='tight')
    plt.show()



def plot_residuals(model):
    frequency = model[0]['frequency']
    min_date = model[0]['data']['ds'].min()
    max_date = model[0]['data']['ds'].max()
    gt = _get_ground_truth(frequency, min_date, max_date)
    min_gt = gt['ds'].min()
    max_gt = gt['ds'].max()
    prediction = model[0]['data']
    prediction = prediction[(prediction['ds'] >= min_gt) & (prediction['ds'] <= max_gt)]
    
    model_name = [c for c in prediction.columns if c not in ["unique_id", "ds"]][0]

    print(type(prediction[model_name]), type(gt['y']))
    plt.scatter(prediction[model_name], gt['y'], color='blue', alpha=0.6)
    min_val = min(prediction[model_name].min(), gt['y'].min())*0.9
    max_val = max(prediction[model_name].max(), gt['y'].max())*1.05


    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    plt.title('Predicted vs Actual Values from ' + str(model_name) + 'with Frequency ' + str(frequency))
    plt.legend([model_name])