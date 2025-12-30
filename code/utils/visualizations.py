import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd

if __package__:
    from .constants import FREQ_DAILY, FREQ_MONTHLY, DEFAULT_START_VAL_DATASET, DEFAULT_START_VAL_VIS, FUTURE_END_VAL_VIS, VISUALIZATION_DIR
    from .visualization_utils import daily_to_monthly, get_ground_truth
else:
    from constants import FREQ_DAILY, FREQ_MONTHLY, DEFAULT_START_VAL_DATASET, DEFAULT_START_VAL_VIS, FUTURE_END_VAL_VIS, VISUALIZATION_DIR
    from visualization_utils import daily_to_monthly, get_ground_truth

def plot_ground_truth(
    frequency: str = FREQ_MONTHLY,
    start_val: str = DEFAULT_START_VAL_DATASET,
    end_val: str = FUTURE_END_VAL_VIS,
    figsize=(14, 5)
):
    plt.figure(figsize=figsize)
    gt = get_ground_truth(frequency, start_val, end_val)

    plt.plot(gt['ds'], gt['y'], linewidth=0.8, color='#2E86AB')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (EUR/MWh)', fontsize=12)
    plt.title('Austrian Electricity Prices (2015-2025)', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, f'ground_truth/ground_truth_{frequency}.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_avg_ground_truth(
    start_val: str = DEFAULT_START_VAL_DATASET,
    end_val: str = FUTURE_END_VAL_VIS,
    figsize=(14, 5)
):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    gt = get_ground_truth(FREQ_MONTHLY, start_val, end_val)

    gt['year'] = gt['ds'].dt.year
    gt['month'] = gt['ds'].dt.month
    
    monthly_avg = gt.groupby('month')["y"].mean()
    axes[0].bar(monthly_avg.index, monthly_avg.values, color='#2E86AB', alpha=0.7)
    axes[0].set_xlabel('Month', fontsize=12)
    axes[0].set_ylabel('Average Price (EUR/MWh)', fontsize=12)
    axes[0].set_title('Average Price by Month', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(1, 13))
    axes[0].grid(alpha=0.3, axis='y')

    yearly_avg = gt.groupby('year')['y'].mean()
    axes[1].bar(yearly_avg.index, yearly_avg.values, color='#2E86AB', alpha=0.7)
    axes[1].set_xlabel('Year', fontsize=12)
    axes[1].set_ylabel('Average Price (EUR/MWh)', fontsize=12)
    axes[1].set_title('Average Price by Year', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, f'ground_truth/avg_ground_truth.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_ground_truth_distribution(
    start_val: str = DEFAULT_START_VAL_DATASET,
    end_val: str = FUTURE_END_VAL_VIS,
    figsize=(14, 5)
):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    gt = get_ground_truth(FREQ_MONTHLY, start_val, end_val)

    axes[0].hist(gt['y'], bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Price (EUR/MWh)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Price Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)

    axes[1].boxplot(gt['y'], vert=True, patch_artist=True, 
                    boxprops=dict(facecolor='#2E86AB', alpha=0.7))
    axes[1].set_ylabel('Price (EUR/MWh)', fontsize=12)
    axes[1].set_title('Price Box Plot', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, f'ground_truth/distribution_ground_truth.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_ground_truth_heatmap(
    start_val: str = DEFAULT_START_VAL_DATASET,
    end_val: str = FUTURE_END_VAL_VIS,
    figsize=(10, 8)
):
    plt.figure(figsize=figsize)
    gt = get_ground_truth(FREQ_MONTHLY, start_val, end_val)

    gt['year'] = gt['ds'].dt.year
    gt['month'] = gt['ds'].dt.month
    pivot_data = gt.pivot_table(values='y', index='month', columns='year', aggfunc='mean')

    im = plt.imshow(pivot_data.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(im, label='Average Price (EUR/MWh)')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Month', fontsize=12)
    plt.title('Average Price Heatmap (Month vs Year)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(pivot_data.columns)), pivot_data.columns, rotation=45)
    plt.yticks(range(len(pivot_data.index)), pivot_data.index)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, f'ground_truth/heatmap_ground_truth.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
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
    gt = get_ground_truth(frequency, start_val, end_val)

    plt.plot(
        gt["ds"],
        gt["y"],
        label=gt_label,
        color="#1D3557",
        linewidth=2
    )

    palette = sns.color_palette("pastel", n_colors=len(models))

    for i, model in enumerate(models.values()):
        if(frequency == FREQ_MONTHLY and model['frequency'] == FREQ_DAILY):
            df = daily_to_monthly(model["data"].copy())
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
            label=f"{i+1}: {model_name} (freq: {model['frequency']}, family: {model['family']})",
            color=palette[i],
            linewidth=2
        )

        plt.axvline(
            df["ds"].min(),
            color=palette[i],
            linestyle=":",
            alpha=0.4
        )

    plt.title("Forecast vs Ground Truth", fontsize=16)
    plt.xlabel("Datum", fontsize=12)
    plt.ylabel("Preis", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    plt.savefig(os.path.join(VISUALIZATION_DIR, f'forecasts/forecast_{name}.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_residuals(model):
    frequency = model[0]['frequency']
    min_date = model[0]['data']['ds'].min()
    max_date = model[0]['data']['ds'].max()
    gt = get_ground_truth(frequency, min_date, max_date)
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
    plt.title("Predicted vs Actual Values")
    plt.legend([f"{model_name} (freq: {model[0]['frequency']}, family: {model[0]['family']})"])
    plt.savefig(os.path.join(VISUALIZATION_DIR, f'forecasts/residuals_{model_name}_{frequency}.png'), dpi=300, bbox_inches='tight')