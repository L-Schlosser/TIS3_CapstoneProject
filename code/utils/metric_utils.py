import pandas as pd
from typing import List

if __package__:
    from .constants import FREQ_DAILY, FREQ_MONTHLY, FAMILY_BASELINE, FAMILY_STATISTICAL, FAMILY_MACHINE_LEARNING, FAMILY_DEEP_LEARNING, SPLIT_VAL, SPLIT_TEST
    from .preprocessing import load_daily_data, load_monthly_data
    from .metrics import calculate_metrics
    from .baseline import run_baseline_forecast_daily, run_baseline_forecast_monthly
    from .statistical import run_statistical_forecast_daily, run_statistical_forecast_monthly
    from .machine_learning import run_machine_learning_forecast_daily, run_machine_learning_forecast_monthly
    from .deep_learning import run_deep_learning_forecast_daily, run_deep_learning_forecast_monthly
else:
    from constants import FREQ_DAILY, FREQ_MONTHLY, FAMILY_BASELINE, FAMILY_STATISTICAL, FAMILY_MACHINE_LEARNING, FAMILY_DEEP_LEARNING, SPLIT_VAL, SPLIT_TEST
    from preprocessing import load_daily_data, load_monthly_data
    from metrics import calculate_metrics
    from baseline import run_baseline_forecast_daily, run_baseline_forecast_monthly
    from statistical import run_statistical_forecast_daily, run_statistical_forecast_monthly
    from machine_learning import run_machine_learning_forecast_daily, run_machine_learning_forecast_monthly
    from deep_learning import run_deep_learning_forecast_daily, run_deep_learning_forecast_monthly

def load_overall_metrics() -> pd.DataFrame:
    base_metrics = pd.concat([
        calculate_metrics(None, FAMILY_BASELINE, FREQ_DAILY, SPLIT_VAL, use_existing=True),
        calculate_metrics(None, FAMILY_BASELINE, FREQ_MONTHLY, SPLIT_VAL, use_existing=True)
    ], ignore_index=True)

    stat_metrics = pd.concat([
        calculate_metrics(None, FAMILY_STATISTICAL, FREQ_DAILY, SPLIT_VAL, use_existing=True),
        calculate_metrics(None, FAMILY_STATISTICAL, FREQ_MONTHLY, SPLIT_VAL, use_existing=True)
    ], ignore_index=True)

    ml_metrics = pd.concat([
        calculate_metrics(None, FAMILY_MACHINE_LEARNING, FREQ_DAILY, SPLIT_VAL, use_existing=True),
        calculate_metrics(None, FAMILY_MACHINE_LEARNING, FREQ_MONTHLY, SPLIT_VAL, use_existing=True)
    ], ignore_index=True)

    dl_metrics = pd.concat([
        calculate_metrics(None, FAMILY_DEEP_LEARNING, FREQ_DAILY, SPLIT_VAL, use_existing=True),
        calculate_metrics(None, FAMILY_DEEP_LEARNING, FREQ_MONTHLY, SPLIT_VAL, use_existing=True)
    ], ignore_index=True)

    return pd.concat([base_metrics, stat_metrics, ml_metrics, dl_metrics], ignore_index=True).sort_values(by=["MAPE"]).reset_index(drop=True)

def merge_prediction_dfs(
        base_df_val: pd.DataFrame, base_df_test: pd.DataFrame,
        stat_df_val: pd.DataFrame, stat_df_test: pd.DataFrame,
        ml_df_val: pd.DataFrame, ml_df_test: pd.DataFrame,
        dl_df_val: pd.DataFrame, dl_df_test: pd.DataFrame
    ) -> pd.DataFrame:
    """Merge prediction dataframes from different model families into a single dataframe."""

    base_df_val["Split"] = SPLIT_VAL
    base_df_test["Split"] = SPLIT_TEST
    base_df = pd.concat([base_df_val.infer_objects(copy=False).fillna(-1), base_df_test.infer_objects(copy=False).fillna(-1)], ignore_index=True)

    stat_df_val["Split"] = SPLIT_VAL
    stat_df_test["Split"] = SPLIT_TEST
    stat_df = pd.concat([stat_df_val.infer_objects(copy=False).fillna(-1), stat_df_test.infer_objects(copy=False).fillna(-1)], ignore_index=True)
    ml_df_val["Split"] = SPLIT_VAL
    ml_df_test["Split"] = SPLIT_TEST
    ml_df = pd.concat([ml_df_val.infer_objects(copy=False).fillna(-1), ml_df_test.infer_objects(copy=False).fillna(-1)], ignore_index=True)

    dl_df_val["Split"] = SPLIT_VAL
    dl_df_test["Split"] = SPLIT_TEST
    dl_df = pd.concat([dl_df_val.infer_objects(copy=False).fillna(-1), dl_df_test.infer_objects(copy=False).fillna(-1)], ignore_index=True)
    merged_df = base_df.merge(
        stat_df,
        on=["unique_id", "ds", "y", "Split"],
        how="left",
    ).merge(
        ml_df,
        on=["unique_id", "ds", "y", "Split"],
        how="left",
    ).merge(
        dl_df,
        on=["unique_id", "ds", "y", "Split"],
        how="left",
    )
    return merged_df

def find_n_best_models(metric_df: pd.DataFrame, n: int, daily_forecasts: pd.DataFrame, monthly_forecasts: pd.DataFrame, metric: str = "MAPE") -> List[str]:
    sorted_df = metric_df.sort_values(by=metric)
    best_models = { }

    for i in range(n):
        if sorted_df.iloc[i]["Frequency"] == FREQ_DAILY:
            data = daily_forecasts
        else:
            data = monthly_forecasts

        data = data[["unique_id", "ds", metric_df.iloc[i]["Model"]]]
        best_models[i] = { "name": metric_df.iloc[i]["Model"], "frequency": metric_df.iloc[i]["Frequency"], "data": data }

    return best_models

if __name__ == "__main__":
    overall_metrics = load_overall_metrics()    
    daily_train, daily_val, daily_test = load_daily_data(use_existing=True)
    monthly_train, monthly_val, monthly_test = load_monthly_data(use_existing=True)

    base_daily_val, base_daily_test = run_baseline_forecast_daily(daily_train, daily_val, daily_test, use_existing=True)
    base_monthly_val, base_monthly_test = run_baseline_forecast_monthly(monthly_train, monthly_val, monthly_test, use_existing=True)

    stat_daily_val, stat_daily_test = run_statistical_forecast_daily(daily_train, daily_val, daily_test, use_existing=True)
    stat_monthly_val, stat_monthly_test = run_statistical_forecast_monthly(monthly_train, monthly_val, monthly_test, use_existing=True)

    ml_daily_val, ml_daily_test = run_machine_learning_forecast_daily(daily_train, daily_val, daily_test, use_existing=True)
    ml_monthly_val, ml_monthly_test = run_machine_learning_forecast_monthly(monthly_train, monthly_val, monthly_test, use_existing=True)

    dl_daily_val, dl_daily_test = run_deep_learning_forecast_daily(daily_train, daily_val, daily_test, use_existing=True)
    dl_monthly_val, dl_monthly_test = run_deep_learning_forecast_monthly(monthly_train, monthly_val, monthly_test, use_existing=True)   

    merged_daily = merge_prediction_dfs(base_daily_val, base_daily_test, stat_daily_val, stat_daily_test, ml_daily_val, ml_daily_test, dl_daily_val, dl_daily_test)
    merged_monthly = merge_prediction_dfs(base_monthly_val, base_monthly_test, stat_monthly_val, stat_monthly_test, ml_monthly_val, ml_monthly_test, dl_monthly_val, dl_monthly_test)

    find_n_best_models(overall_metrics, 3, merged_daily, merged_monthly)