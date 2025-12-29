def find_n_best_models(metric_df: pd.DataFrame, n: int) -> List[str]:
    """Find the names of the n best models based on the 'value' column in the metric dataframe.

    Args:
        metric_df (pd.DataFrame): DataFrame containing model metrics with columns 'model' and 'value'.
        n (int): Number of top models to return.

    Returns:
        List[str]: List of model names corresponding to the n best models.
    """
    sorted_df = metric_df.sort_values(by='value')
    best_models = sorted_df['model'].head(n).tolist()
    return best_models