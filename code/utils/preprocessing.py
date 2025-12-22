import os
import polars as pl
import pandas as pd
from typing import Tuple

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../results")

def _read_original_data() -> pl.DataFrame:
    """Read the original data from a CSV file."""
    daily_df = pl.read_csv(os.path.join(DATA_DIR, "eu_electricity_daily.csv"))
    daily_df = daily_df.with_columns(pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d").alias("Date"))
    daily_df = daily_df.filter(daily_df["ISO3 Code"] == "AUT")
    daily_df = daily_df.select([
        pl.col("Country").alias("unique_id"),
        pl.col("Date").alias("ds"),
        pl.col("Price (EUR/MWhe)").alias("y")
    ]).sort("ds")
    return daily_df

def _read_existing_data(prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read existing preprocessed data from CSV files."""
    train = pd.read_csv(os.path.join(RESULTS_DIR, f"{prefix}_train.csv"), parse_dates=["ds"])
    val = pd.read_csv(os.path.join(RESULTS_DIR, f"{prefix}_val.csv"), parse_dates=["ds"])
    test = pd.read_csv(os.path.join(RESULTS_DIR, f"{prefix}_test.csv"), parse_dates=["ds"])
    return train, val, test

def _write_existing_data(prefix: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Write preprocessed data to CSV files."""
    train.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_train.csv"), index=False)
    val.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_val.csv"), index=False)
    test.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_test.csv"), index=False)

def _split_data(df: pl.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the data into training, validation, and test sets."""
    train_df = df[df["ds"] < pd.to_datetime("2025-01-01")]
    val_df = df[(df["ds"] >= pd.to_datetime("2025-01-01")) & (df["ds"] < pd.to_datetime("2026-01-01"))]
    test_df = pd.DataFrame()
    return train_df, val_df, test_df

def _aggregate_monthly(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.col("ds").dt.truncate("1mo").alias("ds"))
    return df.group_by(["unique_id", "ds"]).agg(pl.col("y").mean().alias("y")).sort("ds")

def load_daily_data(use_existing: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load or create daily data from a CSV file."""
    if use_existing:
        return _read_existing_data("daily")
    
    train_df, val_df, test_df = _split_data(_read_original_data().to_pandas())
    _write_existing_data("daily", train_df, val_df, test_df)
    return train_df, val_df, test_df

def load_monthly_data(use_existing: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load or create monthly data splits."""
    if use_existing:
        return _read_existing_data("monthly")
    
    monthly_df = _aggregate_monthly(_read_original_data()).to_pandas()
    train_df, val_df, test_df = _split_data(monthly_df)
    _write_existing_data("monthly", train_df, val_df, test_df)
    return train_df, val_df, test_df