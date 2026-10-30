"""
Student Name: Matilda Alford
Student FAN: ALFO0043
File: features.py
Date: 30-10-2026
Description: Prepares historical renewable generation for 24-hour forecasting.

Reuse note: add_time_features() and add_lag_features() are copied unchanged
from A1/src/helper_functions.py, written by Shivansh Pant - PANT0108.
The remaining preparation code is for Matilda's A2 renewable subsystem.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[3]
CLEAN_DATA_PATH = PROJECT_DIR / 'A1/data/processed/electricity_cleaned.parquet'
TARGET = 'generation_renewable'
FORECAST_HOURS = 24
LAG_HOURS = [1, 24, 168]
ROLLING_HOURS = 24


def load_renewable_data(path: Path = CLEAN_DATA_PATH) -> pd.DataFrame:
    """Load cleaned hourly data without applying the A1 curtailment cutoff.

    Author: Matilda Alford - ALFO0043
    """
    data = pd.read_parquet(path, columns=['timestamp', TARGET])
    data = data.sort_values('timestamp').reset_index(drop=True)
    if data.empty:
        raise ValueError('The renewable dataset is empty.')
    timestamps = data['timestamp']
    if not isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError('Timestamps must be timezone-aware.')
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError('Missing or duplicate timestamps require investigation.')
    if not timestamps.diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError('The dataset must have a continuous hourly timeline.')
    if not pd.api.types.is_numeric_dtype(data[TARGET]):
        raise ValueError('Renewable generation must be numeric.')
    if np.isinf(data[TARGET]).any():
        raise ValueError('Renewable generation contains infinite values.')
    return data


def coverage_by_year(data: pd.DataFrame) -> pd.DataFrame:
    """Summarise coverage and generation changes before choosing a date window.

    Author: Matilda Alford - ALFO0043
    """
    summary = data.groupby(data['timestamp'].dt.year)[TARGET].agg(
        hours='size', available_hours='count', mean_mw='mean', max_mw='max'
    )
    summary['missing_hours'] = summary['hours'] - summary['available_hours']
    summary['missing_percent'] = 100 * summary['missing_hours'] / summary['hours']
    return summary


def add_time_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Adds cyclical time features so 23:00 and 00:00 are close together.
    Author: Shivansh Pant - PANT0108

    Args:
        data: Electricity dataframe with a timezone-aware timestamp column

    Returns:
        A tuple of the dataframe with time features and the list of columns added.
    """
    # Use Sydney local time, since demand follows the local clock, including daylight saving.
    local = data['timestamp'].dt.tz_convert('Australia/Sydney')
    cycles = {
        'hour': (local.dt.hour, 24),
        'day_of_week': (local.dt.dayofweek, 7),
        'month': (local.dt.month - 1, 12),
    }

    # Represent each time unit as a point on a circle using sine and cosine.
    new_columns = {}
    for name, (values, period) in cycles.items():
        angle = 2 * np.pi * values / period
        new_columns[f'{name}_sin'] = np.sin(angle)
        new_columns[f'{name}_cos'] = np.cos(angle)
    new_columns['is_weekend'] = local.dt.dayofweek.ge(5).astype(int)

    features = pd.concat([data, pd.DataFrame(new_columns, index=data.index)], axis=1)
    return features, list(new_columns)

def add_lag_features(
        data: pd.DataFrame,
        columns: list[str],
        lag_hours: list[int],
        rolling_hours: int
) -> tuple[pd.DataFrame, list[str], int]:
    """
    Adds past values and rolling stats so each row includes what happened before it.
    Author: Shivansh Pant - PANT0108

    Use only past values to prevent information leaking in from the future.

    Args:
        data: Hourly electricity dataframe sorted by timestamp with no gaps
        columns: Columns to build lag and rolling features from
        lag_hours: How many hours back each lag looks
        rolling_hours: Length of the rolling window in hours

    Returns:
        A tuple of the dataframe with lag features, the list of columns added,
        and the number of initial rows dropped.
    """
    new_columns = {}
    for column in columns:
        for lag in lag_hours:
            new_columns[f'{column}_lag_{lag}h'] = data[column].shift(lag)

        # Calculate rolling windows over prior hours, excluding this hour.
        past = data[column].shift(1).rolling(rolling_hours)
        new_columns[f'{column}_rolling_mean_{rolling_hours}h'] = past.mean()
        new_columns[f'{column}_rolling_std_{rolling_hours}h'] = past.std()

    features = pd.concat([data, pd.DataFrame(new_columns, index=data.index)], axis=1)

    # Drop initial rows without enough history.
    warmup_rows = max(max(lag_hours), rolling_hours)
    features = features.iloc[warmup_rows:].reset_index(drop=True)
    return features, list(new_columns), warmup_rows


def prepare_forecast_data(
        data: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Create inputs at hour t and targets for hours t+1 through t+24.

    Author: Matilda Alford - ALFO0043

    Assume hour t has finished and its generation reading is available.
    Keep missing values on the hourly timeline while calculating shifts;
    then exclude incomplete examples rather than interpolating future values.
    """
    features, time_columns = add_time_features(data.copy())
    features, lag_columns, _ = add_lag_features(
        features, [TARGET], LAG_HOURS, ROLLING_HOURS
    )
    feature_columns = [TARGET, *time_columns, *lag_columns]
    target_columns = []
    for hour in range(1, FORECAST_HOURS + 1):
        column = f'{TARGET}_next_{hour}h'
        features[column] = features[TARGET].shift(-hour)
        target_columns.append(column)

    # Keep the original timestamps so exclusions do not conceal historical gaps.
    features = features.dropna(subset=feature_columns + target_columns)
    return features.reset_index(drop=True), feature_columns, target_columns


def split_forecast_data(
        data: pd.DataFrame,
        validation_start: pd.Timestamp,
        test_start: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """Split by time and exclude forecasts whose targets cross a boundary.

    Author: Matilda Alford - ALFO0043

    Boundaries must be timezone-aware. They refer to forecast issue times.
    Historical inputs may precede a boundary; forecast targets may not cross it.
    """
    validation_start = pd.Timestamp(validation_start)
    test_start = pd.Timestamp(test_start)
    if validation_start.tzinfo is None or test_start.tzinfo is None:
        raise ValueError('Split boundaries must be timezone-aware.')
    if validation_start >= test_start:
        raise ValueError('Validation must start before the test period.')
    origin = data['timestamp']
    target_end = origin + pd.Timedelta(hours=FORECAST_HOURS)
    splits = {
        'train': data.loc[target_end < validation_start].copy(),
        'validation': data.loc[
            (origin >= validation_start) & (target_end < test_start)
        ].copy(),
        'test': data.loc[origin >= test_start].copy(),
    }
    if any(part.empty for part in splits.values()):
        raise ValueError('Each split must contain complete forecast examples.')
    return splits
