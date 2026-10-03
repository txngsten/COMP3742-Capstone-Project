"""
Student Names: Oliver Wuttke, Shivansh Pant
Student FANs: WUTT0019, PANT0108
File: helper_functions.py
Date: 22-09-2026
Description: A file containing helpful functions to be used by the DataLoader class.
"""

# Imports
import numpy as np
import pandas as pd

from datetime import timedelta, datetime
from openelectricity import OEClient
from openelectricity.types import DataMetric, MarketMetric

# Network metrics
network_metrics: list[DataMetric] = [
    DataMetric.POWER,
    DataMetric.ENERGY,
    DataMetric.EMISSIONS,
    DataMetric.MARKET_VALUE,
    DataMetric.STORAGE_BATTERY,
]

# Market metrics
market_metrics: list[MarketMetric] = [
    MarketMetric.PRICE,
    MarketMetric.DEMAND,
    MarketMetric.DEMAND_ENERGY,
    MarketMetric.DEMAND_GROSS,
    MarketMetric.DEMAND_GROSS_ENERGY,
    MarketMetric.GENERATION_RENEWABLE,
    MarketMetric.GENERATION_RENEWABLE_ENERGY,
    MarketMetric.GENERATION_RENEWABLE_WITH_STORAGE,
    MarketMetric.GENERATION_RENEWABLE_WITH_STORAGE_ENERGY,
    MarketMetric.RENEWABLE_PROPORTION,
    MarketMetric.RENEWABLE_WITH_STORAGE_PROPORTION,
    MarketMetric.CURTAILMENT,
    MarketMetric.CURTAILMENT_ENERGY,
    MarketMetric.CURTAILMENT_SOLAR_UTILITY,
    MarketMetric.CURTAILMENT_SOLAR_UTILITY_ENERGY,
    MarketMetric.CURTAILMENT_WIND,
    MarketMetric.CURTAILMENT_WIND_ENERGY,
    MarketMetric.FLOW_EXPORTS,
    MarketMetric.FLOW_EXPORTS_ENERGY,
    MarketMetric.FLOW_IMPORTS,
    MarketMetric.FLOW_IMPORTS_ENERGY
]

# Columns that duplicate another column or are mostly missing.
redundant_columns: list[str] = [
    'energy',
    'demand_energy',
    'demand_gross_energy',
    'generation_renewable_energy',
    'generation_renewable_with_storage_energy',
    'curtailment_energy',
    'curtailment_solar_utility_energy',
    'curtailment_wind_energy',
    'flow_exports_energy',
    'flow_imports_energy',
    'market_value',
    'storage_battery',
]

def fetch_in_chunks_market(
        start_date: datetime,
        end_date: datetime,
        interval: str,
        max_days: int,
        api_key: str
) -> pd.DataFrame:
    """
    Fetch market data in chunks respecting API range limits.
    Author: Oliver Wuttke - WUTT0019

    Args:
        start_date: Start of query range
        end_date: End of query range
        interval: Time interval for query
        max_days: Max days allowed for interval type in single call
        api_key: OpenElectricity API key

    Returns:
        A dataframe object containing all metrics indexed by timestamp.
    """
    client = OEClient(api_key)

    all_data = []
    current_start = start_date

    while current_start < end_date:
        # Compute current chunk end date
        current_end = min(
            current_start + timedelta(days=max_days),
            end_date
        )

        try:
            # Make api call using the client SDK
            response = client.get_market(
                network_code='NEM',
                metrics=market_metrics,
                interval=interval,
                date_start=current_start,
                date_end=current_end,
                network_region='NSW1'
            )

            # Unpack response data and append it to data list
            for timeseries in response.data:
                for result in timeseries.results:
                    for data_point in result.data:
                        all_data.append({
                            "timestamp": data_point.timestamp,
                            "metric": timeseries.metric,
                            "value": data_point.value,
                            "unit": timeseries.unit
                        })

        except Exception as e:
            if "Date range is too large" in str(e):
                print(f"Date range error {e}")
                break
            raise

        # Reuse the boundary so no hourly intervals are skipped.
        current_start = current_end

    # Convert to dataframe, indexed by timestamp
    df = (
        pd.DataFrame(all_data)
        # Inclusive request boundaries can return the same observation twice.
        # Keep the later response explicitly instead of averaging duplicates.
        .drop_duplicates(subset=["timestamp", "metric"], keep="last")
        .pivot(index="timestamp", columns="metric", values="value")
        .reset_index()
    )
    df.columns.name = None

    return df


def fetch_in_chunks_network(
        start_date: datetime,
        end_date: datetime,
        interval: str,
        max_days: int,
        api_key: str
) -> pd.DataFrame:
    """
    Fetch network data in chunks respecting API range limits.
    Author: Oliver Wuttke - WUTT0019

    Args:
        start_date: Start of query range
        end_date: End of query range
        interval: Time interval for query
        max_days: Max days allowed for interval type in single call
        api_key: OpenElectricity API key

    Returns:
        A dataframe object containing all metrics indexed by timestamp.
    """
    client = OEClient(api_key)

    all_data = []
    current_start = start_date

    while current_start < end_date:
        # Compute current chunk end date
        current_end = min(
            current_start + timedelta(days=max_days),
            end_date
        )

        try:
            # Make api call using the client SDK
            response = client.get_network_data(
                network_code='NEM',
                metrics=network_metrics,
                interval=interval,
                date_start=current_start,
                date_end=current_end,
                network_region='NSW1'
            )

            # Unpack response data and append it to data list
            for timeseries in response.data:
                for result in timeseries.results:
                    for data_point in result.data:
                        all_data.append({
                            "timestamp": data_point.timestamp,
                            "metric": timeseries.metric,
                            "value": data_point.value,
                            "unit": timeseries.unit
                        })

        except Exception as e:
            if "Date range is too large" in str(e):
                print(f"Date range error {e}")
                break
            raise

        # Reuse the boundary so no hourly intervals are skipped.
        current_start = current_end

    # Convert to dataframe, indexed by timestamp
    df = (
        pd.DataFrame(all_data)
        # Inclusive request boundaries can return the same observation twice.
        # Keep the later response explicitly instead of averaging duplicates.
        .drop_duplicates(subset=["timestamp", "metric"], keep="last")
        .pivot(index="timestamp", columns="metric", values="value")
        .reset_index()
    )
    df.columns.name = None

    return df

def trim_to_curtailment_coverage(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Trims the dataset to the first day with real curtailment data.
    Author: Shivansh Pant - PANT0108

    Before source coverage, curtailment is recorded as 0, not missing;
    these zeros aren't real observations.

    Args:
        data: Cleaned electricity dataframe sorted by timestamp

    Returns:
        A tuple of the trimmed dataframe and the timestamp it starts from.

    Raises:
        ValueError: If curtailment has no nonzero values to trim from
    """
    # Find the first hour with nonzero curtailment.
    nonzero = data['curtailment'].ne(0) & data['curtailment'].notna()
    if not nonzero.any():
        raise ValueError('curtailment has no nonzero values; cannot find coverage start.')
    first_nonzero = data.loc[nonzero, 'timestamp'].min()

    # Start at midnight that day so every kept day is complete.
    coverage_start = first_nonzero.normalize()

    # Keep only rows from the coverage start onwards.
    trimmed = data[data['timestamp'] >= coverage_start].reset_index(drop=True)
    return trimmed, coverage_start


def drop_redundant_columns(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Drops columns that repeat information in another column.
    Author: Shivansh Pant - PANT0108

    At a 1h interval, each energy column is its power column times one hour,
    so they're near-perfect duplicates. market_value is energy times price,
    and storage_battery is mostly missing even after the trim.

    Args:
        data: Electricity dataframe to drop columns from

    Returns:
        A tuple of the reduced dataframe and the list of columns dropped.
    """
    # Only drop columns that are actually in the data.
    dropped = [column for column in redundant_columns if column in data.columns]
    return data.drop(columns=dropped), dropped


def fill_short_gaps(
        data: pd.DataFrame,
        max_gap_hours: int
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """
    Fills short gaps in each column by interpolating over time.
    Author: Shivansh Pant - PANT0108

    Leave longer gaps missing; a straight line across them would be guesswork.

    Args:
        data: Hourly electricity dataframe sorted by timestamp
        max_gap_hours: Longest run of missing hours that will be filled

    Returns:
        A tuple of the filled dataframe with a was_imputed column, the number
        of values filled per column, and the number left missing per column.
    """
    filled = data.copy()
    imputed = pd.Series(False, index=data.index)
    filled_counts = {}
    unfilled_counts = {}
    by_time = data.set_index('timestamp')

    for column in data.columns.drop('timestamp'):
        missing = data[column].isna()
        if not missing.any():
            continue

        # Label each run of missing values and measure its length.
        run_id = missing.ne(missing.shift()).cumsum()
        run_length = missing.groupby(run_id).transform('sum')

        # Interpolate between the values on either side of each gap.
        interpolated = by_time[column].interpolate(method='time', limit_area='inside').to_numpy()

        # Fill only short gaps with real values on both sides.
        fill = missing & run_length.le(max_gap_hours) & ~np.isnan(interpolated)
        filled.loc[fill, column] = interpolated[fill]

        imputed |= fill
        filled_counts[column] = int(fill.sum())
        unfilled_counts[column] = int((missing & ~fill).sum())

    # Flag each row where a value was filled.
    filled['was_imputed'] = imputed.astype(int)
    return filled, filled_counts, unfilled_counts


def flag_negative_curtailment(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Flags hours with any negative curtailment value.
    Author: Shivansh Pant - PANT0108

    Keep negative curtailment values from AEMO's forecast vs dispatch
    calculation, but mark them as suspect.

    Args:
        data: Electricity dataframe with curtailment columns

    Returns:
        A tuple of the dataframe with a curtailment_negative column and the
        number of negative values per curtailment column.
    """
    curtailment_columns = [c for c in data.columns if c.startswith('curtailment')]
    negative = data[curtailment_columns].lt(0)

    flagged = data.copy()
    flagged['curtailment_negative'] = negative.any(axis=1).astype(int)
    return flagged, {c: int(negative[c].sum()) for c in curtailment_columns}


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
