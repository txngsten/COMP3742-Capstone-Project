"""
Student Names: Oliver Wuttke, Shivansh Pant
Student FANs: WUTT0019, PANT0108
File: helper_functions.py
Date: 22-09-2026
Description: A file containing helpful functions to be used by the DataLoader class.
"""

# Imports
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
