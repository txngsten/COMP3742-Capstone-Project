"""
Student Names: Oliver Wuttke,
Student FANs: WUTT0019,
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
            current_start + timedelta(days=max_days + 1),
            end_date
        )

        try:
            # Make api call using the client SDK
            response = client.get_market(
                network_code='NEM',
                metrics=market_metrics,
                interval=interval,
                date_start=current_start,
                date_end=current_end
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

        # Move to next chunk, add 1 to avoid overlap
        current_start = current_end + timedelta(days=1)

    # Convert to dataframe, indexed by timestamp
    df = (
        pd.DataFrame(all_data)
        .pivot_table(index=["timestamp"], columns="metric", values="value")
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
            current_start + timedelta(days=max_days + 1),
            end_date
        )

        try:
            # Make api call using the client SDK
            response = client.get_network_data(
                network_code='NEM',
                metrics=network_metrics,
                interval=interval,
                date_start=current_start,
                date_end=current_end
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

        # Move to next chunk, add 1 to avoid overlap
        current_start = current_end + timedelta(days=1)

    # Convert to dataframe, indexed by timestamp
    df = (
        pd.DataFrame(all_data)
        .pivot_table(index=["timestamp"], columns="metric", values="value")
        .reset_index()
    )
    df.columns.name = None

    return df