"""
Student Names: Oliver Wuttke,
Student FANs: WUTT0019,
File: DataLoader.py
Date: 22-09-2026
Description:
"""

# Imports
import pandas as pd

from helper_functions import fetch_in_chunks_market, fetch_in_chunks_network

from datetime import datetime, timedelta
from openelectricity import OEClient
from openelectricity.types import DataMetric, MarketMetric

class DataLoader:
    def __init__(self, start_date: datetime, end_date: datetime, api_key: str) -> None:
        """
        Constructor for DataLoader class.
        Author: Oliver Wuttke - WUTT0019

        Args:
            start_date: Start date for fetch range
            end_date: End date for fetch range
            api_key: OpenElectricity API key
        """
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key
        self.data_set = pd.DataFrame()

    def fetch(self) -> None:
        """
        Fetches both market and network data and combines into a single pandas dataframe.
        Author: Oliver Wuttke - WUTT0019
        """
        # Fetches market data
        market_df = fetch_in_chunks_market(
            self.start_date,
            self.end_date,
            "1h",
            30,
            self.api_key
        )

        # Fetches network data
        network_df = fetch_in_chunks_network(
            self.start_date,
            self.end_date,
            "1h",
            30,
            self.api_key
        )

        # Combines both market and network data into a dataframe
        # and assigns it to the underlying dataframe object of the class
        self.data_set = market_df.merge(network_df, on="timestamp", how="outer")




