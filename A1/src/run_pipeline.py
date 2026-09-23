"""
Student Names: Oliver Wuttke,
Student FANs: WUTT0019,
File: run_pipeline.py
Date: 22-09-2026
Description:
"""

# Imports
import os

import pandas as pd

from DataLoader import DataLoader
from dotenv import load_dotenv
from datetime import datetime
from openelectricity.types import DataMetric, MarketMetric

# Loads in environment variables
load_dotenv()

# Pipeline constants
API_KEY = os.getenv("OPENELECTRICITY_API_KEY")
START_DATE = datetime(1999, 1, 1)
END_DATE = datetime.now()
GEN_URL = "https://api.openelectricity.org.au/v4/data/network/NEM"
MARKET_URL = "https://api.openelectricity.org.au/v4/market/network/NEM"

# Network metrics
network_metrics = [
    DataMetric.POWER,
    DataMetric.ENERGY,
    DataMetric.EMISSIONS,
    DataMetric.MARKET_VALUE,
    DataMetric.STORAGE_BATTERY,
]

# Market metrics
market_metrics = [
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

def main():
    etl = DataLoader(START_DATE, END_DATE, API_KEY)

if __name__ == "__main__":
    main()
