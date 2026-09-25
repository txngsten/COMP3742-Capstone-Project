"""
Student Names: Oliver Wuttke,
Student FANs: WUTT0019,
File: run_pipeline.py
Date: 22-09-2026
Description:
"""

# Imports
import os

from DataLoader import DataLoader
from dotenv import load_dotenv
from datetime import datetime
from openelectricity.types import DataMetric, MarketMetric

# Loads in environment variables
load_dotenv()

# Pipeline constants
API_KEY: str = os.getenv("OPENELECTRICITY_API_KEY")
START_DATE: datetime = datetime(2000, 1, 1)
END_DATE: datetime = datetime.now()


def main():
    etl = DataLoader(START_DATE, END_DATE, API_KEY)
    etl.fetch()

    # Access underlying dataframe object
    df = etl.data_set

    print(df.head())
    print(df.describe())

if __name__ == "__main__":
    main()
