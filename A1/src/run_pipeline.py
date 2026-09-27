"""
Student Names: Oliver Wuttke, Hans Pujalte
Student FANs: WUTT0019, PUJA0009
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
from pathlib import Path

# Loads in environment variables
load_dotenv()

# Pipeline constants
API_KEY: str | None = os.getenv("OPENELECTRICITY_API_KEY")
START_DATE: datetime = datetime(2000, 1, 1)
END_DATE: datetime = datetime(2026, 9, 24)

# Directory
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
RAW_OUTPUT_PATH = RAW_DATA_DIR / "electricity_raw.parquet"

def main():
    etl = DataLoader(START_DATE, END_DATE, API_KEY)
    etl.fetch()

    # Access and/or assign underlying dataframe object
    df = etl.data_set
    print(df.head())
    print(df.describe())

    # Save to parquet
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_OUTPUT_PATH, index=False)
    print(f"Saved {len(df):,} rows to {RAW_OUTPUT_PATH}")

    # Run data cleaning

    # Return processed data

if __name__ == "__main__":
    main()
