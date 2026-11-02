"""
Student Names: Oliver Wuttke, Hans Pujalte, Shivansh Pant, Matilda Alford
Student FANs: WUTT0019, PUJA0009, PANT0108, ALFO0043
File: fetch_instance.py
Date: 02-11-2026
Description: Fetches latest instance of OpenElectricity data for inference.
"""

# Imports
import os
import json
import pandas as pd

from A2.A1_data_pipeline.DataLoader import DataLoader
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pathlib import Path

# Resolve paths from this file, independently of the launch directory.
PROJECT_DIR: Path = Path(__file__).resolve().parents[1]
load_dotenv()

# Pipeline constants
API_KEY: str | None = os.getenv("OPENELECTRICITY_API_KEY")
START_DATE: datetime = datetime.now() - timedelta(hours=2)
END_DATE: datetime = datetime.now()

# Directory
RAW_DATA_DIR: Path = PROJECT_DIR / "data/instance" / "raw"
RAW_OUTPUT_PATH: Path = RAW_DATA_DIR / "electricity_raw.parquet"

CLEAN_DATA_DIR = PROJECT_DIR / "data" / "processed"
CLEAN_OUTPUT_PATH = CLEAN_DATA_DIR / "electricity_cleaned.parquet"
TRANSFORMED_OUTPUT_PATH = CLEAN_DATA_DIR / "electricity_instance.parquet"


def main():
    etl: DataLoader = DataLoader(START_DATE, END_DATE, API_KEY)
    etl.fetch()

    # Access and/or assign underlying dataframe object
    df: pd.DataFrame = etl.data_set
    print(df.head())
    print(df.describe())

    # Clean data
    df_clean: pd.DataFrame = etl.clean()
    print(df_clean.head())
    print(df_clean.describe())

    # Transform cleaned data into features
    df_transformed: pd.DataFrame = etl.transform()
    print(df_transformed.head())
    print(df_transformed.describe())

    # Save transformed df as parquet
    df_transformed.to_parquet(TRANSFORMED_OUTPUT_PATH, index=False, engine="pyarrow")
    print(f"Saved {len(df_transformed)} transformed rows and {len(df_transformed.columns)} columns to {TRANSFORMED_OUTPUT_PATH}")



if __name__ == "__main__":
    main()
