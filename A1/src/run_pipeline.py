"""
Student Names: Oliver Wuttke, Hans Pujalte, Shivansh Pant
Student FANs: WUTT0019, PUJA0009, PANT0108
File: run_pipeline.py
Date: 22-09-2026
Description:
"""

# Imports
import os
import json

from DataLoader import DataLoader
from dotenv import load_dotenv
from datetime import datetime
from openelectricity.types import DataMetric, MarketMetric
from pathlib import Path

# Resolve paths from this file, independently of the launch directory.
PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / "src" / ".env")

# Pipeline constants
API_KEY: str | None = os.getenv("OPENELECTRICITY_API_KEY")
START_DATE: datetime = datetime(2000, 1, 1)
END_DATE: datetime = datetime(2026, 9, 24)

# Directory
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
RAW_OUTPUT_PATH = RAW_DATA_DIR / "electricity_raw.parquet"

CLEAN_DATA_DIR = PROJECT_DIR / "data" / "processed"
CLEAN_OUTPUT_PATH = CLEAN_DATA_DIR / "electricity_cleaned.parquet"
TRANSFORMED_OUTPUT_PATH = CLEAN_DATA_DIR / "electricity_transformed.parquet"
REPORTS_DIR = PROJECT_DIR / "reports"


def main():
    etl = DataLoader(START_DATE, END_DATE, API_KEY)
    etl.fetch()

    # Access and/or assign underlying dataframe object
    df = etl.data_set
    print(df.head())
    print(df.describe())

    # Save to parquet
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_OUTPUT_PATH, index=False, engine="pyarrow")
    print(f"Saved {len(df):,} rows to {RAW_OUTPUT_PATH}")

    # Clean data
    df_clean = etl.clean()

    # Save cleaned df as parquet
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(CLEAN_OUTPUT_PATH, index=False, engine="pyarrow")
    print(f"Saved {len(df_clean):,} cleaned rows to {CLEAN_OUTPUT_PATH}")

    # Store a report of what was done during cleaning.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "cleaning_report.json").write_text(
        json.dumps(etl.cleaning_report, indent=2) + "\n"
    )

    # Transform cleaned data into features
    df_transformed = etl.transform()

    # Save transformed df as parquet
    df_transformed.to_parquet(TRANSFORMED_OUTPUT_PATH, index=False, engine="pyarrow")
    print(f"Saved {len(df_transformed):,} transformed rows to {TRANSFORMED_OUTPUT_PATH}")

    # Store a report of what the transform did.
    (REPORTS_DIR / "transform_report.json").write_text(
        json.dumps(etl.transform_report, indent=2) + "\n"
    )

if __name__ == "__main__":
    main()
