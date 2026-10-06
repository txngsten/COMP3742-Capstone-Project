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

import pandas as pd

from DataLoader import DataLoader
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

# Resolve paths from this file, independently of the launch directory.
PROJECT_DIR: Path = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / "src" / ".env")

# Pipeline constants
API_KEY: str | None = os.getenv("OPENELECTRICITY_API_KEY")
START_DATE: datetime = datetime(2000, 1, 1)
END_DATE: datetime = datetime(2026, 9, 24)

# Directory
RAW_DATA_DIR: Path = PROJECT_DIR / "data" / "raw"
RAW_OUTPUT_PATH: Path = RAW_DATA_DIR / "electricity_raw.parquet"

CLEAN_DATA_DIR = PROJECT_DIR / "data" / "processed"
CLEAN_OUTPUT_PATH = CLEAN_DATA_DIR / "electricity_cleaned.parquet"
TRANSFORMED_OUTPUT_PATH = CLEAN_DATA_DIR / "electricity_transformed.parquet"
REPORTS_DIR = PROJECT_DIR / "reports"
MODELLED_OUTPUT_PATH = CLEAN_DATA_DIR / "electricity_modelled.parquet"


def main():
    etl: DataLoader = DataLoader(START_DATE, END_DATE, API_KEY)
    etl.fetch()

    # Access and/or assign underlying dataframe object
    df: pd.DataFrame = etl.data_set
    print(df.head())
    print(df.describe())

    # Save to parquet
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_OUTPUT_PATH, index=False, engine="pyarrow")
    print(f"Saved {len(df)} rows and {len(df.columns)} columns to {RAW_OUTPUT_PATH}")

    # Clean data
    df_clean: pd.DataFrame = etl.clean()
    print(df_clean.head())
    print(df_clean.describe())

    # Save cleaned df as parquet
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(CLEAN_OUTPUT_PATH, index=False, engine="pyarrow")
    print(f"Saved {len(df_clean)} cleaned rows and {len(df_clean.columns)} columns to {CLEAN_OUTPUT_PATH}")

    # Store a report of what was done during cleaning.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "cleaning_report.json").write_text(
        json.dumps(etl.cleaning_report, indent=2) + "\n"
    )

    # Transform cleaned data into features
    df_transformed: pd.DataFrame = etl.transform()
    print(df_transformed.head())
    print(df_transformed.describe())

    # Save transformed df as parquet
    df_transformed.to_parquet(TRANSFORMED_OUTPUT_PATH, index=False, engine="pyarrow")
    print(f"Saved {len(df_transformed)} transformed rows and {len(df_transformed.columns)} columns to {TRANSFORMED_OUTPUT_PATH}")

    # Store a report of what the transform did.
    (REPORTS_DIR / "transform_report.json").write_text(
        json.dumps(etl.transform_report, indent=2) + "\n"
    )

    # Perform exploratory modelling
    df_modelled: pd.DataFrame = etl.exploratory_modelling()

    # Save modelled df as parquet
    df_modelled.to_parquet(MODELLED_OUTPUT_PATH, index=False, engine="pyarrow")
    print(f"Saved {len(df_modelled)} modelled rows and {len(df_modelled.columns)} columns to {MODELLED_OUTPUT_PATH}")

    # Store a report of what the modelling did.
    (REPORTS_DIR / "modelling_report.json").write_text(
        json.dumps(etl.modelling_report, indent=2) + "\n"
    )

if __name__ == "__main__":
    main()
