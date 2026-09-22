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

# Loads in environment variables
load_dotenv()

# Pipeline constants
API_KEY = os.getenv("API_KEY")
START_DATE = ""
END_DATE = datetime.now()
GEN_URL = "https://api.openelectricity.org.au/v4/data/network/NEM"
MARKET_URL = "https://api.openelectricity.org.au/v4/market/network/NEM"

def main():
    etl = DataLoader(START_DATE, END_DATE, API_KEY)

if __name__ == "__main__":
    main()
