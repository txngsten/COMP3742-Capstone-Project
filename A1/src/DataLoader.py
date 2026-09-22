"""
Student Names: Oliver Wuttke,
Student FANs: WUTT0019,
File: DataLoader.py
Date: 22-09-2026
Description:
"""

# Imports
import requests

import pandas as pd

class DataLoader:
    def __init__(self, start_date, end_date, api_key):
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key
        self.data_set = pd.DataFrame()

    def fetch(self, gen_url, market_url):
        ...

    def fetch_instance(self):
        ...

    def fetch_clean(self, store_endpoint):
        ...

    def store(self, name):
        ...

    def get_dataframe(self):
        return self.data_set