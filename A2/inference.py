"""
Student Names: Oliver Wuttke, Hans Pujalte, Shivansh Pant, Matilda Alford
Student FANs: WUTT0019, PUJA0009, PANT0108, ALFO0043
File: inference.py
Date: 02-11-2026
Description: Performs one-off inference on pre-trained models.
"""
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from A1_data_pipeline.DataLoader import DataLoader

from models.DemandModel import DemandModel
from models.NetworkStressModel import NetworkStressModel
from models.PriceModel import PriceModel
from models.RenewableModel import RenewableModel

# Load in env variables
load_dotenv()
API_KEY: str | None = os.getenv("OPENELECTRICITY_API_KEY")

# Init dataloader
dl = DataLoader(datetime.now(), datetime.now(), API_KEY)
df = dl.fetch_instance()

# To break up the fetching output from the actual inference
print('===============================================')

# Load in models and perform inference
demand_model = DemandModel()
demand_model.inference(instance=df.set_index("timestamp"))

# Other model inference...