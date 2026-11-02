"""
Student Names: Oliver Wuttke, Hans Pujalte, Shivansh Pant, Matilda Alford
Student FANs: WUTT0019, PUJA0009, PANT0108, ALFO0043
File: training_pipeline.py
Date: 02-11-2026
Description: Runs the automated training pipeline.
"""

import sys
import argparse

import pandas as pd

from models.DemandModel import DemandModel
from models.NetworkStressModel import NetworkStressModel
from models.PriceModel import PriceModel
from models.RenewableModel import RenewableModel

# CLI arg parser for inputting training set path
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="path to training dataset")
args = parser.parse_args()

# Check if training-set is provided
if args.input is not None:
    TRAINING_PATH = args.input
else:
    sys.exit('No dataset argument found. Please ensure you provide the --input data/processsed/electricity_transformed.parquet flag as per the README.md')

# Load in our dataset
df = pd.read_parquet(TRAINING_PATH).set_index("timestamp").sort_index()

# Demand Prediction Model
demand_model = DemandModel()
demand_model.train_eval(training_set=df)

# Other models...
