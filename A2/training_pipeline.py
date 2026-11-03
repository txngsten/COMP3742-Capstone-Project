"""
Student Names: Oliver Wuttke, Hans Pujalte, Shivansh Pant, Matilda Alford
Student FANs: WUTT0019, PUJA0009, PANT0108, ALFO0043
File: training_pipeline.py
Date: 02-11-2026
Description: Runs the automated training pipeline.
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

from models.NetworkStressModel import NetworkStressModel
from models.PriceModel import PriceModel
from models.RenewableModel import RenewableModel

# CLI arg parser for inputting training set path
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="path to training dataset")
parser.add_argument("--models", nargs="+", choices=["demand", "renewable"],
                    default=["demand", "renewable"], help="models to train")
parser.add_argument("--renewable-input", type=str,
                    help="cleaned dataset for renewables; defaults to electricity_cleaned.parquet beside --input")
args = parser.parse_args()

# Check if training-set is provided
if args.input is not None:
    TRAINING_PATH = args.input
else:
    sys.exit('No dataset argument found. Please ensure you provide the --input data/processsed/electricity_transformed.parquet flag as per the README.md')

# Load in our dataset
df = pd.read_parquet(TRAINING_PATH).set_index("timestamp").sort_index()

# Demand Prediction Model
# Demand uses the shared transformed features.
if "demand" in args.models:
    from models.DemandModel import DemandModel
    demand_model = DemandModel()
    demand_model.train_eval(training_set=df)

# Renewables rebuilds features from cleaned data, retaining pre-curtailment history.
if "renewable" in args.models:
    renewable_path = (Path(args.renewable_input) if args.renewable_input
                      else Path(TRAINING_PATH).with_name("electricity_cleaned.parquet"))
    renewable_df = pd.read_parquet(renewable_path)
    RenewableModel().train(training_set=renewable_df)

# Other models...
