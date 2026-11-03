"""
Student Names: Oliver Wuttke, Hans Pujalte, Shivansh Pant, Matilda Alford
Student FANs: WUTT0019, PUJA0009, PANT0108, ALFO0043
Date: 02-11-2026
File: training_pipeline.py
Description: Train all implemented A2 models on the transformed electricity data.
Usage: python training_pipeline --input data/processed/electricity_transformed.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from models.DemandModel import DemandModel
from models.NetworkStressModel import NetworkStressModel
from models.PriceModel import PriceModel
from models.RenewableModel import RenewableModel


def main():
    """Load the shared dataset and train each implemented model."""
    # CLI arg parser for inputting training set path.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="path to training dataset")
    args = parser.parse_args()

    # Require a dataset path before running either training pipeline.
    if args.input is not None:
        TRAINING_PATH = args.input
    else:
        sys.exit(
            "No dataset argument found. Please provide "
            "--input A2/data/processed/electricity_transformed.parquet "
            "from the repository root, as per the README.md."
        )

    # Load and sort the dataset once for both models.
    df = pd.read_parquet(TRAINING_PATH).set_index("timestamp").sort_index()

    print("\n========== Machine Learning Training ==========")
    print(f"Dataset: {Path(TRAINING_PATH).resolve()}")
    print(f"Observations: {len(df):,} | Columns: {len(df.columns)}")
    print(f"Period: {df.index.min()} to {df.index.max()}")
    print("Models: Demand, Price Spike, Network Stress, Renewable Generation")

    # Train and evaluate demand prediction, then save its parameters.
    demand_model = DemandModel()
    demand_model.train_eval(training_set=df)

    # Price-spike Prediction Model
    price_model = PriceModel()
    price_model.train_eval(training_set=df)

    # Compare network-stress classifiers and save the selected model.
    network_stress_model = NetworkStressModel()
    network_stress_model.train_eval(training_set=df)

    RenewableModel().train(training_set=df)
    print("\n========== Training Complete ==========\nAll four training pipelines finished successfully.")

if __name__ == "__main__":
    main()

