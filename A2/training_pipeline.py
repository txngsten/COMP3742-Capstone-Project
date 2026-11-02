"""
Student Names: Oliver Wuttke, Hans Pujalte, Shivansh Pant, Matilda Alford
Student FANs: WUTT0019, PUJA0009, PANT0108, ALFO0043
File: training_pipeline.py
Description: Train all implemented A2 models on the transformed electricity data.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

A2_DIR = Path(__file__).resolve().parent

# Allow this script to run from either the repository root or A2.
if __package__ in (None, ""):
    sys.path.insert(0, str(A2_DIR.parent))



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

    # Load model dependencies only after the input argument is validated.
    from A2.models.DemandModel import DemandModel
    from A2.models.NetworkStressModel import NetworkStressModel

    # Load and sort the dataset once for both models.
    df = pd.read_parquet(TRAINING_PATH).set_index("timestamp").sort_index()

    # Train and evaluate demand prediction, then save its parameters.
    demand_model = DemandModel()
    demand_model.train_eval(
        training_set=df,
        model_path=str(A2_DIR / "demand_gp.npz"),
    )

    # Compare network-stress classifiers and save the selected model.
    network_stress_model = NetworkStressModel()
    network_stress_model.train_eval(training_set=df)

    # Add price and renewable training here when implemented.


if __name__ == "__main__":
    main()
