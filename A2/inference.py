"""
Student Names: Oliver Wuttke, Hans Pujalte, Shivansh Pant, Matilda Alford
Student FANs: WUTT0019, PUJA0009, PANT0108, ALFO0043
File: inference.py
Description: Run all implemented models on the latest electricity data.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

A2_DIR = Path(__file__).resolve().parent

# Allow this script to run from either the repository root or A2.
if __package__ in (None, ""):
    sys.path.insert(0, str(A2_DIR.parent))

from A2.A1_data_pipeline.DataLoader import DataLoader
from A2.models.DemandModel import DemandModel
from A2.models.NetworkStressModel import NetworkStressModel, print_result


def main():
    """Fetch shared observations and run each implemented model."""
    # Load the API key without overriding an existing environment value.
    load_dotenv(A2_DIR / ".env")
    api_key = os.getenv("OPENELECTRICITY_API_KEY")
    if not api_key:
        raise ValueError(
            "Set OPENELECTRICITY_API_KEY in the environment or A2/.env."
        )

    # Fetch and transform the history once for all models.
    now = datetime.now()
    loader = DataLoader(now, now, api_key)
    history = (
        loader.fetch_instance()
        .set_index("timestamp")
        .sort_index()
    )

    # Predict next-hour demand using its saved model parameters.
    demand_model = DemandModel()
    demand_model.inference(
        instance=history,
        model_path=str(A2_DIR / "demand_gp.npz"),
    )

    # Predict next-hour network stress and display its recommendations.
    network_stress_model = NetworkStressModel()
    result = network_stress_model.inference(instance=history)
    print_result(result)

    # Add price and renewable inference here when implemented.


if __name__ == "__main__":
    main()
