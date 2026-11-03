"""
Student Names: Oliver Wuttke, Hans Pujalte, Shivansh Pant, Matilda Alford
Student FANs: WUTT0019, PUJA0009, PANT0108, ALFO0043
Date: 02-11-2026
File: inference.py
Description: Run all implemented models on the latest electricity data.
"""

import os
from datetime import datetime

from dotenv import load_dotenv

from A1_data_pipeline.DataLoader import DataLoader
from models.DemandModel import DemandModel
from models.PriceModel import PriceModel
from models.NetworkStressModel import NetworkStressModel, print_result


def main():
    """Fetch shared observations and run each implemented model."""
    # Load the API key without overriding an existing environment value.
    load_dotenv()
    api_key = os.getenv("OPENELECTRICITY_API_KEY")
    if not api_key:
        raise ValueError(
            "Set OPENELECTRICITY_API_KEY in the environment or .env."
        )

    # Fetch and transform the history once for all models.
    now = datetime.now()
    loader = DataLoader(now, now, api_key)
    history = (
        loader.fetch_instance()
        .set_index("timestamp")
        .sort_index()
    )

    print('\n========== Demand Model Predictions =========')
    # Predict next-hour demand using its saved model parameters.
    demand_model = DemandModel()
    demand_model.inference(
        instance=history,
        model_path="demand_gp.npz",
    )

    # Predict next-hour network stress and display its recommendations.
    network_stress_model = NetworkStressModel()
    result = network_stress_model.inference(instance=history)
    print_result(result)

    # Add price and renewable inference here when implemented.
    print('\n========== Price Spike Inference =====================')
    price_model = PriceModel()
    price_model.inference(instance=history)



# Other model inference...
if __name__ == "__main__":
    main()
