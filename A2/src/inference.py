
# File: inference.py
# Description: ML inference and Prolog reasoning for network stress.

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def find_project_root(start: Path | None = None) -> Path:
    """Find the project root directory."""
    start = (start or Path.cwd()).resolve()

    for directory in (start, *start.parents):
        if (directory / "A1" / "src" / "DataLoader.py").is_file():
            return directory

    raise FileNotFoundError("Could not locate project root.")


def normalise_timestamp(
    timestamp: str,
    timestamps: pd.Series,
) -> pd.Timestamp:
    """Match an input timestamp to the dataset timezone."""
    requested = pd.Timestamp(timestamp)
    timezone = timestamps.dt.tz

    if timezone is not None:
        if requested.tzinfo is None:
            requested = requested.tz_localize(timezone)
        else:
            requested = requested.tz_convert(timezone)
    elif requested.tzinfo is not None:
        raise ValueError(
            "Dataset timestamps are timezone-naive. "
            "Enter a timestamp without a timezone."
        )

    return requested


def load_prolog_class() -> Any:
    """Import PySwip without passing CLI arguments to SWI-Prolog."""
    original_args = sys.argv[:]

    try:
        sys.argv = [original_args[0]]
        from pyswip import Prolog
    finally:
        sys.argv = original_args

    return Prolog


def query_values(
    prolog: Any,
    query: str,
    variable: str,
) -> list[str]:
    """Retrieve unique results from a Prolog query."""
    return list(dict.fromkeys(
        str(result[variable])
        for result in prolog.query(query)
    ))


def run_inference(
    data_path: Path,
    model_path: Path,
    rules_path: Path,
    timestamp: str,
) -> dict[str, Any]:
    """Predict network stress and generate Prolog recommendations."""

    # 1. Load electricity data
    data = pd.read_parquet(data_path)

    requested_timestamp = normalise_timestamp(
        timestamp,
        data["timestamp"],
    )

    observation = data.loc[
        data["timestamp"] == requested_timestamp
    ].iloc[:1]

    if observation.empty:
        raise ValueError(
            f"No observation found for {requested_timestamp}"
        )

    if (
        "was_imputed" in observation.columns
        and bool(observation["was_imputed"].iloc[0])
    ):
        raise ValueError("Selected observation was imputed.")

    # 2. Load trained model
    bundle = joblib.load(model_path)

    model = bundle["model"]
    features = bundle["feature_columns"]
    class_names = bundle["class_names"]

    # 3. Predict future stress
    model_input = observation[features]

    prediction = int(model.predict(model_input)[0])
    predicted_risk = str(class_names[prediction])

    horizon = bundle["prediction_horizon_hours"]

    # 4. Extract relevant electricity observations
    row = observation.iloc[0]

    imports = float(row["flow_imports"])
    gross_demand = float(row["demand_gross"])
    curtailment = max(float(row["curtailment"]), 0.0)

    import_share = (
        imports / gross_demand
        if gross_demand > 0
        else 0.0
    )

    # 5. Initialise Prolog
    Prolog = load_prolog_class()
    prolog = Prolog()

    prolog.consult(str(rules_path.resolve()))

    # 6. Assert ML prediction and operational facts
    prolog.assertz(f"predicted_risk(t1, {predicted_risk})")
    prolog.assertz(f"import_share(t1, {import_share})")
    prolog.assertz(f"curtailment(t1, {curtailment})")

    # 7. Query symbolic rules
    drivers = query_values(
        prolog,
        "stress_driver(t1, Driver)",
        "Driver",
    )

    recommendations = query_values(
        prolog,
        "recommendation(t1, Action)",
        "Action",
    )

    # 8. Return subsystem results
    return {
        "timestamp": str(requested_timestamp),
        "prediction_horizon_hours": horizon,
        "predicted_stress": predicted_risk,
        "stress_drivers": drivers,
        "recommendations": recommendations,
    }


def print_result(result: dict[str, Any]) -> None:
    """Display the ML prediction and symbolic recommendations."""

    print("\n--- Network Stress Prediction ---")
    print(f"Observation: {result['timestamp']}")

    print(
        f"Predicted stress "
        f"(+{result['prediction_horizon_hours']}h): "
        f"{result['predicted_stress']}"
    )

    print("\nStress drivers:")
    if result["stress_drivers"]:
        for driver in result["stress_drivers"]:
            print(f"- {driver}")
    else:
        print("- None identified")

    print("\nRecommendations:")
    if result["recommendations"]:
        for action in result["recommendations"]:
            print(f"- {action}")
    else:
        print("- None available")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    project_root = find_project_root()

    parser = argparse.ArgumentParser(
        description="Predict network stress using ML and Prolog."
    )

    parser.add_argument(
        "--timestamp",
        help="Observation timestamp.",
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=(
            project_root
            / "A1"
            / "data"
            / "processed"
            / "electricity_transformed.parquet"
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=(
            project_root
            / "A2"
            / "models"
            / "network_stress_model.joblib"
        ),
    )

    parser.add_argument(
        "--rules",
        type=Path,
        default=(
            project_root
            / "A2"
            / "src"
            / "stress_rules.pl"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the inference pipeline."""

    args = parse_arguments()

    timestamp = args.timestamp

    if not timestamp:
        timestamp = input(
            "Enter observation timestamp: "
        ).strip()

    result = run_inference(
        data_path=args.data.resolve(),
        model_path=args.model.resolve(),
        rules_path=args.rules.resolve(),
        timestamp=timestamp,
    )

    print_result(result)


if __name__ == "__main__":
    main()
