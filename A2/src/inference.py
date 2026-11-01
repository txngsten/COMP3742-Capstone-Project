# Student Name: [Add your name]
# Student FAN: [Add your FAN]
# File: inference.py
# Date: 01-11-2026
# Description: Run ML inference and symbolic Prolog reasoning for network stress.
# Usage: python inference.py [--timestamp TIMESTAMP] [--reserve-margin-bps BPS]
# Licence: For COMP3742 assessment use.

"""Integrated inference pipeline for the hybrid network-stress system."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


RUNTIME_PREDICATES = (
    "predicted_risk",
    "prediction_confidence",
    "demand",
    "gross_demand",
    "imports",
    "exports",
    "import_share",
    "electricity_price",
    "curtailment",
    "renewable_percentage",
    "reserve_margin",
)
THRESHOLD_PREDICATES = {
    "price_severe_threshold": ("price", "severe_90th"),
    "price_extreme_threshold": ("price", "extreme_97th"),
    "curtailment_severe_threshold": ("curtailment", "severe_90th"),
    "curtailment_extreme_threshold": ("curtailment", "extreme_97th"),
    "renewable_severe_threshold": (
        "renewable_with_storage_proportion",
        "severe_90th",
    ),
}


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root from the current location."""
    start = (start or Path.cwd()).resolve()
    for base in (start, *start.parents):
        if (base / "A1" / "src" / "DataLoader.py").is_file():
            return base
    raise FileNotFoundError("Could not locate the COMP3742 project root.")


def normalise_timestamp(timestamp_text: str, timestamps: pd.Series) -> pd.Timestamp:
    """Parse input and align a naive timestamp with the dataset timezone."""
    requested = pd.Timestamp(timestamp_text)
    data_timezone = timestamps.dt.tz
    if data_timezone is not None and requested.tzinfo is None:
        requested = requested.tz_localize(data_timezone)
    elif data_timezone is not None:
        requested = requested.tz_convert(data_timezone)
    return requested


def get_numeric_fact(row: pd.Series, column_name: str) -> float:
    """Return a finite numeric operational value from an observation."""
    if column_name not in row.index:
        raise KeyError(f"Operational column not found: {column_name}")
    value = float(row[column_name])
    if not math.isfinite(value):
        raise ValueError(f"{column_name} contains an invalid value: {value}")
    return value


def extract_operational_facts(
    observation: pd.DataFrame,
    predicted_class: str,
    confidence: float,
    reserve_margin: float | None = None,
) -> dict[str, float | str]:
    """Extract current measured conditions for the symbolic inference layer."""
    row = observation.iloc[0]
    gross_demand = get_numeric_fact(row, "demand_gross")
    imports = get_numeric_fact(row, "flow_imports")
    facts: dict[str, float | str] = {
        "predicted_risk": predicted_class,
        "prediction_confidence": confidence,
        "demand": get_numeric_fact(row, "demand"),
        "gross_demand": gross_demand,
        "imports": imports,
        "exports": get_numeric_fact(row, "flow_exports"),
        "import_share": imports / gross_demand if gross_demand > 0 else 0.0,
        "electricity_price": get_numeric_fact(row, "price"),
        "curtailment": max(get_numeric_fact(row, "curtailment"), 0.0),
        "renewable_percentage": get_numeric_fact(
            row, "renewable_with_storage_proportion"
        ),
    }
    if reserve_margin is not None:
        if not math.isfinite(reserve_margin):
            raise ValueError("Reserve margin must be finite when supplied.")
        facts["reserve_margin"] = reserve_margin
    return facts


def load_prolog_class() -> Any:
    """Import PySwip without forwarding application CLI flags to SWI-Prolog."""
    original_arguments = sys.argv[:]
    try:
        sys.argv = [original_arguments[0]]
        from pyswip import Prolog
    finally:
        sys.argv = original_arguments
    return Prolog


def assert_runtime_facts(
    prolog: Any,
    case_id: str,
    facts: dict[str, float | str],
) -> None:
    """Clear previous case facts and assert the current inference facts."""
    for predicate in RUNTIME_PREDICATES:
        list(prolog.query(f"retractall({predicate}(_, _))"))
    for predicate, value in facts.items():
        if isinstance(value, str):
            prolog.assertz(f"{predicate}({case_id}, {value})")
        else:
            prolog.assertz(f"{predicate}({case_id}, {value:.12g})")


def assert_threshold_facts(prolog: Any, bundle: dict[str, Any]) -> None:
    """Assert model-versioned training thresholds used by symbolic rules."""
    thresholds = bundle.get("target_thresholds")
    if not thresholds:
        raise ValueError("The saved model bundle does not contain target thresholds.")
    for predicate, (feature, severity) in THRESHOLD_PREDICATES.items():
        try:
            value = float(thresholds[feature][severity])
        except KeyError as error:
            raise ValueError(
                f"Saved model threshold missing for {feature}/{severity}."
            ) from error
        list(prolog.query(f"retractall({predicate}(_))"))
        prolog.assertz(f"{predicate}({value:.12g})")


def query_values(prolog: Any, query: str, variable: str) -> list[str]:
    """Return unique Prolog variable bindings while preserving rule order."""
    values: list[str] = []
    for result in prolog.query(query):
        value = str(result[variable])
        if value not in values:
            values.append(value)
    return values


def run_inference(
    data_path: Path,
    model_path: Path,
    rules_path: Path,
    timestamp_text: str,
    reserve_margin: float | None = None,
) -> dict[str, Any]:
    """Run one complete historical-observation ML-to-Prolog inference."""
    for required_path in (data_path, model_path, rules_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required file not found: {required_path}")

    data = pd.read_parquet(data_path).sort_values("timestamp").reset_index(drop=True)
    requested_timestamp = normalise_timestamp(timestamp_text, data["timestamp"])
    observation = data.loc[data["timestamp"] == requested_timestamp].iloc[:1]
    if observation.empty:
        raise ValueError(f"No observation found for {requested_timestamp}")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_columns = list(bundle["feature_columns"])
    class_names = {int(key): value for key, value in bundle["class_names"].items()}
    missing_features = sorted(set(feature_columns).difference(observation.columns))
    if missing_features:
        raise ValueError(f"Inference data is missing features: {missing_features}")

    model_input = observation.reindex(columns=feature_columns)
    predicted_number = int(model.predict(model_input)[0])
    probabilities = model.predict_proba(model_input)[0]
    probability_by_class = {
        class_names[int(class_number)]: float(probability)
        for class_number, probability in zip(model.classes_, probabilities)
    }
    predicted_class = class_names[predicted_number]
    confidence = probability_by_class[predicted_class]
    facts = extract_operational_facts(
        observation, predicted_class, confidence, reserve_margin
    )

    prolog = load_prolog_class()()
    prolog.consult(str(rules_path.resolve()).replace("\\", "/"))
    case_id = "current_case"
    assert_threshold_facts(prolog, bundle)
    assert_runtime_facts(prolog, case_id, facts)
    final_risks = query_values(
        prolog, f"final_risk({case_id}, Risk)", "Risk"
    )
    if len(final_risks) != 1:
        raise RuntimeError(
            f"Expected one final risk, received {len(final_risks)}: {final_risks}"
        )

    return {
        "timestamp": str(requested_timestamp),
        "prediction_horizon_hours": int(bundle["prediction_horizon_hours"]),
        "ml_prediction": predicted_class,
        "ml_confidence": confidence,
        "probabilities": probability_by_class,
        "operational_facts": facts,
        "stress_driver": query_values(
            prolog, f"stress_driver({case_id}, Driver)", "Driver"
        ),
        "rule_reasons": query_values(prolog, f"reason({case_id}, Reason)", "Reason"),
        "final_risk": final_risks[0],
        "interpretations": query_values(
            prolog, f"interpretation({case_id}, Text)", "Text"
        ),
        "recommendations": query_values(
            prolog, f"recommendation({case_id}, Action)", "Action"
        ),
    }


def print_result(result: dict[str, Any]) -> None:
    """Print a concise trace of the hybrid inference."""
    print(f"Observation time: {result['timestamp']}")
    print(
        f"ML prediction (+{result['prediction_horizon_hours']}h): "
        f"{result['ml_prediction']} ({result['ml_confidence']:.2%})"
    )
    print("\nOperational facts:")
    for name, value in result["operational_facts"].items():
        print(f"- {name}: {value}")
    print(f"\nSymbolic driver: {', '.join(result['stress_driver'])}")
    print(f"Rule reasons: {', '.join(result['rule_reasons'])}")
    print(f"Final hybrid risk: {result['final_risk']}")
    print(f"Interpretation: {', '.join(result['interpretations'])}")
    print("Recommendations:")
    for action in result["recommendations"]:
        print(f"- {action}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    project_root = find_project_root()
    parser = argparse.ArgumentParser(
        description="Run one-hour-ahead hybrid network-stress inference."
    )
    parser.add_argument("--timestamp", help="Prediction-origin timestamp.")
    parser.add_argument(
        "--reserve-margin-bps",
        type=int,
        help=(
            "Optional externally supplied reserve margin in basis points "
            "(for example, 450 means 4.5 percent)."
        ),
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
        default=project_root / "A2" / "models" / "network_stress_model.joblib",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=project_root / "A2" / "src" / "stress_rules.pl",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line hybrid inference pipeline."""
    arguments = parse_arguments()
    timestamp_text = arguments.timestamp
    if not timestamp_text:
        timestamp_text = input("Enter an observation timestamp: ").strip()
    reserve_margin = (
        arguments.reserve_margin_bps / 100
        if arguments.reserve_margin_bps is not None
        else None
    )
    result = run_inference(
        arguments.data.resolve(),
        arguments.model.resolve(),
        arguments.rules.resolve(),
        timestamp_text,
        reserve_margin,
    )
    print_result(result)


if __name__ == "__main__":
    main()
