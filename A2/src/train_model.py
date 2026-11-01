# Student Name: [Add your name]
# Student FAN: [Add your FAN]
# File: train_model.py
# Date: 01-11-2026
# Description: Train, compare, evaluate, and save network-stress classifiers.
# Usage: python train_model.py [--data PATH] [--model-dir PATH]
# Licence: For COMP3742 assessment use.

"""Training pipeline for the one-hour-ahead network-stress proxy."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


RANDOM_STATE = 42
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
STRESS_QUANTILES = (0.75, 0.90, 0.97)
STRESS_CLASSES = ("low", "medium", "high", "critical")
CLASS_TO_NUMBER = {name: number for number, name in enumerate(STRESS_CLASSES)}
NUMBER_TO_CLASS = {number: name for name, number in CLASS_TO_NUMBER.items()}
THRESHOLD_FEATURES = (
    "demand_gross",
    "price",
    "flow_imports",
    "curtailment",
    "flow_exports",
    "renewable_with_storage_proportion",
)
ADEQUACY_WEIGHTS = {
    "demand_gross": 0.40,
    "price": 0.35,
    "flow_imports": 0.25,
}
CONGESTION_WEIGHTS = {
    "curtailment": 0.60,
    "flow_exports": 0.25,
    "renewable_with_storage_proportion": 0.15,
}
STRESS_SCORE_BINS = (-np.inf, 0.75, 1.50, 2.25, np.inf)


@dataclass
class PreparedData:
    """Data and metadata produced by chronological target construction."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    thresholds: pd.DataFrame
    boundaries: dict[str, str]
    excluded_imputed_rows: int


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root from the current location."""
    start = (start or Path.cwd()).resolve()
    for base in (start, *start.parents):
        if (base / "A1" / "src" / "DataLoader.py").is_file():
            return base
    raise FileNotFoundError("Could not locate the COMP3742 project root.")


def severity_points(
    series: pd.Series,
    feature: str,
    thresholds: pd.DataFrame,
) -> pd.Series:
    """Convert a numeric series to training-derived severity points 0-3."""
    q75, q90, q97 = thresholds.loc[feature, list(STRESS_QUANTILES)]
    return pd.Series(
        np.select(
            [series >= q97, series >= q90, series >= q75],
            [3, 2, 1],
            default=0,
        ),
        index=series.index,
        dtype="int8",
    )


def prepare_modelling_data(data: pd.DataFrame) -> PreparedData:
    """Create the target and chronological train/validation/test partitions."""
    required = {"timestamp", "was_imputed", *THRESHOLD_FEATURES}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")

    ordered = data.sort_values("timestamp").reset_index(drop=True).copy()
    if ordered["timestamp"].duplicated().any():
        raise ValueError("Training timestamps must be unique.")

    row_count = len(ordered)
    train_boundary_index = int(row_count * TRAIN_FRACTION)
    validation_boundary_index = int(
        row_count * (TRAIN_FRACTION + VALIDATION_FRACTION)
    )
    train_boundary = ordered.loc[train_boundary_index, "timestamp"]
    validation_boundary = ordered.loc[validation_boundary_index, "timestamp"]

    threshold_data = ordered.iloc[:train_boundary_index][
        list(THRESHOLD_FEATURES)
    ].copy()
    threshold_data["curtailment"] = threshold_data["curtailment"].clip(lower=0)
    thresholds = threshold_data.quantile(list(STRESS_QUANTILES)).T

    adequacy_score = sum(
        weight * severity_points(ordered[feature], feature, thresholds)
        for feature, weight in ADEQUACY_WEIGHTS.items()
    )
    congestion_inputs = {
        "curtailment": ordered["curtailment"].clip(lower=0),
        "flow_exports": ordered["flow_exports"],
        "renewable_with_storage_proportion": ordered[
            "renewable_with_storage_proportion"
        ],
    }
    congestion_score = sum(
        weight * severity_points(congestion_inputs[feature], feature, thresholds)
        for feature, weight in CONGESTION_WEIGHTS.items()
    )

    stress_score = np.maximum(adequacy_score, congestion_score)
    current_class = pd.Series(
        pd.cut(
            stress_score,
            bins=STRESS_SCORE_BINS,
            labels=list(STRESS_CLASSES),
            right=False,
            ordered=True,
        ),
        index=ordered.index,
    )

    ordered["stress_target_1h"] = current_class.shift(-1)
    target_timestamp = ordered["timestamp"].shift(-1)
    target_was_imputed = ordered["was_imputed"].shift(-1).fillna(1).astype(bool)
    current_was_imputed = ordered["was_imputed"].astype(bool)
    has_target = ordered["stress_target_1h"].notna()
    excluded_for_imputation = has_target & (
        current_was_imputed | target_was_imputed
    )
    valid_rows = has_target & ~excluded_for_imputation
    excluded_imputed_rows = int(excluded_for_imputation.sum())

    modelling = ordered.loc[valid_rows].copy()
    modelling["_target_timestamp"] = target_timestamp.loc[valid_rows]

    numeric_columns = modelling.select_dtypes(include="number").columns
    feature_columns = [
        column for column in numeric_columns if column != "was_imputed"
    ]

    train = modelling.loc[
        modelling["_target_timestamp"] < train_boundary
    ].copy()
    validation = modelling.loc[
        (modelling["_target_timestamp"] >= train_boundary)
        & (modelling["_target_timestamp"] < validation_boundary)
    ].copy()
    test = modelling.loc[
        modelling["_target_timestamp"] >= validation_boundary
    ].copy()

    for partition_name, partition in {
        "train": train,
        "validation": validation,
        "test": test,
    }.items():
        if partition.empty:
            raise ValueError(f"The {partition_name} partition is empty.")

    boundaries = {
        "training_target_start": str(train["_target_timestamp"].min()),
        "training_target_end": str(train["_target_timestamp"].max()),
        "validation_target_start": str(validation["_target_timestamp"].min()),
        "validation_target_end": str(validation["_target_timestamp"].max()),
        "test_target_start": str(test["_target_timestamp"].min()),
        "test_target_end": str(test["_target_timestamp"].max()),
    }
    return PreparedData(
        train=train,
        validation=validation,
        test=test,
        feature_columns=feature_columns,
        thresholds=thresholds,
        boundaries=boundaries,
        excluded_imputed_rows=excluded_imputed_rows,
    )


def split_xy(
    partition: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Return model inputs and encoded target labels."""
    features = partition.reindex(columns=feature_columns)
    target = partition["stress_target_1h"].map(CLASS_TO_NUMBER).astype(int)
    return features, target


def build_candidate_models() -> dict[str, Any]:
    """Build the two candidate classifiers using fixed development settings."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            num_class=len(STRESS_CLASSES),
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def fit_model(model: Any, features: pd.DataFrame, target: pd.Series) -> Any:
    """Fit a candidate model with appropriate imbalance handling."""
    if isinstance(model, XGBClassifier):
        sample_weight = compute_sample_weight("balanced", target)
        return model.fit(features, target, sample_weight=sample_weight)
    return model.fit(features, target)


def evaluate_predictions(
    actual: pd.Series,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate classification and ordinal evaluation metrics."""
    per_class_recall = recall_score(
        actual,
        predicted,
        labels=list(NUMBER_TO_CLASS),
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
        "weighted_f1": float(f1_score(actual, predicted, average="weighted")),
        "ordinal_mae": float(mean_absolute_error(actual, predicted)),
        "quadratic_kappa": float(
            cohen_kappa_score(actual, predicted, weights="quadratic")
        ),
        "critical_recall": float(per_class_recall[CLASS_TO_NUMBER["critical"]]),
    }


def thresholds_to_dict(thresholds: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Convert percentile thresholds to JSON-friendly metadata."""
    percentile_names = {
        0.75: "elevated_75th",
        0.90: "severe_90th",
        0.97: "extreme_97th",
    }
    return {
        feature: {
            percentile_names[float(percentile)]: float(value)
            for percentile, value in row.items()
        }
        for feature, row in thresholds.iterrows()
    }


def train_pipeline(
    data_path: Path,
    model_directory: Path,
) -> dict[str, Any]:
    """Run model comparison, final evaluation, and artifact persistence."""
    data = pd.read_parquet(data_path).sort_values("timestamp").reset_index(drop=True)
    prepared = prepare_modelling_data(data)
    x_train, y_train = split_xy(prepared.train, prepared.feature_columns)
    x_validation, y_validation = split_xy(
        prepared.validation, prepared.feature_columns
    )
    x_test, y_test = split_xy(prepared.test, prepared.feature_columns)

    candidates = build_candidate_models()
    validation_metrics: dict[str, dict[str, float]] = {}
    for name, candidate in candidates.items():
        fit_model(candidate, x_train, y_train)
        prediction = candidate.predict(x_validation)
        validation_metrics[name] = evaluate_predictions(y_validation, prediction)

    selected_name = max(
        validation_metrics,
        key=lambda name: validation_metrics[name]["macro_f1"],
    )
    selected_model = clone(candidates[selected_name])
    train_validation = pd.concat(
        [prepared.train, prepared.validation], ignore_index=True
    )
    x_train_validation, y_train_validation = split_xy(
        train_validation, prepared.feature_columns
    )
    fit_model(selected_model, x_train_validation, y_train_validation)

    test_prediction = selected_model.predict(x_test)
    test_metrics = evaluate_predictions(y_test, test_prediction)
    report = classification_report(
        y_test,
        test_prediction,
        labels=list(NUMBER_TO_CLASS),
        target_names=list(STRESS_CLASSES),
        digits=3,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_test,
        test_prediction,
        labels=list(NUMBER_TO_CLASS),
    ).tolist()

    model_directory.mkdir(parents=True, exist_ok=True)
    bundle_path = model_directory / "network_stress_model.joblib"
    metadata_path = model_directory / "network_stress_model_metadata.json"
    bundle = {
        "model": selected_model,
        "model_type": type(selected_model).__name__,
        "feature_columns": prepared.feature_columns,
        "class_names": NUMBER_TO_CLASS,
        "prediction_horizon_hours": 1,
        "target_thresholds": thresholds_to_dict(prepared.thresholds),
    }
    joblib.dump(bundle, bundle_path)

    metadata = {
        "artifact_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": type(selected_model).__name__,
        "selected_candidate": selected_name,
        "selection_metric": "validation_macro_f1",
        "feature_columns": prepared.feature_columns,
        "class_names": {str(k): v for k, v in NUMBER_TO_CLASS.items()},
        "prediction_horizon_hours": 1,
        "split_fractions": {
            "train": TRAIN_FRACTION,
            "validation": VALIDATION_FRACTION,
            "test": 1 - TRAIN_FRACTION - VALIDATION_FRACTION,
        },
        "partition_boundaries": prepared.boundaries,
        "excluded_imputed_rows": prepared.excluded_imputed_rows,
        "target_definition": {
            "quantiles": list(STRESS_QUANTILES),
            "thresholds": thresholds_to_dict(prepared.thresholds),
            "adequacy_weights": ADEQUACY_WEIGHTS,
            "congestion_weights": CONGESTION_WEIGHTS,
            "score_bins": ["-inf", 0.75, 1.50, 2.25, "inf"],
        },
        "validation_metrics": validation_metrics,
        "final_test_metrics": test_metrics,
        "final_test_confusion_matrix": matrix,
        "library_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    if isinstance(selected_model, XGBClassifier):
        selected_model.save_model(model_directory / "network_stress_xgboost.json")

    return {
        "selected_model": selected_name,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "classification_report": report,
        "confusion_matrix": matrix,
        "thresholds": prepared.thresholds,
        "partition_boundaries": prepared.boundaries,
        "bundle_path": bundle_path,
        "metadata_path": metadata_path,
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    project_root = find_project_root()
    parser = argparse.ArgumentParser(
        description="Train and evaluate electricity network-stress classifiers."
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
        "--model-dir",
        type=Path,
        default=project_root / "A2" / "models",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line training pipeline."""
    arguments = parse_arguments()
    result = train_pipeline(arguments.data.resolve(), arguments.model_dir.resolve())
    print("Validation comparison:")
    print(pd.DataFrame(result["validation_metrics"]).T.round(4).to_string())
    print(f"\nSelected model: {result['selected_model']}")
    print("\nUntouched test metrics:")
    print(pd.Series(result["test_metrics"]).round(4).to_string())
    print("\nFinal classification report:")
    print(result["classification_report"])
    print(f"Saved model: {result['bundle_path']}")
    print(f"Saved metadata: {result['metadata_path']}")


if __name__ == "__main__":
    main()
