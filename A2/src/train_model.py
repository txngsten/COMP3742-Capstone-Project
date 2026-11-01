
# File: train_model.py
# Description: Train and evaluate network-stress classifiers.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


# Configuration
RANDOM_STATE = 42
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20

STRESS_QUANTILES = [0.75, 0.90, 0.97]
STRESS_CLASSES = ["low", "medium", "high", "critical"]
SCORE_BINS = [-np.inf, 0.75, 1.50, 2.25, np.inf]

CLASS_TO_NUMBER = {
    name: i for i, name in enumerate(STRESS_CLASSES)
}
NUMBER_TO_CLASS = {
    i: name for name, i in CLASS_TO_NUMBER.items()
}

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

THRESHOLD_FEATURES = list(
    dict.fromkeys([*ADEQUACY_WEIGHTS, *CONGESTION_WEIGHTS])
)


def find_project_root() -> Path:
    """Locate the project root directory."""
    for directory in (Path.cwd(), *Path.cwd().parents):
        if (directory / "A1" / "src" / "DataLoader.py").exists():
            return directory

    raise FileNotFoundError("Project root not found.")


def severity_points(series, feature, thresholds):
    """Assign severity points (0-3) using historical quantiles."""
    q75, q90, q97 = thresholds.loc[feature, STRESS_QUANTILES]

    return np.select(
        [series >= q97, series >= q90, series >= q75],
        [3, 2, 1],
        default=0,
    )


def prepare_modelling_data(data):
    """Create stress targets and chronological dataset partitions."""

    df = data.sort_values("timestamp").reset_index(drop=True).copy()

    # Define chronological split boundaries
    train_end = int(len(df) * TRAIN_FRACTION)
    val_end = int(len(df) * (TRAIN_FRACTION + VALIDATION_FRACTION))

    train_time = df.loc[train_end, "timestamp"]
    val_time = df.loc[val_end, "timestamp"]

    # Calculate thresholds using training data only
    threshold_data = df.iloc[:train_end][THRESHOLD_FEATURES].copy()
    threshold_data["curtailment"] = threshold_data["curtailment"].clip(lower=0)

    thresholds = threshold_data.quantile(STRESS_QUANTILES).T

    # Calculate adequacy score
    adequacy = sum(
        weight * severity_points(df[feature], feature, thresholds)
        for feature, weight in ADEQUACY_WEIGHTS.items()
    )

    # Calculate congestion score
    congestion = sum(
        weight * severity_points(
            df[feature].clip(lower=0) if feature == "curtailment"
            else df[feature],
            feature,
            thresholds,
        )
        for feature, weight in CONGESTION_WEIGHTS.items()
    )

    # Assign stress classifications
    stress_score = np.maximum(adequacy, congestion)

    current_stress = pd.Series(
        pd.cut(
            stress_score,
            bins=SCORE_BINS,
            labels=STRESS_CLASSES,
            right=False,
        ),
        index=df.index,
    )

    # Predict stress one hour into the future
    df["stress_target_1h"] = current_stress.shift(-1)
    df["_target_timestamp"] = df["timestamp"].shift(-1)

    # Keep only consecutive hourly, non-imputed observations
    hourly = (
        df["_target_timestamp"] - df["timestamp"]
    ) == pd.Timedelta(hours=1)

    imputed = df["was_imputed"].astype(bool)
    next_imputed = imputed.shift(-1).fillna(True).astype(bool)

    valid = (
        df["stress_target_1h"].notna()
        & hourly
        & ~imputed
        & ~next_imputed
    )

    df = df.loc[valid].copy()

    # Select numeric features, excluding targets and imputation flags
    features = [
        column for column in df.select_dtypes(include="number").columns
        if column != "was_imputed"
        and not column.startswith("stress_target")
    ]

    # Chronological partitions based on target timestamp
    train = df[df["_target_timestamp"] < train_time].copy()

    validation = df[
        (df["_target_timestamp"] >= train_time)
        & (df["_target_timestamp"] < val_time)
    ].copy()

    test = df[df["_target_timestamp"] >= val_time].copy()

    if any(part.empty for part in (train, validation, test)):
        raise ValueError("One or more dataset partitions are empty.")

    return train, validation, test, features, thresholds


def split_xy(data, features):
    """Separate features and encoded stress labels."""
    X = data[features]
    y = data["stress_target_1h"].map(CLASS_TO_NUMBER).astype(int)
    return X, y


def build_models():
    """Create the candidate classifiers."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
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
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def fit_model(model, X, y):
    """Fit model, accounting for class imbalance."""
    if isinstance(model, XGBClassifier):
        weights = compute_sample_weight("balanced", y)
        model.fit(X, y, sample_weight=weights)
    else:
        model.fit(X, y)


def evaluate_model(y_true, y_pred):
    """Calculate classification metrics."""
    critical = CLASS_TO_NUMBER["critical"]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred,
            labels=list(NUMBER_TO_CLASS),
            average="macro",
            zero_division=0,
        ),
        "critical_recall": recall_score(
            y_true == critical,
            y_pred == critical,
            zero_division=0,
        ),
    }


def train_pipeline(data_path: Path, model_dir: Path):
    """Train, compare, evaluate and save network-stress models."""

    # Prepare datasets
    data = pd.read_parquet(data_path)
    train, validation, test, features, thresholds = (
        prepare_modelling_data(data)
    )

    X_train, y_train = split_xy(train, features)
    X_val, y_val = split_xy(validation, features)
    X_test, y_test = split_xy(test, features)

    # Train and compare candidate models
    models = build_models()
    validation_results = {}

    for name, model in models.items():
        fit_model(model, X_train, y_train)
        predictions = model.predict(X_val)

        validation_results[name] = evaluate_model(
            y_val, predictions
        )

    # Select best model using validation Macro F1
    best_name = max(
        validation_results,
        key=lambda name: validation_results[name]["macro_f1"],
    )

    best_model = clone(models[best_name])

    # Retrain using both training and validation data
    combined = pd.concat([train, validation], ignore_index=True)
    X_combined, y_combined = split_xy(combined, features)

    fit_model(best_model, X_combined, y_combined)

    # Final evaluation on untouched test data
    test_predictions = best_model.predict(X_test)
    test_metrics = evaluate_model(y_test, test_predictions)

    report = classification_report(
        y_test,
        test_predictions,
        labels=list(NUMBER_TO_CLASS),
        target_names=STRESS_CLASSES,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=list(NUMBER_TO_CLASS),
    )

    # Save model
    model_dir.mkdir(parents=True, exist_ok=True)

    threshold_names = [
        "elevated_75th",
        "severe_90th",
        "extreme_97th",
    ]

    threshold_dict = {
        feature: {
            name: float(value)
            for name, value in zip(threshold_names, row)
        }
        for feature, row in thresholds.iterrows()
    }

    bundle = {
        "model": best_model,
        "feature_columns": features,
        "class_names": NUMBER_TO_CLASS,
        "prediction_horizon_hours": 1,
        "target_thresholds": threshold_dict,
    }

    joblib.dump(
        bundle,
        model_dir / "network_stress_model.joblib",
    )

    # Save simplified experiment metadata
    metadata = {
        "selected_model": best_name,
        "features": features,
        "classes": STRESS_CLASSES,
        "prediction_horizon_hours": 1,
        "target_quantiles": STRESS_QUANTILES,
        "target_thresholds": threshold_dict,
        "adequacy_weights": ADEQUACY_WEIGHTS,
        "congestion_weights": CONGESTION_WEIGHTS,
        "score_bins": ["-inf", 0.75, 1.50, 2.25, "inf"],
        "validation_results": validation_results,
        "test_metrics": test_metrics,
        "confusion_matrix": matrix.tolist(),
    }

    (model_dir / "network_stress_model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return best_name, validation_results, test_metrics, report


def main():
    root = find_project_root()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "A1/data/processed/electricity_transformed.parquet",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=root / "A2/models",
    )

    args = parser.parse_args()

    best, validation, test, report = train_pipeline(
        args.data,
        args.model_dir,
    )

    print("\nValidation results:")
    print(pd.DataFrame(validation).T.round(4))

    print(f"\nSelected model: {best}")

    print("\nTest metrics:")
    print(pd.Series(test).round(4))

    print("\nClassification report:")
    print(report)


if __name__ == "__main__":
    main()
