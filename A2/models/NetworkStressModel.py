"""
Student Name: Hans Pujalte
Student FAN: PUJA0009
File: NetworkStressModel.py
Description: Train, evaluate and explain one-hour-ahead regional stress predictions.
"""

from __future__ import annotations

import json
import sys
from typing import Any
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


A2_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = A2_DIR / "saved_models"
DEFAULT_RULES_PATH = A2_DIR / "prolog" / "stress_rules.pl"


def normalise_data(data: pd.DataFrame) -> pd.DataFrame:
    """Accept pipeline DatetimeIndex or parquet timestamp column; preserve input."""
    df = data.copy()
    if "timestamp" not in df.columns:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Expected a timestamp column or DatetimeIndex.")
        df = df.rename_axis("timestamp").reset_index()
    if not isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype):
        raise ValueError("Timestamps must be timezone-aware datetimes.")
    if df.empty or df["timestamp"].isna().any() or df["timestamp"].duplicated().any():
        raise ValueError("Expected nonempty data with unique, nonmissing timestamps.")
    return df.sort_values("timestamp").reset_index(drop=True)


def require_numeric(data: pd.DataFrame, columns: list[str]) -> None:
    """Reject missing or invalid model inputs with a useful error."""
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"Missing network-stress features: {missing}")
    invalid = [c for c in columns if not pd.api.types.is_numeric_dtype(data[c])]
    if invalid:
        raise ValueError(f"Network-stress features must be numeric: {invalid}")
    if not np.isfinite(data[columns].to_numpy(dtype=float)).all():
        raise ValueError("Network-stress inputs contain missing or infinite values.")


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

    df = normalise_data(data)
    if len(df) < 10:
        raise ValueError("Too few observations for chronological training/validation/test splits.")
    require_numeric(df, THRESHOLD_FEATURES + ["was_imputed"])
    if not df["was_imputed"].isin([0, 1]).all():
        raise ValueError("was_imputed must contain only 0 or 1.")

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
        if column not in {"was_imputed", "cluster", "anomaly"}
        and not column.startswith("stress_target")
    ]

    require_numeric(df, features)

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


def train_pipeline(data: pd.DataFrame, model_dir: Path = DEFAULT_MODEL_DIR):
    """Train, compare, evaluate and save network-stress models."""

    # Prepare datasets
    model_dir = Path(model_dir)
    train, validation, test, features, thresholds = (
        prepare_modelling_data(data)
    )

    X_train, y_train = split_xy(train, features)
    X_val, y_val = split_xy(validation, features)
    X_test, y_test = split_xy(test, features)

    if set(y_train.unique()) != set(NUMBER_TO_CLASS):
        raise ValueError("Training partition must contain all four stress classes.")

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

    # Save experiment metadata
    metadata = {
        "selected_model": best_name,
        "data_split": {"train": TRAIN_FRACTION, "validation": VALIDATION_FRACTION, "test": 1 - TRAIN_FRACTION - VALIDATION_FRACTION},
        "partitions": {name: {"rows": len(part), "first_target": str(part["_target_timestamp"].min()), "last_target": str(part["_target_timestamp"].max())} for name, part in [("train", train), ("validation", validation), ("test", test)]},
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

    return {
        **metadata,
        "classification_report": report,
        "bundle_path": str(model_dir / "network_stress_model.joblib"),
        "metadata_path": str(model_dir / "network_stress_model_metadata.json"),
    }



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


class NetworkStressModel:
    """Network-stress subsystem used by A2's shared entry points."""

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, rules_path: Path = DEFAULT_RULES_PATH):
        self.model_dir = Path(model_dir)
        self.rules_path = Path(rules_path)
        self.bundle = None

    def train_eval(self, training_set: pd.DataFrame) -> dict[str, Any]:
        """Compare classifiers, evaluate the selected model and save its bundle."""
        print("\n========== Network Stress | Training ==========")
        print("Candidates: Random Forest, XGBoost | Horizon: 1 hour")
        print("Selection: Highest validation macro-F1")
        result = train_pipeline(training_set, self.model_dir)
        self.bundle = None
        names = {"accuracy": "Accuracy", "macro_f1": "Macro-F1", "critical_recall": "Critical recall"}
        print("\nValidation performance:")
        print(pd.DataFrame(result["validation_results"]).T.rename(columns=names)
              .to_string(float_format=lambda value: f"{value:.4f}"))
        print(f"Selected model: {result['selected_model']}")
        print("\nTest performance:")
        print(pd.DataFrame([result["test_metrics"]], index=[result["selected_model"]])
              .rename(columns=names).to_string(float_format=lambda value: f"{value:.4f}"))
        print("Scores are on a 0–1 scale; higher is better.")
        print("\nTest performance by class:")
        print(result["classification_report"].strip())
        print(f"\nSaved model: {result['bundle_path']}")
        print(f"Saved metadata: {result['metadata_path']}")
        print("Status: Training and evaluation complete")
        return result

    def train(self, training_set: pd.DataFrame) -> dict[str, Any]:
        """Support the original model-placeholder interface."""
        return self.train_eval(training_set)

    def _load_model(self) -> dict[str, Any]:
        """Load the trained bundle once per model instance."""
        # Reuse the loaded model for subsequent predictions.
        if self.bundle is None:
            path = self.model_dir / "network_stress_model.joblib"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Model not found at {path}; run training first."
                )
            self.bundle = joblib.load(path)
        return self.bundle

    def inference(self, instance: pd.DataFrame) -> dict[str, Any]:
        """Predict from the latest row of sorted, timezone-aware DatetimeIndex history."""
        if instance.empty:
            raise ValueError("No observations provided.")
        if not isinstance(instance.index, pd.DatetimeIndex) or instance.index.tz is None:
            raise ValueError("Expected a timezone-aware DatetimeIndex.")
        if not instance.index.is_unique or not instance.index.is_monotonic_increasing:
            raise ValueError("Observation timestamps must be unique and sorted.")

        # The caller selects the historical cutoff; use its latest observation.
        bundle = self._load_model()
        observation = instance.iloc[[-1]]
        features = bundle["feature_columns"]
        required = list(dict.fromkeys(
            features + ["was_imputed", "flow_imports", "demand_gross", "curtailment"]
        ))
        require_numeric(observation, required)
        row = observation.iloc[0]
        if row["was_imputed"] != 0:
            raise ValueError("Selected observation was imputed.")
        if row["demand_gross"] <= 0:
            raise ValueError("Gross demand must be positive.")

        # Keep the training feature order and translate the predicted class number.
        prediction = int(bundle["model"].predict(observation[features])[0])
        class_names = {int(key): value for key, value in bundle["class_names"].items()}
        risk = class_names[prediction]
        if risk not in STRESS_CLASSES:
            raise ValueError("Saved model has an unknown stress class.")

        # Give Prolog the prediction and current conditions for its advisory rules.
        drivers, recommendations = self._explain(
            risk,
            float(row["flow_imports"] / row["demand_gross"]),
            max(float(row["curtailment"]), 0.0),
        )

        # Return the result; the script or notebook decides how to display it.
        timestamp = observation.index[0]
        horizon = bundle["prediction_horizon_hours"]
        return {
            "timestamp": str(timestamp),
            "target_timestamp": str(timestamp + pd.Timedelta(hours=horizon)),
            "prediction_horizon_hours": horizon,
            "predicted_stress": risk,
            "stress_drivers": drivers,
            "recommendations": recommendations,
        }

    def _explain(self, risk: str, import_share: float, curtailment: float):
        """Use isolated facts for this call; recommendations are advisory only."""
        if not self.rules_path.is_file():
            raise FileNotFoundError(f"Stress rules not found: {self.rules_path}")
        prolog = load_prolog_class()()
        prolog.consult(str(self.rules_path.resolve()))
        predicates = ["predicted_risk", "import_share", "curtailment"]
        for predicate in predicates:
            prolog.retractall(f"{predicate}(network_stress_current, _)")
        try:
            prolog.assertz(f"predicted_risk(network_stress_current, {risk})")
            prolog.assertz(f"import_share(network_stress_current, {import_share})")
            prolog.assertz(f"curtailment(network_stress_current, {curtailment})")
            return (
                query_values(prolog, "stress_driver(network_stress_current, Driver)", "Driver"),
                query_values(prolog, "recommendation(network_stress_current, Action)", "Action"),
            )
        finally:
            for predicate in predicates:
                prolog.retractall(f"{predicate}(network_stress_current, _)")


def print_result(result: dict[str, Any]) -> None:
    """Helper to display the ML prediction and symbolic recommendations."""

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
