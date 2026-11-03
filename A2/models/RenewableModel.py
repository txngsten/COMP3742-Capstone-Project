"""
Student Name: Matilda Alford
Student FAN: ALFO0043
File: RenewableModel.py
Description: One-hour renewable generation forecasting and NPZ persistence.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


class RenewableModel:
    HORIZON_HOURS = 1
    DEFAULT_PATH = Path(__file__).resolve().parents[1] / "saved_models" / "renewable_gb.npz"
    FEATURES = [
        f"{column}{suffix}"
        for column in ("generation_renewable", "demand")
        for suffix in (
            "", "_lag_1h", "_lag_24h", "_lag_168h",
            "_mean_24h", "_std_24h", "_change_1h",
        )
    ] + [
        "target_hour_sin", "target_hour_cos", "target_weekday_sin",
        "target_weekday_cos", "target_month_sin", "target_month_cos",
        "target_is_weekend",
    ]

    def __init__(self, model: HistGradientBoostingRegressor | None = None):
        self.model = model
        self.trained_through = None

    @staticmethod
    def _history(data: pd.DataFrame) -> pd.DataFrame:
        """Validate history before row shifts; never fill gaps with future values."""
        history = data.copy()
        if "timestamp" in history.columns:
            history = history.set_index("timestamp")
        if not isinstance(history.index, pd.DatetimeIndex) or history.index.tz is None:
            raise ValueError("History requires a timezone-aware DatetimeIndex or timestamp column.")
        history = history.sort_index()
        if history.empty or history.index.hasnans or history.index.has_duplicates:
            raise ValueError("History must be nonempty with unique, nonmissing timestamps.")
        if not history.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all():
            raise ValueError("History must have a continuous hourly timeline.")
        missing = {"generation_renewable", "demand"}.difference(history.columns)
        if missing:
            raise ValueError(f"Missing required measurements: {sorted(missing)}")
        return history

    @classmethod
    def build_features(cls, data: pd.DataFrame) -> pd.DataFrame:
        """Use observations through t and known local calendar values for t + 1h."""
        history = cls._history(data)
        features = pd.DataFrame(index=history.index)
        for column in ("generation_renewable", "demand"):
            features[column] = history[column]
            for lag in (1, 24, 168):
                features[f"{column}_lag_{lag}h"] = history[column].shift(lag)
            features[f"{column}_mean_24h"] = history[column].rolling(24).mean()
            features[f"{column}_std_24h"] = history[column].rolling(24).std()
            features[f"{column}_change_1h"] = history[column].diff()

        local = (history.index + pd.Timedelta(hours=cls.HORIZON_HOURS)).tz_convert("Australia/Sydney")
        for name, values, period in (
            ("target_hour", local.hour, 24),
            ("target_weekday", local.dayofweek, 7),
            ("target_month", local.month - 1, 12),
        ):
            features[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
            features[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
        features["target_is_weekend"] = (local.dayofweek >= 5).astype(int)
        return features[cls.FEATURES].replace([np.inf, -np.inf], np.nan)

    def train(self, training_set: pd.DataFrame, model_path=None) -> None:
        """Fit CV-selected settings on supplied history and save the fitted ensemble.

        Pass cleaned data, without curtailment trimming. Evaluation belongs in
        renewable_investigation.ipynb; this fits the final deployment artifact.
        """
        print("\n========== Renewable Generation | Training ==========")
        print("Model: Histogram Gradient Boosting | Horizon: 1 hour")
        print("Configuration: Previously selected in the renewable investigation notebook")
        history = self._history(training_set)
        examples = self.build_features(history)
        examples["target"] = history["generation_renewable"].shift(-self.HORIZON_HOURS)
        examples = examples.replace([np.inf, -np.inf], np.nan).dropna()
        if examples.empty:
            raise ValueError("No usable training examples; provide at least 170 hourly rows with valid measurements.")
        self.model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=False, random_state=42,
        )
        with threadpool_limits(limits=2):
            self.model.fit(examples[self.FEATURES], examples["target"])
        self.trained_through = (examples.index[-1] + pd.Timedelta(hours=1)).isoformat()
        self.save(model_path)
        print(f"Training examples: {len(examples):,}")
        print(f"Training targets through: {self.trained_through}")
        print("\nEvaluation: Not performed in this training method.")
        print("See renewable_investigation.ipynb for the recorded model comparison.")
        print(f"\nSaved model: {Path(model_path or self.DEFAULT_PATH).resolve()}")
        print("Status: Deployment model training complete")

    def save(self, model_path=None) -> None:
        """Store the fitted tree state as bytes in NPZ, plus readable metadata.

        Constructor hyperparameters alone cannot reconstruct learned trees.
        Like other pickle-based sklearn artifacts, load only trusted archives.
        """
        if not isinstance(self.model, HistGradientBoostingRegressor) or not hasattr(self.model, "n_iter_"):
            raise ValueError("Fit a HistGradientBoostingRegressor before saving.")
        if list(self.model.feature_names_in_) != self.FEATURES:
            raise ValueError("Fitted feature order does not match RenewableModel.FEATURES.")
        metadata = {
            "format_version": 1, "horizon_hours": self.HORIZON_HOURS,
            "target": "generation_renewable", "units": "MW",
            "calendar_timezone": "Australia/Sydney",
            "sklearn_version": sklearn.__version__,
            "trained_through": self.trained_through,
            "parameters": self.model.get_params(),
        }
        payload = np.frombuffer(pickle.dumps(self.model, protocol=5), dtype=np.uint8)
        path = Path(model_path) if model_path is not None else self.DEFAULT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        # A file handle preserves the requested path rather than appending an extension.
        with path.open("wb") as archive:
            np.savez_compressed(archive, model_state=payload,
                                features=np.asarray(self.FEATURES),
                                metadata=np.asarray(json.dumps(metadata)))

    def load(self, model_path=None):
        """Load a trusted local NPZ archive without refitting the estimator."""
        path = Path(model_path) if model_path is not None else self.DEFAULT_PATH
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata"].item())
            if metadata.get("format_version") != 1 or metadata.get("horizon_hours") != self.HORIZON_HOURS:
                raise ValueError("Unsupported renewable artifact format or forecast horizon.")
            if archive["features"].tolist() != self.FEATURES:
                raise ValueError("Artifact feature order does not match the inference code.")
            if metadata.get("sklearn_version") != sklearn.__version__:
                raise ValueError("Use the scikit-learn version recorded in the artifact, or retrain it.")
            model = pickle.loads(archive["model_state"].tobytes())
        if not isinstance(model, HistGradientBoostingRegressor) or list(model.feature_names_in_) != self.FEATURES:
            raise ValueError("Artifact does not contain the expected fitted renewable model.")
        self.model = model
        self.trained_through = metadata.get("trained_through")
        return self

    def inference(self, instance: pd.DataFrame, model_path=None, verbose: bool = True) -> dict:
        """Forecast one hour after the latest observation; return integration fields."""
        if model_path is not None or self.model is None:
            self.load(model_path)
        history = self._history(instance)
        if len(history) < 169:
            raise ValueError("Renewable inference requires at least 169 hourly observations.")
        latest = self.build_features(history).iloc[[-1]]
        if latest.isna().any().any():
            raise ValueError("Latest renewable predictors are missing or nonfinite; recover measurements before inference.")
        with threadpool_limits(limits=2):
            prediction = float(self.model.predict(latest)[0])
        result = {
            "forecast_time": history.index[-1] + pd.Timedelta(hours=self.HORIZON_HOURS),
            "generation_renewable_mw": prediction,
            "horizon_hours": self.HORIZON_HOURS,
        }
        if verbose:
            print(f"{result['forecast_time']}: renewable generation {prediction:.0f} MW")
        return result
