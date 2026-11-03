"""
Student Name: Shivansh Pant
Student FAN: PANT0108
File: PriceModel.py
Date: 02-11-2026
Description: Class file for training and performing inference for next-hour price-spike predictions.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Resolve saved artifacts and rules from A2, independently of the launch directory.
A2_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = str(A2_DIR / "saved_models" / "price_spike_model.joblib")
DEFAULT_REPORT_DIR = str(A2_DIR / "reports" / "price")


class PriceModel:
    """
    Predicts whether the next hour's NSW spot price will be high (> $300/MWh)
    and turns the prediction into site actions with the Prolog rules in
    prolog/price_spike.pl. The example site is a large electricity user
    (a water utility) with critical and deferrable loads.
    """
    HIGH_PRICE = 300.0                  # $/MWh; label is next-hour price above this

    BASE_COLUMNS = [
        "price", "demand", "demand_gross", "generation_renewable",
        "renewable_proportion", "curtailment", "curtailment_solar_utility",
        "curtailment_wind", "flow_imports", "flow_exports", "emissions", "power",
    ]
    CURTAILMENT = ["curtailment", "curtailment_solar_utility", "curtailment_wind"]
    LAG_COLUMNS = ["price", "demand", "generation_renewable", "curtailment"]
    LAG_HOURS, ROLLING_HOURS = [1, 24, 168], 24
    HISTORY_HOURS = max(max(LAG_HOURS), ROLLING_HOURS)
    MAX_GAP_HOURS = 3
    TIMEZONE = "Australia/Sydney"

    # Chronological split by the hour being predicted: train Sep 2020 - 2023
    # (includes the 2022 energy crisis), validation 2024, test 2025 onwards.
    TRAIN_END, VALIDATION_END = "2024-01-01", "2025-01-01"

    BETA = 2.0                          # F2: a missed spike costs more than a false alarm
    THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)
    CRITICAL_DEMAND_PERCENTILE = 0.95   # grid-stress cut-off from training demand
    RANDOM_STATE = 42

    # Inference input checks
    MAX_DATA_AGE_HOURS = 1
    PRICE_FLOOR, PRICE_CAP = -1000.0, 20000.0

    RULES = str(A2_DIR / "prolog" / "price_spike.pl")
    ALERT_ACTIONS = {"defer_non_essential_loads", "issue_price_warning",
                     "use_site_battery", "monitor_conditions"}

    def __init__(self, site_battery: str = "available"):
        """
        Args:
            site_battery: declared state of the site's own battery
                          ('available' or 'unavailable'); the dataset holds grid data only.
        """
        self.site_battery = site_battery
        self.model = None
        self.metadata: dict = {}
        self._prolog = None

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    @classmethod
    def _causal_fill(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Re-fills short gaps using past values only. The shared transform
        interpolates, which fills hour t with help from hour t + 1; that leaks
        the future and cannot be reproduced live. Rows it flagged as imputed are
        replaced with the previous hour's values instead.
        Args:
            df: hourly data with a 'timestamp' column, sorted by time.

        Returns:
            Copy with short gaps forward-filled and a was_imputed flag.
        """
        data = df.copy()
        flagged = data["was_imputed"].astype(bool) if "was_imputed" in data else \
            pd.Series(False, index=data.index)
        data.loc[flagged, cls.BASE_COLUMNS] = np.nan
        before = data[cls.BASE_COLUMNS].isna()
        data[cls.BASE_COLUMNS] = data[cls.BASE_COLUMNS].ffill(limit=cls.MAX_GAP_HOURS)
        filled = (before & data[cls.BASE_COLUMNS].notna()).any(axis=1)
        data["was_imputed"] = (flagged | filled).astype(int)
        return data

    @classmethod
    def _add_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds the feature table and next-hour label. Every feature for hour t
        uses only observations up to t; the same code runs in training and inference.
        Args:
            df: hourly data with a 'timestamp' column, already causally filled.

        Returns:
            One row per hour t with features, target_timestamp, next_price and
            is_high_price_next (NaN when t + 1 is unknown). Warm-up rows are dropped.
        """
        out = pd.DataFrame({"timestamp": df["timestamp"]})
        for c in cls.BASE_COLUMNS:
            out[c] = df[c]

        # Lags and rolling statistics over prior hours (same definitions as A1 transform)
        for c in cls.LAG_COLUMNS:
            for lag in cls.LAG_HOURS:
                out[f"{c}_lag_{lag}h"] = df[c].shift(lag)
            past = df[c].shift(1).rolling(cls.ROLLING_HOURS)
            out[f"{c}_rolling_mean_{cls.ROLLING_HOURS}h"] = past.mean()
            out[f"{c}_rolling_std_{cls.ROLLING_HOURS}h"] = past.std()

        # Momentum and whether the price is already high
        high = (df["price"] > cls.HIGH_PRICE).astype(int)
        out["price_change_1h"] = df["price"] - df["price"].shift(1)
        out["price_is_high_now"] = high
        out["high_price_hours_last_24h"] = high.rolling(cls.ROLLING_HOURS, min_periods=1).sum()

        # Calendar features of the hour being predicted
        target = (df["timestamp"] + pd.Timedelta(hours=1)).dt.tz_convert(cls.TIMEZONE)
        for name, (values, period) in {"hour": (target.dt.hour, 24),
                                       "day_of_week": (target.dt.dayofweek, 7),
                                       "month": (target.dt.month - 1, 12)}.items():
            out[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
            out[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
        out["is_weekend"] = target.dt.dayofweek.ge(5).astype(int)

        # Data-quality flags, also passed to the symbolic layer
        out["curtailment_negative"] = (df[cls.CURTAILMENT] < 0).any(axis=1).astype(int)
        out["was_imputed"] = df["was_imputed"]

        # Label
        next_price = df["price"].shift(-1)
        out["target_timestamp"] = df["timestamp"] + pd.Timedelta(hours=1)
        out["next_price"] = next_price
        out["is_high_price_next"] = (next_price > cls.HIGH_PRICE).astype(float)
        out.loc[next_price.isna(), "is_high_price_next"] = np.nan
        return out.iloc[cls.HISTORY_HOURS:].reset_index(drop=True)

    @staticmethod
    def _feature_columns(features: pd.DataFrame) -> list:
        """Model input columns: everything except identifiers and labels."""
        excluded = {"timestamp", "target_timestamp", "next_price", "is_high_price_next"}
        return [c for c in features.columns if c not in excluded]

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    @classmethod
    def _candidates(cls) -> dict:
        """
        Hyperparameter candidates. Both impute with the training median; only
        Logistic Regression is scaled. Class weights counter the rarity of spikes.
        Returns:
            Algorithm name -> list of (params, unfitted pipeline).
        """
        logistic = [({"C": c}, Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=c, class_weight="balanced", max_iter=5000)),
        ])) for c in (0.05, 0.5)]
        forest = [({"min_samples_leaf": leaf, "max_depth": depth}, Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200, min_samples_leaf=leaf, max_depth=depth,
                class_weight="balanced_subsample", n_jobs=-1, random_state=cls.RANDOM_STATE)),
        ])) for leaf, depth in ((5, None), (20, 12))]
        return {"logistic_regression": logistic, "random_forest": forest}

    @classmethod
    def _split(cls, features: pd.DataFrame) -> dict:
        """Splits by target hour so no training label falls inside a later period."""
        tz = features["timestamp"].dt.tz
        train_end = pd.Timestamp(cls.TRAIN_END, tz=tz)
        val_end = pd.Timestamp(cls.VALIDATION_END, tz=tz)
        when = features["target_timestamp"]
        return {"train": features[when < train_end],
                "validation": features[(when >= train_end) & (when < val_end)],
                "test": features[when >= val_end]}

    @classmethod
    def _score(cls, y, prob, threshold, high_now=None) -> dict:
        """
        Classification metrics at one threshold. Accuracy is reported but not
        used: high-price hours are under 1% of the test period.
        Args:
            y: true labels.
            prob: predicted spike probabilities.
            threshold: probability at or above which a spike is predicted.
            high_now: optional current-hour high flags, to split recall into
                      spike onsets and continuing spikes.
        """
        pred = (np.asarray(prob) >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        split = {}
        if high_now is not None:
            yy, now = np.asarray(y), np.asarray(high_now).astype(bool)
            for name, mask in (("onset", (yy == 1) & ~now), ("continuing", (yy == 1) & now)):
                split[f"{name}_spikes"] = int(mask.sum())
                split[f"{name}_recall"] = float(pred[mask].mean()) if mask.any() else None
        return {**split,
                "threshold": float(threshold),
                "accuracy": float((tp + tn) / len(y)),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "f1": float(fbeta_score(y, pred, beta=1, zero_division=0)),
                "f2": float(fbeta_score(y, pred, beta=cls.BETA, zero_division=0)),
                "pr_auc": float(average_precision_score(y, prob)),
                "false_alarms_per_week": float(fp / (len(y) / 168)),
                "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}

    @classmethod
    def _tune_threshold(cls, y, prob) -> float:
        """Threshold that maximises F2 on the validation year."""
        scores = [fbeta_score(y, (prob >= t).astype(int), beta=cls.BETA, zero_division=0)
                  for t in cls.THRESHOLDS]
        return float(cls.THRESHOLDS[int(np.argmax(scores))])

    @staticmethod
    def _save_figures(curves: dict, results: dict, report_dir: Path) -> None:
        """Saves validation PR curves and test confusion matrices for the report."""
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for name, (y, prob) in curves.items():
            precision, recall, _ = precision_recall_curve(y, prob)
            ax.plot(recall, precision, label=f"{name} (PR-AUC {average_precision_score(y, prob):.2f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Validation precision-recall curves (2024)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(report_dir / "pr_curves_validation.png", dpi=200)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 3.6))
        for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
            cm = r["confusion_matrix"]
            m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
            ax.imshow(m, cmap="Blues")
            for (i, j), v in np.ndenumerate(m):
                ax.text(j, i, f"{v:,}", ha="center", va="center",
                        color="white" if v > m.max() / 2 else "black")
            ax.set_xticks([0, 1], ["normal", "high"])
            ax.set_yticks([0, 1], ["normal", "high"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{name}\n(threshold {r['threshold']:.2f})", fontsize=9)
        fig.tight_layout()
        fig.savefig(report_dir / "confusion_matrices_test.png", dpi=200)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_eval(self, training_set: pd.DataFrame, model_path: str = DEFAULT_MODEL_PATH,
                   report_dir: str = DEFAULT_REPORT_DIR) -> dict:
        """
        Compares Logistic Regression and Random Forest against persistence and
        always-normal baselines, picks an F2 threshold on the validation year,
        retrains the winner on train + validation, evaluates it once on the test
        period, runs the hybrid (ML + Prolog) evaluation and saves the model.
        Args:
            training_set: hourly data with a DatetimeIndex (the shared transformed dataset).
            model_path: where to save the fitted pipeline and its metadata (.joblib).
            report_dir: where to write metrics, figures and the hybrid evaluation.

        Returns:
            Dictionary of all recorded metrics.
        """
        print("\n========== Price Spike | Training ==========")
        print("Candidates: Logistic Regression, Random Forest | Horizon: 1 hour")
        print("Selection: Highest validation F2, with PR-AUC as the tie-breaker")
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        data = self._causal_fill(training_set.reset_index().sort_values("timestamp"))
        features = self._add_features(data).dropna(subset=["is_high_price_next"])
        columns = self._feature_columns(features)
        splits = self._split(features)
        print("\nDataset partitions:")
        for name, part in splits.items():
            print(f"{name:<10} {len(part):>6} hours, {part.is_high_price_next.mean():.2%} high-price, "
                  f"{part.target_timestamp.min():%Y-%m-%d} to {part.target_timestamp.max():%Y-%m-%d}")
        train, val, test = splits["train"], splits["validation"], splits["test"]
        y_train, y_val, y_test = (p["is_high_price_next"].astype(int) for p in (train, val, test))

        # Baselines: persistence (next hour = this hour) and always "normal",
        # which shows why accuracy is misleading here.
        persistence = {name: self._score(p["is_high_price_next"].astype(int),
                                         p["price_is_high_now"].to_numpy(dtype=float), 0.5,
                                         p["price_is_high_now"])
                       for name, p in (("validation", val), ("test", test))}
        always_normal = self._score(y_test, np.zeros(len(test)), 0.5, test["price_is_high_now"])

        print("\nValidation candidates (F2 and PR-AUC use a 0–1 scale):")
        # Tune each candidate on the validation year
        candidates, curves = [], {}
        for algorithm, options in self._candidates().items():
            best = None
            for params, pipe in options:
                pipe.fit(train[columns], y_train)
                prob = pipe.predict_proba(val[columns])[:, 1]
                result = {"algorithm": algorithm, "params": params,
                          "tuned": self._score(y_val, prob, self._tune_threshold(y_val, prob))}
                candidates.append(result)
                print(f"{algorithm:<20} {params}  F2={result['tuned']['f2']:.3f} "
                      f"PR-AUC={result['tuned']['pr_auc']:.3f} @ {result['tuned']['threshold']:.2f}")
                if best is None or result["tuned"]["f2"] > best[0]["tuned"]["f2"]:
                    best = (result, prob)
            best[0]["best_for_algorithm"] = True
            curves[algorithm] = (y_val, best[1])

        finalists = [c for c in candidates if c.get("best_for_algorithm")]
        selected = max(finalists, key=lambda c: (c["tuned"]["f2"], c["tuned"]["pr_auc"]))
        print(f"Selected {selected['algorithm']} {selected['params']} "
              f"(validation F2 {selected['tuned']['f2']:.3f})")

        # Retrain finalists on train + validation; evaluate once on test
        train_val = pd.concat([train, val])
        results, fitted = {}, {}
        for choice in finalists:
            pipe = clone(next(p for params, p in self._candidates()[choice["algorithm"]]
                              if params == choice["params"]))
            pipe.fit(train_val[columns], train_val["is_high_price_next"].astype(int))
            prob = pipe.predict_proba(test[columns])[:, 1]
            results[choice["algorithm"]] = self._score(y_test, prob, choice["tuned"]["threshold"],
                                                       test["price_is_high_now"])
            fitted[choice["algorithm"]] = pipe
        results["persistence_baseline"] = persistence["test"]
        results["always_normal_baseline"] = always_normal

        table = pd.DataFrame({name: {k: v for k, v in r.items() if k != "confusion_matrix"}
                              for name, r in results.items()}).T
        print("\nTest performance:")
        print(table[["accuracy", "precision", "recall", "f2", "pr_auc",
                     "onset_recall", "continuing_recall"]].astype(float)
              .rename(index={"persistence_baseline": "Persistence", "always_normal_baseline": "Always normal"},
                      columns={"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                               "f2": "F2", "pr_auc": "PR-AUC", "onset_recall": "Onset recall",
                               "continuing_recall": "Continuing recall"})
              .to_string(float_format=lambda value: f"{value:.3f}"))
        print("Scores are on a 0–1 scale; higher is better.")

        # Save the model with everything inference needs
        self.model = fitted[selected["algorithm"]]
        self.metadata = {
            "algorithm": selected["algorithm"],
            "params": selected["params"],
            "classification_threshold": selected["tuned"]["threshold"],
            "high_price_threshold": self.HIGH_PRICE,
            "feature_columns": columns,
            "critical_demand_mw": float(train["demand"].quantile(self.CRITICAL_DEMAND_PERCENTILE)),
            "splits": {name: {"hours": len(p), "high_price_rate": float(p.is_high_price_next.mean()),
                              "start": p.target_timestamp.min().isoformat(),
                              "end": p.target_timestamp.max().isoformat()}
                       for name, p in splits.items()},
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        # Create the output folder before saving the fitted model.
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "metadata": self.metadata}, model_path, compress=3)

        # Figures, importances and metrics for the report
        self._save_figures(curves, {k: v for k, v in results.items()
                                    if k != "always_normal_baseline"}, report_dir)
        est = self.model.named_steps["model"]
        importance = est.feature_importances_ if hasattr(est, "feature_importances_") \
            else np.abs(est.coef_[0])
        (pd.DataFrame({"feature": columns, "importance": importance})
         .sort_values("importance", ascending=False).head(15)
         .to_csv(report_dir / "feature_importance.csv", index=False))
        table.to_csv(report_dir / "model_comparison_test.csv")

        # Hybrid evaluation: every test hour through ML -> Prolog
        summary, decisions = self._hybrid_eval(test, self.model.predict_proba(test[columns])[:, 1])
        decisions.to_csv(report_dir / "hybrid_test_decisions.csv", index=False)
        print("\nHybrid evaluation:")
        hybrid_table = pd.DataFrame({
            "ML only": [summary["ml_only_recall"], summary["ml_only_recall_onset"]],
            "ML + rules": [summary["hybrid_any_alert_recall"], summary["hybrid_any_alert_recall_onset"]],
        }, index=["Any-alert recall", "Onset recall"])
        print(hybrid_table.to_string(float_format=lambda value: f"{value:.3f}"))
        print(f"Constraint violations: {summary['constraint_violations']}")

        metrics = {"selected": selected["algorithm"], "validation_candidates": candidates,
                   "validation_baseline": persistence["validation"], "test": results,
                   "hybrid": summary}
        (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
        print(f"\nSaved model: {Path(model_path).resolve()}")
        print(f"Saved reports: {report_dir.resolve()}")
        print("Status: Training and evaluation complete")
        return metrics

    def _hybrid_eval(self, test: pd.DataFrame, prob: np.ndarray) -> tuple:
        """
        Reasons over every test hour and measures what the rules add to the ML output.
        Args:
            test: test-period feature table.
            prob: spike probabilities from the final model.

        Returns:
            (summary dictionary, per-hour decision table).
        """
        threshold = self.metadata["classification_threshold"]
        rows = []
        for r, p in zip(test.itertuples(index=False), prob):
            issues = [f for f in ("was_imputed", "curtailment_negative") if getattr(r, f)]
            facts = {
                "prediction": "high_price" if p >= threshold else "normal_price",
                "probability": float(p), "threshold": threshold,
                "grid_demand": "critical" if r.demand >= self.metadata["critical_demand_mw"] else "normal",
                "site_battery": self.site_battery,
                "data_quality": "suspect" if issues else "valid", "quality_issues": issues,
            }
            out = self._reason(facts)
            rows.append({"prediction_for": r.target_timestamp, "actual_price": r.next_price,
                         "actual_high_price": int(r.is_high_price_next), "price_now": r.price,
                         "probability": round(float(p), 4), "ml_prediction": facts["prediction"],
                         "grid_demand": facts["grid_demand"], "data_quality": out["data_quality"],
                         "primary_action": out["primary_action"],
                         "actions": "|".join(out["actions"]), "rules": "|".join(sorted(set(out["rules"]))),
                         "violations": len(out["constraint_violations"])})
        t = pd.DataFrame(rows)

        spikes = t.actual_high_price == 1
        onset = spikes & (t.price_now <= self.HIGH_PRICE)
        ml_high = t.ml_prediction == "high_price"
        alerted = t.actions.apply(lambda a: bool(self.ALERT_ACTIONS & set(a.split("|"))))

        def rate(hit, of):
            return float((hit & of).sum() / of.sum()) if of.any() else float("nan")

        summary = {
            "hours": len(t), "actual_high_price_hours": int(spikes.sum()),
            "onset_spike_hours": int(onset.sum()), "continuing_spike_hours": int((spikes & ~onset).sum()),
            "ml_only_recall": rate(ml_high, spikes), "ml_only_recall_onset": rate(ml_high, onset),
            "hybrid_any_alert_recall": rate(alerted, spikes),
            "hybrid_any_alert_recall_onset": rate(alerted, onset),
            "spikes_with_no_alert": int((spikes & ~alerted).sum()),
            "spikes_caught_only_by_near_threshold_rule":
                int((~ml_high & spikes & t.rules.str.contains("r7_near_threshold")).sum()),
            "ml_false_alarms": int((ml_high & ~spikes).sum()),
            "monitor_only_alerts": int((t.primary_action == "monitor_conditions").sum()),
            "grid_stress_warnings_issued": int(t.rules.str.contains("r3_grid_stress_warning").sum()),
            "hours_with_suspect_data": int((t.data_quality == "suspect").sum()),
            "constraint_violations": int(t.violations.sum()),
            "primary_action_counts": t.primary_action.value_counts().to_dict(),
        }
        return summary, t

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _load(self, model_path: str) -> None:
        """Loads the fitted pipeline and metadata written by train_eval()."""
        if self.model is None:
            saved = joblib.load(model_path)
            self.model, self.metadata = saved["model"], saved["metadata"]

    def _validate(self, window: pd.DataFrame, clock: pd.Timestamp) -> tuple:
        """
        Checks completeness, continuity, freshness and plausibility of the input.
        Args:
            window: recent hourly rows with a 'timestamp' column, oldest first.
            clock: the time the prediction is made.

        Returns:
            (quality, issues) with quality 'valid', 'suspect' or 'invalid'.
        """
        need = self.HISTORY_HOURS + 1
        if len(window) < need:
            return "invalid", [f"only {len(window)} of {need} required hours available"]
        window = window.tail(need)
        issues = []
        if not window["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
            issues.append("timestamps are not continuous hourly steps")
        age = clock - window["timestamp"].iloc[-1]
        if age > pd.Timedelta(hours=self.MAX_DATA_AGE_HOURS):
            issues.append(f"latest observation is {age} old (limit {self.MAX_DATA_AGE_HOURS} h)")
        latest = window.iloc[-1]
        if pd.isna(latest["price"]) or pd.isna(latest["demand"]):
            issues.append("latest hour is missing price or demand")
        price = window["price"].dropna()
        if ((price < self.PRICE_FLOOR) | (price > self.PRICE_CAP)).any():
            issues.append("price outside the NEM market floor/cap range")
        if issues:
            return "invalid", issues

        # Usable, but the rules should know about weaker inputs
        if latest["was_imputed"] or window[self.BASE_COLUMNS].isna().any().any():
            issues.append("some recent values were missing and have been filled or imputed")
        if (latest[self.CURTAILMENT] < 0).any():
            issues.append("negative curtailment in the latest hour")
        return ("suspect" if issues else "valid"), issues

    def inference(self, instance: pd.DataFrame, model_path: str = DEFAULT_MODEL_PATH,
                  clock: pd.Timestamp | None = None, verbose: bool = True) -> dict:
        """
        Predicts whether the hour after the last row will be high-priced and
        recommends site actions with Prolog. Invalid input skips the model and
        falls back to the safe default (manual review, keep loads running).
        Args:
            instance: hourly history (DatetimeIndex), at least the last 169 rows.
            model_path: .joblib file written by train_eval().
            clock: time the prediction is made (default: now). Pass a past time to replay history.

        Returns:
            Dictionary with the prediction, probability and recommendation.
        """
        self._load(model_path)
        window = self._causal_fill(instance.reset_index().sort_values("timestamp"))
        tz = window["timestamp"].dt.tz
        clock = pd.Timestamp.now(tz=tz).floor("h") if clock is None else clock
        quality, issues = self._validate(window, clock)

        facts = {"prediction": "unavailable", "probability": 0.0,
                 "threshold": self.metadata["classification_threshold"],
                 "grid_demand": "normal", "site_battery": self.site_battery,
                 "data_quality": quality, "quality_issues": issues}
        latest = {}
        if quality != "invalid":
            row = self._add_features(window).iloc[-1]
            p = float(self.model.predict_proba(row[self.metadata["feature_columns"]].to_frame().T)[0, 1])
            facts["probability"] = p
            facts["prediction"] = "high_price" if p >= facts["threshold"] else "normal_price"
            facts["grid_demand"] = "critical" if row["demand"] >= self.metadata["critical_demand_mw"] \
                else "normal"
            latest = {"price": round(float(row["price"]), 1), "demand_mw": round(float(row["demand"]), 1),
                      "grid_renewable_pct": round(float(row["renewable_proportion"]), 1)}

        # The model forecasts the hour after the latest observation; without a
        # usable observation the safe default covers the hour after the clock.
        observed_at = window["timestamp"].iloc[-1]
        target = observed_at if quality != "invalid" else clock
        result = {"prediction_for": (target + pd.Timedelta(hours=1)).isoformat(),
                  "observed_at": observed_at.isoformat(),
                  **facts, "inputs": latest, "recommendation": self._reason(facts)}

        if verbose:
            # Print prediction and explanation
            rec = result["recommendation"]
            print(f"{result['prediction_for']}: {facts['prediction']} "
                  f"(P(spike) = {facts['probability']:.2f}, threshold {facts['threshold']:.2f}, "
                  f"grid demand {facts['grid_demand']}, data {rec['data_quality']})")
            print(f"Recommended: {rec['primary_action']}  (all actions: {', '.join(rec['actions'])}; "
                  f"confidence {rec['confidence']})")
            for rule, text in zip(rec["rules"], rec["explanations"]):
                print(f"  - [{rule}] {text}")
            if rec["deferred_loads"]:
                print(f"  Defer: {', '.join(rec['deferred_loads'])}. Critical loads keep running.")
            for issue in rec["quality_issues"]:
                print(f"  ! {issue}")
        return result

    # ------------------------------------------------------------------
    # Symbolic reasoning
    # ------------------------------------------------------------------

    def _reason(self, facts: dict) -> dict:
        """
        Asks the Prolog knowledge base for a recommendation. If Prolog cannot
        run, returns the same safe default as rule r5 instead of failing.
        Args:
            facts: prediction, probability, threshold, grid_demand,
                   site_battery, data_quality and quality_issues.
        """
        try:
            if self._prolog is None:
                if not Path(self.RULES).exists():
                    raise FileNotFoundError(self.RULES)
                from pyswip import Prolog
                prolog = Prolog()
                # Raise errors to Python instead of opening the interactive debugger
                list(prolog.query("set_prolog_flag(debug_on_error, false)"))
                prolog.consult(self.RULES)
                self._prolog = prolog
            issues = ", ".join(_prolog_string(i) for i in facts["quality_issues"])
            query = (f"assess({_prolog_atom(facts['prediction'])}, {_prolog_number(facts['probability'])}, "
                     f"{_prolog_number(facts['threshold'])}, {_prolog_atom(facts['grid_demand'])}, "
                     f"{_prolog_atom(facts['site_battery'])}, {_prolog_atom(facts['data_quality'])}, "
                     f"[{issues}], Primary, Actions, Rules, Texts, Deferred, Violations, "
                     f"Confidence, Quality, Issues)")
            r = list(self._prolog.query(query))[0]
            return {"primary_action": _text(r["Primary"]),
                    "actions": [_text(a) for a in r["Actions"]],
                    "rules": [_text(a) for a in r["Rules"]],
                    "explanations": [_text(a) for a in r["Texts"]],
                    "deferred_loads": [_text(a) for a in r["Deferred"]],
                    "constraint_violations": [_text(a) for a in r["Violations"]],
                    "confidence": _text(r["Confidence"]),
                    "data_quality": _text(r["Quality"]),
                    "quality_issues": [_text(a) for a in r["Issues"]]}
        except Exception as error:  # Prolog missing or failed: never act on an unchecked prediction
            reason = f"knowledge base unavailable ({type(error).__name__}: {error})"
            return {"primary_action": "request_manual_review",
                    "actions": ["request_manual_review", "maintain_operations"],
                    "rules": ["python_fail_safe"], "explanations": [reason],
                    "deferred_loads": [], "constraint_violations": [], "confidence": "none",
                    "data_quality": facts.get("data_quality", "invalid"),
                    "quality_issues": list(facts.get("quality_issues", [])) + [reason]}


def _prolog_atom(value) -> str:
    """Passes only plain lowercase atoms; anything else becomes an atom Prolog rejects as invalid."""
    return value if isinstance(value, str) and re.fullmatch(r"[a-z][a-z_]*", value) else "unrecognised"


def _prolog_number(value) -> str:
    """Formats a number for Prolog; non-finite values become -1, which Prolog rejects."""
    return repr(float(value)) if np.isfinite(value) else "-1"


def _prolog_string(value) -> str:
    """Quotes text as a Prolog string."""
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def _text(value) -> str:
    """Converts a pyswip atom, string or term to Python text."""
    return value.decode() if isinstance(value, bytes) else str(value)
