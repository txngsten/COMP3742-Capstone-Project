"""
Student Name: Oliver Wuttke
Student FAN: WUTT0019
File: DemandModel.py
Date: 02-11-2026
Description: Class file for training and performing inference for next-hour demand predictions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


# Resolve saved parameters and rules from A2, independently of the launch directory.
A2_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = str(A2_DIR / "saved_models" / "demand_gp.npz")
DEFAULT_RULES_PATH = str(A2_DIR / "prolog" / "uncertainty.pl")


class DemandModel:
    FEATURES = [
        "demand",
        "demand_lag_1h",
        "demand_lag_23h",
        "demand_lag_167h",
                "hour_sin",
        "hour_cos",
        "is_weekend"
    ]

    WINDOW, STEP = 24 * 30, 24 * 7      # rolling train window
    TEST_START = "2025-09-24"           # start of test period
    CAL_WEEKS, CAL_MIN = 12, 4          # weeks used for interval scaling, min before applying

    @staticmethod
    def _add_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Lags columns for GP model training
        Args:
            df: Dataframe used for training

        Returns:
            Dataframe with lagged features and encoded hours.
        """
        df = df.copy()
        df["demand_lag_23h"] = df["demand"].shift(23)
        df["demand_lag_167h"] = df["demand"].shift(167)

        # Cos/sin encode the hour being predicted since 24 hour time is periodic
        h = (df.index + pd.Timedelta(hours=1)).hour
        df["hour_sin"] = np.sin(2 * np.pi * h / 24)
        df["hour_cos"] = np.cos(2 * np.pi * h / 24)
        return df

    @staticmethod
    def _make_kernel(n_features: int):
        """
        Constructs a kernel function to measure variance.
        Args:
            n_features: Number of features

        Returns:
            ConstantKernel for GP regression model.
        """
        return ConstantKernel() * Matern(length_scale=np.ones(n_features), nu=2.5) + WhiteKernel()

    @staticmethod
    def _gp(kernel, tune: bool) -> GaussianProcessRegressor:
        """
        Builds the GP Regression Model for a given kernel.
        Args:
            kernel: Constant kernel function
            tune: Boolean for flagging if optimization is needed.

        Returns:
            Sklearn GP Regression object.
        """
        return GaussianProcessRegressor(
            kernel, normalize_y=True, optimizer="fmin_l_bfgs_b" if tune else None)

    @staticmethod
    def _hour_scale(past: list) -> np.ndarray:
        """
        Per-hour std multiplier so 95% of past standardized errors fall within 1.96.
        Args:
            past: List of past observations

        Returns:
            The newly scaled values.
        """
        z = np.concatenate([p[0] for p in past])
        hz = np.concatenate([p[1] for p in past])
        return np.array([np.quantile(z[hz == i], 0.95) / 1.96 for i in range(24)])


    def train_eval(self, training_set: pd.DataFrame, model_path: str = DEFAULT_MODEL_PATH) -> None:
        """
        Trains a GP Regression model to predict next hour demand, evaluates it
        walk-forward over the test period and saves the final model.
        Args:
            training_set: hourly data with a DatetimeIndex and columns
                          demand, demand_lag_1h, is_weekend.
            model_path: where to save the final model (.npz).
        """
        print("\n========== Demand | Training ==========")
        print("Model: Gaussian Process Regression | Horizon: 1 hour")
        print("Evaluation: Walk-forward comparison with persistence")

        # Features and target (change in demand over the next hour)
        df = self._add_features(training_set)
        df["y"] = df["demand"].shift(-1)
        df = df.dropna(subset=self.FEATURES + ["y"])

        X = df[self.FEATURES].to_numpy()
        now = df["demand"].to_numpy()
        y = df["y"].to_numpy() - now

        W, S = self.WINDOW, self.STEP
        start = df.index.searchsorted(pd.Timestamp(self.TEST_START, tz=df.index.tz))

        # Tune hyperparameters once, on the window before the test period
        sub = slice(start - W, start)
        sx = StandardScaler().fit(X[sub])
        kernel = self._gp(self._make_kernel(X.shape[1]), tune=True) \
            .fit(sx.transform(X[sub]), y[sub]).kernel_
        print(f"Fitted kernel: {kernel}")

        # Rolling forecast with per-hour interval scaling
        preds, stds, past = [], [], []
        for o in range(start, len(df), S):
            sx = StandardScaler().fit(X[o - W:o])
            model = self._gp(kernel, tune=False).fit(sx.transform(X[o - W:o]), y[o - W:o])
            m, s_raw = model.predict(sx.transform(X[o:o + S]), return_std=True)

            # Scale std per hour using errors from previous weeks (already observed)
            hrs = (df.index[o:o + S] + pd.Timedelta(hours=1)).hour.to_numpy()
            if len(past) >= self.CAL_MIN:
                s = s_raw * self._hour_scale(past[-self.CAL_WEEKS:])[hrs]
            else:
                s = s_raw

            past.append((np.abs(y[o:o + S] - m) / s_raw, hrs))
            preds.append(m)
            stds.append(s)

        # Create the output folder before saving the trained parameters.
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            model_path,
            theta=kernel.theta,
            X_train=X[-W:],
            y_train=y[-W:],
            hour_scale=self._hour_scale(past[-self.CAL_WEEKS:]),
        )

        # Results
        res = pd.DataFrame({
            "actual": df["y"].to_numpy()[start:],
            "pred": now[start:] + np.concatenate(preds),
            "std": np.concatenate(stds),
            "persistence": now[start:],
        }, index=df.index[start:] + pd.Timedelta(hours=1))

        def metrics(p):
            e = res.actual - p
            return {
                "MAE": np.mean(np.abs(e)),
                "RMSE": np.sqrt(np.mean(e ** 2)),
                "MAPE %": 100 * np.mean(np.abs(e / res.actual)),
                "R^2": 1 - np.sum(e ** 2) / np.sum((res.actual - res.actual.mean()) ** 2),
            }

        table = pd.DataFrame({"GP": metrics(res.pred), "persistence": metrics(res.persistence)}).T
        table["skill vs persistence"] = 1 - table["RMSE"] / table.loc["persistence", "RMSE"]
        print("\nTest performance:")
        display = table.rename(index={"GP": "Gaussian Process", "persistence": "Persistence"},
                               columns={"MAE": "MAE (MW)", "RMSE": "RMSE (MW)",
                                        "R^2": "R²", "skill vs persistence": "RMSE skill"})
        print(display.to_string(float_format=lambda value: f"{value:.3f}"))
        print("RMSE skill is relative to persistence; higher is better.")

        coverage = np.mean(np.abs(res.actual - res.pred) <= 1.96 * res["std"])
        print(f"95% prediction-interval coverage: {coverage:.1%}")
        print(f"\nSaved model: {Path(model_path).resolve()}")
        print("Status: Training and evaluation complete")
        print("Plot: Last 14 days (close the plot window to continue if it blocks).")

        # Plot the last 14 days
        z = res.iloc[-24 * 14:]
        plt.figure(figsize=(14, 4))
        plt.plot(z.index, z.actual, "k", lw=1, label="actual")
        plt.plot(z.index, z.pred, lw=1, label="GP")
        plt.fill_between(z.index, z.pred - 1.96 * z["std"], z.pred + 1.96 * z["std"],
                         alpha=0.25, label="95% band")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


    def inference(self, instance: pd.DataFrame, model_path: str = DEFAULT_MODEL_PATH, explain: bool = True, verbose: bool = True) -> dict:
        """
        Predicts demand for the hour after the last row of new instance and explains its uncertainty.
        Args:
            instance: hourly history (DatetimeIndex), at least the last 168 rows,
                      with columns demand, demand_lag_1h, is_weekend.
            model_path: .npz file written by train().
            explain: Run the existing Prolog uncertainty explanation (default True).
            verbose: Print output when True; shared inference formats returned results.

        Returns:
            Timestamped next-hour demand prediction and calibrated interval.
        """
        # Rebuild the GP from saved params
        d = np.load(model_path)
        kernel = self._make_kernel(len(self.FEATURES)).clone_with_theta(d["theta"])
        sx = StandardScaler().fit(d["X_train"])
        model = self._gp(kernel, tune=False).fit(sx.transform(d["X_train"]), d["y_train"])

        # Predict from the latest row
        df = self._add_features(instance)
        last = df.iloc[-1]
        x = df[self.FEATURES].iloc[[-1]].to_numpy()
        m, s = model.predict(sx.transform(x), return_std=True)
        target_time = df.index[-1] + pd.Timedelta(hours=1)

        # Print prediction
        pred = last["demand"] + m[0]
        band = 1.96 * s[0] * d["hour_scale"][target_time.hour]
        if verbose:
            print(f"{target_time}: {pred:.0f} MW ± {band:.0f}")

        # Explain the uncertainty with Prolog
        demand_train = d["X_train"][:, self.FEATURES.index("demand")]
        explanation = None
        if explain:
            explanation = self._explain({
                "pred": pred,
                "band": band,
                "hour_scale": d["hour_scale"][target_time.hour],
                "demand": last["demand"],
                "lag_1h": last["demand_lag_1h"],
                "lag_23h": last["demand_lag_23h"],
                "is_weekend": last["is_weekend"],
                "train_min": demand_train.min(),
                "train_max": demand_train.max(),
            }, verbose=verbose)
        return {
            "forecast_time": target_time,
            "horizon_hours": 1,
            "demand_mw": float(pred),
            "lower_95_mw": float(pred - band),
            "upper_95_mw": float(pred + band),
            "uncertainty": explanation,
        }


    @staticmethod
    def _explain(facts: dict, rules: str = DEFAULT_RULES_PATH, verbose: bool = True) -> dict:
        """
        Classifies and explains forecast uncertainty using Prolog rules.
        Args:
            facts: forecast values and context from inference().
            rules: path to the Prolog rules file.
        """
        from pyswip import Prolog

        prolog = Prolog()
        prolog.consult(rules)
        prolog.retractall("forecast(_, _)")
        for k, v in facts.items():
            prolog.assertz(f"forecast({k}, {float(v):.6f})")

        result = list(prolog.query("explanation(Level, Texts)"))[0]
        reasons = [str(t) for t in result["Texts"]]
        reasons = reasons or ["no specific risk factors; band reflects normal model noise"]
        explanation = {"level": str(result["Level"]), "reasons": reasons}
        if verbose:
            print(f"Uncertainty: {explanation['level']}")
            for reason in reasons:
                print(f"  - {reason}")
        return explanation
