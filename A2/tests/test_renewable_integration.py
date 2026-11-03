"""Offline checks for renewable inference with explicitly supplied demand."""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import run_inference


class RenewableInferenceTests(unittest.TestCase):
    def test_external_demand_assessment(self):
        time = pd.Timestamp("2026-11-03T14:00:00+10:00")
        history = pd.DataFrame({"timestamp": [time - pd.Timedelta(hours=1)]})
        renewable = {"forecast_time": time, "generation_renewable_mw": 2400., "horizon_hours": 1}
        demand = {"forecast_time": time, "demand_mw": 5000., "horizon_hours": 1}
        with patch("inference.RenewableModel") as model:
            model.return_value.inference.return_value = renewable
            results = run_inference(history, ["renewable"], external_demand=demand)
            self.assertEqual(results["renewable_assessment"]["renewable_shortfall_mw"], 100)
            pd.testing.assert_frame_equal(
                model.return_value.inference.call_args.args[0], history.set_index("timestamp")
            )

    def test_no_implicit_demand_forecast(self):
        history = pd.DataFrame(index=pd.date_range("2026-01-01", periods=1, tz="UTC"))
        with patch("inference.RenewableModel") as model:
            model.return_value.inference.return_value = {"generation_renewable_mw": 2400.}
            results = run_inference(history, ["renewable"])
            self.assertNotIn("renewable_assessment", results)
            self.assertNotIn("demand", results)

    def test_olis_forecast_drives_sufficiency(self):
        time = pd.Timestamp("2026-11-03T14:00:00+10:00")
        history = pd.DataFrame({"timestamp": [time - pd.Timedelta(hours=1)]})
        renewable = {"forecast_time": time, "generation_renewable_mw": 2400., "horizon_hours": 1}
        demand = {"forecast_time": time, "demand_mw": 5000., "horizon_hours": 1,
                  "lower_95_mw": 4000., "upper_95_mw": 6000.}
        with patch("inference.RenewableModel") as renewable_model, \
             patch("models.DemandModel.DemandModel") as demand_model:
            renewable_model.return_value.inference.return_value = renewable
            demand_model.return_value.inference.return_value = demand
            results = run_inference(history, ["demand", "renewable"], explain_demand=False)
            self.assertEqual(results["demand"], demand)
            assessment = results["renewable_assessment"]
            self.assertEqual(assessment["status"], "insufficient")
            self.assertEqual(assessment["renewable_shortfall_mw"], 100)
            self.assertEqual(assessment["demand_sensitivity"], "depends_on_demand")
            pd.testing.assert_frame_equal(
                demand_model.return_value.inference.call_args.args[0],
                renewable_model.return_value.inference.call_args.args[0],
            )
            self.assertFalse(demand_model.return_value.inference.call_args.kwargs["explain"])
            demand_model.return_value.inference.return_value = {
                **demand, "forecast_time": time + pd.Timedelta(hours=1)
            }
            with self.assertRaisesRegex(ValueError, "timestamps must match"):
                run_inference(history, ["demand", "renewable"], explain_demand=False)

    def test_demand_returns_prediction_and_preserves_default_explanation(self):
        from models.DemandModel import DemandModel
        index = pd.date_range("2026-01-01", periods=200, freq="h", tz="Etc/GMT-10")
        history = pd.DataFrame({"demand": 5000., "demand_lag_1h": 5000.,
                                "is_weekend": 0}, index=index)
        archive = {"theta": np.zeros(1), "X_train": np.zeros((10, 7)),
                   "y_train": np.zeros(10), "hour_scale": np.ones(24)}
        gp = MagicMock()
        gp.fit.return_value = gp
        gp.predict.return_value = (np.array([10.]), np.array([20.]))
        model = DemandModel()
        with patch("models.DemandModel.np.load", return_value=archive), \
             patch.object(model, "_make_kernel", return_value=MagicMock()), \
             patch.object(model, "_gp", return_value=gp), \
             patch.object(model, "_explain") as explain:
            forecast = model.inference(history, explain=False)
            explain.assert_not_called()
            self.assertEqual(forecast["forecast_time"], index[-1] + pd.Timedelta(hours=1))
            self.assertEqual(forecast["demand_mw"], 5010.)
            self.assertAlmostEqual(forecast["upper_95_mw"] - forecast["demand_mw"], 39.2)
            model.inference(history)
            explain.assert_called_once()


if __name__ == "__main__":
    unittest.main()
