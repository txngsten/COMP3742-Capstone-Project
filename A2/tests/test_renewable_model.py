"""Checks for forecast alignment, fitted-state persistence and inference inputs."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from A2.models.RenewableModel import RenewableModel


class RenewableModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp_dir.name) / "renewable.npz"
        index = pd.date_range("2025-09-01", periods=1000, freq="h", tz="Etc/GMT-10")
        hours = np.arange(len(index))
        cls.history = pd.DataFrame({
            "generation_renewable": 1500 + 600 * np.sin(2 * np.pi * hours / 24),
            "demand": 7000 + 1000 * np.cos(2 * np.pi * hours / 24),
        }, index=index)
        cls.fitted = RenewableModel()
        cls.fitted.train(cls.history, cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_features_use_past_measurements_and_target_local_hour(self):
        features = RenewableModel.build_features(self.history)
        row = features.iloc[-1]
        self.assertEqual(row.generation_renewable_lag_168h,
                         self.history.generation_renewable.iloc[-169])
        self.assertAlmostEqual(row.generation_renewable_mean_24h,
                               self.history.generation_renewable.iloc[-24:].mean())
        target = (self.history.index[-1] + pd.Timedelta(hours=1)).tz_convert("Australia/Sydney")
        self.assertAlmostEqual(row.target_hour_sin, np.sin(2 * np.pi * target.hour / 24))
        changed_future = self.history.copy()
        changed_future.iloc[-1] *= 10
        pd.testing.assert_frame_equal(
            features.iloc[:-1], RenewableModel.build_features(changed_future).iloc[:-1]
        )

    def test_npz_round_trip_does_not_refit(self):
        with np.load(self.path, allow_pickle=False) as archive:
            self.assertEqual(archive["model_state"].dtype, np.uint8)
            self.assertEqual(archive["features"].tolist(), RenewableModel.FEATURES)
        expected = self.fitted.inference(self.history)
        with patch("sklearn.ensemble.HistGradientBoostingRegressor.fit",
                   side_effect=AssertionError("Inference must not fit")):
            restored = RenewableModel().load(self.path)
            actual = restored.inference(self.history)
            short_actual = restored.inference(self.history.tail(169))
        self.assertEqual(actual, expected)
        self.assertEqual(short_actual, self.fitted.inference(self.history.tail(169)))
        self.assertEqual(actual["forecast_time"], self.history.index[-1] + pd.Timedelta(hours=1))
        self.assertEqual(restored.trained_through, self.history.index[-1].isoformat())

    def test_missing_or_insufficient_latest_history_rejected(self):
        with self.assertRaisesRegex(ValueError, "169"):
            self.fitted.inference(self.history.tail(168))
        missing = self.history.tail(169).copy()
        missing.iloc[-1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "missing or nonfinite"):
            self.fitted.inference(missing)

    def test_hourly_gaps_and_duplicate_timestamps_rejected(self):
        with self.assertRaisesRegex(ValueError, "continuous hourly"):
            RenewableModel.build_features(self.history.drop(self.history.index[10]))
        with self.assertRaisesRegex(ValueError, "unique"):
            RenewableModel.build_features(pd.concat([self.history, self.history.tail(1)]))

    def test_wrong_feature_order_in_archive_rejected(self):
        path = Path(self.temp_dir.name) / "wrong_features.npz"
        with np.load(self.path, allow_pickle=False) as archive:
            entries = {name: archive[name] for name in archive.files}
        entries["features"] = entries["features"][::-1]
        np.savez_compressed(path, **entries)
        with self.assertRaisesRegex(ValueError, "feature order"):
            RenewableModel().load(path)


if __name__ == "__main__":
    unittest.main()
