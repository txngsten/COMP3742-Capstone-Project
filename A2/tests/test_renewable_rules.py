"""Boundary, forecast alignment and uncertainty checks for the symbolic layer."""
import unittest

from A2.prolog.RenewableRules import RenewableRules


class RenewableRulesTests(unittest.TestCase):
    def forecasts(self, renewable=2400, demand=5000):
        common = {"forecast_time": "2026-11-03T14:00:00+10:00", "horizon_hours": 1}
        return ({**common, "generation_renewable_mw": renewable},
                {**common, "demand_mw": demand})

    def test_project_example_and_scenario_thresholds(self):
        renewable, demand = self.forecasts()
        cases = [(0.40, "sufficient", 0), (0.50, "insufficient", 100),
                 (0.82, "insufficient", 1700)]
        for share, status, shortfall in cases:
            with self.subTest(share=share):
                result = RenewableRules(share).assess(renewable, demand)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["renewable_shortfall_mw"], shortfall)
                self.assertEqual(result["coverage_percent"], 48)
                self.assertTrue(result["recommendations"])
                self.assertEqual(result["rule_ids"], ["target_met" if shortfall == 0 else "target_missed"])

    def test_equality_meets_target_and_facts_do_not_leak(self):
        rules = RenewableRules()
        for generation, expected in [(2500, "sufficient"), (2499.9, "insufficient"),
                                     (0, "insufficient"), (6000, "sufficient")]:
            r, d = self.forecasts(renewable=generation)
            self.assertEqual(rules.assess(r, d)["status"], expected)

    def test_forecast_alignment_and_horizon(self):
        r, d = self.forecasts()
        d["forecast_time"] = "2026-11-03T15:00:00+11:00"
        self.assertEqual(RenewableRules().assess(r, d)["coverage_percent"], 48)
        d["forecast_time"] = "2026-11-03T15:00:00+10:00"
        with self.assertRaisesRegex(ValueError, "timestamps must match"):
            RenewableRules().assess(r, d)
        d["forecast_time"] = "2026-11-03T14:00:00"
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            RenewableRules().assess(r, d)
        r, d = self.forecasts()
        d["horizon_hours"] = 24
        with self.assertRaisesRegex(ValueError, "one-hour"):
            RenewableRules().assess(r, d)

    def test_invalid_numbers_and_targets(self):
        for share in [0, -0.5, 1.01, float("nan"), float("inf")]:
            with self.subTest(share=share), self.assertRaises(ValueError):
                RenewableRules(share)
        for renewable, demand in [(-1, 5000), (1000, 0), (1000, -1),
                                   (float("nan"), 5000), (1000, float("inf"))]:
            with self.subTest(renewable=renewable, demand=demand), self.assertRaises(ValueError):
                r, d = self.forecasts(renewable, demand)
                RenewableRules().assess(r, d)

    def test_demand_band_sensitivity_boundaries(self):
        for renewable, expected in [(1999, "below_across_demand_band"),
                                     (2000, "depends_on_demand"),
                                     (2500, "depends_on_demand"),
                                     (3000, "meets_across_demand_band")]:
            r, d = self.forecasts(renewable=renewable)
            d.update(lower_95_mw=4000, upper_95_mw=6000)
            result = RenewableRules().assess(r, d)
            self.assertEqual(result["demand_sensitivity"], expected)
            self.assertIn("not quantified", result["uncertainty_note"])
        r, d = self.forecasts()
        d["lower_95_mw"] = 4000
        with self.assertRaisesRegex(ValueError, "both"):
            RenewableRules().assess(r, d)
        d["upper_95_mw"] = 4500
        with self.assertRaisesRegex(ValueError, "contain"):
            RenewableRules().assess(r, d)


if __name__ == "__main__":
    unittest.main()
