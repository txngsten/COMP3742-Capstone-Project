"""Matilda Alford (ALFO0043): hourly renewable coverage rules using miniKanren.

Thresholds are project scenarios, not reliability standards. The ratio compares
forecast renewable generation with forecast demand, not annual generation mix.
"""
from math import isfinite

import pandas as pd
from kanren import Relation, conde, eq, facts, lall, run, var
from unification import reify


def _compare(left, right, predicate):
    """A numeric miniKanren goal, evaluated after forecast facts bind its terms."""
    def goal(state):
        if predicate(reify(left, state), reify(right, state)):
            yield state
    return goal


class RenewableRules:
    DEFAULT_TARGET_SHARE = 0.50

    def __init__(self, target_share=DEFAULT_TARGET_SHARE):
        self.target_share = self._number(target_share, "target_share")
        if not 0 < self.target_share <= 1:
            raise ValueError("target_share must be greater than zero and at most one.")

    @staticmethod
    def _number(value, name):
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a finite number.") from None
        if not isfinite(number):
            raise ValueError(f"{name} must be a finite number.")
        return number

    def assess(self, renewable: dict, demand: dict) -> dict:
        """Derive status and recommendations for aligned one-hour forecasts.

        Optional demand bounds give sensitivity to demand uncertainty only.
        They do not form a joint prediction interval for renewable coverage.
        """
        times = []
        for name, forecast in (("renewable", renewable), ("demand", demand)):
            timestamp = pd.Timestamp(forecast["forecast_time"])
            if pd.isna(timestamp) or timestamp.tzinfo is None:
                raise ValueError(f"{name} forecast_time must be timezone-aware and nonmissing.")
            if forecast.get("horizon_hours") != 1:
                raise ValueError("Both forecasts must have a one-hour horizon.")
            times.append(timestamp)
        if times[0] != times[1]:
            raise ValueError("Renewable and demand forecast timestamps must match.")

        renewable_mw = self._number(renewable["generation_renewable_mw"], "renewable generation")
        demand_mw = self._number(demand["demand_mw"], "demand")
        if renewable_mw < 0 or demand_mw <= 0:
            raise ValueError("Renewable generation must be nonnegative and demand must be positive.")

        # Facts are local to this assessment; repeated calls cannot leak old values.
        forecast = Relation()
        facts(forecast, ("renewable", renewable_mw), ("demand", demand_mw),
              ("target_share", self.target_share))
        responses = Relation()
        facts(responses,
              ("target_met", "sufficient", (
                  "Continue monitoring renewable generation and the next forecast.",
                  "Check total supply and reserve adequacy separately.",
              )),
              ("target_missed", "insufficient", (
                  "Review available renewable supply and renewable-charged storage before considering a response.",
                  "Consider shifting flexible demand to hours with higher renewable availability.",
                  "Check other supply and reserves separately; a renewable target shortfall alone does not imply a blackout.",
              )))
        r, d, share = var(), var(), var()
        rule, status, actions = var(), var(), var()
        coverage_rule = lall(
            forecast("renewable", r), forecast("demand", d), forecast("target_share", share),
            conde(
                [_compare((r, d, share), None,
                          lambda values, _: values[0] >= values[1] * values[2]), eq(rule, "target_met")],
                [_compare((r, d, share), None,
                          lambda values, _: values[0] < values[1] * values[2]), eq(rule, "target_missed")],
            ),
            responses(rule, status, actions),
        )
        solutions = run(0, (rule, status, actions), coverage_rule)
        if len(solutions) != 1:
            raise RuntimeError("Expected one renewable coverage rule to match.")
        rule_id, classification, recommendations = solutions[0]
        required = demand_mw * self.target_share
        coverage = 100 * renewable_mw / demand_mw
        shortfall = max(0.0, required - renewable_mw)
        result = {
            "forecast_time": times[0], "horizon_hours": 1,
            "status": classification, "coverage_percent": coverage,
            "target_percent": 100 * self.target_share,
            "required_renewable_mw": required, "renewable_shortfall_mw": shortfall,
            "rule_ids": [rule_id],
            "explanation": (
                f"Forecast renewable generation {renewable_mw:.1f} MW covers {coverage:.1f}% "
                f"of forecast demand {demand_mw:.1f} MW. The project target is "
                f"{100 * self.target_share:g}% ({required:.1f} MW). "
                f"Renewable target shortfall: {shortfall:.1f} MW."
            ),
            "recommendations": list(recommendations),
            "scope": "Hourly renewable coverage of demand; total supply reliability is not assessed.",
            "demand_sensitivity": None,
        }

        # Derive a separate sensitivity flag using demand's optional interval.
        has_lower = "lower_95_mw" in demand
        has_upper = "upper_95_mw" in demand
        if has_lower != has_upper:
            raise ValueError("Provide both demand interval bounds or neither.")
        if has_lower:
            lower = self._number(demand["lower_95_mw"], "demand lower bound")
            upper = self._number(demand["upper_95_mw"], "demand upper bound")
            if not 0 < lower <= demand_mw <= upper:
                raise ValueError("Demand bounds must be positive and contain the point forecast.")
            sensitivity = var()
            sensitivity_rules = conde(
                [_compare(renewable_mw, self.target_share * upper, lambda a, b: a >= b),
                 eq(sensitivity, "meets_across_demand_band")],
                [_compare(renewable_mw, self.target_share * lower, lambda a, b: a < b),
                 eq(sensitivity, "below_across_demand_band")],
                [_compare(renewable_mw, (self.target_share * lower, self.target_share * upper),
                          lambda a, bounds: bounds[0] <= a < bounds[1]),
                 eq(sensitivity, "depends_on_demand")],
            )
            result["demand_sensitivity"] = run(1, sensitivity, sensitivity_rules)[0]
            if result["demand_sensitivity"] == "depends_on_demand":
                result["recommendations"].append(
                    "The target outcome changes within the demand interval; monitor the next forecast before relying on the point classification."
                )
            result["uncertainty_note"] = (
                "Demand-band sensitivity holds renewable generation fixed. Renewable forecast uncertainty is not quantified."
            )
        return result

    @staticmethod
    def print_assessment(result):
        print(f"Renewable sufficiency ({result['forecast_time']}): {result['status'].upper()}")
        print(result["explanation"])
        if result["demand_sensitivity"]:
            print(f"Demand sensitivity: {result['demand_sensitivity']}")
            print(result["uncertainty_note"])
        for recommendation in result["recommendations"]:
            print(f"  - {recommendation}")
        print(result["scope"])
