# Student Name: [Add your name]
# Student FAN: [Add your FAN]
# File: test_hybrid.py
# Date: 01-11-2026
# Description: Smoke tests for the symbolic network-stress inference layer.
# Usage: python -m unittest A2.src.test_hybrid
# Licence: For COMP3742 assessment use.

"""Tests for optional reserve input and symbolic risk escalation."""

from pathlib import Path
import unittest

from A2.src.inference import assert_runtime_facts, load_prolog_class, query_values


class HybridRulesTest(unittest.TestCase):
    """Validate core Prolog integration and safety escalation rules."""

    @classmethod
    def setUpClass(cls) -> None:
        project_root = Path(__file__).resolve().parents[2]
        cls.prolog = load_prolog_class()()
        cls.prolog.consult(
            str(project_root / "A2" / "src" / "stress_rules.pl").replace(
                "\\", "/"
            )
        )
        for fact in (
            "price_severe_threshold(229)",
            "price_extreme_threshold(353)",
            "curtailment_severe_threshold(217)",
            "curtailment_extreme_threshold(491)",
            "renewable_severe_threshold(54)",
        ):
            cls.prolog.assertz(fact)

    def facts(self, predicted_risk: str = "low") -> dict[str, float | str]:
        """Return a normal-condition fact set for one test case."""
        return {
            "predicted_risk": predicted_risk,
            "prediction_confidence": 0.80,
            "demand": 7000.0,
            "gross_demand": 8000.0,
            "imports": 600.0,
            "exports": 50.0,
            "import_share": 0.075,
            "electricity_price": 100.0,
            "curtailment": 0.0,
            "renewable_percentage": 30.0,
        }

    def final_risk(self, facts: dict[str, float | str]) -> str:
        """Assert facts and return the unique final risk."""
        assert_runtime_facts(self.prolog, "test_case", facts)
        results = query_values(
            self.prolog, "final_risk(test_case, Risk)", "Risk"
        )
        self.assertEqual(len(results), 1)
        return results[0]

    def test_missing_reserve_margin_defaults_without_error(self) -> None:
        self.assertEqual(self.final_risk(self.facts()), "low")

    def test_reserve_margin_below_five_escalates_to_critical(self) -> None:
        facts = self.facts()
        facts["reserve_margin"] = 4.5
        self.assertEqual(self.final_risk(facts), "critical")

    def test_severe_curtailment_escalates_medium_to_high(self) -> None:
        facts = self.facts("medium")
        facts["curtailment"] = 525.0
        self.assertEqual(self.final_risk(facts), "high")

    def test_price_and_import_predicates_use_python_fact_names(self) -> None:
        facts = self.facts("medium")
        facts["import_share"] = 0.21
        facts["electricity_price"] = 400.0
        self.assertEqual(self.final_risk(facts), "high")


if __name__ == "__main__":
    unittest.main()
