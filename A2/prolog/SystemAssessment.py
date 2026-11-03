"""
Student Name: Hans Pujalte
Student FAN: PUJA0009
File: SystemAssessment.py
Date: 03-11-2026
Description: Validate and combine subsystem evidence through Prolog rules.
"""
from pathlib import Path

import pandas as pd


class SystemAssessment:
    RULES = Path(__file__).with_name("system_assessment.pl")

    def assess(self, results: dict, errors: dict | None = None) -> dict:
        """Require aligned one-hour forecasts; missing evidence triggers review."""
        issues = [f"{name}: {message}" for name, message in (errors or {}).items()]
        times = []
        for name, field in [("demand", "forecast_time"), ("network", "target_timestamp"),
                            ("price", "prediction_for"), ("renewable", "forecast_time"),
                            ("coverage", "forecast_time")]:
            value = results.get(name)
            try:
                if not value:
                    raise ValueError("result unavailable")
                timestamp = pd.Timestamp(value[field])
                if pd.isna(timestamp) or timestamp.tzinfo is None:
                    raise ValueError("forecast time must be timezone-aware")
                times.append(timestamp)
                if name != "price":
                    horizon = value.get("prediction_horizon_hours" if name == "network" else "horizon_hours")
                    if horizon != 1:
                        raise ValueError("forecast horizon must be one hour")
            except (KeyError, TypeError, ValueError) as error:
                issues.append(f"{name}: {error}")
        if times and any(t != times[0] for t in times):
            issues.append("Subsystem forecast timestamps do not match.")

        # Only recognised atoms reach the Prolog query; no free text is interpolated.
        def category(value, allowed, label):
            if value in allowed:
                return value
            issues.append(f"{label}: missing or unrecognised category")
            return "unknown"

        network = category(results.get("network", {}).get("predicted_stress"),
                           {"low", "medium", "high", "critical"}, "network stress")
        uncertainty = category((results.get("demand", {}).get("uncertainty") or {}).get("level"),
                               {"low", "medium", "high"}, "demand uncertainty")
        price_result = results.get("price", {})
        quality = category(price_result.get("recommendation", {}).get("data_quality"),
                           {"valid", "suspect", "invalid"}, "price data quality")
        price = category(price_result.get("prediction"),
                         {"normal_price", "high_price", "unavailable"}, "price prediction")
        if price == "unavailable":
            issues.append("Price prediction unavailable.")
        renewable = category(results.get("coverage", {}).get("status"),
                             {"sufficient", "insufficient"}, "renewable coverage")
        if "python_fail_safe" in price_result.get("recommendation", {}).get("rules", []):
            issues.append("Price symbolic reasoning was unavailable; its safe fallback was used.")
        completeness = "incomplete" if issues else "complete"
        try:
            from pyswip import Prolog
            prolog = Prolog()
            prolog.consult(str(self.RULES))
            query = f"system_assessment:assess({completeness}, {network}, {uncertainty}, {quality}, {price}, {renewable}, Priority, Rules, Texts)"
            row = list(prolog.query(query))[0]
            decode = lambda x: x.decode() if isinstance(x, bytes) else str(x)
            priority = decode(row["Priority"])
            rule_ids = [decode(x) for x in row["Rules"]]
            actions = [decode(x) for x in row["Texts"]]
        except Exception as error:
            issues.append(f"Combined reasoning unavailable: {type(error).__name__}: {error}")
            completeness = "incomplete"
            priority = "manual_review_required"
            rule_ids = ["combined_reasoning_unavailable"]
            actions = ["Request operator review; no combined conclusion is available."]

        # Never merge a cost-saving action into an instruction overriding safety review.
        if priority == "routine_monitoring":
            summary = "No elevated network-stress or price warning in the aligned forecasts."
        else:
            summary = "Operator attention is required; review the priorities and evidence below."
        if renewable == "insufficient":
            summary += " The renewable-share target is not met."
        if completeness == "incomplete":
            summary = "The combined assessment is incomplete; do not treat it as an all-clear."
        return {
            "forecast_time": str(times[0]) if times and len(set(times)) == 1 else "Not aligned or unavailable",
            "priority": priority, "summary": summary, "completeness": completeness,
            "rule_ids": rule_ids, "recommendations": actions, "issues": issues,
            "scope": "Advisory only. Critical services remain active; an authorised operator must approve operational changes. Regional forecasts do not establish site-specific safety or reserve adequacy.",
        }
