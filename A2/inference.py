"""
Student Names: Oliver Wuttke, Hans Pujalte, Shivansh Pant, Matilda Alford
Student FANs: WUTT0019, PUJA0009, PANT0108, ALFO0043
Date: 02-11-2026
File: inference.py
Description: Run all implemented models on the latest electricity data.
Usage: python inference.py
Date: 02-11-2026
"""

import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from A1_data_pipeline.DataLoader import DataLoader
from models.DemandModel import DemandModel
from models.PriceModel import PriceModel
from models.NetworkStressModel import NetworkStressModel
from models.RenewableModel import RenewableModel
from prolog.RenewableRules import RenewableRules
from prolog.SystemAssessment import SystemAssessment

# Shared formatting keeps every subsystem and the final assessment consistent.
def label(value):
    return str(value).replace("_", " ").capitalize()


def section(title, forecast_time, prediction, details=(), explanations=(), recommendations=()):
    """Use the same headings and time representation for every subsystem."""
    print(f"\n========== {title} ==========")
    try:
        forecast_time = pd.Timestamp(forecast_time).strftime("%Y-%m-%d %H:%M %z")
    except (TypeError, ValueError):
        pass
    print(f"Forecast for: {forecast_time}")
    print(f"Assessment: {prediction}")
    for detail in details:
        print(f"  {detail}")
    print("Explanation:")
    for reason in explanations or ["No additional explanation supplied."]:
        print(f"  - {reason}")
    print("Recommendations:")
    for action in recommendations or ["See the combined assessment below."]:
        print(f"  - {action}")


def print_assessment(results, errors, combined):
    """Display subsystem evidence first, followed by the coordinated decision."""
    for key, title in [("demand", "Demand Forecast"), ("network", "Network Stress"),
                       ("price", "Price Spike"), ("renewable", "Renewable Coverage")]:
        r = results.get(key)
        if not r:
            section(title, "Unavailable", "Unavailable", explanations=[errors.get(key, "No result returned.")],
                    recommendations=["Resolve this subsystem failure before relying on the combined assessment."])
            continue
        if key == "demand":
            u = r.get("uncertainty") or {}
            section(title, r["forecast_time"], f"{r['demand_mw']:,.0f} MW",
                    [f"95% interval: {r['lower_95_mw']:,.0f}–{r['upper_95_mw']:,.0f} MW",
                     f"Uncertainty: {label(u.get('level', 'unavailable'))}"], u.get("reasons", []))
        elif key == "network":
            drivers = [label(x) for x in r["stress_drivers"]]
            section(title, r["target_timestamp"], label(r["predicted_stress"]),
                    explanations=["Potential drivers from current observations: " + (", ".join(drivers) or "None identified")],
                    recommendations=[label(x) for x in r["recommendations"]])
        elif key == "price":
            rec = r["recommendation"]
            section(title, r["prediction_for"], label(r["prediction"]),
                    [f"Spike probability: {r['probability']:.1%} | Decision threshold: {r['threshold']:.1%}",
                     f"Data quality: {label(rec['data_quality'])} | Confidence: {label(rec['confidence'])}"],
                    [f"[{rule}] {text}" for rule, text in zip(rec["rules"], rec["explanations"])] + rec["quality_issues"],
                    [label(x) for x in rec["actions"]] +
                    (["Deferrable loads: " + ", ".join(label(x) for x in rec["deferred_loads"])] if rec["deferred_loads"] else []))
        else:
            coverage = results.get("coverage")
            if not coverage:
                section(title, r["forecast_time"], f"{r['generation_renewable_mw']:,.0f} MW; coverage unavailable",
                        explanations=[errors.get("coverage", "Demand-linked assessment unavailable.")])
            else:
                reasons = [coverage["explanation"], coverage["scope"]]
                if coverage.get("uncertainty_note"):
                    reasons.append(coverage["uncertainty_note"])
                section(title, r["forecast_time"], label(coverage["status"]) + " against the renewable-share target",
                        [f"Generation: {r['generation_renewable_mw']:,.0f} MW",
                         f"Coverage: {coverage['coverage_percent']:.1f}% | Target: {coverage['target_percent']:.1f}%",
                         f"Target shortfall: {coverage['renewable_shortfall_mw']:,.0f} MW",
                         f"Demand sensitivity: {label(coverage.get('demand_sensitivity') or 'not assessed')}"],
                        reasons, coverage["recommendations"])

    section("Combined System Assessment", combined["forecast_time"], label(combined["priority"]),
            [f"Evidence: {label(combined['completeness'])}"],
            [combined["summary"], combined["scope"]] + combined["issues"],
            [f"[{rule}] {action}" for rule, action in zip(combined["rule_ids"], combined["recommendations"])])


def run_inference(history):
    """Run independent subsystems, retain failures, and coordinate their evidence."""
    results, errors = {}, {}
    # Keep one failed subsystem from hiding other results. Failures prevent an all-clear.
    jobs = {
        "demand": lambda: DemandModel().inference(history, verbose=False),
        "network": lambda: NetworkStressModel().inference(history),
        "price": lambda: PriceModel().inference(history, verbose=False),
        "renewable": lambda: RenewableModel().inference(history, verbose=False),
    }
    for name, predict in jobs.items():
        try:
            results[name] = predict()
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"

    # Coverage requires forecasts from both modules for the same hour.
    if "demand" in results and "renewable" in results:
        try:
            results["coverage"] = RenewableRules(target_share=0.5).assess(
                results["renewable"], results["demand"]
            )
        except Exception as error:
            errors["coverage"] = f"{type(error).__name__}: {error}"
    else:
        errors["coverage"] = "Demand and renewable forecasts are both required."

    combined = SystemAssessment().assess(results, errors)
    print_assessment(results, errors, combined)
    return {"subsystems": results, "errors": errors, "combined": combined}


def main():
    """Fetch shared observations and display the complete hybrid assessment."""
    load_dotenv()
    api_key = os.getenv("OPENELECTRICITY_API_KEY")
    if not api_key:
        raise ValueError("Set OPENELECTRICITY_API_KEY in the environment or .env.")

    # Fetch and transform the same history for every model.
    now = datetime.now()
    history = (DataLoader(now, now, api_key).fetch_instance()
               .set_index("timestamp").sort_index())
    return run_inference(history)


if __name__ == "__main__":
    main()
