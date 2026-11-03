"""
Student Names: Oliver Wuttke, Hans Pujalte, Shivansh Pant, Matilda Alford
Student FANs: WUTT0019, PUJA0009, PANT0108, ALFO0043
File: inference.py
Date: 02-11-2026
Description: Forecast from saved models and assess hourly renewable coverage.
"""
import argparse
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from A1_data_pipeline.DataLoader import DataLoader
from models.RenewableModel import RenewableModel


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=["demand", "renewable"],
                        default=["demand", "renewable"], help="models to run")
    parser.add_argument("--renewable-target-percent", type=float, default=50.0,
                        help="project-defined hourly renewable coverage target (default: 50)")
    parser.add_argument("--demand-mw", type=float,
                        help="external demand forecast for renewable-only assessment")
    parser.add_argument("--demand-time", type=str,
                        help="timezone-aware timestamp for --demand-mw; must match renewable forecast")
    parser.add_argument("--skip-demand-explanation", action="store_true",
                        help="use demand prediction without its Prolog explanation")
    return parser


def run_inference(instance, models, target_percent=50.0, external_demand=None,
                  explain_demand=True):
    """Run models on shared history; this function does not fetch live data."""
    from prolog.RenewableRules import RenewableRules

    rules = RenewableRules(target_share=target_percent / 100)
    history = instance.set_index("timestamp") if "timestamp" in instance.columns else instance
    history = history.sort_index()
    # Load in models and perform inference
    results = {}
    if "demand" in models:
        from models.DemandModel import DemandModel
        results["demand"] = DemandModel().inference(history, explain=explain_demand)
    if "renewable" in models:
        results["renewable"] = RenewableModel().inference(history)
        demand_forecast = results.get("demand", external_demand)
        if demand_forecast is not None:
            results["renewable_assessment"] = rules.assess(results["renewable"], demand_forecast)
            rules.print_assessment(results["renewable_assessment"])
        else:
            print("Sufficiency not assessed: run --models demand renewable, or supply --demand-mw and --demand-time.")
    # Other model inference...
    return results


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not 0 < args.renewable_target_percent <= 100:
        parser.error("--renewable-target-percent must be greater than 0 and at most 100")
    if (args.demand_mw is None) != (args.demand_time is None):
        parser.error("--demand-mw and --demand-time must be supplied together")
    external_demand = None
    if args.demand_mw is not None:
        if "demand" in args.models or "renewable" not in args.models:
            parser.error("external demand is supported with --models renewable only")
        if not 0 < args.demand_mw < float("inf"):
            parser.error("--demand-mw must be finite and positive")
        try:
            timestamp = pd.Timestamp(args.demand_time)
            if pd.isna(timestamp) or timestamp.tzinfo is None:
                raise ValueError("Timezone required")
        except (ValueError, TypeError):
            parser.error("--demand-time must be a timezone-aware timestamp")
        external_demand = {"forecast_time": timestamp, "demand_mw": args.demand_mw,
                           "horizon_hours": 1}
    # Load in env variables
    load_dotenv()
    # Init dataloader
    loader = DataLoader(datetime.now(), datetime.now(), os.getenv("OPENELECTRICITY_API_KEY"))
    instance = loader.fetch_instance()
    # To break up the fetching output from the actual inference
    print('===============================================')
    run_inference(instance, args.models, args.renewable_target_percent,
                  external_demand, explain_demand=not args.skip_demand_explanation)


if __name__ == "__main__":
    main()
