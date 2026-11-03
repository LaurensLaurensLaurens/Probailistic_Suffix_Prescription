"""
XGBoost der Werkstatt (repair) mit den Stundensätzen der Option als Eingabe (estimators/xgb_model.py, params option_rates), auf dem
Outcome kosten bzw. kosten_b (prescriptive/costs/repair). Dieselben Trainingsfälle wie scripts/main.py --steps train
--train-cases 6250 (helpers/training: sample_cases mit TRAIN_SEED, ohne test_cases), aus den vorhandenen RCT-Daten.
Gespeichert unter eigenem Namen (--out), das XGBoost der Pipeline (models/xgb_model_repair_kosten.pkl) bleibt.

Aufruf aus process-simulation-engine/:
    .venv/bin/python -u ../prescriptive/scripts/train_xgb_rates.py --rates jetzt --out ../prescriptive/models/xgb_model_repair_kosten_satz.pkl
Auswerten mit main.py --xgb-model und repair_truth.py --xgb-model.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "process-simulation-engine"), os.path.join(ROOT, "prescriptive"), ROOT]
RCT = os.path.join(ROOT, "prescriptive", "data", "rct_repair_nacharbeit_delta50", "log")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome", default="kosten", help="outcome_optionen von repair, deren Tarif die Labels und die Sätze im Training haben")
    parser.add_argument("--rates", choices=("jetzt", "jetzt_48h"), required=True)
    parser.add_argument("--cases", type=int, default=6250)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.environ["PRESCRIPTIVE_PROFILE"] = "repair"  # vor dem Import von enriched_log (PROFILE)
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    from prescriptive import profiles
    profiles.set_outcome("repair", args.outcome)
    from helpers.rct import load_data
    from helpers.training import TRAIN_SEED, sample_cases, test_cases
    from prescriptive.enriched_log import CASE, with_current_costs
    from prescriptive.estimators.xgb_model import XGBModel
    data = sample_cases(with_current_costs(load_data(RCT)), args.cases, TRAIN_SEED)
    held_out = test_cases(data)
    print(f"xgb mit Stundensätzen ({args.rates}): Training auf {data[CASE].nunique() - len(held_out):,} Fällen, {len(held_out):,} Testfälle ...", flush=True)
    model = XGBModel("repair", seed=TRAIN_SEED, params={"option_rates": args.rates}).fit(data[~data[CASE].isin(held_out)])
    model.test_cases = held_out
    model.trained_on = {"rct": os.path.basename(os.path.dirname(RCT)), "outcome": args.outcome, "cases": int(data[CASE].nunique()),
                        "points": list(model.points), "option_rates": args.rates}
    model.save(args.out)
    for point, row in model.summary.items():
        print(point, {k: round(v, 1) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    print(f"gespeichert in {args.out}", flush=True)


if __name__ == "__main__":
    main()
