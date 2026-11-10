"""
XGBoost (estimators/xgb_model.py) auf dem echten Event-Log statt auf den RCT-Daten des Simulators trainieren, wie
scripts/train_psp_real.py für das PSP: TEST_SHARE der Fälle des echten Logs bleiben als Testfälle draußen (test_cases,
gleiche Saat), aus den übrigen werden --cases Fälle gezogen (helpers/training.sample_cases, --seed).

Die Wahlpunkte kommen aus dem Katalog (interventions.choice_points); für einen Lauf neben anderen eine Kopie über die
Umgebungsvariable PSP_CATALOG (data/bpic17_xgb_real.sh). XGBoost erkennt die Punkte am Namen: das Modell passt zu
einer Online-Auswertung mit denselben Punkten.

Aufruf (im Ordner process-simulation-engine):
    PSP_CATALOG=... .venv/bin/python -u ../prescriptive/scripts/train_xgb_real.py --tag real_10k --cases 10000
Ergebnis: models/xgb_model_durchlaufzeit_<tag>.pkl
"""
import argparse
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
PRESCRIPTIVE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PRESCRIPTIVE)

from helpers.datasets import CASE, add_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="bpic17")
    parser.add_argument("--tag", required=True, help="Name: models/xgb_model_durchlaufzeit_<tag>.pkl")
    parser.add_argument("--cases", type=int, default=10000, help="Trainingsfälle aus dem echten Log (ohne Testfälle)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    add_paths()

    import interventions
    from helpers.datasets import load_dataset
    from helpers.training import MODELS, sample_cases, test_cases
    from preprocessing import preprocess
    from prescriptive.estimators.xgb_model import XGBModel

    path = os.path.join(MODELS, f"xgb_model_durchlaufzeit_{args.tag}.pkl")
    if os.path.exists(path):
        sys.exit(f"{path} gibt es schon, Abbruch")
    print(f"Katalog: {interventions.CANDIDATES_FILE}", flush=True)

    print(f"######## Echter Log {args.dataset}: Laden und Preprocessing ({time.strftime('%H:%M:%S')})", flush=True)
    raw, _, _ = load_dataset(args.dataset)
    log = preprocess(args.dataset, raw)
    del raw
    held_out = test_cases(log)
    train = log[~log[CASE].astype(str).isin(held_out)]
    print(f"Echter Log: {log[CASE].nunique():,} Fälle, davon {len(held_out):,} Testfälle; Training auf {args.cases:,} von "
          f"{train[CASE].nunique():,} Fällen", flush=True)
    train = sample_cases(train, args.cases, args.seed)

    points = interventions.choice_points(args.dataset)
    print(f"Wahlpunkte ({len(points)}): {', '.join(point.name for point in points)}", flush=True)
    print(f"######## Training ({time.strftime('%H:%M:%S')})", flush=True)
    model = XGBModel(args.dataset, points, seed=args.seed).fit(train)
    model.test_cases = held_out
    model.trained_on = {"echter Log": args.dataset, "outcome": "durchlaufzeit", "cases": int(train[CASE].nunique()), "points": [point.name for point in points]}
    model.save(path)
    for name, row in model.summary.items():
        print(f"  {name}: {row}", flush=True)
    missing = [point.name for point in points if not model.knows(point)]
    if missing:
        print(f"Ohne Proben (dort wählt die Policy nicht): {', '.join(missing)}", flush=True)
    print(f"######## Gespeichert in {path} ({time.strftime('%H:%M:%S')})", flush=True)


if __name__ == "__main__":
    main()
