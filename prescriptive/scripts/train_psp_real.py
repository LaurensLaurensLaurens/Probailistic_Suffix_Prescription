"""
Das PSP (estimators/psp_model.py) auf dem echten Event-Log statt auf den RCT-Daten des Simulators trainieren, sonst wie
data/bpic17_psp_w3_10k.sh: dieselben 12 Wahlpunkte (8 Mitarbeiter, 4 Control Flow; hier fest gesetzt, nicht aus dem
Katalog gelesen, der während anderer Läufe umgestellt sein kann), train_cases Fälle, Epochen, Lernrate, Gewicht der
Wahlpunkte, GPU (TRAIN device auto).

Geteilt wird wie in helpers/training.train_estimators: TEST_SHARE der Fälle des echten Logs bleiben als Testfälle draußen
(test_cases, gleiche Saat), aus den übrigen zieht PSPModel.fit train_cases Fälle und Validierungsfälle wie im Original.
Eigener Checkpoint-Ordner und eigener Modellname, damit dieser Lauf parallel zu den Läufen der Pipeline laufen kann
(models/psp_model.pkl, models/psp_training und der Katalog bleiben unberührt).

Mit --catalog-points die Wahlpunkte des Katalogs (z. B. eine Kopie über PSP_CATALOG, wie data/bpic17_psp_real_neue_punkte.sh).
Mit --remaining-head lernt ein Restzeit-Kopf auf dem Encoder mit (psp_model REMAINING_HEAD, Gewicht --remaining-weight).
Mit --gap-classes sagt der Decoder je Schritt auch die Klasse des Abstands zum vorigen Event voraus (psp_model GAP_CLASS).

Aufruf (im Ordner process-simulation-engine):
    .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag real_w3_10k --cases 10000 --epochs 100 \
        --learning-rate 1e-5 --point-weight 3
Ergebnis: models/psp_model_durchlaufzeit_<tag>.pkl, models/psp_training_<tag>/ (last.pt jede Epoche).
"""
import argparse
import copy
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
PRESCRIPTIVE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PRESCRIPTIVE)

from helpers.datasets import CASE, add_paths

# Die Wahlpunkte von bpic17_psp_w5.sh / bpic17_psp_w3_10k.sh (lauf.mitarbeiterwahl und lauf.control_flow)
EMPLOYEES = ["Mitarbeiter: W_Call incomplete files", "Mitarbeiter: W_Complete application", "Mitarbeiter: W_Handle leads",
             "Mitarbeiter: W_Validate application", "Mitarbeiter: W_Call after offers (resume)", "Mitarbeiter: W_Call incomplete files (resume)",
             "Mitarbeiter: W_Complete application (resume)", "Mitarbeiter: W_Validate application (resume)"]
CONTROL = ["handle_leads", "annehmen_nach_vervollstaendigung", "angebot_nachlegen", "neues_angebot_beim_nachfassen"]


def fixed_points(dataset):
    """Die Wahlpunkte mit EMPLOYEES und CONTROL, unabhängig vom Stand von catalog.yaml (Katalog im Speicher ersetzt)."""
    import interventions
    fixed = copy.deepcopy(interventions.catalog())
    lauf = fixed[dataset].setdefault("lauf", {})
    lauf["mitarbeiterwahl"], lauf["control_flow"] = list(EMPLOYEES), list(CONTROL)
    interventions.catalog = lambda: fixed
    return interventions.choice_points(dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="bpic17")
    parser.add_argument("--tag", required=True, help="Name: models/psp_model_durchlaufzeit_<tag>.pkl, models/psp_training_<tag>")
    parser.add_argument("--cases", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--point-weight", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suffix-size", type=int, default=None, help="Schritte des Decoders im Training (Standard wie im Original: 5)")
    parser.add_argument("--log-gaps", action="store_true", help="Abstände zum vorigen Event log-normal statt normal (psp_model properties)")
    parser.add_argument("--time-from", choices=("elapsed", "gaps"), default=None,
                        help="Durchlaufzeit aus der gezogenen Zeit seit Fallbeginn (Original) oder der Summe der gezogenen Abstände")
    parser.add_argument("--statics-without-future", action="store_true",
                        help="nur die Fallattribute, die zu Fallbeginn feststehen (psp_model.STATICS_WITHOUT_FUTURE), statt der des Originals")
    parser.add_argument("--remaining-head", action="store_true",
                        help="Restzeit-Kopf auf dem Zustand des Encoders mittrainieren (psp_model REMAINING_HEAD, TRAIN remaining_head)")
    parser.add_argument("--gap-classes", action="store_true",
                        help="Abstandsklasse als zusätzliche kategoriale Ausgabe des Decoders, Abstand aus den echten Abständen der Klasse (psp_model GAP_CLASS)")
    parser.add_argument("--remaining-weight", type=float, default=None, help="Gewicht des Restzeit-Verlusts (Standard REMAINING_HEAD weight)")
    parser.add_argument("--catalog-points", action="store_true",
                        help="die Wahlpunkte des Katalogs (interventions.choice_points, auch über PSP_CATALOG) statt EMPLOYEES und CONTROL")
    args = parser.parse_args()
    add_paths()

    from helpers.datasets import load_dataset
    from helpers.training import MODELS, test_cases
    from preprocessing import preprocess
    from prescriptive.estimators.psp_model import PSPModel

    path = os.path.join(MODELS, f"psp_model_durchlaufzeit_{args.tag}.pkl")
    checkpoints = os.path.join(MODELS, f"psp_training_{args.tag}")
    for taken in (path, checkpoints):
        if os.path.exists(taken):
            sys.exit(f"{taken} gibt es schon, Abbruch")

    print(f"######## Echter Log {args.dataset}: Laden und Preprocessing ({time.strftime('%H:%M:%S')})", flush=True)
    raw, _, _ = load_dataset(args.dataset)
    log = preprocess(args.dataset, raw)
    del raw
    held_out = test_cases(log)
    train = log[~log[CASE].astype(str).isin(held_out)]
    print(f"Echter Log: {log[CASE].nunique():,} Fälle, davon {len(held_out):,} Testfälle; Training zieht {args.cases:,} aus "
          f"{train[CASE].nunique():,} Fällen", flush=True)

    if args.catalog_points:
        import interventions
        points = interventions.choice_points(args.dataset)
        print(f"Katalog: {interventions.CANDIDATES_FILE}", flush=True)
    else:
        points = fixed_points(args.dataset)
    print(f"Wahlpunkte ({len(points)}): {', '.join(point.name for point in points)}", flush=True)
    params = {"train_cases": args.cases, "epochs": args.epochs, "learning_rate": args.learning_rate,
              "point_weight": args.point_weight, "checkpoints": checkpoints}
    if args.suffix_size:
        params["suffix_size"] = args.suffix_size
    if args.log_gaps:
        params["log_gaps"] = True
    if args.time_from:
        params["time_from"] = args.time_from
    if args.remaining_head:
        params["remaining_head"] = True
    if args.gap_classes:
        params["gap_classes"] = True
    if args.remaining_weight is not None:
        params["remaining_weight"] = args.remaining_weight
    if args.statics_without_future:
        from prescriptive.estimators.psp_model import STATICS_WITHOUT_FUTURE
        params["statics"] = STATICS_WITHOUT_FUTURE
    print(f"######## Training: {params} ({time.strftime('%H:%M:%S')})", flush=True)
    model = PSPModel(args.dataset, points, seed=args.seed, params=params).fit(train)
    model.test_cases, model.trained_on = held_out, {"echter Log": args.dataset, "outcome": "durchlaufzeit"}
    model.save(path)
    print(f"Zusammenfassung: {model.summary}", flush=True)
    print(f"######## Gespeichert in {path} ({time.strftime('%H:%M:%S')})", flush=True)


if __name__ == "__main__":
    main()
