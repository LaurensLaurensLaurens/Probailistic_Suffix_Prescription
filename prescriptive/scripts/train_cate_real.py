"""
Die kausalen Schätzer nach Bozorgi et al. (2023) (estimators/cate_model.py: S-Learner, X-Learner, Causal Forest) auf dem
echten Event-Log trainieren, als Vergleich zum PSP: wie scripts/train_xgb_real.py und scripts/train_psp_real.py bleiben
TEST_SHARE der Fälle als Testfälle draußen (test_cases, gleiche Saat), aus den übrigen werden --cases Fälle gezogen
(0: alle, wie beim PSP der Stufen 2 und 3).

Die Wahlpunkte kommen aus dem Katalog (interventions.choice_points; eine Kopie über PSP_CATALOG, z. B.
data/catalog_neue_punkte.yaml). Nach dem Training je Wahlpunkt auf den Testfällen: der geschätzte Effekt (Mittel und
Streuung über die Entscheidungen), wie oft er das Vorzeichen des AIPW-Effekts hat (scripts/psp_diagnose.AIPW, soweit
dort vorhanden) und wie oft die Policy die Aktion wählen würde.

Aufruf (im Ordner process-simulation-engine):
    PSP_CATALOG=... .venv/bin/python -u ../prescriptive/scripts/train_cate_real.py --tag real_alle --learner s x cf --cases 0
Ergebnis: models/cate_model_<learner>_durchlaufzeit_<tag>.pkl, results_100k/cate_<tag>/effekte_testfaelle.csv

ERM statt Erwartungswert: zusätzlich --erm-gamma 0.1. Trainiert CATEERMModel auf exp(γ · Durchlaufzeit - b).
Ergebnis: models/cate_erm_model_<learner>_gamma0.1_durchlaufzeit_<tag>.pkl,
results_100k/cate_erm_gamma0.1_<tag>/effekte_testfaelle.csv. Effekte in skaliertem exponentiellem Verlust,
keine Tage und kein Vergleich mit dem AIPW-Effekt auf die mittlere Durchlaufzeit. Kleinerer Effekt ist besser.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
PRESCRIPTIVE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PRESCRIPTIVE)
sys.path.insert(0, os.path.join(PRESCRIPTIVE, "scripts"))

import numpy as np
import pandas as pd

from helpers.datasets import CASE, add_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="bpic17")
    parser.add_argument("--tag", required=True, help="Name: models/cate_model_<learner>_durchlaufzeit_<tag>.pkl")
    parser.add_argument("--learner", nargs="+", default=["s", "x", "cf"], choices=("s", "x", "cf"))
    parser.add_argument("--cases", type=int, default=0, help="Trainingsfälle aus dem echten Log (ohne Testfälle); 0: alle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--erm-gamma", type=float, default=None,
                        help="ERM-Modelle statt EV trainieren: Risikoaversion γ in 1/Tag (z. B. 0.1, wie PSP-ERM); neues Training je γ")
    args = parser.parse_args()
    erm = args.erm_gamma is not None
    if erm and (not np.isfinite(args.erm_gamma) or args.erm_gamma <= 0):
        parser.error("--erm-gamma muss endlich und größer 0 sein")
    add_paths()

    import interventions
    from helpers.datasets import load_dataset
    from helpers.training import MODELS, sample_cases, test_cases
    from preprocessing import outcome_spec, preprocess
    from prescriptive.estimators.cate_model import LEARNERS, CATEModel
    if erm:
        from prescriptive.estimators.cate_erm_model import CATEERMModel
        if outcome_spec(args.dataset)["type"] != "negative_cycle_days":
            parser.error("CATE-ERM braucht das Outcome negative_cycle_days (γ in 1/Tag)")
    else:
        from psp_diagnose import AIPW

    model_class = CATEERMModel if erm else CATEModel
    settings = {"gamma": args.erm_gamma} if erm else {}
    paths = {learner: os.path.join(MODELS, f"cate_erm_model_{learner}_gamma{args.erm_gamma}_durchlaufzeit_{args.tag}.pkl" if erm else
                                  f"cate_model_{learner}_durchlaufzeit_{args.tag}.pkl") for learner in args.learner}
    for path in paths.values():
        if os.path.exists(path):
            sys.exit(f"{path} gibt es schon, Abbruch")
    print(f"Katalog: {interventions.CANDIDATES_FILE}", flush=True)

    print(f"######## Echter Log {args.dataset}: Laden und Preprocessing ({time.strftime('%H:%M:%S')})", flush=True)
    raw, _, _ = load_dataset(args.dataset)
    log = preprocess(args.dataset, raw)
    del raw
    held_out = test_cases(log)
    is_test = log[CASE].astype(str).isin(held_out)
    train, test = log[~is_test], log[is_test]
    print(f"Echter Log: {log[CASE].nunique():,} Fälle, davon {len(held_out):,} Testfälle; Training auf "
          f"{'allen' if not args.cases else f'{args.cases:,} von'} {train[CASE].nunique():,} Fällen", flush=True)
    if args.cases:
        train = sample_cases(train, args.cases, args.seed)

    points = interventions.choice_points(args.dataset)
    print(f"Wahlpunkte ({len(points)}): {', '.join(point.name for point in points)}", flush=True)
    first = model_class(args.dataset, points, seed=args.seed, learner=args.learner[0], **settings)
    train_samples, test_samples = first.samples(train), first.samples(test)
    rows = []
    for learner, path in paths.items():
        print(f"######## {LEARNERS[learner]}{' ERM γ=' + str(args.erm_gamma) if erm else ''}: Training ({time.strftime('%H:%M:%S')})", flush=True)
        model = model_class(args.dataset, points, seed=args.seed, learner=learner, **settings).fit(train, samples=train_samples)
        model.test_cases = held_out
        model.trained_on = {"echter Log": args.dataset, "outcome": "durchlaufzeit", "cases": int(train[CASE].nunique()),
                            "points": [point.name for point in points], "learner": learner}
        if erm:
            model.trained_on.update(objective=model.objective, gamma=model.gamma)
        model.save(path)
        for name, row in model.summary.items():
            print(f"  {name}: {row}", flush=True)
        missing = [point.name for point in points if not model.knows(point)]
        if missing:
            print(f"Ohne Modell (dort wählt die Policy nicht): {', '.join(missing)}", flush=True)
        for point in points:
            if not model.knows(point) or not test_samples.get(point.name):
                continue
            _, features, options = zip(*test_samples[point.name])
            effects = model.effects(point, list(features))
            if erm:
                values = np.column_stack([np.zeros(len(effects)), effects])
                best = values == values.min(axis=1, keepdims=True)
                shares = best / best.sum(axis=1, keepdims=True)  # wie die Policy: bei Gleichstand gleichverteilt
            for k in model.models[point.name]:
                effect = effects[:, k - 1]
                if erm:
                    details = {"ERM γ (1/Tag)": model.gamma, "ERM Log-Skalierung": model.log_scales[point.name],
                               "Effekt Mittel (skalierter exp-Verlust)": float(effect.mean()),
                               "Effekt Streuung (skalierter exp-Verlust)": float(effect.std()),
                               "Anteil Aktion nach Policy": float(shares[:, k].mean())}
                else:
                    aipw = AIPW.get(point.name) if len(point.options) == 2 else None
                    details = {"Effekt laut AIPW (Tage)": aipw, "Effekt Mittel (Tage)": float(effect.mean()),
                               "Effekt Streuung (Tage)": float(effect.std()),
                               "Vorzeichen wie AIPW": float(np.mean(np.sign(effect) == np.sign(aipw))) if aipw else np.nan,
                               "Anteil Aktion nach Policy": float(np.mean(effect > 0))}
                rows.append({"Schätzer": LEARNERS[learner], "Wahlpunkt": point.name, "Aktion": point.options[k], "Referenz": point.options[0],
                             "Entscheidungen (Testfälle)": len(effect), "Anteil Aktion im Log": float(np.mean(np.array(options) == point.options[k])),
                             "Effekt P10": float(np.quantile(effect, 0.1)), "Effekt P90": float(np.quantile(effect, 0.9)),
                             **details})
        print(f"######## Gespeichert in {path} ({time.strftime('%H:%M:%S')})", flush=True)
    table = pd.DataFrame(rows)
    out = os.path.join(PRESCRIPTIVE, "scripts", "results_100k", f"cate_erm_gamma{args.erm_gamma}_{args.tag}" if erm else f"cate_{args.tag}")
    os.makedirs(out, exist_ok=True)
    table.to_csv(os.path.join(out, "effekte_testfaelle.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.drop(columns=["Aktion", "Referenz"], errors="ignore").round(3).to_string(index=False), flush=True)
    print(f"######## Fertig ({time.strftime('%H:%M:%S')}); Tabelle in {out}", flush=True)


if __name__ == "__main__":
    main()
