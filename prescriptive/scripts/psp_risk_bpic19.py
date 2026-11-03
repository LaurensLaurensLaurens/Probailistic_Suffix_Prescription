"""
Derselbe Vergleich wie prescriptive/scripts/psp_risk_check.py, auf BPIC19: Erkennt das PSP besser als XGBoost, welche laufenden Fälle ihre Frist reißen?
Grund für den zweiten Versuch: Auf BPIC17 zieht das PSP bis zu 50 Schritte weit und verliert klar; BPIC19 hat im Median 5 Events je Fall, so viele wie
die Schritte des Decoders im Training.

Vorher festgelegt (wie bei BPIC17): klar besser heißt, bei der Hauptfrist (P90 des echten Logs: 133 Tage) mindestens 0,02 höhere AUC als XGBoost mit denselben
Trainingsfällen und ein 95-%-KI der Differenz ohne die 0, auf Bestellungen, die kein Modell gesehen hat.

Anders als bei BPIC17, begründet: Beide lernen direkt aus dem echten Log, nicht aus einem RCT (das Ergebnis hängt so nicht am Simulator). Getrennt wird nach
Bestellung (case:Purchasing Document), nicht nach Fall: Positionen einer Bestellung laufen meist gleichzeitig, sonst sickerte die Antwort durch. Fälle mit mehr als
MAX_EVENTS Events fehlen bei allen Modellen (0,9 % der Fälle, das feste Fenster des PSP). Frist und Ziel wie das Outcome: Tage bis zum letzten Clear Invoice, ohne ein
solches bis zum letzten Event.

Aufruf aus process-simulation-engine/, erst Training, dann Vergleich; Ergebnisse in prescriptive/scripts/results_risk_bpic19/:
    .venv/bin/python -u ../prescriptive/scripts/psp_risk_bpic19.py --train
    .venv/bin/python -u ../prescriptive/scripts/psp_risk_bpic19.py
"""
import argparse
import contextlib
import io
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import psp_risk_check as base  # noqa: E402
from psp_risk_check import ACTIVITY, CASE, DAY, PSPModel, ROOT, XGBModel, cases_of, difference, features, load_dataset, matrix, preprocess, scores, train_xgb  # noqa: E402

OUT = os.path.join(ROOT, "prescriptive", "scripts", "results_risk_bpic19")
MODEL = os.path.join(ROOT, "prescriptive", "models", "psp_model_bpic19_risk.pkl")
CHECKPOINTS = os.path.join(ROOT, "prescriptive", "models", "psp_training_bpic19_risk")
MAX_EVENTS, HELD_OUT, TRAIN_CUTS = 30, 0.2, 3
TAUS, MAIN_TAU = (77.0, 108.0, 133.0, 160.0), 133.0
base.FIRST_DAYS = 200.0  # die Zeitpunkte der Prefixe: die ersten 200 Tage des Falls


def prepared(cases_wanted, seed=7):
    """Das Log (Fälle bis MAX_EVENTS Events), Trainings- und Testfälle nach Bestellung getrennt und die Fälle, auf denen das PSP lernt."""
    with contextlib.redirect_stdout(io.StringIO()):
        raw, _, _ = load_dataset("bpic19")
        log = preprocess("bpic19", raw, verbose=False)
    sizes = log.groupby(CASE)[CASE].transform("size")
    log = log[sizes <= MAX_EVENTS].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    documents = np.sort(log["case:Purchasing Document"].astype(str).unique())
    held = set(rng.choice(documents, int(HELD_OUT * len(documents)), replace=False))
    of_case = log.groupby(CASE)["case:Purchasing Document"].first().astype(str)
    test = set(of_case.index[of_case.isin(held)])
    train = sorted(set(of_case.index) - test)
    chosen = set(rng.choice(train, min(cases_wanted, len(train)), replace=False))
    return log, set(train), test, chosen


def target(activities, seconds_since, reached):
    """Tage bis zum letzten Clear Invoice eines Verlaufs, ohne ein solches bis zum letzten Event (wie das Outcome)."""
    hits = [s for a, s in zip(activities, seconds_since) if str(a).startswith("Clear Invoice")]
    return (hits[-1] if hits else reached) / DAY


def psp_targets(model, log, times, spots, draws=30, batch=64):
    names = log[ACTIVITY].astype(str).to_numpy()
    result, started = [], time.time()
    for first in range(0, len(spots), batch):
        chunk = spots[first:first + batch]
        encoded, reached = [], []
        for begin, count in chunk:
            prefix, static, mask, length, elapsed = model._encode(log.iloc[begin:begin + count].to_dict("records"))
            encoded.append((prefix, static, mask, length))
            reached.append(elapsed)
        for (begin, count), suffixes, elapsed in zip(chunk, model.draw_cases(encoded, draws), reached):
            known = [(names[i], times[i] - times[begin]) for i in range(begin, begin + count)]
            row = []
            for suffix in suffixes:
                labels = [a for a, _ in known] + [label.split("|")[0] for label, _, _ in suffix]
                moments = [s for _, s in known] + [seconds for _, seconds, _ in suffix]
                row.append(target(labels, moments, max(moments)))
            result.append(row)
        print(f"  {min(first + batch, len(spots))} von {len(spots)}, {time.time() - started:.0f} s", flush=True)
    return np.array(result)


def fit(args):
    log, train, test, chosen = prepared(args.cases)
    print(f"Log: {log[CASE].nunique():,} Fälle, Training {len(train):,}, Test {len(test):,}, PSP lernt auf {len(chosen):,}", flush=True)
    part = log[log[CASE].isin(chosen)]
    params = {"train_cases": None, "epochs": args.epochs, "learning_rate": args.lr, "threads": 16, "checkpoints": CHECKPOINTS, "lifecycle_tokens": True, "suffix_size": 5}
    started = time.time()
    model = PSPModel("bpic19", points=[], seed=42, params=params).fit(part)
    print(f"PSP trainiert in {(time.time() - started) / 60:.1f} min", model.summary, flush=True)
    model.test_cases = test
    model.save(args.model or MODEL)


def compare(args):
    os.makedirs(OUT, exist_ok=True)
    log, train, test, chosen = prepared(args.cases)
    log, times, cases = cases_of(log)
    model = PSPModel.load(args.model or MODEL)
    model.set_threads(16)
    frame = XGBModel("bpic19")
    outcome = -log.groupby(CASE)["outcome"].first()
    rng = np.random.default_rng(11)

    def spot(a, b):
        count = base.cut_of(times[a:b], rng)
        return (a, count), (times[a + count - 1] - times[a]) / DAY

    test_cases = [c for c in cases if c[0] in test]
    picks = rng.choice(len(test_cases), min(args.test, len(test_cases)), replace=False)
    test_spots, test_elapsed, test_y = [], [], []
    for i in picks:
        name, a, b = test_cases[i]
        one, age = spot(a, b)
        test_spots.append(one); test_elapsed.append(age); test_y.append(outcome[name])
    test_elapsed, test_y = np.array(test_elapsed), np.array(test_y)
    print(f"{len(test_spots)} Prefixe, PSP zieht ...", flush=True)
    drawn = psp_targets(model, log, times, test_spots)
    test_X_rows = features(frame, log, times, test_spots)

    print("Merkmale für das Training ...", flush=True)
    spots, ends, ages, inside = [], [], [], []
    for name, a, b in cases:
        if name in train:
            for _ in range(TRAIN_CUTS):
                one, age = spot(a, b)
                spots.append(one); ends.append(outcome[name]); ages.append(age); inside.append(name in chosen)
    X, columns, categories = matrix(features(frame, log, times, spots))
    test_X = matrix(test_X_rows, columns, categories)[0]
    ends, ages, inside = np.array(ends), np.array(ages), np.array(inside)
    same = f"XGBoost {len(chosen):,} Fälle"
    print(f"{len(X):,} Trainingszeilen, {inside.sum():,} für {same}", flush=True)

    rows, details = [], {}
    for tau in TAUS:
        keep = test_elapsed < tau
        y = (test_y[keep] > tau).astype(int)
        risks = {"nur Alter des Falls": test_elapsed[keep] / tau}
        for label, mask in ((same, inside), ("XGBoost alle Fälle", np.ones(len(X), bool))):
            print(f"tau {tau:.0f}: {label} ...", flush=True)
            risks[label] = train_xgb(X[mask], ends[mask], ages[mask], tau).predict_proba(test_X[keep])[:, 1]
        risks["PSP"] = (drawn[keep] > tau).mean(axis=1)
        risks["PSP (nur Mittelwert)"] = drawn[keep].mean(axis=1) / tau
        details[tau] = (y, risks, test_elapsed[keep])
        rows += [{"Frist": tau, "Modell": label, "Prefixe": len(y), **scores(y, risk)} for label, risk in risks.items()]
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "ergebnis.csv"), index=False)
    with pd.option_context("display.width", 170, "display.max_columns", 20, "display.max_rows", 60, "display.max_colwidth", 30):
        print(table.round(3).to_string(index=False), flush=True)

    y, risks, age = details[MAIN_TAU]
    print(f"\nHürde: Frist {MAIN_TAU:.0f} Tage, AUC-Differenz mindestens 0,02 und KI ohne 0", flush=True)
    gates = []
    for name in ("PSP", "PSP (nur Mittelwert)"):
        for rival in (same, "XGBoost alle Fälle", "nur Alter des Falls"):
            value, low, high = difference(y, risks[name], risks[rival])
            gates.append({"PSP": name, "gegen": rival, "Differenz": value, "KI unten": low, "KI oben": high, "bestanden": value >= 0.02 and low > 0})
    gates = pd.DataFrame(gates)
    gates.to_csv(os.path.join(OUT, "huerde.csv"), index=False)
    print(gates.round(4).to_string(index=False), flush=True)

    from sklearn.metrics import roc_auc_score
    stage = []
    for low, high in ((0, 7), (7, 30), (30, 60), (60, 133)):
        part = (age >= low) & (age < high)
        if part.sum() > 30 and 0 < y[part].mean() < 1:
            stage.append({"Alter (Tage)": f"{low}-{high}", "Prefixe": int(part.sum()), "verspätet": int(y[part].sum()), **{k: roc_auc_score(y[part], r[part]) for k, r in risks.items()}})
    stage = pd.DataFrame(stage)
    stage.to_csv(os.path.join(OUT, "nach_alter.csv"), index=False)
    with pd.option_context("display.width", 170, "display.max_columns", 20):
        print("\nAUC nach Alter des Falls:", flush=True)
        print(stage.round(3).to_string(index=False), flush=True)
    print("FERTIG", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--cases", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--test", type=int, default=4000)
    parser.add_argument("--model", default=None)
    arguments = parser.parse_args()
    (fit if arguments.train else compare)(arguments)
