"""
Erkennt das PSP besser als XGBoost, welche laufenden Fälle ihre Frist reißen werden? Die Annahme hinter einer PSP-Triage (PSP sagt, wo es sich
lohnt, XGBoost, was zu tun ist), geprüft auf dem echten BPIC17-Log, bevor etwas dafür gebaut wird.

Vorher festgelegt: Das PSP ist klar besser, wenn es bei tau = 37 Tagen auf dem echten Log eine um mindestens 0,02 höhere AUC erreicht als XGBoost
und das 95-%-KI dieser Differenz die 0 nicht einschließt.

Fair gemacht: Beide lernen aus denselben RCT-Daten (data/rct_bpic17_channel_delta50_AllePunkte), das echte Log hat keines von beiden gesehen.
XGBoost bekommt das Risiko direkt als Ziel, je Frist ein Modell, einmal mit so vielen Fällen wie das PSP (5.000), einmal mit allen. Alle sehen dieselben
Prefixe: je Fall ein zufälliger Zeitpunkt in den ersten 45 Tagen, der Prefix endet am letzten Event davor. Das PSP zieht je Prefix Suffixe, sein Risiko
ist der Anteil der Ziehungen über der Frist. Fälle, die die Frist schon gerissen haben, zählen dort nicht. Referenz: nur das Alter des Falls.

Aufruf aus process-simulation-engine/, Ergebnisse in prescriptive/scripts/results_risk/:
    .venv/bin/python -u ../prescriptive/scripts/psp_risk_check.py --real 4000 --rct 2000
"""
import argparse
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "prescriptive")]

from helpers.datasets import ACTIVITY, CASE, LIFECYCLE, RESOURCE, TIMESTAMP, add_paths, load_dataset  # noqa: E402

add_paths()
from preprocessing import preprocess  # noqa: E402
from prescriptive.estimators.psp_model import PSPModel  # noqa: E402
from prescriptive.estimators.xgb_model import Prefix, XGBModel  # noqa: E402

OUT = os.path.join(ROOT, "prescriptive", "scripts", "results_risk")
DATA = os.path.join(ROOT, "prescriptive", "data", "rct_bpic17_channel_delta50_AllePunkte", "log.pkl")
MODELS = {"PSP AllePunkte": "psp_model_durchlaufzeit_PSP_AllePunkte.pkl", "PSP lr1e4": "psp_model_durchlaufzeit_PSP_lr1e4.pkl"}
TAUS = (20.0, 30.0, 37.0, 45.0)
MAIN_TAU = 37.0
DAY = 86400.0
DRAWS, BATCH = 30, 64
FIRST_DAYS = 45.0
TRAIN_CUTS = 3  # so viele Prefixe je Trainingsfall
PARAMS = {"n_estimators": 2000, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.9, "min_child_weight": 20,
          "early_stopping_rounds": 50, "tree_method": "hist", "enable_categorical": True, "n_jobs": 16, "eval_metric": "logloss"}


def seconds(frame):
    return (pd.to_datetime(frame[TIMESTAMP]) - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()


def cases_of(log):
    """Je Fall die Zeilen, nach Zeit sortiert: (Fall, Zeilen, Sekunden)."""
    log = log.sort_values([CASE, TIMESTAMP], kind="stable").reset_index(drop=True)
    times, values = seconds(log), log[CASE].to_numpy()
    bounds = np.flatnonzero(np.r_[True, values[1:] != values[:-1], True])
    return log, times, [(values[a], a, b) for a, b in zip(bounds[:-1], bounds[1:])]


def cut_of(times, rng):
    """Letztes Event vor einem zufälligen Zeitpunkt in den ersten FIRST_DAYS Tagen des Falls: Zahl der Events im Prefix."""
    span = min(times[-1] - times[0], FIRST_DAYS * DAY)
    return int(np.searchsorted(times, times[0] + rng.random() * span, side="right"))


def features(model, log, times, spots):
    """Merkmale wie estimators/xgb_model.py (Prefix.row) je (Beginn, Zahl der Events)."""
    names = log[ACTIVITY].astype(str).to_numpy()
    cycles = log[LIFECYCLE].astype(str).to_numpy()
    values = log.reindex(columns=model.event_numeric).to_numpy(dtype=float)
    rows = []
    for begin, count in spots:
        prefix = Prefix(model._attributes(log.iloc[begin].to_dict()), model.event_numeric)
        for i in range(begin, begin + count):
            prefix.add(f"{names[i]}|{cycles[i]}", times[i], dict(zip(model.event_numeric, values[i])))
        rows.append(prefix.row(times[begin + count - 1]))
    return rows


def matrix(rows, columns=None, categories=None):
    """Tabelle der Merkmale; columns und categories aus dem Training."""
    X = pd.DataFrame(rows)
    if columns is None:
        columns = list(X.columns)
    X = X.reindex(columns=columns)
    for column in [c for c in X.columns if c.startswith("n:")]:
        X[column] = X[column].fillna(0.0)
    labels = [c for c in X.columns if X[c].dtype == object] if categories is None else list(categories)
    categories = {c: sorted(X[c].dropna().astype(str).unique()) for c in labels} if categories is None else categories
    for column, values in categories.items():
        X[column] = pd.Categorical(X[column].astype(str), categories=values)
    X.columns = [str(c).replace("[", "(").replace("]", ")").replace("<", "lt") for c in X.columns]
    return X, columns, categories


def train_xgb(X, ends, elapsed, tau, seed=42):
    from xgboost import XGBClassifier
    keep = elapsed < tau
    y = (ends[keep] > tau).astype(int)
    order = np.random.default_rng(seed).permutation(int(keep.sum()))
    stop = len(order) // 10
    data = X[keep]
    model = XGBClassifier(**PARAMS, random_state=seed)
    model.fit(data.iloc[order[stop:]], y[order[stop:]], eval_set=[(data.iloc[order[:stop]], y[order[:stop]])], verbose=False)
    return model


def psp_ends(model, log, spots):
    """Je Prefix die gezogenen Fallenden in Tagen seit Fallbeginn (Zeilen: Prefixe, Spalten: Ziehungen)."""
    ends, started = [], time.time()
    for first in range(0, len(spots), BATCH):
        chunk = spots[first:first + BATCH]
        encoded, reached = [], []
        for begin, count in chunk:
            prefix, static, mask, length, elapsed = model._encode(log.iloc[begin:begin + count].to_dict("records"))
            encoded.append((prefix, static, mask, length))
            reached.append(elapsed)
        for suffixes, elapsed in zip(model.draw_cases(encoded, DRAWS), reached):
            ends.append([max([elapsed] + [seconds_since for _, seconds_since, _ in suffix]) / DAY for suffix in suffixes])
        print(f"  {min(first + BATCH, len(spots))} von {len(spots)} Prefixen, {time.time() - started:.0f} s", flush=True)
    return np.array(ends)


def scores(y, risk):
    from sklearn.metrics import brier_score_loss, roc_auc_score
    top = np.argsort(-risk, kind="stable")[:max(1, len(y) // 10)]
    return {"AUC": roc_auc_score(y, risk), "Brier": brier_score_loss(y, np.clip(risk, 0, 1)), "Risiko im Mittel": float(np.mean(risk)), "Rate": float(np.mean(y)),
            "Top 10 % erfassen": float(y[top].sum() / max(y.sum(), 1))}


def difference(y, a, b, reps=2000, seed=3):
    """AUC(a) - AUC(b) mit 95-%-Bootstrap-KI (je Prefix ein Fall, also unabhängig)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        pick = rng.integers(len(y), size=len(y))
        if y[pick].min() != y[pick].max():
            values.append(roc_auc_score(y[pick], a[pick]) - roc_auc_score(y[pick], b[pick]))
    return roc_auc_score(y, a) - roc_auc_score(y, b), np.quantile(values, .025), np.quantile(values, .975)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=int, default=2500)
    parser.add_argument("--rct", type=int, default=1500)
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(11)
    models = {name: PSPModel.load(os.path.join(ROOT, "prescriptive", "models", file)) for name, file in MODELS.items()}
    for model in models.values():
        model.set_threads(16)
    held_out = set(next(iter(models.values())).test_cases)
    frame = XGBModel("bpic17")

    print("Lade die Logs ...", flush=True)
    raw, _, _ = load_dataset("bpic17")
    real, real_times, real_cases = cases_of(preprocess("bpic17", raw, verbose=False))
    del raw
    with open(DATA, "rb") as f:
        rct = pickle.load(f)
    rct, rct_times, rct_cases = cases_of(rct)
    print(f"Echt: {len(real_cases):,} Fälle; RCT: {len(rct_cases):,} Fälle, davon Testfälle {len(held_out):,}", flush=True)

    def sample(cases, times, n):
        chosen = [cases[i] for i in rng.choice(len(cases), min(n, len(cases)), replace=False)]
        spots, ends, elapsed = [], [], []
        for _, a, b in chosen:
            count = cut_of(times[a:b], rng)
            spots.append((a, count))
            ends.append((times[b - 1] - times[a]) / DAY)
            elapsed.append((times[a + count - 1] - times[a]) / DAY)
        return spots, np.array(ends), np.array(elapsed)

    sources = {"echtes Log": (real, real_times, *sample(real_cases, real_times, args.real)),
               "RCT-Testfälle": (rct, rct_times, *sample([c for c in rct_cases if c[0] in held_out], rct_times, args.rct))}

    # XGBoost: dieselben RCT-Trainingsfälle, 5.000 wie das PSP und alle
    train_cases = [c for c in rct_cases if c[0] not in held_out]
    print(f"Merkmale für das Training: {len(train_cases):,} Fälle ...", flush=True)
    spots, ends, elapsed, which = [], [], [], []
    for number, (_, a, b) in enumerate(train_cases):
        for _ in range(TRAIN_CUTS):
            count = cut_of(rct_times[a:b], rng)
            spots.append((a, count))
            ends.append((rct_times[b - 1] - rct_times[a]) / DAY)
            elapsed.append((rct_times[a + count - 1] - rct_times[a]) / DAY)
            which.append(number)
    X, columns, categories = matrix(features(frame, rct, rct_times, spots))
    ends, elapsed, which = np.array(ends), np.array(elapsed), np.array(which)
    subset = np.isin(which, rng.choice(len(train_cases), 5000, replace=False))
    print(f"{len(X):,} Trainingszeilen, {subset.sum():,} für die 5.000 Fälle, {X.shape[1]} Merkmale", flush=True)

    rows, details, trained = [], {}, {}
    for source, (log, times, test_spots, test_ends, test_elapsed) in sources.items():
        print(f"{source}: {len(test_spots)} Prefixe, Merkmale ...", flush=True)
        test_X = matrix(features(frame, log, times, test_spots), columns, categories)[0]
        drawn = {}
        for name, model in models.items():
            print(f"{source}: {name} zieht Suffixe ...", flush=True)
            drawn[name] = psp_ends(model, log, test_spots)
        for tau in TAUS:
            keep = test_elapsed < tau
            y = (test_ends[keep] > tau).astype(int)
            risks = {"nur Alter des Falls": test_elapsed[keep] / tau}
            for label, mask in (("XGBoost 5.000 Fälle", subset), ("XGBoost alle Fälle", np.ones(len(X), bool))):
                if (label, tau) not in trained:
                    print(f"tau {tau:.0f}: {label} ...", flush=True)
                    trained[(label, tau)] = train_xgb(X[mask], ends[mask], elapsed[mask], tau)
                risks[label] = trained[(label, tau)].predict_proba(test_X[keep])[:, 1]
            for name in models:
                risks[name] = (drawn[name][keep] > tau).mean(axis=1)
                risks[name + " (nur Mittelwert)"] = drawn[name][keep].mean(axis=1) / tau
            details[(source, tau)] = (y, risks, test_elapsed[keep])
            for label, risk in risks.items():
                rows.append({"Daten": source, "Frist": tau, "Modell": label, "Prefixe": len(y), **scores(y, risk)})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "ergebnis.csv"), index=False)
    with pd.option_context("display.width", 180, "display.max_columns", 20, "display.max_rows", 120, "display.max_colwidth", 34):
        print(table.round(3).to_string(index=False), flush=True)

    print(f"\nHürde: Frist {MAIN_TAU:.0f} Tage, echtes Log, AUC-Differenz mindestens 0,02 und KI ohne 0", flush=True)
    y, risks, age = details[("echtes Log", MAIN_TAU)]
    gates = []
    for name in models:
        for rival in ("XGBoost 5.000 Fälle", "XGBoost alle Fälle", "nur Alter des Falls"):
            value, low, high = difference(y, risks[name], risks[rival])
            passed = value >= 0.02 and low > 0
            gates.append({"PSP": name, "gegen": rival, "Differenz": value, "KI unten": low, "KI oben": high, "bestanden": passed})
    gates = pd.DataFrame(gates)
    gates.to_csv(os.path.join(OUT, "huerde.csv"), index=False)
    print(gates.round(4).to_string(index=False), flush=True)

    from sklearn.metrics import roc_auc_score
    print("\nAUC nach Alter des Falls am Prefix (Frist 37 Tage, echtes Log):", flush=True)
    bins = [(0, 1), (1, 7), (7, 14), (14, 37)]
    stage = []
    for low, high in bins:
        part = (age >= low) & (age < high)
        if part.sum() > 30 and 0 < y[part].mean() < 1:
            stage.append({"Alter (Tage)": f"{low}-{high}", "Prefixe": int(part.sum()), "Rate": float(y[part].mean()),
                          **{label: roc_auc_score(y[part], risk[part]) for label, risk in risks.items() if "Mittelwert" not in label}})
    stage = pd.DataFrame(stage)
    stage.to_csv(os.path.join(OUT, "nach_alter.csv"), index=False)
    with pd.option_context("display.width", 180, "display.max_columns", 20):
        print(stage.round(3).to_string(index=False), flush=True)
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
