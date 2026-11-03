"""
Training und Kennzahlen der Modelle (scripts/main.py, Schritte train und models): XGBoost (estimators/xgb_model.py),
LSTM (estimators/lstm_model.py) und PSP (estimators/psp_model.py), alle mit denselben Stellen der Proben und derselben
Schnittstelle (samples, fit, predict, values). Das PSP schätzt den Outcome über gezogene Suffixe; es bekommt zusätzlich
Kennzahlen der Suffixe und wird auf höchstens max_eval_samples Proben je Wahlpunkt und Daten ausgewertet.

- train_estimators: die RCT-Daten (helpers/rct.py) nach Fällen in Training und Test teilen (TEST_SHARE, TEST_SEED), auf
  dem Trainingsteil trainieren und in prescriptive/models/<modell>_model<outcome>.pkl speichern (model_path; mit
  --outcome bpic17=gewinn z. B. xgb_model_gewinn.pkl, sonst xgb_model.pkl). Die Testfälle und worauf trainiert
  wurde stehen im Modell (test_cases, trained_on). Vorhanden und auf denselben Daten trainiert: geladen (retrain=True: neu).
- evaluate_estimators: Kennzahlen je Wahlpunkt (estimators/kpis.py), auf den RCT-Testfällen (simuliert) und dem echten
  Log. Jedes Erreichen eines Wahlpunkts ist eine Probe (XGBModel.samples): der geschätzte Outcome der gewählten Option
  gegen den tatsächlichen des Falls; auf den randomisierten RCT-Testfällen am Control-Flow-Wahlpunkt dazu der
  beobachtete und geschätzte Effekt und der Wert der Policy (IPW). Beim Mitarbeiter gibt es den Wert der Policy nicht:
  wer zur Wahl stand (die freien Kompetenten), steht nicht im Log.

    from helpers.training import train_estimators, evaluate_estimators
    table = train_estimators(simulator, rct_data, delta=0.5)
    table, values = evaluate_estimators(simulator, rct_data, real_log)
"""
import os

import numpy as np
import pandas as pd

from .datasets import CASE, PRESCRIPTIVE, add_paths
from .parameters import DELTA, TRAIN_SEED
from .rct import DATA_FOLDER, rct_folder

ESTIMATORS = ("xgb", "lstm", "psp")
MODELS = os.path.join(PRESCRIPTIVE, "models")
TEST_SHARE = 0.2  # Anteil der RCT-Fälle, die das Training nicht sieht
TEST_SEED = 17


def train_estimators(simulator, data, estimators=ESTIMATORS, delta=DELTA, seed=TRAIN_SEED, retrain=False, cases=None, params=None):
    """
    Trainiert die Modelle estimators auf den RCT-Daten data (simulate_rct mit demselben delta) und gibt je Modell und
    Wahlpunkt Proben, Fälle, Optionen, Bäume und den Fehler auf dem Validierungsteil zurück. cases: nur so viele zufällig
    gezogene Fälle der RCT-Daten (mit seed; None: alle). params: je Modell Einstellungen des Trainings, z. B.
    {"psp": {"epochs": 30}} (nur Modelle mit params, LSTM und PSP).
    """
    add_paths()
    from prescriptive import profiles
    if cases:
        data = sample_cases(data, cases, seed)
    trained_on = {"rct": os.path.basename(rct_folder(simulator, delta, DATA_FOLDER)), "outcome": profiles.outcome_tag(simulator.dataset) or "Standard",
                  "cases": int(data[CASE].nunique()), "points": [point.name for point in simulator.points]}
    rows = {}
    for name in estimators:
        path = model_path(name, simulator.dataset)
        if os.path.exists(path) and not retrain:
            model = model_class(name).load(path)
            if getattr(model, "trained_on", None) == trained_on:
                print(f"{name}: vorhanden, auf denselben Daten trainiert ({path}; neu trainieren: retrain=True)", flush=True)
                rows.update({(name, point): row for point, row in model.summary.items()})
                continue
            print(f"{name}: vorhanden, aber auf {getattr(model, 'trained_on', 'unbekannten Daten')} trainiert: neu", flush=True)
        held_out = test_cases(data)
        print(f"{name}: Training auf {data[CASE].nunique() - len(held_out):,} Fällen, {len(held_out):,} Testfälle ...", flush=True)
        settings = {"params": (params or {})[name]} if (params or {}).get(name) else {}
        model = model_class(name)(simulator.dataset, simulator.points, seed=seed, **settings).fit(data[~data[CASE].isin(held_out)])
        model.test_cases, model.trained_on = held_out, trained_on
        model.save(path)
        print(f"{name}: gespeichert in {path}", flush=True)
        rows.update({(name, point): row for point, row in model.summary.items()})
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.names = ["Modell", "Wahlpunkt"]
    return table


def evaluate_estimators(simulator, data, real_log=None, estimators=ESTIMATORS):
    """
    Kennzahlen der trainierten Modelle (siehe oben). data: die RCT-Daten des Trainings (ausgewertet nur die Testfälle);
    real_log: vorverarbeitet mit Spalte outcome (helpers.preprocessing). Liefert die Tabelle (Zeilen: Bereich, Kennzahl;
    Spalten: Daten, Modell) und je (Daten, Modell) {Wahlpunkt: (tatsächlich, geschätzt)} für die Diagramme.
    """
    add_paths()
    from prescriptive.estimators import kpis
    columns, values = {}, {}
    for name in estimators:
        path = model_path(name, simulator.dataset)
        if not os.path.exists(path):
            print(f"{name}: kein Modell ({path}), übersprungen", flush=True)
            continue
        model = model_class(name).load(path)
        sources = {"RCT (Test)": (data[data[CASE].isin(model.test_cases)], True)}
        if real_log is not None:
            sources["echter Log"] = (real_log, False)
        for source, (log, randomized) in sources.items():
            print(f"{name} auf {source} ({log[CASE].nunique():,} Fälle) ...", flush=True)
            sections, values[(source, name)] = point_kpis(model, log, randomized, simulator.intervention)
            if hasattr(model, "prefix_kpis"):
                # PSP: die Suffixe auch auf zufälligen Prefixen jeder Länge, wie die Auswertung des PSP-Papers
                sections = {"Suffix auf Prefixen jeder Länge": model.prefix_kpis(log), **sections}
            columns[(source, name)] = kpis.table(sections)
    order = list(dict.fromkeys(key for column in columns.values() for key in column.index))
    return pd.DataFrame(columns).reindex(order), values


# Je Wahlpunkt des Modells: die Kennzahlen (Bereich -> {Kennzahl: Wert}) und (tatsächlich, geschätzt) der Proben des Logs
def point_kpis(model, log, randomized, intervention):
    from prescriptive.estimators import kpis
    outcome = log.groupby(CASE)["outcome"].first()
    random_cases = set(log.loc[log["rct"].astype(bool), CASE]) if randomized and "rct" in log else set(log[CASE]) if randomized else set()
    sections, pairs = {}, {}
    limit = getattr(model, "max_eval_samples", None)
    for name, rows in model.samples(log).items():
        point = model.points[name]
        if not rows or not model.knows(point):
            continue
        if limit and len(rows) > limit:
            # Teure Modelle (PSP): eine zufällige Auswahl der Proben, immer dieselbe
            rows = [rows[k] for k in sorted(np.random.default_rng(0).choice(len(rows), limit, replace=False))]
        cases, features, options = zip(*rows)
        actual = outcome.reindex(list(cases)).to_numpy(dtype=float)
        if hasattr(model, "suffix_kpis"):
            predicted, details = model.predict_details(point, list(features), list(options))
            sections[f"Suffix an {name}"] = model.suffix_kpis(list(features), details)
        else:
            predicted = model.predict(point, list(features), list(options))
        sections[f"Outcome an {name}"] = kpis.regression(actual, predicted)
        pairs[name] = (actual, predicted)
        if randomized and point.kind == "control_flow":
            keep = np.array([case in random_cases for case in cases])
            kept = [features[i] for i in np.flatnonzero(keep)]
            q = np.column_stack([model.predict(point, kept, [option] * len(kept)) for option in point.options])
            chosen = np.array([point.options.index(option) for option in options])[keep]
            labels = intervention["actions"] if intervention.get("action_activities") == list(point.options) else list(point.options)
            sections[f"Effekt an {name} (RCT, randomisiert)"] = kpis.effect(chosen, actual[keep], q, labels)
    return sections, pairs


# Die Klasse des Modells name (estimators/<name>_model.py)
def model_class(name):
    if name == "xgb":
        from prescriptive.estimators.xgb_model import XGBModel
        return XGBModel
    if name == "lstm":
        from prescriptive.estimators.lstm_model import LSTMModel
        return LSTMModel
    if name == "psp":
        from prescriptive.estimators.psp_model import PSPModel
        return PSPModel
    raise ValueError(f"Unbekanntes Modell {name}, möglich: {', '.join(ESTIMATORS)}")


# Die Datei des Modells name für das gewählte Outcome des Datensatzes (profiles.outcome_tag: "" oder z. B. _gewinn)
def model_path(name, dataset):
    add_paths()
    from prescriptive import profiles
    # BPIC17 ohne Datensatz im Namen, wie die vorhandenen Modelle
    return os.path.join(MODELS, f"{name}_model{'' if dataset == 'bpic17' else '_' + dataset}{profiles.outcome_tag(dataset)}.pkl")


# Die Testfälle: TEST_SHARE der Fälle, zufällig nach Fällen (immer dieselben bei gleichen Fällen)
def test_cases(data):
    cases = np.sort(pd.unique(data[CASE].astype(str)))
    return set(np.random.default_rng(TEST_SEED).choice(cases, int(TEST_SHARE * len(cases)), replace=False))


# cases zufällig gezogene Fälle der RCT-Daten (immer dieselben bei gleichem seed); hat data nicht mehr Fälle: alle
def sample_cases(data, cases, seed=TRAIN_SEED):
    available = data[CASE].unique()
    if cases >= len(available):
        print(f"RCT-Daten haben nur {len(available):,} Fälle: alle für das Training", flush=True)
        return data
    chosen = np.random.default_rng(seed).choice(available, cases, replace=False)
    data = data[data[CASE].isin(set(chosen))].reset_index(drop=True)
    print(f"Training auf {cases:,} zufällig gezogenen von {len(available):,} Fällen der RCT-Daten ({len(data):,} Events)", flush=True)
    return data


# Name der RCT-Daten, mit cases dazu die Zahl der Fälle (10000 -> _10k); für das Lagebild (scripts/main.py)
def model_tag(simulator, delta=DELTA, cases=None, data_folder=DATA_FOLDER):
    tag = os.path.basename(rct_folder(simulator, delta, data_folder))
    if not cases:
        return tag
    return tag + (f"_{cases // 1000}k" if cases % 1000 == 0 else f"_{cases}")
