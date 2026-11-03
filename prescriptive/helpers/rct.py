"""
RCT-Daten auf dem Simulator (BPIC17, process-simulation-engine/simulation/dataset_simulator.py): die Trainingsdaten der
Modelle (z. B. estimators/xgb_model.py).

In runs Läufen (simulierten Jahren) mit je n_cases Fällen läuft der Simulator ganz normal unter der Policy rct
(policies/rct_policy.py): an jedem Wahlpunkt (Control Flow der Intervention, Mitarbeiterwahl) eine zufällige Option, so
kommt jede Option in jeder Lage vor (Positivity); mit Wahrscheinlichkeit delta je Fall stattdessen wie die Bank
(confounding "point" wie in SCOPE). Spalte rct: der Fall war randomisiert. Alle Fälle eines Laufs konkurrieren um die
Mitarbeiter; ein Lauf je Arbeitsprozess. Lauf r zieht mit seed + 1000 + r, die Fall-IDs beginnen mit R<r>_.

Ergebnis: das angereicherte Event-Log (enriched_log.enrich, alle Lifecycle-Events, Spalte outcome mit den gewählten
Kosten), gespeichert in prescriptive/data/rct_<datensatz>_<intervention>[_delta<Prozent>]/log.pkl, dazu die Parameter in
rct.json. Vorhanden und mit denselben Parametern (auch denselben Wahlpunkten) erzeugt: geladen, sonst neu simuliert
(regenerate=True: immer neu).

    from helpers.rct import rct_summary, simulate_rct
    data = simulate_rct(simulator, runs=3, costs=COSTS)
    rct_summary(data, simulator.intervention)
"""
import json
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .datasets import CASE, PRESCRIPTIVE, add_paths
from .parameters import DELTA, RCT_CASES, RCT_RUNS, RCT_SEED, WORKERS

DATA_FOLDER = os.path.join(PRESCRIPTIVE, "data")
_SHARED = {}  # was die Arbeitsprozesse der Engine brauchen (_init_worker)


# Ordner der RCT-Daten des Simulators: rct_<datensatz>_<intervention>, mit delta _delta<Prozent>
def rct_folder(simulator, delta=DELTA, data_folder=DATA_FOLDER):
    suffix = f"_delta{int(round(100 * delta))}" if delta > 0 else ""
    return os.path.join(data_folder, f"rct_{simulator.dataset}_{simulator.intervention['key']}" + suffix)


def save_data(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".pkl", "wb") as f:
        pickle.dump(data, f)


def load_data(path):
    with open(path + ".pkl", "rb") as f:
        return pickle.load(f)


def simulate_rct(simulator, n_cases=RCT_CASES, runs=RCT_RUNS, seed=RCT_SEED, delta=DELTA, workers=WORKERS, costs=None,
                 regenerate=False, data_folder=DATA_FOLDER):
    """
    RCT-Daten des Simulators (siehe oben): vorhanden geladen, sonst simuliert und gespeichert. n_cases: Fälle je Lauf
    (None: so viele, wie der echte Log Fälle hat); costs: Kostensätze des Outcomes (Standard helpers/parameters.py).
    """
    add_paths()
    from prescriptive import profiles
    from interventions import arrival_load
    from prescriptive.enriched_log import enrich, with_current_costs
    profile = profiles.get(simulator.dataset)
    folder = rct_folder(simulator, delta, data_folder)
    path = os.path.join(folder, "log")
    n_cases = n_cases or simulator.log[CASE].nunique()
    wanted = {"n_cases": int(n_cases), "runs": int(runs), "seed": int(seed), "delta": float(delta), "intervention": simulator.intervention["key"],
              "points": [point.name for point in simulator.points]}
    load = arrival_load(simulator.dataset)
    if load != 1:
        wanted["load"] = load
    if os.path.exists(path + ".pkl") and not regenerate:
        saved = _read_parameters(folder)
        if saved and saved != wanted:
            # Andere Parameter (z. B. andere Wahlpunkte im Katalog): die Daten passen nicht, neu simulieren
            print(f"RCT-Daten in {folder} mit anderen Parametern erzeugt ({saved}), gewählt ist {wanted}: neu simuliert", flush=True)
        else:
            print(f"RCT-Daten vorhanden, geladen: {path}.pkl ({'erzeugt mit ' + str(saved) if saved else 'Parameter unbekannt'}; "
                  f"neu erzeugen: regenerate=True)", flush=True)
            return with_current_costs(load_data(path), profile, costs)

    print(f"RCT {simulator.dataset}, Wahlpunkte {', '.join(wanted['points'])}: {runs} Läufe mit je {n_cases:,} Fällen, delta {delta}, "
          f"{min(workers, runs)} Prozesse ...", flush=True)
    jobs = [(run, seed + 1000 + run, n_cases, delta) for run in range(runs)]
    config = (simulator.dataset, simulator.log, simulator.intervention["key"])
    if min(workers, runs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, runs), mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker,
                                 initargs=config) as pool:
            logs = list(pool.map(_run, jobs))
    else:
        _init_worker(*config)
        logs = [_run(job) for job in jobs]
    data = enrich(pd.concat(logs, ignore_index=True), n_stages=1, profile=profile, intervention=simulator.intervention, costs=costs)
    save_data(data, path)
    with open(os.path.join(folder, "rct.json"), "w") as f:
        json.dump(wanted, f)
    print(f"Gespeichert: {path}.pkl", flush=True)
    return data


def rct_summary(data, intervention):
    """Kennzahlen der RCT-Daten: Umfang, Anteil randomisiert und je Aktion am ersten Entscheidungspunkt Fälle und Outcome."""
    cases = data.groupby("case_nr").first()
    decisions = data[data["stage"] == 0]
    rows = {"Fälle": len(cases), "Events": len(data), "Fälle mit Entscheidungspunkt": len(decisions)}
    # Ältere RCT-Daten von main.py ohne Spalte rct sind ganz randomisiert (delta 0)
    if "rct" in decisions:
        rows["davon randomisiert (Anteil)"] = decisions["rct"].mean()
    rows["Outcome je Fall"] = cases["outcome"].mean()
    for action, name in enumerate(intervention["actions"]):
        part = decisions[decisions["treatment"] == action]
        rows[f"{name}: Anteil der Entscheidungspunkte"] = len(part) / max(len(decisions), 1)
        rows[f"{name}: Outcome je Fall"] = part["outcome"].mean()
        if "approved" in part:
            rows[f"{name}: Ziel erreicht (Anteil)"] = part["approved"].mean()
    if "call_resource" in cases and cases["call_resource"].notna().any():
        rows["Mitarbeiter am Arbeitsschritt der Intervention"] = cases["call_resource"].nunique()
    return pd.Series(rows, name="Wert").to_frame().round(3)


# ------------------------------------------------------------------ Läufe

def _read_parameters(folder):
    path = os.path.join(folder, "rct.json")
    return json.load(open(path)) if os.path.exists(path) else None


# Wird aufgerufen von simulate_rct -> ProcessPoolExecutor -> _init_worker (je Arbeitsprozess einmal): der Simulator des
# Datensatzes (das Modell ist schon trainiert)
def _init_worker(dataset, log, intervention):
    add_paths()
    from simulation.dataset_simulator import Simulator
    _SHARED["simulator"] = Simulator(dataset, log, intervention)


# Ein Lauf: der Simulator unter der Policy rct
# Wird aufgerufen von simulate_rct -> _run (je Lauf, im Arbeitsprozess)
def _run(job):
    from prescriptive.policies.rct_policy import RCTPolicy
    run, seed, n_cases, delta = job
    policy = RCTPolicy([seed, 1], delta)  # eigener Zufall, unabhängig vom Zufall der Simulation (seed)
    events, choices = _SHARED["simulator"].run(n_cases, policy=policy, seed=seed)
    events["rct"] = events[CASE].isin(policy.randomized)
    events[CASE] = f"R{run}_" + events[CASE].astype(str)
    print(f"RCT-Lauf {run}: {n_cases:,} Fälle, {len(events):,} Events, randomisiert {len(policy.randomized):,}, "
          f"Wahlen {int(choices['choice'].notna().sum()):,}", flush=True)
    return events
