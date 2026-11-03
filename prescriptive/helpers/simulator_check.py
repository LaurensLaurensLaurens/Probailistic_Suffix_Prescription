"""
Den Simulator (process-simulation-engine/simulation/dataset_simulator.py) gegen den echten Log prüfen: dieselben
Kennzahlen für den echten Log und die Simulation wie die Bank (das Modell entscheidet die Aktion wie im echten Log); mit
policies=("bank", 0, 1) auch die Simulation mit vorgegebener Aktion (immer 0, immer 1).

    from simulation.dataset_simulator import Simulator
    from helpers.simulator_check import check_simulator
    simulator = Simulator(DATASET, log)
    check = check_simulator(simulator, log, n_cases=2000)

Wie die Bank sollte der Simulator den echten Log treffen (Anteil Aktion, Outcome mit und ohne Aktion). Mit
vorgegebener Aktion zeigt "Anteil Aktion", ob er sie befolgt, und der Unterschied im Outcome den Effekt im Simulator.
Die simulierten Logs bekommen die gemeinsamen Schritte des Preprocessings und das Outcome wie der echte Log.

evaluate_simulator: die volle Auswertung (process-simulation-engine/evaluation/evaluation.py, die KPIs, mit denen bisher
BPIC17 ausgewertet wurde) für den Datensatz des Simulators, mit Diagrammen.
"""
import os

import pandas as pd

from .datasets import ACTIVITY, CASE, PRESCRIPTIVE, TIMESTAMP, dataset_info
from .parameters import N_CASES, SEED
from preprocessing import chosen_outcome_spec
from preprocessing.outcome import add_outcome
from preprocessing.steps import COMMON_STEPS
from .screening.events import actions


def check_simulator(simulator, log, n_cases=N_CASES, seed=SEED, costs=None, policies=("bank",)):
    info = dataset_info(simulator.dataset)
    test = decision_test(simulator.intervention)
    names = simulator.intervention["actions"]
    logs = {"echt": log}
    for policy in policies:
        label = "simuliert, wie die Bank" if policy == "bank" else f"simuliert, immer {names[policy]}"
        print(f"{label} ...", flush=True)
        logs[label] = prepare_simulated(simulate(simulator, n_cases, policy, seed), chosen_outcome_spec(simulator.dataset), costs)
    return pd.DataFrame({label: statistics(frame, test, info["screening"]) for label, frame in logs.items()}).round(3)


# Ein Lauf des Simulators: policy "bank" (wie die Bank) oder die Nummer einer Option der Intervention (an ihrem Wahlpunkt
# immer diese; Mitarbeiter wie die Bank). Liefert das Event-Log.
def simulate(simulator, n_cases, policy, seed):
    if policy == "bank":
        return simulator.run(n_cases, seed=seed)[0]
    return simulator.run(n_cases, policy=_Always(int(policy)), points=[simulator.intervention["key"]], seed=seed)[0]


class _Always:
    def __init__(self, index):
        self.index = index

    def choose(self, point, case, options, time, engine):
        return options[self.index]


# Entscheidungs-Event, Aktion und Regel der Intervention, wie im Feld test des Katalogs
def decision_test(intervention):
    action = intervention.get("action_activities") or ([intervention["action_activity"]] if intervention.get("action_activity") else None)
    return {"decision": intervention["decision_activity"], "lifecycle": intervention.get("decision_lifecycle", "complete"), "rule": intervention["rule"], "action": action,
            "action_contains": intervention.get("action_contains")}


def prepare_simulated(events, outcome_spec, costs=None):
    for step in COMMON_STEPS:
        events = step(events)
    return add_outcome(events, outcome_spec, costs)


def statistics(log, test, spec):
    log = log.sort_values([CASE, TIMESTAMP], kind="stable").reset_index(drop=True)
    log["_pos"] = log.groupby(CASE).cumcount()
    outcome = log.groupby(CASE)["outcome"].first()
    decision_pos, treated, lag, _, _ = actions(log, test, spec)
    days = (log.groupby(CASE)[TIMESTAMP].max() - log.groupby(CASE)[TIMESTAMP].min()).dt.total_seconds() / 86400
    with_action = outcome.reindex(treated.index[treated])
    without = outcome.reindex(treated.index[~treated])
    return {"Fälle": float(log[CASE].nunique()), "Events je Fall": len(log) / log[CASE].nunique(), "Aktivitäten": float(log[ACTIVITY].nunique()),
            "Durchlaufzeit (Tage)": days.mean(), "Entscheidung erreicht": len(decision_pos) / log[CASE].nunique(),
            "Anteil Aktion": treated.mean() if len(treated) else float("nan"), "Verzögerung bis Aktion, Median (h)": lag.median() if len(lag) else float("nan"),
            "Outcome je Fall": outcome.mean(), "Outcome mit Aktion": with_action.mean(), "Outcome ohne Aktion": without.mean()}


def evaluate_simulator(simulator, log, n_cases=N_CASES, policy="bank", seed=SEED, folder=None, first_cases=None, costs=None, show=True):
    """
    Der simulierte Log (policy: bank = wie im echten Log) gegen den echten: alle KPIs der Auswertung für den Datensatz
    des Simulators (Fallattribute, Arbeitsschritte, Systemnutzer, Entscheidungspunkt und Outcome des Datensatzes).
    Tabellen und Diagramme in folder (Standard prescriptive/data/evaluation/<datensatz>), die Diagramme mit show im
    Notebook (show: erst die Tabelle, dann die Diagramme). first_cases: nur so viele erste Fälle des echten Logs
    vergleichen. costs: Kostensätze des Outcomes wie für den echten Log (preprocessing.apply_costs).
    """
    from evaluation.evaluation import evaluate
    simulated = prepare_simulated(simulate(simulator, n_cases, policy, seed), chosen_outcome_spec(simulator.dataset), costs)
    folder = folder or os.path.join(PRESCRIPTIVE, "data", "evaluation", simulator.dataset)
    table = evaluate(log, simulated, simulator.dataset, folder, intervention=simulator.intervention["key"], first_cases=first_cases)
    if show:
        from IPython.display import Image, display
        with pd.option_context("display.max_rows", None):
            display(table.round(2))
        for name in sorted((f for f in os.listdir(folder) if f.endswith(".png")), key=lambda f: int(f.split("_")[0])):
            display(Image(os.path.join(folder, name)))
    return table
