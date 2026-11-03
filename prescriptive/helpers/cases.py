"""
Ein zufälliger echter Fall je Datensatz: der Fall, an dem das Notebook die Entscheidung zeigt. Gezogen wird unter den
Fällen, die den Entscheidungspunkt der Intervention (decision_activity) erreichen; vorgegeben (case_id) geht auch.

    from helpers.cases import random_case, show_case
    case = random_case("bpic17", log, simulator.intervention, seed=None, costs=COSTS)
    show_case(case)

Ebenso ein simulierter Fall: simulated_log(simulator, n_cases, policy="bank") simuliert einen Lauf (bank: das Modell
entscheidet die Aktion wie im echten Log) und bereitet ihn auf wie den echten Log; random_case zieht daraus.

Je Fall:
    summary  Fallattribute, Umfang, Entscheidungspunkt, die Aktion dort im Log, Outcome
    events   alle Events wie im angereicherten Log (enriched_log.enrich): Aktivität, Lifecycle, Zeit, Mitarbeiter,
             Stunden seit Fallbeginn, das aktuelle Angebot (nur mit offers), Spalte Entscheidung am Entscheidungspunkt
    prefix   die Events des Falls im Log bis einschließlich zum Entscheidungspunkt: was zur Entscheidung bekannt ist
    log      alle Events des Falls im Log
"""
import numpy as np
import pandas as pd

from .datasets import ACTIVITY, CASE, LIFECYCLE, RESOURCE, TIMESTAMP, add_paths


def random_case(dataset, log, intervention, seed=None, case_id=None, costs=None):
    """
    Ein Fall des vorverarbeiteten Logs: case_id oder zufällig (seed None: jedes Mal ein anderer) unter den Fällen mit
    Entscheidungspunkt. intervention: die des Simulators (simulator.intervention); costs: Kostensätze des Outcomes.
    """
    add_paths()
    from prescriptive import profiles
    from prescriptive.enriched_log import current_offer, enrich
    profile = profiles.get(dataset)
    decision = intervention.get("decision_activity")
    complete = log[log[LIFECYCLE] == "complete"] if LIFECYCLE in log else log
    candidates = np.sort(complete.loc[complete[ACTIVITY] == decision, CASE].unique()) if decision else np.sort(log[CASE].unique())
    if case_id is None:
        case_id = candidates[np.random.default_rng(seed).integers(len(candidates))]
    case_log = log[log[CASE] == case_id].sort_values(TIMESTAMP, kind="stable")
    if case_log.empty:
        raise ValueError(f"Fall {case_id} nicht im Log von {dataset}")
    enriched = enrich(case_log, n_stages=1, profile=profile, intervention=intervention, costs=costs)

    columns = ["event_nr", ACTIVITY, LIFECYCLE, TIMESTAMP] + ([RESOURCE] if enriched[RESOURCE].notna().any() else []) + ["elapsed_hours"]
    events = enriched[columns + list(current_offer(profile).values())].copy()
    point = enriched["stage"] == 0
    events["Entscheidung"] = ""
    events.loc[point, "Entscheidung"] = [f"Entscheidungspunkt: {intervention['actions'][action]}" for action in enriched.loc[point, "treatment"]]
    events = events.rename(columns={"event_nr": "Nr.", ACTIVITY: "Aktivität", LIFECYCLE: "Lifecycle", TIMESTAMP: "Zeit", RESOURCE: "Mitarbeiter",
                                    "elapsed_hours": "Stunden seit Beginn"}).set_index("Nr.")

    first = enriched.iloc[0]
    summary = {"Fall": case_id, "Unter Fällen mit Entscheidungspunkt": f"{len(candidates):,}"}
    summary.update({column: first[column] for column in list(profile["static_categorical"]) + list(profile["static_numeric"]) if column in enriched})
    summary["Events (davon complete)"] = f"{len(enriched)} ({int((enriched[LIFECYCLE] == 'complete').sum())})"
    if point.any():
        at = enriched[point].iloc[0]
        summary[f"Entscheidungspunkt ({decision})"] = f"Event Nr. {int(at['event_nr'])}, nach {at['elapsed_hours']:.1f} h"
        summary["Aktion im Log"] = intervention["actions"][int(at["treatment"])]
        # Das angereicherte Log hat die Events in derselben Reihenfolge wie der Log (nach Zeit, stabil)
        prefix = case_log.iloc[:int(at["event_nr"]) + 1]
    else:
        summary["Entscheidungspunkt"] = "keiner"
        prefix = case_log.iloc[0:0]
    summary["Outcome"] = round(float(first["outcome"]), 2)
    if "approved" in enriched:
        summary["Ziel erreicht"] = bool(first["approved"])
    summary["Durchlaufzeit (Tage)"] = round(float(first["cycle_days"]), 1)
    return {"dataset": dataset, "case_id": case_id, "summary": pd.Series(summary, name="Wert").to_frame(), "events": events,
            "prefix": prefix, "log": case_log}


def simulated_log(simulator, n_cases, policy="bank", seed=0, costs=None):
    """Ein Lauf des Simulators mit policy (bank oder die Nummer einer Option), aufbereitet wie der echte Log (check_simulator)."""
    from preprocessing import chosen_outcome_spec
    from .simulator_check import prepare_simulated, simulate
    return prepare_simulated(simulate(simulator, n_cases, policy, seed), chosen_outcome_spec(simulator.dataset), costs)


def show_case(case):
    """Im Notebook: erst die Angaben zum Fall, dann seine Events."""
    try:
        from IPython.display import display
    except ImportError:
        display = print
    display(case["summary"])
    display(case["events"].round(2))
