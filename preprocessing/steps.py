"""
Die Maßnahmen des Preprocessings: COMMON_STEPS für alle Datensätze (hier), DATASET_STEPS je Datensatz (aus dem Ordner des
Datensatzes, z. B. BPIC2013/steps.py), optional zuletzt work_sessions (Lifecycle bereinigt; Standard: aus, die Lifecycles
wie im Log). Jede Maßnahme ist eine
Funktion log -> log; ihr Docstring (erste Zeile) erscheint im Protokoll von preprocess.
"""
import numpy as np
import pandas as pd

from prescriptive.helpers.datasets import ACTIVITY, CASE, LIFECYCLE, RESOURCE, TIMESTAMP
from .BPIC2013.steps import STEPS as BPIC2013_STEPS
from .BPIC2015.steps import STEPS as BPIC2015_STEPS
from .BPIC2019.steps import STEPS as BPIC2019_STEPS


# ------------------------------------------------------------------ Für alle Datensätze

def drop_incomplete_cases(log):
    """Fälle mit einem Event ohne Aktivität oder Zeitstempel ganz entfernen (der Ablauf hätte eine Lücke); Events ohne Fall-ID auch."""
    log = log[log[CASE].notna()]
    broken = log.loc[log[ACTIVITY].isna() | log[TIMESTAMP].isna(), CASE].unique()
    return log[~log[CASE].isin(broken)].copy()


def naive_utc_timestamps(log):
    """Zeitstempel naiv in UTC."""
    log[TIMESTAMP] = pd.to_datetime(log[TIMESTAMP], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    return log


def sort_events(log):
    """Nach Fall und Zeit sortieren, stabil: bei gleicher Zeit bleibt die Reihenfolge im Log."""
    return log.sort_values([CASE, TIMESTAMP], kind="stable").reset_index(drop=True)


def ensure_lifecycle(log):
    """Logs ohne Lifecycle-Spalte (z. B. BPIC19): alle Events sind complete-Events (der Simulator braucht Aktivität und Lifecycle)."""
    if LIFECYCLE not in log:
        log[LIFECYCLE] = "complete"
    return log


def strip_activity_names(log):
    """Leerzeichen am Rand von Aktivitätsnamen entfernen (BPIC17: "W_Shortened completion ")."""
    log[ACTIVITY] = log[ACTIVITY].astype(str).str.strip()
    return log


# ------------------------------------------------------------------ Lifecycle bereinigen: Arbeitsabschnitte

OPENING, CLOSING, ENDING = ("start", "resume"), ("suspend", "complete"), ("ate_abort", "withdraw")
ORIGINAL_LIFECYCLE, WORK_ITEM_END = "lifecycle:original", "work_item_end"


# Die Arbeitsschritte eines Logs: Tätigkeiten mit einem anderen Lifecycle als complete
def work_item_activities(log):
    return set(log.loc[log[LIFECYCLE] != "complete", ACTIVITY].unique()) if LIFECYCLE in log else set()


def work_sessions(log, activities=None):
    """
    Lifecycle bereinigt: je Arbeitsabschnitt ein Paar start -> complete mit einem Mitarbeiter (schedule, ate_abort, withdraw entfernt).

    Gilt für jede Tätigkeit mit einem anderen Lifecycle als complete (BPIC17: die Arbeitsschritte W_...; activities: diese
    Tätigkeiten statt der aus dem Log, z. B. für einen einzelnen Fall); Tätigkeiten nur mit complete-Events (BPIC17: A_...,
    O_...; BPIC13, BPIC19: alle) bleiben, wie sie sind. Eine Instanz ohne Arbeitsabschnitt (BPIC17: W_Handle leads mit
    schedule und withdraw) fällt ganz weg. Ein Arbeitsabschnitt geht von
    start bzw. resume bis zum nächsten suspend bzw. complete derselben Tätigkeit im Fall (wie die Arbeitszeit in
    work_time.py): das öffnende Event wird start, das schließende complete. Ein Abschnitt, den ate_abort oder withdraw
    schließt, endet dort; sein complete bekommt den Mitarbeiter des Abschnitts (ate_abort bucht meist, wer den nächsten
    Arbeitsschritt beginnt). Entfernt werden schedule, ate_abort und withdraw nach einem geschlossenen Abschnitt (Buchungen
    ohne Arbeit), ein zweites Öffnen ohne Schließen dazwischen (das erste fällt weg) und Abschnitte, die bis zum Fallende
    offen bleiben. Neue Spalten: lifecycle:original (der Lifecycle vorher) und work_item_end am complete-Event des letzten
    Abschnitts einer Instanz (wie sie endete: complete, ate_abort, withdraw; offen, wenn nicht).
    """
    sessions = set(activities) if activities is not None else work_item_activities(log)
    if not sessions:
        return log
    log = log.copy()
    log[ORIGINAL_LIFECYCLE] = log[LIFECYCLE]
    log[WORK_ITEM_END] = None
    keep = np.ones(len(log), dtype=bool)
    lifecycle = log[LIFECYCLE].to_numpy(dtype=object).copy()
    resource = log[RESOURCE].to_numpy(dtype=object).copy() if RESOURCE in log else None
    ends = log[WORK_ITEM_END].to_numpy(dtype=object)
    items = log[ACTIVITY].isin(sessions).to_numpy()
    rows = np.flatnonzero(items)
    keep[rows] = False
    # je Fall und Tätigkeit in Log-Reihenfolge (das Log ist nach Fall und Zeit sortiert)
    for _, positions in pd.Series(rows).groupby([log[CASE].to_numpy()[rows], log[ACTIVITY].to_numpy()[rows]], sort=False):
        opened, last_close = None, None
        for i in positions.to_numpy():
            transition = lifecycle[i]
            if transition in OPENING:
                opened = i
            elif transition in CLOSING + ENDING and opened is not None:
                keep[opened], keep[i] = True, True
                lifecycle[opened], lifecycle[i] = "start", "complete"
                if transition in ENDING and resource is not None:
                    resource[i] = resource[opened]
                last_close, opened = i, None
                if transition != "suspend":
                    ends[i] = transition
            if transition in ("complete",) + ENDING and last_close is not None:
                # Die Instanz ist zu Ende: am letzten Abschnitt steht, wie
                ends[last_close] = transition
                last_close = None
            elif transition == "schedule" and last_close is not None:
                # neue Instanz, die vorige endete ohne complete, ate_abort oder withdraw
                ends[last_close] = ends[last_close] or "offen"
                last_close = None
        if last_close is not None and ends[last_close] is None:
            ends[last_close] = "offen"
    log[LIFECYCLE] = lifecycle
    if resource is not None:
        log[RESOURCE] = resource
    log[WORK_ITEM_END] = ends
    return log[keep].reset_index(drop=True)


COMMON_STEPS = [drop_incomplete_cases, naive_utc_timestamps, sort_events, ensure_lifecycle, strip_activity_names]


# Je Datensatz: die Maßnahmen aus seinem Ordner (BPIC17 braucht keine eigenen)
DATASET_STEPS = {
    "bpic17": [],
    "bpic13": BPIC2013_STEPS,
    "bpic15": BPIC2015_STEPS,
    "bpic19": BPIC2019_STEPS,
    "repair": [],  # synthetisch (process-simulation-engine/simulation/repair_workshop.py), nichts zu bereinigen
}
