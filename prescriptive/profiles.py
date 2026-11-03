"""
Das Profil eines Event-Logs für die Pipeline in prescriptive/ (psp.py, enriched_log.py, main.py, oracle_evaluation.py): was an einem Log
besonders ist, an einer Stelle zusammengesetzt. Der Simulator braucht es nicht, er liest seine Tabellen selbst. Eigene
Daten hat profiles nicht, es liest:
- event_log, intervention (standard_intervention), outcome (type) und outcome_spec (der ganze Block):
  der Katalog prescriptive/catalog.yaml
- engine: ob der Datensatz Arbeitsschritte hat und Fälle um Mitarbeiter konkurrieren (work_item_prefix; sonst läuft jeder Fall für sich, simulation/dataset_simulator.py DATASETS)
- static_categorical, static_numeric, dynamic_numeric, dynamic_categorical, caller, work_item_prefix, max_events,
  max_gap_days, model_folder, trainable:
  next_event_prediction/datasets.py (NEXT_EVENT)
Gewählt mit --profile (Umgebungsvariable PRESCRIPTIVE_PROFILE, damit auch die Arbeitsprozesse es kennen); Standard bpic17.
Das Outcome ist das Standard-Outcome des Katalogs (outcome) oder eine seiner outcome_optionen, gewählt mit set_outcome
(Umgebungsvariable PRESCRIPTIVE_OUTCOME_<DATENSATZ>, ebenfalls für die Arbeitsprozesse); outcome_option: deren Name,
"" beim Standard.

prepare(): der Log nach dem Preprocessing des Datensatzes (prescriptive/helpers/preprocessing).
"""
import os
from typing import List, Optional

import pandas as pd

import interventions
from next_event_prediction.datasets import NEXT_EVENT

PROFILE_ENV = "PRESCRIPTIVE_PROFILE"
OUTCOME_ENV = "PRESCRIPTIVE_OUTCOME_{}"  # je Datensatz die gewählte outcome_option (leer: Standard)
DEFAULT = "bpic17"


def names() -> List[str]:
    return list(interventions.catalog())


def get(name: Optional[str] = None) -> dict:
    if name in (None, ""):
        name = DEFAULT
    catalog = interventions.catalog()
    if name not in catalog:
        raise ValueError(f"Unbekanntes Profil '{name}', möglich: {', '.join(catalog)}")
    entry = catalog[name]
    option = os.environ.get(OUTCOME_ENV.format(name.upper())) or ""
    spec = entry["outcome"] if not option else (entry.get("outcome_optionen") or {})[option]
    return {"key": name, "event_log": entry["event_log"], "intervention": entry["standard_intervention"],
            "outcome": spec["type"], "outcome_spec": dict(spec), "outcome_option": option, "engine": bool(NEXT_EVENT[name]["work_item_prefix"]), **NEXT_EVENT[name]}


# Das Outcome eines Datensatzes für diesen Prozess und seine Arbeitsprozesse: eine der outcome_optionen des Katalogs,
# None oder "": das Standard-Outcome. Vor dem ersten Import von enriched_log setzen (es liest das Profil beim Import).
# Wird aufgerufen von scripts/main.py (--outcome)
def set_outcome(name, option=None):
    options = interventions.catalog()[name].get("outcome_optionen") or {}
    if option and option not in options:
        raise ValueError(f"Unbekanntes Outcome '{option}' für {name}, möglich: {', '.join(options) or 'keine Optionen'}")
    os.environ[OUTCOME_ENV.format(name.upper())] = option or ""


# Zusatz der Ordner von Modellen, die das Outcome direkt lernen (XGBoost, LSTM): "" beim Standard-Outcome, sonst
# "_<Option>" (z. B. xgb_rct_lc_gewinn)
def outcome_tag(name=None):
    option = get(name).get("outcome_option")
    return f"_{option}" if option else ""


# Das Profil dieses Prozesses (gesetzt mit --profile, siehe prescriptive/__init__.py)
def current() -> dict:
    return get(os.environ.get(PROFILE_ENV))


def prepare(df: pd.DataFrame, profile: Optional[dict] = None, work_sessions: bool = False) -> pd.DataFrame:
    """
    Der Log nach dem Preprocessing des Datensatzes (prescriptive/helpers/preprocessing, mit der Spalte outcome).
    work_sessions: mit Arbeitsabschnitten, wenn das Simulationsmodell darauf trainiert ist
    (simulation/dataset_simulator.uses_work_sessions); sonst die Lifecycles wie im Log.
    """
    profile = profile or current()
    from preprocessing import preprocess
    return preprocess(profile["key"], df, verbose=False, work_sessions=work_sessions)
