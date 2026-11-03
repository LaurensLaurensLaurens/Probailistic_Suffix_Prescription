"""
BPIC17: das Outcome profit (Option gewinn von bpic17 im Katalog prescriptive/catalog.yaml), für das Notebook
(../outcome.py, OUTCOMES) und die BPIC17-Pipeline in prescriptive/ (enriched_log.enrich, estimators/, main.py, ...).

profit (BPIC17): Gewinn der Bank in €
    Gewinn = bewilligt × Zinsertrag − CALL_COST × Anrufe − HOURLY_RATE × Arbeitsstunden − DAILY_COST × Tage
    bewilligt       der Fall erreicht OUTCOME_ACTIVITY (A_Pending)
    Zinsertrag      MonthlyCost × NumberOfTerms − OfferedAmount des zuletzt erstellten Angebots, mindestens 0 (im echten
                    Log im Median 2.800 € bei bewilligten Fällen)
    Anrufe          Zahl der CALL_ACTIVITY (A_Incomplete, Anruf wegen fehlender Unterlagen)
    Arbeitsstunden  Arbeitsabschnitte der Arbeitsschritte (W_..., ../work_time.py), ohne die Systemnutzer
                    (screening.system_users im Katalog)
    Tage            Durchlaufzeit vom ersten bis zum letzten complete-Event
    Aktivitäten und CALL_COST stehen im Katalog (Option gewinn von bpic17) und werden beim Import gelesen; HOURLY_RATE und
    DAILY_COST gelten für alle Datensätze (helpers/parameters.py, im Notebook wählbar).
    Für die vom PSP gezogenen Suffixe (Tokens aus Aktivität und Lifecycle, ohne Mitarbeiter) wird die Arbeitszeit aus
    den Tokens geschätzt (../work_time.py: fit_hours_per_activity, estimated_hours).

Für die Pipeline in prescriptive/ (über enriched_log.py bzw. estimators/psp.py, je Outcome-Typ):
    enriched_columns   Gewinn und seine Bestandteile als Spalten des angereicherten Logs
    with_current_costs Gewinn neu aus den Bestandteilen (gespeicherte Logs)
    offer_income       was am Prefix schon feststeht: der Zinsertrag des aktuellen Angebots (Eingang von PSP und XGBoost)
    suffix_outcome     Gewinn eines Falls aus Prefix und gezogenem Suffix (PSP)
"""
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from prescriptive.helpers.datasets import ACTIVITY, CASE, LIFECYCLE, TIMESTAMP
from prescriptive.helpers.parameters import DAILY_COST, HOURLY_RATE
from ..work_time import estimated_hours, working_hours

from interventions import CANDIDATES_FILE as CATALOG  # prescriptive/catalog.yaml


# Der profit-Block von bpic17: outcome oder eine der outcome_optionen (derzeit gewinn)
def _bpic17_outcome():
    import yaml
    with open(CATALOG) as f:
        entry = yaml.safe_load(f)["bpic17"]
    blocks = [entry["outcome"], *(entry.get("outcome_optionen") or {}).values()]
    return next(block for block in blocks if block["type"] == "profit")


# ------------------------------------------------------------------ Gewinn der Bank (BPIC17)

_SPEC = _bpic17_outcome()
OUTCOME_ACTIVITY = _SPEC["outcome_activity"]  # Kredit bewilligt und vom Kunden angenommen
CALL_ACTIVITY = _SPEC["call_activity"]  # Anruf wegen fehlender Unterlagen
CALL_COST = float(_SPEC["call_cost"])
# HOURLY_RATE, DAILY_COST: Standardwerte für alle Datensätze (helpers/parameters.py), im Notebook wählbar (costs)
WORK_ITEM_PREFIX = "W_"  # Arbeitsschritte (wie work_item_prefix von bpic17 in next_event_prediction/datasets.py)


def _system_users(log):
    from resources.datasets import system_users
    return system_users("bpic17", log)


def interest_income(monthly_cost, number_of_terms, offered_amount):
    """Zinsertrag eines Angebots (nie negativ, 0 ohne Angebot)."""
    income = np.asarray(monthly_cost, dtype=float) * np.asarray(number_of_terms, dtype=float) - np.asarray(offered_amount, dtype=float)
    return np.nan_to_num(np.maximum(income, 0.0))


def profit(approved, income, calls, hours, days, hourly_rate: float = HOURLY_RATE, call_cost: float = CALL_COST,
           daily_cost: float = DAILY_COST):
    return (np.asarray(approved, dtype=float) * np.asarray(income, dtype=float) - call_cost * np.asarray(calls, dtype=float)
            - hourly_rate * np.asarray(hours, dtype=float) - daily_cost * np.asarray(days, dtype=float))


def outcome_profit(log, spec):
    """Gewinn je Fall aus einem vorverarbeiteten Log mit allen Lifecycle-Events; Aktivitäten und Kostensätze aus spec."""
    hours = working_hours(log, WORK_ITEM_PREFIX, _system_users(log))
    complete = log[log[LIFECYCLE] == "complete"] if LIFECYCLE in log else log
    cases = complete.groupby(CASE)
    approved = cases[ACTIVITY].apply(lambda a: spec["outcome_activity"] in set(a)).astype(float)
    # Das letzte Angebot: je Attribut der letzte vorhandene Wert (die Attribute stehen nur auf O_Create Offer)
    offer = cases[["MonthlyCost", "NumberOfTerms", "OfferedAmount"]].agg(lambda s: pd.to_numeric(s, errors="coerce").dropna().iloc[-1] if s.notna().any() else np.nan)
    income = interest_income(offer["MonthlyCost"], offer["NumberOfTerms"], offer["OfferedAmount"])
    calls = (complete[ACTIVITY] == spec["call_activity"]).groupby(complete[CASE]).sum().astype(float)
    days = (cases[TIMESTAMP].max() - cases[TIMESTAMP].min()).dt.total_seconds() / 86400
    value = profit(approved, income, calls, hours.reindex(approved.index).fillna(0.0), days,
                   hourly_rate=float(spec.get("hourly_rate", HOURLY_RATE)), call_cost=float(spec["call_cost"]),
                   daily_cost=float(spec.get("daily_cost", DAILY_COST)))
    return pd.Series(value, index=approved.index)


# ------------------------------------------------------------------ Angereichertes Log (prescriptive/enriched_log.py)

COMPONENTS = ["approved", "interest_income", "calls", "working_hours", "cycle_days"]


def enriched_columns(df: pd.DataFrame, current_offer: Dict[str, str], costs: Optional[dict] = None, complete: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Gewinn (outcome) und seine Bestandteile als Spalten auf allen Zeilen eines Falls. df: das angereicherte Log mit
    case_nr, dem aktuellen Angebot (current_offer: Attribut -> Spalte), working_hours und cycle_days (generisch,
    enriched_log.enrich). Zinsertrag aus dem zuletzt erstellten Angebot des Falls; Anrufe = Anzahl der CALL_ACTIVITY
    (kosten bei jeder Intervention). costs: Kostensätze statt der Standardwerte (hourly_rate, daily_cost, call_cost).
    complete: welche Zeilen complete-Events sind (nur sie zählen; Standard: alle).
    """
    offer = [current_offer[name] for name in ("MonthlyCost", "NumberOfTerms", "OfferedAmount")]
    complete = complete if complete is not None else pd.Series(True, index=df.index)
    df["approved"] = ((df[ACTIVITY] == OUTCOME_ACTIVITY) & complete).groupby(df["case_nr"]).transform("any").astype(float)
    last_offer = df.groupby("case_nr")[offer].transform("last")
    df["interest_income"] = interest_income(*(last_offer[column] for column in offer))
    df["calls"] = ((df[ACTIVITY] == CALL_ACTIVITY) & complete).groupby(df["case_nr"]).transform("sum").astype(float)
    df["outcome"] = profit(*(df[column] for column in COMPONENTS), **(costs or {}))
    return df


def with_current_costs(log: pd.DataFrame, costs: Optional[dict] = None) -> pd.DataFrame:
    """Gewinn neu aus seinen Bestandteilen mit den aktuellen Kostensätzen (gespeicherte Logs); costs wie bei enriched_columns."""
    log["outcome"] = profit(*(log[column] for column in COMPONENTS), **(costs or {}))
    return log


def offer_income(frame, current_offer: Dict[str, str]):
    """Zinsertrag des aktuellen Angebots (eine Zeile oder je Fall die letzte Zeile des angereicherten Logs)."""
    return interest_income(*(frame[current_offer[name]] for name in ("MonthlyCost", "NumberOfTerms", "OfferedAmount")))


def suffix_outcome(activities: List[str], elapsed_hours: Sequence[float], income: float, hours_per_activity: Dict[str, float]):
    """
    Gewinn eines Falls aus Prefix und gezogenem Suffix (activities: die Tokens der Spalte event, je Event Stunden seit
    Fallbeginn): Zinsertrag des aktuellen Angebots am Prefix (income), wenn OUTCOME_ACTIVITY vorkommt, abzüglich der
    Anrufe, der aus den Tokens geschätzten Arbeitszeit und der Durchlaufzeit (am letzten complete-Event, wie
    enriched_log.enrich). Liefert (Gewinn, bewilligt, (Anrufe, Stunden, Tage)).
    """
    from prescriptive.enriched_log import is_complete_token
    approved = OUTCOME_ACTIVITY in activities
    calls = sum(activity == CALL_ACTIVITY for activity in activities)
    completes = [hours for activity, hours in zip(activities, elapsed_hours) if is_complete_token(activity)]
    days = max(completes or elapsed_hours) / 24
    hours = estimated_hours(activities, hours_per_activity)
    return float(profit(approved, income, calls, hours, days)), approved, (calls, hours, days)
