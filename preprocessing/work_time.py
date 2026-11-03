"""
Arbeitszeit der Mitarbeiter, für alle Datensätze mit Arbeitsschritten (work_item_prefix in next_event_prediction/datasets.py,
bei BPIC17 "W_"): Ein Arbeitsabschnitt geht von start/resume bis suspend/complete/ate_abort/withdraw desselben
Arbeitsschritts im Fall. Abschnitte der Systemnutzer (screening.system_users im Katalog) zählen nicht. Braucht alle
Lifecycle-Events.

working_hours  Arbeitsstunden je Fall (Gewinn BPIC17, enriched_log.enrich)
work_segments  die einzelnen Abschnitte (Mitarbeiter, Aktivität, Sekunden)
tempo_classes  Tempo-Klasse fast / medium / slow je Mitarbeiter aus den Abschnitten (für Auswertungen; bei BPIC17
               brauchen langsame Mitarbeiter je Abschnitt im Mittel 1,72-mal so lange wie der Durchschnitt, schnelle 0,62-mal)
fit_hours_per_activity, estimated_hours
               Arbeitszeit eines gezogenen Suffixes (Tokens aus Aktivität und Lifecycle, ohne Mitarbeiter), geschätzt
               aus den Tokens
"""
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from prescriptive.helpers.datasets import ACTIVITY, CASE, LIFECYCLE, RESOURCE, TIMESTAMP

OPENING, CLOSING = ("start", "resume"), ("suspend", "complete", "ate_abort", "withdraw")
CLASSES = ("fast", "medium", "slow")
MIN_SEGMENTS = 30


def _segments(df: pd.DataFrame, prefix: str):
    """Je Fall (in Reihenfolge) die Abschnitte (Fall, Mitarbeiter, Aktivität, Sekunden, ggf. <= 0) der Arbeitsschritte mit prefix."""
    df = df.sort_values([CASE, TIMESTAMP], kind="stable")
    timestamps = pd.to_datetime(df[TIMESTAMP])
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert("UTC").dt.tz_localize(None)
    seconds = (timestamps - pd.Timestamp("1970-01-01")).dt.total_seconds()
    resources = df[RESOURCE] if RESOURCE in df.columns else pd.Series(None, index=df.index)
    for case, activities, lifecycles, times, employees in zip(
        *(column.groupby(df[CASE], sort=False).agg(list) for column in (df[CASE], df[ACTIVITY], df[LIFECYCLE], seconds, resources))
    ):
        opened = {}
        for activity, lifecycle, time, employee in zip(activities, lifecycles, times, employees):
            if not str(activity).startswith(prefix):
                continue
            if lifecycle in OPENING:
                opened[activity] = (time, employee)
            elif lifecycle in CLOSING and activity in opened:
                since, who = opened.pop(activity)
                yield case[0], who, activity, time - since


def working_hours(df: pd.DataFrame, prefix: Optional[str], system_users: Iterable[str] = ()) -> pd.Series:
    """Arbeitsstunden je Fall (Index: Fall); 0 ohne Arbeitsschritte (prefix None) oder ohne Lifecycle-Spalte."""
    cases = df[CASE].unique()
    if not prefix or LIFECYCLE not in df.columns:
        return pd.Series(0.0, index=cases)
    system_users = set(system_users)
    hours = {case: 0.0 for case in cases}
    for case, who, _, seconds in _segments(df, prefix):
        if who not in system_users:
            hours[case] += max(seconds, 0.0)
    return pd.Series(hours).reindex(cases, fill_value=0.0) / 3600.0


def work_segments(df: pd.DataFrame, prefix: str, system_users: Iterable[str] = ()) -> pd.DataFrame:
    """Die Arbeitsabschnitte (resource, activity, seconds) mit positiver Dauer, ohne die Systemnutzer."""
    system_users = set(system_users)
    rows = [(who, activity, seconds) for _, who, activity, seconds in _segments(df, prefix) if who not in system_users and seconds > 0]
    return pd.DataFrame(rows, columns=["resource", "activity", "seconds"])


def tempo_classes(df: pd.DataFrame, prefix: str, system_users: Iterable[str] = ()) -> dict:
    """
    Tempo-Klasse je Mitarbeiter (dict Mitarbeiter -> "fast" / "medium" / "slow"): wie schnell er dieselben Arbeitsschritte
    erledigt wie seine Kollegen (mittlere log-Dauer relativ zum Mittel der Aktivität), in Dritteln der Abschnitte.
    Mitarbeiter mit weniger als MIN_SEGMENTS Abschnitten: medium.
    """
    segments = work_segments(df, prefix, system_users)
    segments["relative"] = np.log(segments["seconds"]) - segments.groupby("activity")["seconds"].transform(lambda s: np.log(s).mean())
    counts = segments.groupby("resource").size()
    tempo = segments.groupby("resource")["relative"].mean()[counts >= MIN_SEGMENTS].sort_values()
    share = counts[tempo.index].cumsum() / counts[tempo.index].sum()
    classes = {resource: CLASSES[0] if s <= 1 / 3 else CLASSES[1] if s <= 2 / 3 else CLASSES[2] for resource, s in share.items()}
    classes.update({resource: CLASSES[1] for resource in counts.index if resource not in classes})
    return classes


def fit_hours_per_activity(data: pd.DataFrame) -> Dict[str, float]:
    """
    Schätzt aus dem Event-Log (enriched_log.enrich, mit Spalte working_hours je Fall) die Arbeitsstunden, die ein
    Event jedes Tokens (Spalte event: Aktivität mit Lifecycle, z. B. "W_Validate application:start") im Mittel mit
    sich bringt: nicht-negative kleinste Quadrate von Arbeitsstunden je Fall auf die Anzahl der Events je Token.
    """
    from scipy.optimize import nnls

    # enriched_log.EVENT; ältere Logs ohne die Spalte: die Aktivität
    counts = pd.crosstab(data["case_nr"], data["event"] if "event" in data.columns else data[ACTIVITY])
    hours = data.groupby("case_nr")["working_hours"].first().reindex(counts.index)
    coefficients, _ = nnls(counts.to_numpy(dtype=float), hours.to_numpy(dtype=float))
    return dict(zip(counts.columns, coefficients))


def estimated_hours(activities: Iterable[str], hours_per_activity: Dict[str, float]) -> float:
    return float(sum(hours_per_activity.get(activity, 0.0) for activity in activities))
