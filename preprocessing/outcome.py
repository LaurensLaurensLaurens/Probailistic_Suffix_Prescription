"""
Das Outcome eines Falls, für alle Datensätze. Die einzige Stelle mit dieser Logik: das Notebook (preprocess -> add_outcome)
und die Pipeline in prescriptive/ (enriched_log.enrich, estimators/psp.py, evaluation/oracle_evaluation.py, ...) nutzen
dieselben Funktionen. Was nur ein Datensatz nutzt, steht in dessen Ordner (BPIC2017/outcome.py: der Gewinn).

Je Datensatz im Katalog (prescriptive/catalog.yaml) unter outcome festgelegt, Typen in OUTCOMES:

profit (BPIC17 als Option gewinn): Gewinn der Bank in €, in BPIC2017/outcome.py (dort die Definition)

negative_days_until (derzeit von keinem Datensatz genutzt): negative Tage vom Fallbeginn bis zum ersten bzw. letzten Event (which), das mit
    prefix beginnt; ohne ein solches bis zum letzten Event des Falls

negative_cycle_days (Standard für BPIC17 und BPIC19): negative Durchlaufzeit in Tagen, vom ersten bis zum letzten Event
    des Falls

negative_repair_cost (Werkstatt als Option kosten): -(Lohnkosten + Satz je Tag x Durchlaufzeit) in €, in
    prescriptive/costs/repair/outcome.py (dort die Definition, die Stundensätze in rates.py)

Neben outcome kann ein Datensatz im Katalog outcome_optionen haben (Name -> Block wie outcome); gewählt mit
preprocess(..., outcome=Name), im Notebook mit OUTCOME.
"""
from prescriptive.helpers.datasets import ACTIVITY, CASE, TIMESTAMP
from prescriptive.costs.repair.outcome import outcome_negative_repair_cost
from .BPIC2017.outcome import outcome_profit


def outcome_negative_days_until(log, spec):
    """Negative Tage vom Fallbeginn bis zum ersten bzw. letzten Event (which), das mit prefix beginnt; sonst bis zum letzten Event."""
    start = log.groupby(CASE)[TIMESTAMP].min()
    hit = log[log[ACTIVITY].astype(str).str.startswith(spec["prefix"])].groupby(CASE)[TIMESTAMP]
    end = (hit.min() if spec.get("which", "first") == "first" else hit.max()).reindex(start.index)
    end = end.fillna(log.groupby(CASE)[TIMESTAMP].max())
    return -(end - start).dt.total_seconds() / 86400


def outcome_negative_cycle_days(log, spec):
    """Negative Durchlaufzeit in Tagen: vom ersten bis zum letzten Event des Falls."""
    span = log.groupby(CASE)[TIMESTAMP].agg(["min", "max"])
    return -(span["max"] - span["min"]).dt.total_seconds() / 86400


def hours_until(activities, elapsed_hours, spec) -> float:
    """
    Wie outcome_negative_days_until für einen Fall als Folge (Aktivitäten und Stunden seit Fallbeginn, z. B. Prefix und
    simulierter Suffix in prescriptive/estimators/psp.py, oracle_evaluation.py): Stunden bis zum ersten bzw. letzten Event (which), das mit
    prefix beginnt, sonst bis zum letzten Event. Das Outcome ist -hours_until / 24.
    """
    hits = [hours for activity, hours in zip(activities, elapsed_hours) if str(activity).startswith(spec["prefix"])]
    if hits:
        return float(hits[0] if spec.get("which", "first") == "first" else hits[-1])
    return float(max(elapsed_hours)) if len(elapsed_hours) else 0.0


def suffix_outcome_until(activities, elapsed_hours, spec):
    """
    Outcome negative_days_until eines Falls aus Prefix und gezogenem Suffix (prescriptive/estimators/psp.py), wie
    suffix_outcome in BPIC2017/outcome.py: (Outcome, Ziel-Event erreicht, (0, 0, Tage bis dahin)).
    """
    hours = hours_until(activities, elapsed_hours, spec)
    resolved = any(str(activity).startswith(spec["prefix"]) for activity in activities)
    return -hours / 24, resolved, (0.0, 0.0, hours / 24)


OUTCOMES = {"profit": outcome_profit, "negative_days_until": outcome_negative_days_until, "negative_cycle_days": outcome_negative_cycle_days,
            "negative_repair_cost": outcome_negative_repair_cost}


def add_outcome(log, spec, costs=None):
    """
    Spalte outcome (Wert des Falls auf allen seinen Events) nach spec (Block outcome des Datensatzes im Katalog).
    costs: Kostensätze für alle Datensätze (hourly_rate, daily_cost; Standard helpers/parameters.py).
    """
    spec = {**spec, **(costs or {})}
    if spec["type"] not in OUTCOMES:
        raise ValueError(f"Unbekannter Outcome-Typ {spec['type']}, möglich: {', '.join(OUTCOMES)}")
    log["outcome"] = log[CASE].map(OUTCOMES[spec["type"]](log, spec))
    return log
