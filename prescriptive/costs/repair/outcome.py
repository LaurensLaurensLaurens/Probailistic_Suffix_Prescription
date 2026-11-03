"""
Outcome kosten der Werkstatt (Katalog repair, outcome_optionen kosten und kosten_b; Typ negative_repair_cost):

    Outcome = -(Lohnkosten aller Arbeitsschritte + cost_per_day x Durchlaufzeit in Tagen), in €

Lohnkosten: je Arbeitsschritt (W_..., start bis complete) der Satz seines Mitarbeiters (rates.py, Tarif tarif), Stunde für
Stunde aufsummiert. Durchlaufzeit wie negative_cycle_days: vom ersten bis zum letzten Event des Falls.

Eine Stelle für alle:
- case_costs / outcome_negative_repair_cost: aus einem Log (echter Log, RCT-Daten, Online-Auswertung, echte Werkstatt);
  eingehängt in preprocessing/outcome.py (OUTCOMES) und prescriptive/enriched_log.py (OUTCOME_COLUMNS, with_current_costs)
- SuffixCosts: für das PSP aus dem Prefix, dem Options-Event und je gezogenem Suffix (estimators/psp_model.py outcome).
  Das PSP zieht keine Mitarbeiter: Beginnt der Arbeitsschritt des Wahlpunkts im Suffix neu (Nacharbeit), macht ihn der
  gewählte Mitarbeiter, alle anderen Schritte kosten den mittleren Satz ihrer Rolle (rates.ROLES). Schritte, die am Ende
  des Suffixes noch offen sind, enden mit seinem letzten Event. Die Zeiten des Suffixes: Options-Event plus die gezogenen
  Abstände aufsummiert (nicht die gezogene Zeit seit Fallbeginn, die die Unterschiede zwischen Mitarbeitern staucht).
"""
import numpy as np
import pandas as pd

from . import rates

CASE, ACTIVITY, RESOURCE, TIMESTAMP, LIFECYCLE = "case:concept:name", "concept:name", "org:resource", "time:timestamp", "lifecycle:transition"
WORK = "W_"  # Präfix der Arbeitsschritte (start, complete)
DAY_HOURS = 24.0


def work_intervals(log):
    """Die Arbeitsschritte des Logs: je Fall, Schritt und Durchgang Mitarbeiter, Beginn und Ende (Stunden seit rates.EPOCH)."""
    work = log[log[ACTIVITY].astype(str).str.startswith(WORK) & log[LIFECYCLE].isin(["start", "complete"])]
    work = work[[CASE, ACTIVITY, LIFECYCLE, RESOURCE, TIMESTAMP]].sort_values([CASE, TIMESTAMP], kind="stable").copy()
    work["k"] = work.groupby([CASE, ACTIVITY, LIFECYCLE]).cumcount()
    starts, ends = work[work[LIFECYCLE] == "start"], work[work[LIFECYCLE] == "complete"]
    pairs = starts.merge(ends[[CASE, ACTIVITY, "k", TIMESTAMP]], on=[CASE, ACTIVITY, "k"], suffixes=("", "_end"))
    return pd.DataFrame({CASE: pairs[CASE].to_numpy(), ACTIVITY: pairs[ACTIVITY].to_numpy(), RESOURCE: pairs[RESOURCE].astype(str).to_numpy(),
                         "start": rates.hours(pairs[TIMESTAMP]), "end": rates.hours(pairs[TIMESTAMP + "_end"])})


def case_costs(log, spec):
    """Je Fall Lohnkosten (€), Durchlaufzeit (Tage) und Kosten (€) nach spec (tarif, cost_per_day)."""
    plan = rates.plan(spec.get("tarif", "A"))
    intervals = work_intervals(log)
    labor = pd.Series(plan.cost(plan.rows(intervals[RESOURCE]), intervals["start"], intervals["end"]), index=intervals.index)
    span = log.groupby(CASE)[TIMESTAMP].agg(["min", "max"])
    table = pd.DataFrame({"Lohnkosten (€)": labor.groupby(intervals[CASE]).sum().reindex(span.index).fillna(0.0),
                          "Durchlaufzeit (Tage)": (pd.to_datetime(span["max"]) - pd.to_datetime(span["min"])).dt.total_seconds() / 86400})
    table["Kosten (€)"] = table["Lohnkosten (€)"] + float(spec["cost_per_day"]) * table["Durchlaufzeit (Tage)"]
    return table


def outcome_negative_repair_cost(log, spec):
    """Outcome je Fall: -(Lohnkosten + cost_per_day x Durchlaufzeit) in € (preprocessing/outcome.py OUTCOMES)."""
    return -case_costs(log, spec)["Kosten (€)"]


def enriched_columns(df, spec):
    """Auf dem angereicherten Log (enriched_log): Spalten labor_cost und outcome je Fall."""
    table = case_costs(df, spec)
    df["labor_cost"] = df[CASE].map(table["Lohnkosten (€)"])
    df["outcome"] = -df[CASE].map(table["Kosten (€)"])
    return df


class SuffixCosts:
    """
    Kosten eines Falls für das PSP. events: die Events des Falls bis jetzt (dicts wie engine.events); option_event: das
    Options-Event am Ende des Prefix (psp_model._option_event); employee: der gewählte Mitarbeiter (Mitarbeiterpunkt) oder
    None (Control Flow); activity: der Arbeitsschritt des Wahlpunkts. Aufruf mit der Zeit seit Fallbeginn am Options-Event
    (Sekunden) und einem gezogenen Suffix (Liste (Token Aktivität|Lifecycle, Sekunden seit Fallbeginn, Abstand in Sekunden)):
    Outcome in €, die Zeiten aus den Abständen.
    """

    def __init__(self, spec, events, option_event, employee=None, activity=None):
        self.plan, self.per_day = rates.plan(spec.get("tarif", "A")), float(spec["cost_per_day"])
        self.employee, self.activity = employee, activity
        self.start = rates.hours(events[0][TIMESTAMP])
        closed, self.open = [], {}
        for event in list(events) + [option_event]:
            name, lifecycle = str(event[ACTIVITY]), event.get(LIFECYCLE) or "complete"
            if not name.startswith(WORK):
                continue
            at = rates.hours(event[TIMESTAMP])
            if lifecycle == "start":
                resource = self._who(name) if event is option_event else event.get(RESOURCE)
                self.open[name] = (at, resource)
            elif lifecycle == "complete" and name in self.open:
                began, resource = self.open.pop(name)
                closed.append((resource, began, at))
        self.sunk = self._cost(closed)

    def _who(self, name):
        return self.employee if self.employee is not None and name == self.activity else rates.role(name)

    def _cost(self, intervals):
        if not intervals:
            return 0.0
        resources, began, ended = zip(*intervals)
        return float(self.plan.cost(self.plan.rows(list(resources)), began, ended).sum())

    def __call__(self, prefix_elapsed, suffix):
        # Zeitachse aus den gezogenen Abständen ab dem Options-Event, nicht aus der gezogenen Zeit seit Fallbeginn: diese staucht
        # die Dauer der Reparatur je Mitarbeiter (neutral minus stark 0,6 statt 1,2 Tage), die Abstände treffen sie (1,18)
        running, closed, last = dict(self.open), [], prefix_elapsed
        for label, _, gap in suffix:
            last += gap
            seconds = last
            name, _, lifecycle = str(label).partition("|")
            if not name.startswith(WORK):
                continue
            at = self.start + seconds / 3600
            if lifecycle == "start":
                running[name] = (at, self._who(name))
            elif lifecycle == "complete" and name in running:
                began, resource = running.pop(name)
                closed.append((resource, began, at))
        end = self.start + last / 3600
        closed += [(resource, began, max(end, began)) for began, resource in running.values()]
        return -(self.sunk + self._cost(closed) + self.per_day * last / 86400)
