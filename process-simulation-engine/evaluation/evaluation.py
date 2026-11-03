"""
Compares a simulated event log with the real one, for every dataset: KPIs and diagrams.

The KPIs (size, events per case, throughput time, arrivals and completions per day, directly-follows edges, trace
variants, start and end activities, activity durations, activity distribution, resources), plus the split of the case
time into working, pausing and waiting, the shares and the outcome at the first decision point, and the case
attributes. What a dataset has comes from its settings (Setting): case attributes and work items
(next_event_prediction/datasets.py), system users (catalog prescriptive/catalog.yaml), decision point (the
intervention, interventions/), outcome (column outcome of the preprocessing). What only one dataset has lives in its
folder (DATASET_KPIS; BPIC2017/kpis.py: approval, offer attributes, callers).

    from evaluation.evaluation import evaluate
    table = evaluate(real_log, simulated_log, "bpic13", "evaluation/output/bpic13")   # logs after the preprocessing

Usage (from process-simulation-engine/):
    python -m evaluation.evaluation --dataset bpic17 --simulated <log.csv or .xes> [--real <log>] [--first-cases N]
"""
import argparse
import importlib
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

Events = List[Tuple[str, str]]  # (activity, lifecycle)

# State of a case between two events, read from the lifecycle of its work items (Setting.work_item_prefix)
WORKING, PAUSED, WAITING_FOR_START, NO_OPEN_ITEM = "in Arbeit", "pausiert", "wartet auf Start", "kein offener Arbeitsschritt"
_ITEM_STATE = {"schedule": WAITING_FOR_START, "start": WORKING, "resume": WORKING, "suspend": PAUSED}

REAL, SIMULATED = "echt", "simuliert"
# Okabe-Ito blue and orange: distinguishable with every common colour vision deficiency
COLORS = {REAL: "#0072B2", SIMULATED: "#E69F00"}
INK, MUTED = "#222222", "#6b6b6b"

# What only one dataset has: module with kpis(real, simulated), plots(real, simulated, save) and prepare(df, log)
DATASET_KPIS = {"bpic17": "evaluation.BPIC2017.kpis"}


@dataclass
class Setting:
    """How a dataset is evaluated (setting())."""
    dataset: str
    categorical: Dict[str, str]  # case attributes: column -> short name
    numeric: Dict[str, str]
    work_item_prefix: Optional[str]  # work items with sessions (None: no time per state, no work segments)
    system_users: set
    intervention: dict  # decision point (interventions/)

    @property
    def extension(self):
        return importlib.import_module(DATASET_KPIS[self.dataset]) if self.dataset in DATASET_KPIS else None


def setting(dataset: str, real_df: pd.DataFrame, intervention: Optional[str] = None) -> Setting:
    """The settings of the dataset; intervention: default the standard_intervention of the catalog."""
    import interventions
    from next_event_prediction.datasets import model_spec
    from resources.datasets import system_users
    spec = model_spec(dataset)
    short = lambda columns: {column: column.replace("case:", "", 1) for column in columns}
    name = intervention or interventions.catalog()[dataset]["standard_intervention"]
    return Setting(dataset, short(spec["static_categorical"]), short(spec["static_numeric"]), spec["work_item_prefix"],
                   system_users(dataset, real_df), interventions.get(name))


@dataclass
class Log:
    events: List[Events]  # per case, in order
    gaps: List[List[float]]  # per case: seconds before each event (0 for the first)
    starts: np.ndarray  # timestamp of the first event per case
    resources: Optional[List[List[str]]] = None
    # One row per case in the order of events: case attributes (short names) and start month
    attributes: Optional[pd.DataFrame] = None
    outcome: Optional[np.ndarray] = None  # per case, in order: column outcome of the preprocessing
    extra: Dict[str, object] = field(default_factory=dict)  # of the dataset (DATASET_KPIS, prepare)

    @property
    def ends(self) -> np.ndarray:
        return self.starts + np.array([sum(gaps) for gaps in self.gaps]).astype("timedelta64[s]")

    def head(self, n: int) -> "Log":
        """The first n cases (by start)."""
        attributes = self.attributes.iloc[:n] if self.attributes is not None else None
        extra = {key: value[value["case:concept:name"].isin(attributes.index)] if attributes is not None and isinstance(value, pd.DataFrame) else value
                 for key, value in self.extra.items()}
        return Log(self.events[:n], self.gaps[:n], self.starts[:n], self.resources[:n] if self.resources else None, attributes,
                   self.outcome[:n] if self.outcome is not None else None, extra)


def read_log(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    import pm4py

    return pm4py.convert_to_dataframe(pm4py.read_xes(path))


def from_dataframe(df: pd.DataFrame, setting: Optional[Setting] = None) -> Log:
    df = df.copy()
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True, format="mixed").dt.tz_localize(None)
    if "lifecycle:transition" not in df.columns:
        df["lifecycle:transition"] = "complete"
    df = df.sort_values(["case:concept:name", "time:timestamp"], kind="stable")
    cases = df.groupby("case:concept:name", sort=False)
    seconds = cases["time:timestamp"].diff().dt.total_seconds().fillna(0.0)
    order = cases["time:timestamp"].min().sort_values(kind="stable").index
    activities, lifecycles = cases["concept:name"].agg(list), cases["lifecycle:transition"].agg(list)
    gaps = seconds.groupby(df["case:concept:name"], sort=False).agg(list)
    resources = cases["org:resource"].agg(list) if "org:resource" in df.columns and df["org:resource"].notna().any() else None
    attributes = None
    if setting is not None:
        columns = {**setting.categorical, **setting.numeric}
        if columns and all(column in df.columns for column in columns):
            attributes = cases[list(columns)].first().loc[order].rename(columns=columns)
            for name in setting.numeric.values():
                attributes[name] = pd.to_numeric(attributes[name], errors="coerce")
            for name in setting.categorical.values():
                attributes[name] = attributes[name].astype(str)
        elif not columns:
            attributes = pd.DataFrame(index=order)
        if attributes is not None:
            attributes["Monat"] = cases["time:timestamp"].min().loc[order].dt.month
    outcome = cases["outcome"].first().loc[order].to_numpy(dtype=float) if "outcome" in df.columns else None
    log = Log(
        events=[list(zip(activities[c], lifecycles[c])) for c in order],
        gaps=[gaps[c] for c in order],
        starts=cases["time:timestamp"].min()[order].values.astype("datetime64[s]"),
        resources=[resources[c] for c in order] if resources is not None else None,
        attributes=attributes,
        outcome=outcome,
    )
    if setting is not None and setting.extension is not None:
        setting.extension.prepare(df, log)
    return log


# ---------------------------------------------------------------------------------------- KPIs


def gap_states(events: Events, prefix: str) -> List[str]:
    """
    For every event: the state of the case until the next event. A case is working if one of its
    work items (activities with prefix) was started or resumed, paused if one is suspended (mostly waiting
    for the customer), and waiting for start if one is scheduled but nobody has picked it up yet.
    """
    states, items = [], {}
    for activity, lifecycle in events:
        if activity.startswith(prefix):
            if lifecycle in _ITEM_STATE:
                items[activity] = _ITEM_STATE[lifecycle]
            else:
                items.pop(activity, None)
        open_states = set(items.values())
        states.append(next((state for state in (WORKING, PAUSED, WAITING_FOR_START) if state in open_states), NO_OPEN_ITEM))
    return states


def _share_distance(a: Counter, b: Counter) -> float:
    """Total variation distance between two frequency distributions, in percent (0 = identical)."""
    total_a, total_b = sum(a.values()), sum(b.values())
    return 50.0 * sum(abs(a[key] / total_a - b[key] / total_b) for key in set(a) | set(b))


def _overlap(a: Counter, b: Counter, top: int) -> Dict[str, float]:
    """a = real, b = generated."""
    common, union = set(a) & set(b), set(a) | set(b)
    top_common = {key for key, _ in a.most_common(top)} & {key for key, _ in b.most_common(top)}
    return {"count": len(b), "overlap": 100.0 * len(common) / len(a), "jaccard": 100.0 * len(common) / len(union), "top": len(top_common)}


def _edges(traces: List[Events]) -> Counter:
    return Counter((a[0], b[0]) for trace in traces for a, b in zip(trace, trace[1:]))


def control_flow_kpis(generated: List[Events], real: List[Events]) -> Dict[str, float]:
    """
    Activities, directly-follows edges and trace variants of generated against real traces, defined as
    in SimulationBenchmark: on the activity names of all events of a case, in order.
    """
    def counters(traces):
        activities = Counter(activity for trace in traces for activity, _ in trace)
        variants = Counter(tuple(activity for activity, _ in trace) for trace in traces)
        complete_variants = Counter(tuple(activity for activity, lifecycle in trace if lifecycle == "complete") for trace in traces)
        return activities, _edges(traces), variants, complete_variants, np.array([len(trace) for trace in traces])

    real_activities, real_edges, real_variants, real_complete, real_lengths = counters(real)
    activities, edges, variants, complete_variants, lengths = counters(generated)
    activity_overlap = _overlap(real_activities, activities, top=len(real_activities))
    edge_overlap = _overlap(real_edges, edges, top=20)
    variant_overlap = _overlap(real_variants, variants, top=10)
    complete_overlap = _overlap(real_complete, complete_variants, top=10)
    known_variants = sum(count for variant, count in variants.items() if variant in real_variants)
    known_complete = sum(count for variant, count in complete_variants.items() if variant in real_complete)
    return {
        "Fälle": len(generated),
        "Events": int(lengths.sum()),
        "Event-Verhältnis (%)": 100.0 * lengths.sum() / real_lengths.sum(),
        "verschiedene Aktivitäten": len(activities),
        "Events/Fall Mittel": lengths.mean(),
        "Events/Fall Median": float(np.median(lengths)),
        "Events/Fall P90": float(np.quantile(lengths, 0.9)),
        "Events/Fall Max": int(lengths.max()),
        "Aktivitäten Jaccard (%)": activity_overlap["jaccard"],
        "Aktivitäten Abstand der Anteile (%)": _share_distance(real_activities, activities),
        "Kanten Anzahl": edge_overlap["count"],
        "Kanten Überlappung (%)": edge_overlap["overlap"],
        "Kanten Jaccard (%)": edge_overlap["jaccard"],
        "Kanten Top 20 getroffen": edge_overlap["top"],
        "Kanten Abstand der Anteile (%)": _share_distance(real_edges, edges),
        "Varianten Anzahl": variant_overlap["count"],
        "Varianten Überlappung (%)": variant_overlap["overlap"],
        "Varianten Jaccard (%)": variant_overlap["jaccard"],
        "Varianten Top 10 getroffen": variant_overlap["top"],
        "Fälle mit echter Variante (%)": 100.0 * known_variants / len(generated),
        "Varianten nur complete: Anzahl": complete_overlap["count"],
        "Varianten nur complete: Jaccard (%)": complete_overlap["jaccard"],
        "Varianten nur complete: Top 10 getroffen": complete_overlap["top"],
        "Fälle mit echter Variante, nur complete (%)": 100.0 * known_complete / len(generated),
    }


def decision_kpis(traces: List[Events], outcome: Optional[np.ndarray], intervention: dict) -> Dict[str, float]:
    """
    At the first decision event of the intervention (interventions/): how often it is reached, how often the action
    follows (rule of the intervention) and the mean outcome of the cases, overall and per action.
    """
    import interventions
    reached, acted = [], []
    for trace in traces:
        labels = [label for label in interventions.decision_labels(trace, intervention) if label is not None]
        reached.append(bool(labels))
        acted.append(bool(labels) and bool(labels[0]))
    reached, acted = np.array(reached), np.array(acted)
    rows = {
        "erreicht Entscheidungspunkt (%)": 100.0 * reached.mean(),
        "Aktion am ersten Entscheidungspunkt (%)": 100.0 * acted[reached].mean() if reached.any() else float("nan"),
    }
    if outcome is not None:
        with_decision, action = outcome[reached], acted[reached]
        rows["Outcome je Fall: Mittel"] = float(np.nanmean(outcome))
        rows["Outcome ohne Aktion: Mittel"] = float(np.nanmean(with_decision[~action])) if (~action).any() else float("nan")
        rows["Outcome mit Aktion: Mittel"] = float(np.nanmean(with_decision[action])) if action.any() else float("nan")
    return rows


def time_per_state(traces: List[Events], gaps: List[List[float]], prefix: str) -> Dict[str, np.ndarray]:
    """Hours per case in each state (see gap_states), and in total."""
    per_case = {WORKING: [], PAUSED: [], WAITING_FOR_START: [], "gesamt": []}
    for events, case_gaps in zip(traces, gaps):
        totals = {WORKING: 0.0, PAUSED: 0.0, WAITING_FOR_START: 0.0}
        # the state after event j lasts until event j+1
        for state, gap in zip(gap_states(events, prefix)[:-1], case_gaps[1:]):
            if state in totals:
                totals[state] += gap / 3600.0
        for state, hours in totals.items():
            per_case[state].append(hours)
        per_case["gesamt"].append(sum(case_gaps[1:]) / 3600.0)
    return {state: np.array(hours) for state, hours in per_case.items()}


def time_kpis(traces: List[Events], gaps: List[List[float]], prefix: str) -> Dict[str, float]:
    """Time per case in hours: working, paused (mostly waiting for the customer), waiting for somebody to start a work item."""
    result = {}
    for name, hours in time_per_state(traces, gaps, prefix).items():
        result[f"{name}: Mittel (h)"] = hours.mean()
        result[f"{name}: Median (h)"] = float(np.median(hours))
        result[f"{name}: P90 (h)"] = float(np.quantile(hours, 0.9))
    return result


def daily_counts(timestamps: np.ndarray) -> pd.Series:
    """Cases per calendar day (days without a case are not counted, as in SimulationBenchmark)."""
    return pd.Series(pd.to_datetime(timestamps)).dt.floor("D").value_counts().sort_index()


def activity_durations(log: Log) -> pd.DataFrame:
    """Per activity: time until the next event of the case in hours (mean, median, P90)."""
    durations: Dict[str, List[float]] = {}
    for events, gaps in zip(log.events, log.gaps):
        for (activity, _), gap in zip(events[:-1], gaps[1:]):
            durations.setdefault(activity, []).append(gap / 3600.0)
    return pd.DataFrame({
        activity: {"Mittel (h)": np.mean(hours), "Median (h)": np.median(hours), "P90 (h)": np.quantile(hours, 0.9), "Anzahl": len(hours)}
        for activity, hours in durations.items()
    }).T


def _quantile_rows(values, unit: str, quantiles=(("Median", 0.5), ("P90", 0.9))) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    rows = {f"Mittel{unit}": values.mean()}
    rows.update({f"{name}{unit}": float(np.quantile(values, q)) for name, q in quantiles})
    rows[f"Maximum{unit}"] = values.max()
    return rows


def _ks(a, b) -> float:
    """Kolmogorov-Smirnov distance of two samples: largest gap between their distribution functions (0 = identical, 1 = disjoint)."""
    a, b = np.sort(np.asarray(a, dtype=float)), np.sort(np.asarray(b, dtype=float))
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    values = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, values, side="right") / len(a) - np.searchsorted(b, values, side="right") / len(b))))


_OPENING, _CLOSING = ("start", "resume"), ("suspend", "complete", "ate_abort", "withdraw")
MIN_SEGMENTS = 30


def work_segments(log: Log, prefix: str, system_users: set) -> pd.DataFrame:
    """One row per work segment: from start/resume of a work item to its suspend/complete/ate_abort/withdraw, with its employee."""
    rows = []
    for case, (events, gaps, resources) in enumerate(zip(log.events, log.gaps, log.resources)):
        clock, opened = 0.0, {}
        for (activity, lifecycle), gap, resource in zip(events, gaps, resources):
            clock += gap
            if not activity.startswith(prefix):
                continue
            if lifecycle in _OPENING:
                opened[activity] = (resource, clock)
            elif lifecycle in _CLOSING and activity in opened:
                employee, since = opened.pop(activity)
                rows.append((case, activity, employee, clock - since))
    segments = pd.DataFrame(rows, columns=["case", "activity", "resource", "seconds"])
    return segments[(segments["seconds"] > 0) & ~segments["resource"].isin(system_users)]


def resource_effect_kpis(real: Log, simulated: Log, setting: Setting) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """Do employees differ as in the real log? Working time per work segment by employee and activity."""
    rows: Dict[Tuple[str, str], Tuple[float, float]] = {}
    if real.resources is None or simulated.resources is None or not setting.work_item_prefix:
        return rows
    area = "Mitarbeiter-Effekte"

    def tempo(log: Log):
        # Per activity: share of the variance of the log working time explained by the employee, and the
        # employee's tempo (mean log time minus that of the activity); only employees with enough segments
        segments = work_segments(log, setting.work_item_prefix, setting.system_users)
        segments["log"] = np.log(segments["seconds"])
        explained, total, factors = 0.0, 0.0, {}
        for activity, group in segments.groupby("activity"):
            counts = group["resource"].value_counts()
            group = group[group["resource"].isin(counts[counts >= MIN_SEGMENTS].index)]
            if group["resource"].nunique() < 5:
                continue
            means = group.groupby("resource")["log"].mean()
            overall = group["log"].mean()
            explained += ((group.groupby("resource")["log"].transform("mean") - overall) ** 2).sum()
            total += ((group["log"] - overall) ** 2).sum()
            factors.update({(resource, activity): value for resource, value in (means - overall).items()})
        return (explained / total * 100 if total else float("nan")), pd.Series(factors)

    (share_real, factors_real), (share_simulated, factors_simulated) = tempo(real), tempo(simulated)
    rows[(area, "Arbeitsabschnitt: Streuung durch den Mitarbeiter (%)")] = (share_real, share_simulated)
    common = factors_real.index.intersection(factors_simulated.index)
    rows[(area, "Tempo je Mitarbeiter und Aktivität: Korrelation mit echt")] = (
        1.0, float(np.corrcoef(factors_real[common], factors_simulated[common])[0, 1]) if len(common) > 2 else float("nan"))
    return rows


def _shares(column: str, frame_real: pd.DataFrame, frame_simulated: pd.DataFrame) -> pd.DataFrame:
    detail = pd.DataFrame({"Anteil echt (%)": frame_real[column].value_counts(normalize=True) * 100,
                           "Anteil simuliert (%)": frame_simulated[column].value_counts(normalize=True) * 100}).fillna(0.0)
    detail["Differenz"] = detail["Anteil simuliert (%)"] - detail["Anteil echt (%)"]
    return detail.sort_values("Anteil echt (%)", ascending=False)


def attribute_kpis(real: Log, simulated: Log, setting: Setting) -> Tuple[Dict[Tuple[str, str], Tuple[float, float]], Dict[str, pd.DataFrame]]:
    """
    The case attributes of the dataset (Setting) in both logs: distributions, their distances, how two attributes fit
    together and the change over the year. Comparative KPIs (distances) have 0 for the real log, the reference.
    """
    rows: Dict[Tuple[str, str], Tuple[float, float]] = {}
    details: Dict[str, pd.DataFrame] = {}
    if real.attributes is None or simulated.attributes is None:
        return rows, details
    r, s = real.attributes, simulated.attributes
    categorical, numeric = list(setting.categorical.values()), list(setting.numeric.values())
    area = "Fallattribute"
    for name in categorical:
        details[name] = _shares(name, r, s)
        rows[(area, f"{name}: Abstand der Anteile (%)")] = (0.0, details[name]["Differenz"].abs().sum() / 2)
    if len(categorical) >= 2:
        first, second = categorical[:2]
        pairs = _shares("Paar", r.assign(Paar=r[first] + " | " + r[second]), s.assign(Paar=s[first] + " | " + s[second]))
        rows[(area, f"{first} × {second}: Abstand der Anteile (%)")] = (0.0, pairs["Differenz"].abs().sum() / 2)
    for name in numeric:
        for label, function in (("Mittel", np.mean), ("Median", np.median), ("P90", lambda v: np.quantile(v, 0.9))):
            rows[(area, f"{name}: {label}")] = tuple(float(function(frame[name].dropna())) for frame in (r, s))
        rows[(area, f"{name}: Anteil 0 (%)")] = tuple(float((frame[name] == 0).mean() * 100) for frame in (r, s))
        rows[(area, f"{name}: KS-Abstand")] = (0.0, _ks(r[name], s[name]))
        details[f"{name} je Monat"] = pd.DataFrame({REAL: r.groupby("Monat")[name].mean(), SIMULATED: s.groupby("Monat")[name].mean()})
        if categorical:
            # Does the amount fit the first categorical attribute? Median per category, mean relative deviation
            medians = pd.DataFrame({REAL: r.groupby(categorical[0])[name].median(), SIMULATED: s.groupby(categorical[0])[name].median()}).dropna()
            medians = medians[medians[REAL] > 0]
            rows[(area, f"{name} je {categorical[0]}: mittl. Abweichung des Medians (%)")] = (0.0, float((medians[SIMULATED] / medians[REAL] - 1).abs().mean() * 100))
    if categorical:
        # Change over the year: per start month the distance of the shares of the first categorical attribute
        monthly = [_shares(categorical[0], r[r["Monat"] == month], s[s["Monat"] == month])["Differenz"].abs().sum() / 2
                   for month in sorted(set(r["Monat"]) & set(s["Monat"]))]
        rows[(area, f"{categorical[0]}: Abstand der Anteile je Startmonat, Mittel (%)")] = (0.0, float(np.mean(monthly)) if monthly else float("nan"))
    return rows, details


def compare(real: Log, simulated: Log, setting: Setting) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    All KPIs of both logs. Returns the KPI table (index: area and KPI; columns: real, simulated) and
    the detail tables per activity and per edge.
    """
    rows: Dict[Tuple[str, str], Tuple[float, float]] = {}

    def add(area: str, values_real: Dict[str, float], values_simulated: Dict[str, float]) -> None:
        for name in values_real:
            rows[(area, name)] = (values_real[name], values_simulated.get(name, float("nan")))

    def both(function):
        return function(real), function(simulated)

    add("Umfang", *both(lambda log: {
        "Fälle": len(log.events), "Events": sum(len(e) for e in log.events),
        "verschiedene Aktivitäten": len({a for e in log.events for a, _ in e}),
    }))
    add("Events pro Fall", *both(lambda log: _quantile_rows([len(e) for e in log.events], "")))
    throughput = (("Median", 0.5), ("P75", 0.75), ("P90", 0.9), ("P95", 0.95))
    add("Durchlaufzeit pro Fall", *both(lambda log: _quantile_rows([sum(g) / 3600.0 for g in log.gaps], " (h)", throughput)))
    add("Ankünfte pro Tag", *both(lambda log: _quantile_rows(daily_counts(log.starts), "")))
    add("Abschlüsse pro Tag", *both(lambda log: _quantile_rows(daily_counts(log.ends), "")))

    # Control flow: the comparative KPIs have no value of their own for the real log (it is the reference)
    flow = control_flow_kpis(simulated.events, real.events)
    reference = control_flow_kpis(real.events, real.events)
    for area, names in (
        ("Kontrollfluss (Kanten)", ["Kanten Anzahl", "Kanten Überlappung (%)", "Kanten Jaccard (%)", "Kanten Top 20 getroffen", "Kanten Abstand der Anteile (%)"]),
        ("Trace-Varianten", ["Varianten Anzahl", "Varianten Überlappung (%)", "Varianten Jaccard (%)", "Varianten Top 10 getroffen", "Fälle mit echter Variante (%)",
                             "Varianten nur complete: Anzahl", "Varianten nur complete: Jaccard (%)", "Fälle mit echter Variante, nur complete (%)"]),
        ("Aktivitätsverteilung", ["Aktivitäten Jaccard (%)", "Aktivitäten Abstand der Anteile (%)"]),
    ):
        for name in names:
            rows[(area, name)] = (reference[name], flow[name])

    def first_last(log: Log, index: int) -> Counter:
        return Counter(events[index][0] for events in log.events if events)

    for area, index in (("Startaktivitäten", 0), ("Endaktivitäten", -1)):
        counts_real, counts_simulated = first_last(real, index), first_last(simulated, index)
        rows[(area, "verschiedene Aktivitäten")] = (len(counts_real), len(counts_simulated))
        rows[(area, "Abstand der Anteile (%)")] = (0.0, _share_distance(counts_real, counts_simulated))

    durations = activity_durations(real).join(activity_durations(simulated), lsuffix=" echt", rsuffix=" simuliert", how="outer")
    ratio = durations["Mittel (h) simuliert"] / durations["Mittel (h) echt"]
    # Many activities are followed by the next event within a second; there a relative deviation says nothing
    close = ((ratio > 0.75) & (ratio < 1.25)) | ((durations["Mittel (h) simuliert"] - durations["Mittel (h) echt"]).abs() < 1 / 60)
    rows[("Aktivitätsdauern", "Aktivitäten mit mittlerer Dauer innerhalb ±25 % oder 1 Minute")] = (len(durations), int(close.sum()))
    rows[("Aktivitätsdauern", "mittlere Zeit bis zum nächsten Event (h)")] = tuple(
        np.mean([gap for gaps in log.gaps for gap in gaps[1:]]) / 3600.0 for log in (real, simulated)
    )

    if real.resources is not None and simulated.resources is not None:
        def resource_kpis(log: Log) -> Dict[str, float]:
            counts = Counter(r for case in log.resources for r in case)
            # Workload of the employees without the system users, who do a large share of all events
            workload = np.sort([count for resource, count in counts.items() if resource not in setting.system_users])
            gini = (2 * np.arange(1, len(workload) + 1) - len(workload) - 1).dot(workload) / (len(workload) * workload.sum())
            handovers = [sum(a != b for a, b in zip(case, case[1:])) for case in log.resources]
            return {
                "Anzahl": len(counts),
                **{f"Events pro Ressource, {name}": value for name, value in _quantile_rows(list(counts.values()), "").items()},
                "Übergaben pro Fall": float(np.mean(handovers)),
                "Gini der Arbeitslast (0 = gleich verteilt)": float(gini),
            }

        add("Ressourcen", *both(resource_kpis))
        rows.update(resource_effect_kpis(real, simulated, setting))
    if setting.work_item_prefix:
        add("Zeit pro Fall nach Zustand", *both(lambda log: time_kpis(log.events, log.gaps, setting.work_item_prefix)))
    add("Entscheidungspunkt", *both(lambda log: decision_kpis(log.events, log.outcome, setting.intervention)))
    attribute_rows, attribute_details = attribute_kpis(real, simulated, setting)
    rows.update(attribute_rows)
    if setting.extension is not None:
        rows.update(setting.extension.kpis(real, simulated))
    # Rows of the dataset next to the general ones of their area
    areas = list(dict.fromkeys(area for area, _ in rows))
    rows = dict(sorted(rows.items(), key=lambda item: areas.index(item[0][0])))

    table = pd.DataFrame(rows, index=[REAL, SIMULATED]).T
    table.index.names = ["Bereich", "Kennzahl"]

    def shares(counter_real: Counter, counter_simulated: Counter) -> pd.DataFrame:
        detail = pd.DataFrame({"Anteil echt (%)": pd.Series(counter_real) / sum(counter_real.values()) * 100,
                               "Anteil simuliert (%)": pd.Series(counter_simulated) / sum(counter_simulated.values()) * 100}).fillna(0.0)
        detail["Differenz"] = detail["Anteil simuliert (%)"] - detail["Anteil echt (%)"]
        return detail.sort_values("Anteil echt (%)", ascending=False)

    activity_counts = [Counter(a for events in log.events for a, _ in events) for log in (real, simulated)]
    details = {
        "Aktivitäten": shares(*activity_counts),
        "Kanten": shares(_edges(real.events), _edges(simulated.events)),
        "Aktivitätsdauern": durations,
        "Startaktivitäten": shares(first_last(real, 0), first_last(simulated, 0)),
        "Endaktivitäten": shares(first_last(real, -1), first_last(simulated, -1)),
        **attribute_details,
    }
    return table, details


# ---------------------------------------------------------------------------------------- diagrams


def _axes(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cfcfcf")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color="#ececec", linewidth=0.8)
    ax.set_axisbelow(True)


def _dumbbell(ax, detail: pd.DataFrame, title: str, top: int = 20) -> None:
    """One row per category: the real and the simulated share as two dots, joined by a line."""
    detail = detail.head(top).iloc[::-1]
    labels = [" → ".join(index) if isinstance(index, tuple) else (str(index) if str(index).strip() else "(leer)") for index in detail.index]
    positions = np.arange(len(detail))
    ax.hlines(positions, detail["Anteil echt (%)"], detail["Anteil simuliert (%)"], color="#b5b5b5", linewidth=1.5, zorder=1)
    # The real dot is larger, so it stays visible as a ring where both values coincide
    for name, column, size in ((REAL, "Anteil echt (%)", 70), (SIMULATED, "Anteil simuliert (%)", 22)):
        ax.scatter(detail[column], positions, s=size, color=COLORS[name], label=name, zorder=2, edgecolor="white", linewidth=0.8)
    ax.set_yticks(positions, labels)
    ax.set_xlim(0, detail[["Anteil echt (%)", "Anteil simuliert (%)"]].to_numpy().max() * 1.08)
    _axes(ax, title, xlabel="Anteil (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")


def cdf(ax, values_per_log, title: str, xlabel: str, log_x: bool = False) -> None:
    for name, values in values_per_log:
        values = np.sort(np.asarray(values, dtype=float))
        values = values[np.isfinite(values) & ((values > 0) if log_x else True)]
        ax.plot(values, np.arange(1, len(values) + 1) / len(values) * 100, linewidth=2, color=COLORS[name], label=name)
    if log_x:
        ax.set_xscale("log")
    _axes(ax, title, xlabel=xlabel, ylabel="höchstens so groß (%)")


def plot_all(real: Log, simulated: Log, table: pd.DataFrame, details: Dict[str, pd.DataFrame], folder: str, setting: Setting) -> List[str]:
    """Writes one diagram per KPI area into the folder and returns the file names."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(folder, exist_ok=True)
    files = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(os.path.join(folder, name), dpi=140)
        plt.close(fig)
        files.append(name)

    logs = ((REAL, real), (SIMULATED, simulated))

    # Events per case and throughput time: distributions
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    lengths = {name: np.array([len(e) for e in log.events]) for name, log in logs}
    bins = np.arange(0, np.quantile(lengths[REAL], 0.995) + 4, 4)
    for name, values in lengths.items():
        left.hist(values, bins=bins, density=True, histtype="step", linewidth=2, color=COLORS[name], label=name)
    _axes(left, "Events pro Fall", xlabel="Events", ylabel="Anteil der Fälle (Dichte)")
    left.legend(frameon=False, fontsize=9)
    for name, log in logs:
        days = np.sort([sum(g) / 86400.0 for g in log.gaps])
        right.plot(days, np.arange(1, len(days) + 1) / len(days) * 100, linewidth=2, color=COLORS[name], label=name)
    right.set_xlim(0, np.quantile([sum(g) / 86400.0 for g in real.gaps], 0.995))
    _axes(right, "Durchlaufzeit pro Fall", xlabel="Tage", ylabel="Fälle, die höchstens so lange dauern (%)")
    right.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, "1_events_und_durchlaufzeit.png")

    # Arrivals and completions per day: two panels, one measure each
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, title, timestamps in ((axes[0], "Ankünfte pro Tag", lambda log: log.starts), (axes[1], "Abschlüsse pro Tag", lambda log: log.ends)):
        for name, log in logs:
            counts = daily_counts(timestamps(log))
            ax.plot(counts.index, counts.values, linewidth=1.3, color=COLORS[name], label=name)
        _axes(ax, title, ylabel="Fälle")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "2_ankuenfte_und_abschluesse.png")

    # Activity distribution and most frequent edges
    fig, ax = plt.subplots(figsize=(9, 8))
    _dumbbell(ax, details["Aktivitäten"], "Anteil der Aktivitäten an allen Events", top=30)
    save(fig, "3_aktivitaetsverteilung.png")
    fig, ax = plt.subplots(figsize=(10, 7))
    _dumbbell(ax, details["Kanten"], "Die 20 häufigsten Übergänge zwischen Aktivitäten", top=20)
    save(fig, "4_kanten_top20.png")

    # Start and end activities
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
    _dumbbell(left, details["Startaktivitäten"], "Erste Aktivität eines Falls", top=6)
    _dumbbell(right, details["Endaktivitäten"], "Letzte Aktivität eines Falls", top=10)
    save(fig, "5_start_und_endaktivitaeten.png")

    # Activity durations: one dot per activity, real against simulated mean
    durations = details["Aktivitätsdauern"].dropna(subset=["Mittel (h) echt", "Mittel (h) simuliert"])
    fig, ax = plt.subplots(figsize=(7, 6))
    x, y = durations["Mittel (h) echt"].clip(lower=1e-4), durations["Mittel (h) simuliert"].clip(lower=1e-4)
    limits = [min(x.min(), y.min()) / 2, max(x.max(), y.max()) * 2]
    ax.plot(limits, limits, color="#b5b5b5", linewidth=1, zorder=1)
    ax.scatter(x, y, s=36, color=COLORS[SIMULATED], edgecolor="white", linewidth=0.8, zorder=2)
    # Label only activities that are off by more than a factor of 1.5 and take longer than a minute
    far = (np.abs(np.log(y / x)) > np.log(1.5)) & (np.maximum(x, y) > 1 / 60)
    for activity in durations.index[far]:
        ax.annotate(activity, (x[activity], y[activity]), xytext=(5, 3), textcoords="offset points", fontsize=7, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _axes(ax, "Mittlere Zeit bis zum nächsten Event, je Aktivität\n(auf der Linie: simuliert = echt)", xlabel="echt (Stunden)", ylabel="simuliert (Stunden)")
    save(fig, "6_aktivitaetsdauern.png")

    if setting.work_item_prefix:
        # Time per case by state: the three states differ by orders of magnitude, so one panel each
        states = {name: time_per_state(log.events, log.gaps, setting.work_item_prefix) for name, log in logs}
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        for ax, state, title in zip(axes, (WORKING, PAUSED, WAITING_FOR_START), ("Bearbeitung", "Pausen (v. a. Warten auf Kunden)", "Warten auf den Start")):
            positions = np.arange(2)
            for offset, (name, _) in zip((-0.2, 0.2), logs):
                values = [states[name][state].mean(), np.median(states[name][state])]
                bars = ax.bar(positions + offset, values, width=0.36, color=COLORS[name], label=name)
                ax.bar_label(bars, fmt="%.1f", fontsize=8, color=INK, padding=2)
            ax.set_xticks(positions, ["Mittel", "Median"])
            _axes(ax, title, ylabel="Stunden pro Fall")
            ax.grid(axis="x", visible=False)
        axes[0].legend(frameon=False, fontsize=9)
        save(fig, "7_zeit_nach_zustand.png")

    # Decision point: the shares in percent, and the mean outcome per action
    decision = table.loc["Entscheidungspunkt"]
    percent = decision[[label.endswith("(%)") for label in decision.index]]
    outcome = decision[[label.startswith("Outcome") for label in decision.index]]
    fig, axes = plt.subplots(1, 2 if len(outcome) else 1, figsize=(13 if len(outcome) else 9, 3.8), squeeze=False)
    for ax, values, title, ylabel, fmt in ((axes[0, 0], percent, "Entscheidungspunkt: Anteile in Prozent", "%", "%.0f"),
                                           *(((axes[0, 1], outcome, f"Outcome je Fall ({setting.dataset})", "Outcome", "%.1f"),) if len(outcome) else ())):
        positions = np.arange(len(values))
        for offset, name in zip((-0.2, 0.2), (REAL, SIMULATED)):
            bars = ax.bar(positions + offset, values[name], width=0.36, color=COLORS[name], label=name)
            ax.bar_label(bars, fmt=fmt, fontsize=8, color=INK, padding=2)
        ax.set_xticks(positions, [label.replace(" (%)", "").replace(": Mittel", "").replace(", ", ",\n").replace(" am ", "\nam ").replace("erreicht ", "erreicht\n")
                                  for label in values.index], fontsize=8)
        if ylabel == "%":
            ax.set_ylim(0, 100)
        _axes(ax, title, ylabel=ylabel)
        ax.grid(axis="x", visible=False)
        ax.axhline(0, color="#cfcfcf", linewidth=0.8)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper right")
    save(fig, "8_entscheidungspunkt.png")

    categorical, numeric = list(setting.categorical.values()), list(setting.numeric.values())
    if categorical and categorical[0] in details:
        # Case attributes: shares of the first categorical attribute; with a numeric one its distribution and the year
        fig, axes = plt.subplots(1, 3 if numeric else 1, figsize=(15 if numeric else 7, 4.6), squeeze=False,
                                 gridspec_kw={"width_ratios": [1.3, 1, 1]} if numeric else None)
        _dumbbell(axes[0, 0], details[categorical[0]], f"{categorical[0]}: Anteil der Fälle", top=14)
        if numeric:
            name = numeric[0]
            cdf(axes[0, 1], [(log_name, log.attributes[name]) for log_name, log in logs], f"{name} (ohne 0)", "Wert (log)", log_x=True)
            axes[0, 1].legend(frameon=False, fontsize=9, loc="lower right")
            monthly = details[f"{name} je Monat"]
            for log_name in (REAL, SIMULATED):
                axes[0, 2].plot(monthly.index, monthly[log_name], marker="o", markersize=4, linewidth=2, color=COLORS[log_name], label=log_name)
            axes[0, 2].set_xticks(range(1, 13))
            _axes(axes[0, 2], f"{name} im Mittel je Startmonat", xlabel="Monat", ylabel="Wert")
            axes[0, 2].legend(frameon=False, fontsize=9)
        save(fig, "9_fallattribute.png")

    if setting.extension is not None:
        setting.extension.plots(real, simulated, save)
    return files


def evaluate(real_df: pd.DataFrame, simulated_df: pd.DataFrame, dataset: str, folder: str, intervention: Optional[str] = None,
             first_cases: Optional[int] = None) -> pd.DataFrame:
    """
    Computes all KPIs of the dataset, writes the tables and the diagrams into the folder and returns the KPI table.
    real_df, simulated_df: both after the preprocessing (prescriptive/helpers/preprocessing; with the column outcome).
    first_cases: compare only the first cases of the real log (by start), e.g. as many as were simulated.
    """
    chosen = setting(dataset, real_df, intervention)
    real, simulated = from_dataframe(real_df, chosen), from_dataframe(simulated_df, chosen)
    if first_cases:
        real = real.head(first_cases)
    os.makedirs(folder, exist_ok=True)
    table, details = compare(real, simulated, chosen)
    table.to_csv(os.path.join(folder, "kennzahlen.csv"))
    for name, detail in details.items():
        detail.to_csv(os.path.join(folder, f"detail_{name.replace('/', '_')}.csv"))
    plot_all(real, simulated, table, details, folder, chosen)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bpic17", help="Datensatz (Katalog prescriptive/catalog.yaml)")
    parser.add_argument("--simulated", required=True, help="Simulated event log (.csv, .xes or .xes.gz)")
    parser.add_argument("--real", default=None, help="Real event log (default: the one of the dataset, after the preprocessing)")
    parser.add_argument("--intervention", default=None, help="Decision point (default: standard_intervention of the dataset)")
    parser.add_argument("--output", default=None, help="Folder for tables and diagrams (default: evaluation/output/<dataset>)")
    parser.add_argument("--first-cases", type=int, default=None,
                        help="Compare only the first cases of the real log (by start), e.g. as many as were simulated")
    args = parser.parse_args()

    from next_event_prediction.train import prepared_log
    project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import sys
    if project not in sys.path:
        sys.path.insert(0, project)
    from prescriptive.helpers.datasets import dataset_info
    from prescriptive.helpers.simulator_check import prepare_simulated
    real_df = prepared_log(args.dataset, args.real)
    simulated_df = prepare_simulated(read_log(args.simulated), dataset_info(args.dataset)["outcome"])
    output = args.output or os.path.join("evaluation", "output", args.dataset)
    kpis = evaluate(real_df, simulated_df, args.dataset, output, args.intervention, args.first_cases)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(kpis.round(1).to_string())
    print(f"\nTabellen und Diagramme: {output}")
