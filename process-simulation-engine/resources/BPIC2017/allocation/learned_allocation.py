"""
Zuteilung "learned": wen die echte Bank für einen Arbeitsschritt wählt, gelernt aus dem echten Log.

Die Zuteilung "bank" gewichtet nur nach der Erfahrung je Aktivität. Die echte Bank folgt weiteren Regeln, etwa:
- Vier-Augen-Prinzip: Wer validiert (W_Validate application), hat den Fall vorher fast nie bearbeitet (8 %).
- Spezialisierung: Die 10 aktivsten Mitarbeiter übernehmen 62 % der Validierungen und 66 % der Anrufe.
- Kontinuität: Wer den Antrag erfasst, macht ihn meist auch fertig (W_Complete application, 93 %).

Modell: je Arbeitsschritt und Öffnung (start / resume) ein bedingtes Logit über die Kandidaten. Wahrscheinlichkeit eines
Kandidaten proportional zu exp(Gewichte · Merkmale). Merkmale je Kandidat (FEATURES):
- experience: log1p, wie oft er den Arbeitsschritt im echten Log begonnen oder fortgesetzt hat
- share: welcher Anteil seiner eigenen Arbeit auf diesen Arbeitsschritt entfällt (Spezialist gegen Allrounder)
- last: ob er zuletzt einen Arbeitsschritt des Falls begonnen oder fortgesetzt hat
- handled: ob er schon einen Arbeitsschritt des Falls begonnen oder fortgesetzt hat
- same_item: ob er diesen Arbeitsschritt im Fall zuletzt begonnen oder fortgesetzt hat
Nur Beginn und Fortsetzung zählen, keine Abschlüsse: Wer eine Validierung beginnt, schließt im selben Moment oft noch
offene Arbeitsschritte des Falls (z. B. W_Call after offers|ate_abort). Mit Abschlüssen war "last" deshalb fast immer der
Gewählte selbst; das Merkmal verriet die Antwort, und in der Simulation entstand zu viel Kontinuität.
Im Training sind die Kandidaten wie in der Simulation die freien Mitarbeiter im Dienst mit Erfahrung für den Schritt
(MIN_EXPERIENCE): im Dienst = eigene Events höchstens ON_DUTY_HOURS vor oder nach dem Moment, frei = kein anderer
Arbeitsschritt offen (start/resume ohne suspend/complete). Mit allen, die an dem Tag arbeiteten, auch belegten, hat das
Modell die Spezialisierung unterschätzt: Die erfahrensten Mitarbeiter sind oft belegt und wurden deshalb "nicht gewählt".
Mit dem Mitarbeiter als Eingang des Simulationsmodells erzeugt diese Zuteilung die Kombinationen aus Mitarbeiter und
Fall, die das Modell aus dem echten Log kennt.
"""
import os
import pickle
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

from resources.case_allocation import Allocation, Task, work_counts

FEATURES = ("experience", "share", "last", "handled", "same_item")
MIN_EXPERIENCE = 30
MIN_CHOICES = 200
ON_DUTY_HOURS = 2.0  # weniger Entscheidungen in einer Gruppe: Gewichte aller Gruppen derselben Öffnung
_OPENING, _CLOSING = ("start", "resume"), ("suspend", "complete", "ate_abort", "withdraw")


def _features(experience: float, total: float, last: bool, handled: bool, same_item: bool) -> np.ndarray:
    return np.array([np.log1p(experience), experience / total if total else 0.0, float(last), float(handled), float(same_item)])


# Wie oft jeder Mitarbeiter insgesamt Arbeitsschritte begonnen oder fortgesetzt hat (Nenner von share)
def _totals(counts):
    totals = Counter()
    for (resource, _), count in counts.items():
        totals[resource] += count
    return totals


def _choice_situations(event_log, counts, system_users, prefix):
    """
    Je Beginn oder Fortsetzung eines Arbeitsschritts im echten Log: Gruppe (Aktivität, Öffnung), Merkmale aller Kandidaten
    und Index des gewählten. Kandidaten: Erfahrung >= MIN_EXPERIENCE, im Dienst und frei, dazu immer der Gewählte.
    """
    import bisect
    log = event_log[~event_log["org:resource"].isin(system_users)].sort_values(["case:concept:name", "time:timestamp"], kind="stable")
    seconds = (log["time:timestamp"].dt.tz_localize(None) if log["time:timestamp"].dt.tz is not None else log["time:timestamp"]).astype("int64").to_numpy() / 1e9
    seconds = seconds / (1e3 if seconds.max() > 1e11 else 1)  # Mikro- oder Nanosekunden
    rows = list(zip(log["case:concept:name"], log["concept:name"], log["lifecycle:transition"], log["org:resource"], seconds))
    # Im Dienst: Zeitpunkte aller Events je Mitarbeiter; belegt: offene Arbeitsabschnitte je Mitarbeiter
    times, sessions, opened = defaultdict(list), defaultdict(list), {}
    for case, activity, lifecycle, resource, time in rows:
        times[resource].append(time)
        if activity.startswith(prefix):
            if lifecycle in _OPENING:
                opened[(case, activity)] = (resource, time)
            elif lifecycle in _CLOSING and (case, activity) in opened:
                who, since = opened.pop((case, activity))
                sessions[who].append((since, time))
    times = {resource: np.sort(values) for resource, values in times.items()}
    busy_starts, busy_ends = {}, {}
    for resource, intervals in sessions.items():
        intervals.sort()
        busy_starts[resource] = [start for start, _ in intervals]
        busy_ends[resource] = np.maximum.accumulate([end for _, end in intervals]).tolist()

    def on_duty(resource, time):
        values = times.get(resource)
        if values is None:
            return False
        index = np.searchsorted(values, time - ON_DUTY_HOURS * 3600)
        return index < len(values) and values[index] <= time + ON_DUTY_HOURS * 3600

    def busy(resource, time):
        starts = busy_starts.get(resource)
        if not starts:
            return False
        index = bisect.bisect_left(starts, time) - 1
        return index >= 0 and busy_ends[resource][index] > time

    totals = _totals(counts)
    experienced = defaultdict(set)
    for (resource, activity), count in counts.items():
        if count >= MIN_EXPERIENCE and resource not in system_users:
            experienced[activity].add(resource)
    situations = defaultdict(list)
    previous_case, last, handled, holders = None, None, set(), {}
    for case, activity, lifecycle, resource, time in rows:
        if case != previous_case:
            previous_case, last, handled, holders = case, None, set(), {}
        if not activity.startswith(prefix):
            continue
        if lifecycle in _OPENING:
            candidates = sorted({r for r in experienced[activity] if on_duty(r, time) and not busy(r, time)} | {resource})
            features = np.stack([_features(counts.get((r, activity), 0), totals[r], r == last, r in handled, r == holders.get(activity)) for r in candidates])
            situations[(activity, lifecycle)].append((features, candidates.index(resource)))
        if lifecycle in _OPENING:
            last = resource
            handled.add(resource)
            holders[activity] = resource
    return situations


def _fit_logit(choices) -> np.ndarray:
    """Gewichte des bedingten Logits (Maximum-Likelihood, L-BFGS) aus einer Liste von (Merkmale, Index des Gewählten)."""
    from scipy.optimize import minimize
    features = np.concatenate([f for f, _ in choices])
    groups = np.repeat(np.arange(len(choices)), [len(f) for f, _ in choices])
    chosen = np.cumsum([0] + [len(f) for f, _ in choices[:-1]]) + np.array([c for _, c in choices])

    def loss(weights):
        scores = features @ weights
        top = np.maximum.reduceat(scores, np.r_[0, np.flatnonzero(np.diff(groups)) + 1])
        exp = np.exp(scores - top[groups])
        sums = np.bincount(groups, weights=exp)
        probabilities = exp / sums[groups]
        value = -(scores[chosen] - top - np.log(sums)).sum() + 0.01 * weights @ weights
        gradient = -(features[chosen].sum(axis=0) - (probabilities[:, None] * features).sum(axis=0)) + 0.02 * weights
        return value, gradient

    return minimize(loss, np.zeros(features.shape[1]), jac=True, method="L-BFGS-B").x


def fit_learned_allocation(event_log, system_users, prefix) -> Dict:
    """
    Gewichte je (Aktivität, Öffnung); Gruppen mit weniger als MIN_CHOICES Entscheidungen bekommen die ihrer Öffnung.
    system_users: Systemnutzer (resources/datasets.py); prefix: Arbeitsschritte (next_event_prediction/datasets.py).
    """
    counts = work_counts(event_log, prefix)
    situations = _choice_situations(event_log, counts, system_users, prefix)
    weights = {}
    for opening in _OPENING:
        pooled = [choice for (activity, lifecycle), choices in situations.items() if lifecycle == opening for choice in choices]
        # Im Log mit Arbeitsabschnitten (preprocess mit work_sessions) gibt es nur start, kein resume
        if pooled:
            weights[("*", opening)] = _fit_logit(pooled)
    for group, choices in situations.items():
        if len(choices) >= MIN_CHOICES:
            weights[group] = _fit_logit(choices)
    return {"weights": weights, "counts": counts, "choices": {group: len(choices) for group, choices in situations.items()}}


class LearnedAllocation(Allocation):
    """Wählt unter den freien Kandidaten mit den Wahrscheinlichkeiten des gelernten Logits (fit_learned_allocation)."""

    def __init__(self, model: Dict):
        # Kontinuität ist ein Merkmal des Modells, keine feste Regel
        super().__init__(keep_last_employee=False)
        self.weights, self.counts = model["weights"], model["counts"]
        self.totals = _totals(self.counts)

    def resource_for_task(self, task: Task, free: List[str], time: float) -> str:
        case = self.engine.cases.get(task.case_id)
        handled = case.handlers if case is not None else set()
        holders = case.item_holders if case is not None else {}
        # Ohne eigene Gewichte die der Öffnung; fehlt die Öffnung im Training (resume bei Arbeitsabschnitten), die von start
        weights = self.weights.get((task.activity, task.opening), self.weights.get(("*", task.opening), self.weights[("*", "start")]))
        last = case.last_opener if case is not None else None
        scores = np.array([_features(self.counts.get((r, task.activity), 0), self.totals[r], r == last, r in handled, r == holders.get(task.activity)) @ weights
                           for r in free])
        # Python's random, seeded by the engine like the other selection strategies
        return random.choices(free, weights=np.exp(scores - scores.max()))[0]


def load_learned_allocation(event_log, path: str, system_users, prefix: str) -> LearnedAllocation:
    """Gespeicherte Gewichte laden (path: resources/datasets.py), sonst aus dem echten Log fitten und speichern."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
    else:
        model = fit_learned_allocation(event_log, system_users, prefix)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
    return LearnedAllocation(model)
