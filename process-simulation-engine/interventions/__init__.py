"""
Katalog der Interventionspunkte: wo im Fall entschieden wird und woran man die gewählte Aktion im Log erkennt.
Gemeinsam für den Simulator (Entscheidungspunkt der Engine) und prescriptive/ (treatment, Tokens des PSP, Schätzer,
Auswertung). Gewählt wird mit --intervention.

Je Intervention:
- decision_activity: Am complete-Event dieser Aktivität wird entschieden.
- rule: woran die Aktion im Log erkennbar ist (Label je Entscheidungs-Event: 0/1 = Option, None = kein Entscheidungspunkt)
    next_event: direkt nach dem Entscheidungs-Event kommt eine der beiden Optionen (action_activities); nur dann ist
                es ein Entscheidungspunkt, Label = Nummer der Option. Im Simulator: zieht das Modell nach dem
                Entscheidungs-Event eine der Optionen, wählt die Policy, welche (case_engine.set_policy)
    follows_before_next_decision: action_activity folgt, bevor decision_activity wieder vorkommt
    follows_later: action_activity kommt nach dem Entscheidungs-Event noch irgendwann vor
- action_activity: die Aktivität der Aktion (bei follows_...)
- action_activities: bei next_event die beiden Optionen (0, 1); bei follows_... mehrere Aktivitäten der Aktion
- action_contains: statt action_activity: jede Aktivität, die einen dieser Texte enthält, ist die Aktion (BPIC13)
- action_lifecycle: Lifecycle der Aktions-Events (Standard complete; bei Arbeitsschritten z. B. start = bearbeitet)
- dataset: Datensatz (Schlüssel im Katalog catalog.yaml), für den die Intervention definiert ist (Standard bpic17)
- actions: Namen der beiden Optionen (0, 1), auch für die Tokens des PSP
Im Simulator umgesetzt sind nur Interventionen mit next_event; die anderen dienen dem Screening.

Unabhängig davon die Wahl des Mitarbeiters (employee_points): an den Arbeitsschritten aus lauf.mitarbeiterwahl im
Katalog wählt die Policy unter den Mitarbeitern, die kompetent, im Dienst und frei sind.

Beides zusammen sind die Wahlpunkte eines Laufs (choice_points, ChoicePoint): an ihnen fragt der Simulator die Policy
(simulation/case_engine.py set_policy, policy.choose).
"""
import functools
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

DEFAULT = "channel"


def get(name: Optional[str]) -> dict:
    """
    Die Intervention name aus dem Katalog (catalog.yaml): ein Control-Flow-Kandidat mit test (Entscheidungs-Event, Aktion,
    Regel). Mit der Regel next_event ist sie ein Wahlpunkt im Simulator, ohne Eintrag hier.
    """
    if name in (None, ""):
        name = DEFAULT
    found = _from_catalog(name, os.path.getmtime(CANDIDATES_FILE))
    if found is None:
        raise ValueError(f"Unbekannte Intervention '{name}', möglich: die Control-Flow-Kandidaten des Katalogs")
    return dict(found)


# Zwischengespeichert, solange sich der Katalog nicht ändert: get wird beim Kodieren je Fall aufgerufen
@functools.lru_cache(maxsize=None)
def _from_catalog(name: str, mtime: float) -> Optional[dict]:
    for row in candidates(art="control_flow"):
        test = row.get("test") or {}
        if row["name"] == name and test.get("rule") in ("next_event", "follows_before_next_decision", "follows_later"):
            return from_test(name, row["dataset"], test)
    return None


def from_test(name: str, dataset: str, test: dict) -> dict:
    """Intervention aus dem Feld test eines Control-Flow-Kandidaten im Katalog."""
    actions = test.get("action") or []
    if test["rule"] == "next_event" and len(actions) != 2:
        raise ValueError(f"{name}: next_event braucht genau zwei Optionen in action, nicht {actions}")
    return {"key": name, "name": name, "dataset": dataset, "decision_activity": test["decision"], "decision_lifecycle": test.get("lifecycle", "complete"), "rule": test["rule"],
            "action_activity": actions[0] if len(actions) == 1 and test["rule"] != "next_event" else None, "action_activities": actions if len(actions) > 1 or test["rule"] == "next_event" else None,
            "action_contains": test.get("action_contains"), "action_lifecycle": test.get("action_lifecycle"),
            "actions": test.get("names") or [f"no_{name}", name]}


# Der Katalog liegt bei der Pipeline (prescriptive/catalog.yaml), neben process-simulation-engine; die Umgebungsvariable
# PSP_CATALOG setzt einen anderen (z. B. eine Kopie mit anderen lauf-Punkten für einen Lauf neben anderen; gilt auch in den
# Arbeitsprozessen, die die Umgebung erben)
CANDIDATES_FILE = os.environ.get("PSP_CATALOG") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "prescriptive", "catalog.yaml")


def catalog() -> dict:
    """Der ganze Katalog (catalog.yaml) je Datensatz; nur lesen, nicht ändern (zwischengespeichert, solange die Datei gleich bleibt)."""
    return _catalog(os.path.getmtime(CANDIDATES_FILE))


@functools.lru_cache(maxsize=None)
def _catalog(mtime: float) -> dict:
    import yaml
    with open(CANDIDATES_FILE) as f:
        return yaml.safe_load(f)


def candidates(dataset: Optional[str] = None, art: Optional[str] = None, simulator: Optional[str] = None) -> List[dict]:
    """
    Katalog aller realistisch beeinflussbaren Interventionen (catalog.yaml), wahlweise gefiltert nach
    Datensatz (bpic17, bpic13, bpic19), Art (control_flow, variable) und Stand im Simulator (umgesetzt, moeglich,
    nicht_lernbar). Je Eintrag mit "dataset".
    """
    import yaml
    with open(CANDIDATES_FILE) as f:
        catalog = yaml.safe_load(f)
    rows = [{"dataset": name, **entry} for name, data in catalog.items() for entry in data["interventionen"]]
    return [row for row in rows if (dataset is None or row["dataset"] == dataset) and (art is None or row["art"] == art)
            and (simulator is None or row.get("simulator") == simulator)]


def is_action(activity, intervention: dict):
    """Ist die Aktivität (str oder pandas Series) die Aktion der Intervention? Bei next_event: die Option 1."""
    contains, several = intervention.get("action_contains"), intervention.get("action_activities")
    if intervention.get("rule") == "next_event":
        return activity == several[1]
    if hasattr(activity, "str"):
        if contains:
            return np.logical_or.reduce([activity.astype(str).str.contains(text, regex=False) for text in contains])
        if several:
            return activity.isin(several)
        return activity == intervention.get("action_activity")
    if contains:
        return any(text in str(activity) for text in contains)
    if several:
        return activity in several
    return activity == intervention.get("action_activity")


def option_token(option: str) -> str:
    """Das Event einer Control-Flow-Option als Aktivität|Lifecycle; eine Option ohne | ist das complete-Event der Aktivität."""
    return option if "|" in option else option + "|complete"


def option_at(activity: str, lifecycle: str, options: Sequence[str]) -> Optional[str]:
    """Die Option, die das Event (activity, lifecycle) ist, sonst None."""
    return next((option for option in options if option_token(option) == f"{activity}|{lifecycle}"), None)


def decision_labels(events: Sequence[Tuple[str, str]], intervention: dict) -> List[Optional[int]]:
    """
    Je Event None (kein Entscheidungs-Event) oder das Label der Aktion (0/1) an jedem complete-Event der
    decision_activity (next_event: mit decision_lifecycle). events: (Aktivität, Lifecycle) in Log-Reihenfolge. Bei next_event
    nur dort, wo direkt danach eine der Optionen kommt (Label: ihre Nummer).
    """
    decision, rule = intervention["decision_activity"], intervention["rule"]
    labels: List[Optional[int]] = [None] * len(events)
    if rule == "next_event":
        options = intervention["action_activities"]
        for i, (activity, lifecycle) in enumerate(events[:-1]):
            option = option_at(*events[i + 1], options)
            if lifecycle == intervention.get("decision_lifecycle", "complete") and activity == decision and option is not None:
                labels[i] = options.index(option)
        return labels
    # Aktion: Events mit action_lifecycle (Standard complete), wie treatment in prescriptive/enriched_log.enrich
    action_lifecycle = intervention.get("action_lifecycle") or "complete"
    seen = False
    for i in range(len(events) - 1, -1, -1):
        activity, lifecycle = events[i]
        if lifecycle == "complete" and activity == decision:
            labels[i] = int(seen)
            if rule == "follows_before_next_decision":
                seen = False
        if lifecycle == action_lifecycle and is_action(activity, intervention):
            seen = True
    return labels


def employee_points(dataset: str) -> List[dict]:
    """
    Die Wahl des Mitarbeiters in diesem Lauf (lauf.mitarbeiterwahl im Katalog, Namen von Mitarbeiter-Kandidaten): je
    Punkt name, activity (der Arbeitsschritt) und lifecycle (Standard start). Die Policy wählt jedes Mal, wenn der
    Arbeitsschritt so beginnt, unabhängig von der Intervention.
    """
    entry = catalog()[dataset]
    names = (entry.get("lauf") or {}).get("mitarbeiterwahl") or []
    rows = {row["name"]: row for row in entry["interventionen"]}
    unknown = [name for name in names if name not in rows or (rows[name].get("test") or {}).get("kind") != "resource"]
    if unknown:
        raise ValueError(f"lauf.mitarbeiterwahl ({dataset}): keine Mitarbeiter-Kandidaten im Katalog: {', '.join(unknown)}")
    return [{"name": name, "activity": rows[name]["test"]["decision"], "lifecycle": rows[name]["test"].get("lifecycle") or "complete"} for name in names]


def arrival_load(dataset: str) -> float:
    """Ankunftslast des Laufs: lauf.ankunft_faktor im Katalog, 1 wie im Log, 3 dreimal so viele Fälle je Tag."""
    return float((catalog()[dataset].get("lauf") or {}).get("ankunft_faktor", 1))


def run_allocation(dataset: str) -> str:
    """
    Wie die Engine die Mitarbeiter zuteilt, wo keine Policy wählt (lauf.zuteilung im Katalog, Standard random): eine
    Strategie aus resources/selection_strategies.py (random, round_robin, shortest_queue), als Name für
    resources/case_allocation.py create_allocation.
    """
    from resources.selection_strategies import STRATEGY_REGISTRY
    name = (catalog()[dataset].get("lauf") or {}).get("zuteilung") or "random"
    allowed = set(STRATEGY_REGISTRY) | {"learned"}  # learned: wen die echte Bank wählt (resources/BPIC2017/allocation/learned_allocation.py)
    if name not in allowed:
        raise ValueError(f"lauf.zuteilung ({dataset}): '{name}' ist keine Zuteilung, möglich: {', '.join(sorted(allowed))}")
    return name


PERMISSIONS = ("ordinor", "erfahrung")


def run_permissions(dataset: str) -> str:
    """
    Wer einen Arbeitsschritt darf (lauf.berechtigung im Katalog, Standard ordinor): ordinor das Berechtigungsmodell des
    Datensatzes (resources/datasets.py) mit der Erfahrungsschwelle der Engine, erfahrung nur die Erfahrung im echten Log
    (resources/case_allocation.experienced_employees). Gilt für Engine und Screening (Pool kompetent).
    """
    name = (catalog()[dataset].get("lauf") or {}).get("berechtigung") or "ordinor"
    if name not in PERMISSIONS:
        raise ValueError(f"lauf.berechtigung ({dataset}): '{name}', möglich: {', '.join(PERMISSIONS)}")
    return name


# Antwort einer Policy an einem Mitarbeiter-Wahlpunkt: jetzt niemanden zuteilen, der Fall wartet weiter und wird erneut
# gefragt, sobald sich etwas ändert (case_engine_lifecycle_aware: ein Mitarbeiter wird frei oder volle Stunde; repair_workshop:
# eine Reparatur endet oder ein Fall wird bereit). Gleich dem WAIT von simulation/repair_engine.py
WAIT = "warten"


@dataclass(frozen=True)
class ChoicePoint:
    """
    Ein Wahlpunkt des Simulators: an dem Event activity|lifecycle fragt die Engine die Policy (policy.choose).
    kind control_flow: direkt nach dem Event, wenn das Modell eine der options gezogen hat, welche passiert (Regel
        next_event; BPIC17 channel: nach O_Created|complete die beiden O_Sent). Optionen sind Aktivitäten (ihr complete-Event)
        oder Aktivität|Lifecycle (option_token).
    kind employee: jedes Mal, wenn der Arbeitsschritt activity mit lifecycle (start) einen Mitarbeiter
        braucht, welcher; die Optionen (freie Kompetente) entstehen erst zur Laufzeit, options ist leer.
    """
    name: str
    kind: str
    activity: str
    lifecycle: str
    options: Tuple[str, ...] = ()


def choice_points(dataset: str, intervention=None) -> List[ChoicePoint]:
    """
    Die Wahlpunkte eines Laufs: die Intervention (Schlüssel oder dict aus get; Standard: standard_intervention des
    Datensatzes im Katalog) und die Einträge aus lauf.control_flow als control_flow, wenn sie die Regel next_event haben,
    dazu je Eintrag aus lauf.mitarbeiterwahl ein Punkt employee. Mit lauf.nur_mitarbeiter gibt es keine Control-Flow-Punkte.
    """
    chosen = intervention if isinstance(intervention, dict) else get(intervention or catalog()[dataset]["standard_intervention"])
    lauf = catalog()[dataset].get("lauf") or {}
    further = lauf.get("control_flow") or []
    control = {}  # (Entscheidungs-Event) -> [Name, Optionen]; Interventionen am selben Event werden eine Wahl mit allen Optionen
    for item in [] if lauf.get("nur_mitarbeiter") else [chosen] + [get(name) for name in further if name != chosen["key"]]:
        if item.get("rule") == "next_event":
            entry = control.setdefault((item["decision_activity"], item.get("decision_lifecycle", "complete")), [None, []])
            entry[0] = item["key"] if entry[0] is None else f"{entry[0]}+{item['key']}"
            entry[1] += [option for option in item["action_activities"] if option not in entry[1]]
    points = [ChoicePoint(name, "control_flow", activity, lifecycle, tuple(options)) for (activity, lifecycle), (name, options) in control.items()]
    points += [ChoicePoint(point["name"], "employee", point["activity"], point["lifecycle"]) for point in employee_points(dataset)]
    return points
