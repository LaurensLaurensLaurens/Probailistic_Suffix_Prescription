"""
Die Reparaturwerkstatt (Weiterentwicklung von SCOPE_WORKSPACE/SimRepair): ein synthetischer Prozess mit bekannter Wahrheit, um
zu zeigen, wann der Blick auf die anderen laufenden Fälle etwas bringt (prescriptive/policies/psp_lookahead_policy.py). Anders
als die beiden Engines mit Next-Event-Modell ist das hier kein gelerntes Modell: Die Regeln unten SIND der Prozess. Die Schnittstelle
ist dieselbe (run, set_policy, other_cases, choices, waits), Policies laufen also unverändert darauf.

Ein Fall:
    initiate_case, intake, warranty_check und safety_check, order_stock_parts oder order_factory_parts,
    parts_received, W_repair (start, complete), shipping, customer_acceptance oder customer_refusal
oder er bricht während der Wartezeit ab (case_cancelled).

Mit process["work_steps"] sind Annahme, Prüfungen, Qualitätsprüfung und Versand Arbeitsschritte mit Mitarbeitern (start, complete):
    initiate_case, W_intake, W_warranty_check und W_safety_check, order_..., parts_received, W_repair, W_quality_check
    (mit Anteil rework zurück zu W_repair, Nacharbeit kürzer: rework_share), W_shipping, customer_acceptance oder customer_refusal
    W_intake, W_warranty_check, W_shipping   Sachbearbeiter (CLERKS)
    W_safety_check, W_quality_check          Prüfer (INSPECTORS)
    W_repair                                 Mechaniker und Expertin, wie oben
Wer frei ist, übernimmt den ältesten wartenden Arbeitsschritt seiner Gruppe (Dauer: step_days, gestreut wie die Reparatur). Die Frist
gilt bis zum Ende der letzten Reparatur. Ohne work_steps ist alles wie vorher (dieselben Zufallszahlen, dasselbe Log).

Was den Prozess ausmacht (Herleitung: Spielmodelle vom 1.11.2026; Lookahead wirkt nur, wenn alles davon zusammenkommt):
- Wenige wertvolle Fälle: Premium ist 20-mal so viel wert wie normal. Premium steht nirgends im Log, nur in der Reihenfolge der
  Prüfungen: warranty_check vor safety_check bei 90 % der Premium-Fälle, aber nur bei 3 % der normalen.
- Knappheit: Premium kann nur die Expertin, normale Fälle alle. Ein Beginn lässt sich nicht zurücknehmen, die Bindung dauert Tage.
- Die Wartezeit hängt vom Verlauf ab: Lagerteile kommen in Tagen, Werksteile in Wochen, und ein Teil der Lagerbestellungen verzögert
  sich ohne Vorwarnung auf Wochen. Die Ankunft ist also oft zweigipflig, ihr Mittelwert sagt wenig.
- Harte Fristen ab parts_received: bis dahin nimmt der Kunde an, danach lehnt er ab (Wert 0).

Der einzige Wahlpunkt: ein normaler Fall wartet, kein Mechaniker ist frei, die Expertin schon. Die Policy nimmt sie (EXPERT) oder lässt
sie frei (WAIT); gefragt wird dann beim nächsten fertigen Fall oder bei der nächsten Reparatur wieder. Ohne Policy nimmt die Werkstatt
sie immer ("immer nehmen"). Premium-Fälle haben Vorrang, Mechaniker gleich schnell, also gibt es sonst nichts zu wählen.

Jeder Fall zieht aus eigenen Zufallszahlen: Verlauf und Reparaturdauer eines Falls sind unter allen Policies gleich, nur Wartezeiten
unterscheiden sich. truth (nur für das Orakel) und case_table (für die Auswertung) kennen, was im Log nicht steht.

    engine = RepairEngine(rate=1.3, seed=0)                               # mit Arbeitsschritten: process={"work_steps": True}
    engine.set_policy(policy, [engine.point], window=(800, 11200))
    events = pd.DataFrame(engine.run(12000))
    table = engine.case_table()
"""
import heapq
import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from interventions import ChoicePoint
from simulation.case_engine_lifecycle_aware import to_datetime

DAY = 86400.0
START = 1704067200.0  # 1.1.2024
WORK_ITEM = "W_repair"
EXPERT, MECHANICS, SYSTEM = "expert_1", ("mechanic_1", "mechanic_2", "mechanic_3"), "system"
CLERKS, INSPECTORS = ("clerk_1", "clerk_2"), ("inspector_1", "inspector_2")
POOLS = {"W_intake": CLERKS, "W_warranty_check": CLERKS, "W_shipping": CLERKS, "W_safety_check": INSPECTORS, "W_quality_check": INSPECTORS}
WAIT = "warten"
ARRIVAL, EVENT, REPAIR_END, OBSERVE, TASK_END = 0, 1, 2, 3, 4

# Tage, wenn nicht anders genannt
PROCESS = {
    "premium": 0.12, "premium_value": 20.0, "normal_value": 1.0,
    "pattern": (0.90, 0.03),  # Anteil warranty_check vor safety_check: Premium, normal
    "cancel": 0.10,  # bricht während der Wartezeit ab
    "stock": 0.5, "stock_days": (0.3, 3.0), "factory_days": (12.0, 20.0),
    "delay": 0.2,  # Anteil der Lagerbestellungen, die wie Werksbestellungen dauern
    "premium_days": 2.0, "normal_days": 3.0, "noise": 0.3,  # Reparatur: Mittel und Streuung (log) der Dauer
    "premium_budget": 3.0, "normal_budget": 9.0,  # Frist ab parts_received bis zum Ende der Reparatur
    "step": (0.01, 0.03),  # Abstand der Events vor der Wartezeit und nach der Reparatur
    # Nur mit work_steps: mittlere Dauer der Arbeitsschritte (Auslastung bei rate 1.3 je Gruppe knapp 60 %), Nacharbeit
    "work_steps": False,
    "step_days": {"W_intake": 0.35, "W_warranty_check": 0.3, "W_shipping": 0.25, "W_safety_check": 0.5, "W_quality_check": 0.35},
    "rework": 0.1, "rework_share": 0.4,
}


def draw_case(process: Dict, rng: np.random.Generator) -> Dict:
    """Was einen Fall ausmacht. Auch für die Wahrscheinlichkeitstabellen der Policies (prescriptive/policies/repair_rules.py)."""
    premium = rng.random() < process["premium"]
    pattern = rng.random() < process["pattern"][0 if premium else 1]
    stock = rng.random() < process["stock"]
    delayed = stock and rng.random() < process["delay"]
    lead = rng.uniform(*(process["factory_days"] if delayed or not stock else process["stock_days"]))
    cancel_after = lead * rng.random() if rng.random() < process["cancel"] else None
    # Mit Mittel 1, wie die Streuung um die mittlere Reparaturdauer
    factor = rng.lognormal(-process["noise"] ** 2 / 2, process["noise"])
    return {"premium": premium, "pattern": pattern, "stock": stock, "lead": lead, "cancel_after": cancel_after, "factor": factor}


@dataclass
class Case:
    case_id: str
    number: int
    draw: Dict
    rng: np.random.Generator
    arrived: float
    plan: deque = field(default_factory=deque)  # die nächsten Events: (Zeit, Aktivität)
    prefix: List[Dict] = field(default_factory=list)  # die Events bis jetzt, wie in engine.events; bekommen die Policies
    ready: Optional[float] = None  # parts_received
    began: Optional[float] = None  # W_repair start (der laufenden Reparatur)
    resource: Optional[str] = None
    in_repair: bool = False
    queued: Optional[float] = None  # seit wann er auf W_repair wartet
    first_began: Optional[float] = None  # erste W_repair start
    # Nur mit work_steps
    checks: deque = field(default_factory=deque)  # die noch offenen Prüfungen, in Reihenfolge
    task: Optional[tuple] = None  # der laufende Arbeitsschritt außer W_repair: (Aktivität, Mitarbeiter)
    repaired: Optional[float] = None  # Ende der letzten Reparatur
    reworks: int = 0


class RepairEngine:
    # Mitarbeiter: Gruppen der Arbeitsschritte außer W_repair (nur mit work_steps) und die Mechaniker (repair_workshop.py hat mehr)
    pools, mechanics = POOLS, MECHANICS

    def __init__(self, rate: float = 1.3, seed: int = 0, process: Optional[Dict] = None):
        """rate: Fälle je Tag. process: Abweichungen von PROCESS."""
        self.rate, self.seed = rate, seed
        self.process = {**PROCESS, **(process or {})}
        self.steps = bool(self.process["work_steps"])
        self.point = ChoicePoint("expertin_oder_warten", "employee", WORK_ITEM, "start")
        self.observer, self.observe_seconds = None, None
        self.set_policy(None, [])
        self._reset()

    def _reset(self) -> None:
        self.queue: List[tuple] = []
        self._sequence = itertools.count()
        self.cases: Dict[str, Case] = {}
        self.events: List[Dict] = []
        self.choices: List[Dict] = []
        self.results: List[Dict] = []
        self.premium_queue, self.normal_queue = deque(), deque()
        self.expert_free, self.free_mechanics = True, list(self.mechanics)
        self.busy_expert = 0.0
        # Nur mit work_steps: wer frei ist, und die wartenden Arbeitsschritte außer W_repair (Fall, Aktivität, seit wann)
        self.idle = {resource: True for pool in self.pools.values() for resource in pool}
        self.tasks = deque()
        # Dieselben Größen wie die Engines mit Next-Event-Modell
        self.waits: List[Dict] = []
        self.waiting_seconds: Dict[str, float] = {"alle beschäftigt": 0.0}
        self.forced_assignments = 0

    def set_policy(self, policy, points: List, window: Optional[tuple] = None) -> None:
        """policy.choose(point, case, options, time, engine) gibt EXPERT oder WAIT zurück, None: wie die Werkstatt. window: wie in den Engines."""
        self.policy, self.window = (policy if points else None), window or (0, float("inf"))

    def run(self, num_cases: int) -> List[Dict]:
        """num_cases Ankünfte simulieren, alle Fälle gemeinsam. Liefert das Event-Log als Liste von Zeilen."""
        self._reset()
        times = START + np.cumsum(np.random.default_rng([self.seed, 0]).exponential(DAY / self.rate, num_cases))
        for number, time in enumerate(times):
            self._schedule(float(time), ARRIVAL, f"Case_{number}")
        if self.observer is not None and self.observe_seconds and num_cases:
            self._schedule(float(times[0]) + self.observe_seconds, OBSERVE, None)
        while self.queue:
            time, _, kind, case_id = heapq.heappop(self.queue)
            self.now = time
            if kind == ARRIVAL:
                self._arrive(case_id, time)
            elif kind == EVENT:
                self._next_event(self.cases[case_id], time)
            elif kind == REPAIR_END:
                self._repair_end(self.cases[case_id], time)
            elif kind == TASK_END:
                self._task_end(self.cases[case_id], time)
            else:
                self.observer(time, self)
                if self.queue:
                    self._schedule(time + self.observe_seconds, OBSERVE, None)
            # Ganz am Ende fragt niemand mehr nach: was noch wartet, nimmt die Expertin
            if not self.queue and self.normal_queue and self.expert_free:
                self.policy = None
                self._dispatch(time)
        return self.events

    def _schedule(self, time: float, kind: int, case_id: Optional[str]) -> None:
        heapq.heappush(self.queue, (time, next(self._sequence), kind, case_id))

    def _step(self, case: Case) -> float:
        return float(case.rng.uniform(*self.process["step"])) * DAY

    def _arrive(self, case_id: str, time: float) -> None:
        number = int(case_id.rsplit("_", 1)[1])
        rng = np.random.default_rng([self.seed, number + 1])
        draw = self._draw(rng)
        case = Case(case_id, number, draw, rng, time)
        checks = ["warranty_check", "safety_check"] if draw["pattern"] else ["safety_check", "warranty_check"]
        if self.steps:
            # Nur initiate_case geplant; danach die Arbeitsschritte (_task_end), die Bestellung nach der zweiten Prüfung
            case.plan.append((time, "initiate_case"))
            case.checks = deque("W_" + check for check in checks)
            self.cases[case_id] = case
            self._schedule(time, EVENT, case_id)
            return
        moment = time
        case.plan.append((moment, "initiate_case"))
        for activity in ["intake"] + checks + ["order_stock_parts" if draw["stock"] else "order_factory_parts"]:
            moment += self._step(case)
            case.plan.append((moment, activity))
        if draw["cancel_after"] is not None:
            case.plan.append((moment + draw["cancel_after"] * DAY, "case_cancelled"))
        else:
            case.plan.append((moment + draw["lead"] * DAY, "parts_received"))
        self.cases[case_id] = case
        self._schedule(case.plan[0][0], EVENT, case_id)

    def _draw(self, rng: np.random.Generator) -> Dict:
        return draw_case(self.process, rng)

    def _attributes(self, case: Case) -> Dict:
        """Fallattribute an jedem Event (hier keine; repair_workshop.py: der Gerätetyp)."""
        return {}

    def _record(self, case: Case, activity: str, lifecycle: str, resource: str, time: float) -> None:
        event = {"case:concept:name": case.case_id, "concept:name": activity, "org:resource": resource, "time:timestamp": to_datetime(time),
                 "lifecycle:transition": lifecycle, **self._attributes(case)}
        self.events.append(event)
        case.prefix.append(event)

    def _next_event(self, case: Case, time: float) -> None:
        _, activity = case.plan.popleft()
        self._record(case, activity, "complete", SYSTEM, time)
        if case.plan:
            self._schedule(case.plan[0][0], EVENT, case.case_id)
        elif activity == "parts_received":
            case.ready = time
            self._queue_repair(case, time)
        elif activity == "initiate_case":  # nur mit work_steps (sonst folgt intake im Plan)
            self.tasks.append((case, "W_intake", time))
            self._dispatch_tasks(time)
        else:
            self._close(case, time)

    def _queue_repair(self, case: Case, time: float) -> None:
        case.queued = time
        (self.premium_queue if case.draw["premium"] else self.normal_queue).append(case)
        self._dispatch(time)

    def _dispatch(self, now: float) -> None:
        """Wer jetzt anfängt: Premium zur Expertin, normale Fälle zu Mechanikern; ist nur noch die Expertin frei, der Wahlpunkt."""
        if self.expert_free and self.premium_queue:
            self._begin(self.premium_queue.popleft(), EXPERT, now)
        while self.free_mechanics and self.normal_queue:
            self._begin(self.normal_queue.popleft(), self.free_mechanics.pop(0), now)
        if self.expert_free and self.normal_queue:
            if self._ask(self.normal_queue[0], [EXPERT, WAIT], now) != WAIT:
                self._begin(self.normal_queue.popleft(), EXPERT, now)

    def _ask(self, case: Case, options: List[str], time: float) -> Optional[str]:
        if self.policy is None or not self.window[0] <= case.number < self.window[1]:
            return None
        choice = self.policy.choose(self.point, case, options, time, self)
        self.choices.append({"case": case.case_id, "point": self.point.name, "kind": self.point.kind, "choice": choice,
                             "index": options.index(choice) if choice in options else None, "options": len(options), "time": time})
        return choice

    def _begin(self, case: Case, resource: str, now: float) -> None:
        if resource == EXPERT:
            self.expert_free = False
        case.resource, case.began, case.in_repair = resource, now, True
        if case.first_began is None:
            case.first_began = now
        self._record(case, WORK_ITEM, "start", resource, now)
        self._note_wait(case, WORK_ITEM, case.queued, now)
        mean = self.process["premium_days" if case.draw["premium"] else "normal_days"]
        share = self.process["rework_share"] if case.reworks else 1.0
        self._schedule(now + mean * share * case.draw["factor"] * DAY, REPAIR_END, case.case_id)

    def _note_wait(self, case: Case, activity: str, since: float, now: float) -> None:
        if now > since:
            self.waits.append({"case": case.case_id, "activity": activity, "lifecycle": "start", "seconds": now - since, "reason": "alle beschäftigt"})
            self.waiting_seconds["alle beschäftigt"] += now - since

    def _repair_end(self, case: Case, time: float) -> None:
        self._record(case, WORK_ITEM, "complete", case.resource, time)
        case.in_repair = False
        if case.resource == EXPERT:
            self.expert_free = True
            self.busy_expert += time - case.began
        else:
            self.free_mechanics.append(case.resource)
        if self.steps:
            case.repaired = time
            self.tasks.append((case, "W_quality_check", time))
            self._dispatch_tasks(time)
            self._dispatch(time)
            return
        budget = self.process["premium_budget" if case.draw["premium"] else "normal_budget"]
        on_time = (time - case.ready) / DAY <= budget
        case.draw["on_time"] = on_time
        moment = time + self._step(case)
        case.plan.append((moment, "shipping"))
        case.plan.append((moment + self._step(case), "customer_acceptance" if on_time else "customer_refusal"))
        self._schedule(case.plan[0][0], EVENT, case.case_id)
        self._dispatch(time)

    # ------------------------------------------------------------------ Arbeitsschritte außer W_repair (nur mit work_steps)

    def _dispatch_tasks(self, now: float) -> None:
        """Je wartendem Arbeitsschritt, der Reihe nach: der erste freie Mitarbeiter seiner Gruppe fängt an."""
        waiting = deque()
        while self.tasks:
            case, activity, since = self.tasks.popleft()
            resource = next((r for r in self.pools[activity] if self.idle[r]), None)
            if resource is None:
                waiting.append((case, activity, since))
                continue
            self.idle[resource] = False
            case.task = (activity, resource)
            self._record(case, activity, "start", resource, now)
            self._note_wait(case, activity, since, now)
            factor = case.rng.lognormal(-self.process["noise"] ** 2 / 2, self.process["noise"])
            self._schedule(now + self.process["step_days"][activity] * factor * DAY, TASK_END, case.case_id)
        self.tasks = waiting

    def _task_end(self, case: Case, time: float) -> None:
        (activity, resource), case.task = case.task, None
        self._record(case, activity, "complete", resource, time)
        self.idle[resource] = True
        draw = case.draw
        if activity in ("W_intake", "W_warranty_check", "W_safety_check"):
            if case.checks:
                self.tasks.append((case, case.checks.popleft(), time))
            else:  # beide Prüfungen fertig: bestellen, dann warten wie ohne work_steps
                moment = time + self._step(case)
                case.plan.append((moment, "order_stock_parts" if draw["stock"] else "order_factory_parts"))
                if draw["cancel_after"] is not None:
                    case.plan.append((moment + draw["cancel_after"] * DAY, "case_cancelled"))
                else:
                    case.plan.append((moment + draw["lead"] * DAY, "parts_received"))
                self._schedule(case.plan[0][0], EVENT, case.case_id)
        elif activity == "W_quality_check":
            if case.rng.random() < self._rework_probability(case):
                case.reworks += 1
                self._queue_repair(case, time)
            else:
                budget = self.process["premium_budget" if draw["premium"] else "normal_budget"]
                draw["on_time"] = (case.repaired - case.ready) / DAY <= budget
                self.tasks.append((case, "W_shipping", time))
        else:  # W_shipping
            case.plan.append((time + self._step(case), "customer_acceptance" if draw["on_time"] else "customer_refusal"))
            self._schedule(case.plan[0][0], EVENT, case.case_id)
        self._dispatch_tasks(time)

    def _rework_probability(self, case: Case) -> float:
        return self.process["rework"]

    def _close(self, case: Case, time: float) -> None:
        draw = case.draw
        value = self.process["premium_value" if draw["premium"] else "normal_value"] * draw.get("on_time", False)
        self.results.append({"case": case.case_id, "number": case.number, "premium": draw["premium"], "cancelled": draw["cancel_after"] is not None,
                             "on_time": draw.get("on_time", False), "value": value, "days": (time - case.arrived) / DAY,
                             "waited": 0.0 if case.ready is None else (case.first_began - case.ready) / DAY, "resource": case.resource,
                             "reworks": case.reworks})
        self.cases.pop(case.case_id)

    # ------------------------------------------------------------------ für Policies und Auswertung

    def other_cases(self, case_id: str, activity: str) -> List[Dict]:
        """Wie in der Engine mit Lifecycles: je anderem laufenden Fall sein Prefix, ob er gerade repariert wird und ob er darauf wartet."""
        waiting = {case.case_id for case in itertools.chain(self.premium_queue, self.normal_queue)}
        return [{"case": other.case_id, "prefix": other.prefix, "open": other.in_repair, "waiting": other.case_id in waiting}
                for other in self.cases.values() if other.case_id != case_id and other.prefix]

    def truth(self, case_id: str) -> Dict:
        """Nur für das Orakel: was im Log nicht steht. ready: wann der Fall bereit wird, None: er bricht ab."""
        case = self.cases[case_id]
        last = case.plan[-1] if case.plan else None
        return {"premium": case.draw["premium"], "ready": last[0] if last and last[1] == "parts_received" else None}

    def case_table(self) -> pd.DataFrame:
        """Je Fall die Wahrheit und das Ergebnis, nach Ankunft sortiert."""
        return pd.DataFrame(self.results).sort_values("number").reset_index(drop=True)
