"""
Die Werkstatt als Datensatz für die Pipeline (Katalog: repair, Log: eventlog/repair/repair_workshop.xes.gz): die Reparaturwerkstatt
(repair_engine.py) mit Arbeitsschritten, dreimal so groß, mit Gerätetypen und Mechanikern, die sich je Typ unterscheiden. So macht die
Zuteilung bei W_repair für den Fall selbst etwas aus (Dauer und Nacharbeit) und nebenbei für die anderen (wer belegt ist, fehlt ihnen).
Die Wahrheit ist bekannt: Policies, die im gelernten Simulator trainiert und ausgewertet wurden, laufen auch hier (dieselbe Schnittstelle).

Anders als in repair_engine.py:
- Gerätetyp case:device (A, B, C zu 50, 30, 20 %), an jedem Event wie die Fallattribute der BPIC-Logs
- 6 Sachbearbeiter, 6 Prüfer, 9 Mechaniker, 3 Expertinnen. Mechaniker 1-3 sind stark bei A und schwach bei C, 4-6 stark bei B und
  schwach bei A, 7-9 stark bei C und schwach bei B (SKILLS: Faktor auf die Reparaturdauer, Anteil Nacharbeit nach der Qualitätsprüfung).
  Expertinnen sind bei allen Typen schnell und gründlich
- Wahlpunkt POINT bei W_repair start: wartet ein normaler Fall und sind mindestens zwei Mechaniker oder Expertinnen frei, wählt die Policy
  einen von ihnen (auch bei der Nacharbeit). Premium-Fälle gehen wie bisher zu einer Expertin, ohne Wahl
- Ohne Policy die Hausregel: der Mechaniker, der am längsten frei ist; eine Expertin nur, wenn kein Mechaniker frei ist. Sie achtet nicht
  auf den Typ, so bearbeitet im Log jeder Mechaniker Geräte jedes Typs (sonst könnte der gelernte Simulator andere Zuteilungen nicht kennen)
- Warten: antwortet die Policy WAIT (interventions.WAIT, z. B. prescriptive/policies/psp_lookahead_policy.py), bleibt der Fall in der
  Warteschlange auf seinem Platz und wird erst wieder gefragt, wenn seitdem ein Mitarbeiter frei geworden ist (eine
  Reparatur endet), nicht bei jedem neuen bereiten Fall; die Fälle hinter ihm werden trotzdem gefragt. Policies mit may_wait werden auch bei nur einem freien Mitarbeiter gefragt

    engine = RepairWorkshop(seed=0)
    engine.set_policy(policy, [engine.point], window=(1000, 29000))
    events = pd.DataFrame(engine.run(30000))
    table = engine.case_table()

Das Log für den Katalog (aus process-simulation-engine/):
    .venv/bin/python -m simulation.repair_workshop --cases 30000 --out ../eventlog/repair/repair_workshop.xes.gz
"""
from typing import Dict

import numpy as np

from interventions import WAIT, ChoicePoint
from simulation.repair_engine import DAY, REPAIR_END, WORK_ITEM, Case, RepairEngine

RATE = 3.6  # Fälle je Tag
DEVICES, DEVICE_SHARES = ("A", "B", "C"), (0.5, 0.3, 0.2)
CLERKS = tuple(f"clerk_{k}" for k in range(1, 7))
INSPECTORS = tuple(f"inspector_{k}" for k in range(1, 7))
MECHANICS = tuple(f"mechanic_{k}" for k in range(1, 10))
EXPERTS = ("expert_1", "expert_2", "expert_3")
# (Faktor auf die Dauer der Reparatur, Anteil Nacharbeit)
SKILLS = {"stark": (0.6, 0.05), "neutral": (1.0, 0.10), "schwach": (1.6, 0.35), "expertin": (0.5, 0.03)}
POINT = "Mitarbeiter: W_repair"  # wie im Katalog (repair, lauf.mitarbeiterwahl)


def skill(resource: str, device: str) -> tuple:
    """(Faktor auf die Dauer, Anteil Nacharbeit) von resource bei Geräten des Typs device."""
    if resource in EXPERTS:
        return SKILLS["expertin"]
    group = (int(resource.rsplit("_", 1)[1]) - 1) // 3
    strong, weak = DEVICES[group], DEVICES[(group + 2) % 3]
    return SKILLS["stark" if device == strong else "schwach" if device == weak else "neutral"]


class RepairWorkshop(RepairEngine):
    pools = {"W_intake": CLERKS, "W_warranty_check": CLERKS, "W_shipping": CLERKS, "W_safety_check": INSPECTORS, "W_quality_check": INSPECTORS}
    mechanics = MECHANICS

    def __init__(self, rate: float = RATE, seed: int = 0, process: Dict = None, point: ChoicePoint = None):
        """process: Abweichungen von repair_engine.PROCESS (work_steps immer an). point: der Wahlpunkt (Standard POINT bei W_repair start)."""
        super().__init__(rate, seed, {**(process or {}), "work_steps": True})
        self.point = point or ChoicePoint(POINT, "employee", WORK_ITEM, "start")

    def _reset(self) -> None:
        super()._reset()
        self.free_experts = list(EXPERTS)
        self.busy = dict.fromkeys(MECHANICS + EXPERTS, 0.0)  # Tage in Reparatur je Mechaniker und Expertin
        self.held = {}  # Fälle, für die die Policy warten wollte -> Zahl der beendeten Reparaturen da (interventions.WAIT)
        self.releases = 0  # beendete Reparaturen (jede macht einen Mitarbeiter frei)

    def _draw(self, rng: np.random.Generator) -> Dict:
        draw = super()._draw(rng)
        draw["device"] = DEVICES[int(rng.choice(len(DEVICES), p=DEVICE_SHARES))]
        return draw

    def _attributes(self, case: Case) -> Dict:
        return {"case:device": case.draw["device"]}

    def _dispatch(self, now: float) -> None:
        """Premium zur Expertin; dann je wartendem normalem Fall die Wahl unter allen freien (Policy) oder die Hausregel."""
        while self.free_experts and self.premium_queue:
            self._begin(self.premium_queue.popleft(), self.free_experts[0], now)
        waiting = []
        while self.normal_queue and (self.free_mechanics or self.free_experts):
            case = self.normal_queue.popleft()
            options = self.free_mechanics + self.free_experts
            # Wollte die Policy warten, wird erst wieder gefragt, wenn seitdem ein Mitarbeiter frei geworden ist
            if self.held.get(case.case_id) == self.releases:
                waiting.append(case)
                continue
            ask = len(options) > 1 or getattr(self.policy, "may_wait", False)
            choice = self._ask(case, options, now) if ask else None
            if choice == WAIT:
                self.held[case.case_id] = self.releases
                waiting.append(case)
                continue
            self.held.pop(case.case_id, None)
            self._begin(case, choice if choice in options else options[0], now)
        self.normal_queue.extendleft(reversed(waiting))  # die Wartenden behalten ihren Platz vorn

    def _begin(self, case: Case, resource: str, now: float) -> None:
        (self.free_experts if resource in EXPERTS else self.free_mechanics).remove(resource)
        case.resource, case.began, case.in_repair = resource, now, True
        if case.first_began is None:
            case.first_began = now
        self._record(case, WORK_ITEM, "start", resource, now)
        self._note_wait(case, WORK_ITEM, case.queued, now)
        mean = self.process["premium_days" if case.draw["premium"] else "normal_days"]
        share = self.process["rework_share"] if case.reworks else 1.0
        self._schedule(now + mean * share * case.draw["factor"] * skill(resource, case.draw["device"])[0] * DAY, REPAIR_END, case.case_id)

    def _repair_end(self, case: Case, time: float) -> None:
        self._record(case, WORK_ITEM, "complete", case.resource, time)
        case.in_repair = False
        self.releases += 1
        (self.free_experts if case.resource in EXPERTS else self.free_mechanics).append(case.resource)
        self.busy[case.resource] += (time - case.began) / DAY
        case.repaired = time
        self.tasks.append((case, "W_quality_check", time))
        self._dispatch_tasks(time)
        self._dispatch(time)

    def _rework_probability(self, case: Case) -> float:
        return skill(case.resource, case.draw["device"])[1]

    def _close(self, case: Case, time: float) -> None:
        super()._close(case, time)
        self.results[-1]["device"] = case.draw["device"]


def export_log(path: str, n_cases: int, seed: int = 0):
    """Das Log der Werkstatt unter der Hausregel als XES (Zeitstempel in UTC wie in den BPIC-Logs). Liefert das Log und die Fälle."""
    import os

    import pandas as pd
    import pm4py
    engine = RepairWorkshop(seed=seed)
    log = pd.DataFrame(engine.run(n_cases))
    log["time:timestamp"] = pd.to_datetime(log["time:timestamp"]).dt.tz_localize("UTC")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pm4py.write_xes(log, path, case_id_key="case:concept:name")
    return log, engine.case_table()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="../eventlog/repair/repair_workshop.xes.gz")
    args = parser.parse_args()
    log, table = export_log(args.out, args.cases, args.seed)
    print(f"{args.out}: {table.shape[0]:,} Fälle, {len(log):,} Events, {log['time:timestamp'].min():%Y-%m-%d} bis {log['time:timestamp'].max():%Y-%m-%d}")
