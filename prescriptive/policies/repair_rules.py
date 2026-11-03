"""
Vergleichs-Policies der Reparaturwerkstatt (process-simulation-engine/simulation/repair_engine.py), alle ohne gelerntes Modell: sie zeigen,
was am Wahlpunkt "Expertin nehmen oder warten" überhaupt zu holen ist, bevor ein PSP ins Spiel kommt.

    HoldAlways      die Expertin bleibt für Premium frei, immer
    CountRule       Handregel: warten, wenn ein Premium-Muster läuft und höchstens limit Fälle warten. imminent: nur Fälle, die während der
                    Bindung kommen können: Lagerbestellungen in den ersten Tagen, alle Bestellungen ab dem Beginn der Werkszeiten. Wer sie
                    schreibt, muss Prozess und Lieferzeiten kennen
    TrueLookahead   die Formel von policies/psp_lookahead_policy.py mit den wahren Wahrscheinlichkeiten statt gezogener Suffixe, aus dem
                    Prefix: Ist der erwartete Schaden an ankommenden Premium-Fällen größer als der Preis des Wartens?
    OracleLookahead wie TrueLookahead, aber es weiß, welche Fälle Premium sind und wann sie kommen: die Obergrenze dieser Greedy-Regel

Der Preis des Wartens ist queue_weight · (wartende normale Fälle). Ohne ihn läuft die Warteschlange weg: Sobald normale Fälle ohnehin zu spät
sind, kostet weiteres Warten rechnerisch nichts mehr (Spielmodell vom 1.11.2026: 98 % gewartet, 7 % pünktlich).
"""
import numpy as np
import pandas as pd

from simulation.repair_engine import EXPERT, PROCESS, WAIT, draw_case

GRID_STEP = 0.1  # Tage: Raster der Tabellen
GRID_DAYS = 30.0
PRICE_SAMPLES = 100_000


def _seconds(timestamp):
    return (pd.Timestamp(timestamp) - pd.Timestamp("1970-01-01")).total_seconds()


def pending(others):
    """Die anderen Fälle, die noch nicht bereit sind: (Fall, wer zuerst geprüft wurde, Lagerbestellung?, Tage seit der Bestellung bis jetzt ohne now)."""
    rows = []
    for other in others:
        names = [event["concept:name"].removeprefix("W_") for event in other["prefix"]]  # mit work_steps W_warranty_check usw.
        if "parts_received" in names or "case_cancelled" in names or "order_stock_parts" not in names and "order_factory_parts" not in names:
            continue
        order = next(k for k, name in enumerate(names) if name.startswith("order_"))
        rows.append((other["case"], names.index("warranty_check") < names.index("safety_check"), names[order] == "order_stock_parts",
                     _seconds(other["prefix"][order]["time:timestamp"])))
    return rows


def waiting_cases(others):
    """Wie viele normale Fälle warten, mit dem gefragten (Premium wartet nie, wenn die Expertin frei ist)."""
    return 1 + sum(other["waiting"] for other in others)


class HoldAlways:
    def choose(self, point, case, options, time, engine):
        return WAIT


class CountRule:
    def __init__(self, limit=float("inf"), imminent=False, process=None):
        self.limit, self.imminent = limit, imminent
        process = {**PROCESS, **(process or {})}
        self.early = process["stock_days"][1] * 86400.0
        self.late = (process["factory_days"][0] - process["normal_days"]) * 86400.0

    def choose(self, point, case, options, time, engine):
        others = engine.other_cases(case.case_id, point.activity)
        if waiting_cases(others) > self.limit:
            return EXPERT
        for _, pattern, stock, ordered in pending(others):
            waited = time - ordered
            if pattern and (not self.imminent or (stock and waited <= self.early) or waited >= self.late):
                return WAIT
        return EXPERT


class TrueLookahead:
    def __init__(self, queue_weight=1.0, process=None, seed=5):
        self.queue_weight = queue_weight
        self.process = {**PROCESS, **(process or {})}
        p = self.process
        self.bind = p["normal_days"]
        premium, (with_premium, with_normal) = p["premium"], p["pattern"]
        self.p_premium = {True: with_premium * premium / (with_premium * premium + with_normal * (1 - premium)),
                          False: (1 - with_premium) * premium / ((1 - with_premium) * premium + (1 - with_normal) * (1 - premium))}
        rng = np.random.default_rng(seed)
        draws = [draw_case(p, rng) for _ in range(PRICE_SAMPLES)]
        factor = np.array([d["factor"] for d in draws])
        # Was es kostet, wenn ein ankommender Premium-Fall im Mittel die halbe Bindung warten muss
        self.harm = p["premium_value"] * float(np.mean(p["premium_days"] * factor <= p["premium_budget"]) - np.mean(self.bind / 2 + p["premium_days"] * factor <= p["premium_budget"]))
        self.arrive = {stock: self._table([d for d in draws if d["stock"] == stock]) for stock in (True, False)}

    def _table(self, draws):
        """Je Wartezeit bisher (Raster): P(der Fall kommt während der Bindung an | er läuft noch)."""
        lead = np.array([d["lead"] for d in draws])
        cancel = np.array([np.inf if d["cancel_after"] is None else d["cancel_after"] for d in draws])
        grid = np.arange(0.0, GRID_DAYS + GRID_STEP, GRID_STEP)
        alive = np.array([(np.minimum(lead, cancel) > w).sum() for w in grid])
        came = np.array([((lead > w) & (lead <= w + self.bind) & (cancel > lead)).sum() for w in grid])
        return came / np.maximum(alive, 1)

    def _probability(self, stock, waited):
        return float(self.arrive[stock][min(int(waited / GRID_STEP), len(self.arrive[stock]) - 1)])

    def risk(self, others, time, engine):
        total = 0.0
        for _, pattern, stock, ordered in pending(others):
            total += self._probability(stock, (time - ordered) / 86400.0) * self.p_premium[pattern] * self.harm
        return total

    def choose(self, point, case, options, time, engine):
        others = engine.other_cases(case.case_id, point.activity)
        return WAIT if self.risk(others, time, engine) > self.queue_weight * waiting_cases(others) else EXPERT


class OracleLookahead(TrueLookahead):
    def risk(self, others, time, engine):
        total = 0.0
        for case, _, _, _ in pending(others):
            truth = engine.truth(case)
            if truth["premium"] and truth["ready"] is not None and time < truth["ready"] <= time + self.bind * 86400.0:
                total += self.harm
        return total
