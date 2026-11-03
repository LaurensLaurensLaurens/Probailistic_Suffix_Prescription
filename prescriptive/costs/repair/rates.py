"""
Stundensätze der Werkstatt (repair) je Mitarbeiter und Zeitpunkt, für das Outcome kosten (outcome.py): Grundsatz x Verlauf
über die Tage (je Mitarbeiter) x Faktor der Tagesstunde x Wochenende. Die Werkstatt arbeitet rund um die Uhr
(simulation/repair_engine.py): jede Stunde in einem Arbeitsschritt kostet.

- Grundsätze (€ je Stunde): Sachbearbeiter 40, Prüfer 50, Expertinnen 120. Mechaniker je Stelle in ihrer Fähigkeitsgruppe
  (1-3, 4-6, 7-9 wie simulation/repair_workshop.skill) günstig, mittel, teuer: Tarif A 45 / 60 / 75, Tarif B umgekehrt
- Verlauf über die Tage: je Mitarbeiter ein Faktor je Tag, im Log ein AR(1) zurück zu 1 (KEEP je Tag), Rauschen SD je Tag,
  mit Wahrscheinlichkeit JUMP je Tag ein Sprung um x 0,7 bis x 1,4, begrenzt auf x 0,6 bis x 1,8. Fest ausgelost je Tarif
  (TARIFFS: seed): der Plan ist im Voraus bekannt, PSP und Auswertung rechnen mit denselben Sätzen
- Tagesstunde: 22-6 Uhr x 1,5; 6-18 Uhr x 1,0; 18-22 Uhr x 1,25; Samstag und Sonntag zusätzlich x 1,5
- Rollen (ROLES): für Arbeitsschritte ohne bekannten Mitarbeiter (gezogene Suffixe des PSP) das Mittel der Sätze der Rolle

Zeit: Stunden seit EPOCH (1.1.2024, ein Montag, Beginn von Log und Simulationen wie repair_engine.START), naive UTC wie im
Log. Kosten eines Intervalls in O(1): je Mitarbeiter die aufsummierten Kosten ganzer Tage (Plan.cum) und innerhalb des
Tages die aufsummierten Stundenfaktoren (DAY_GRID, DAY_CUM).

    from prescriptive.costs.repair import rates
    rates.rate("mechanic_3", "2024-03-01 20:00")                              # € je Stunde, Tarif A
    rates.labor_cost("mechanic_3", "2024-03-01 08:00", "2024-03-03 14:00")   # € für das Intervall
"""
import functools

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("2024-01-01")
DAYS = 365 * 30  # Plan bis Ende 2053 (der Log der Werkstatt reicht bis 2046)
# Stundenfaktoren: 0-6 Uhr 1,5; 6-18 Uhr 1,0; 18-22 Uhr 1,25; 22-24 Uhr 1,5. DAY_CUM: aufsummiert bis zur Stunde in DAY_GRID
DAY_GRID = np.array([0.0, 6.0, 18.0, 22.0, 24.0])
DAY_CUM = np.concatenate([[0.0], np.cumsum(np.diff(DAY_GRID) * np.array([1.5, 1.0, 1.25, 1.5]))])
WEEKEND = 1.5
# Mitarbeiter wie in simulation/repair_workshop.py
ROLES = {"clerk": tuple(f"clerk_{k}" for k in range(1, 7)), "inspector": tuple(f"inspector_{k}" for k in range(1, 7)),
         "mechanic": tuple(f"mechanic_{k}" for k in range(1, 10)), "expert": tuple(f"expert_{k}" for k in range(1, 4))}
BASE = {"clerk": 40.0, "inspector": 50.0, "expert": 120.0}
MECHANIC_BASE = {"A": (45.0, 60.0, 75.0), "B": (75.0, 60.0, 45.0)}  # je Stelle in der Fähigkeitsgruppe
TARIFFS = {"A": 1, "B": 2}  # Tarif -> seed des Verlaufs
KEEP, SD, JUMP, JUMP_RANGE, BOUNDS = 0.02, 0.04, 0.02, (0.7, 1.4), (0.6, 1.8)
# Wer welchen Arbeitsschritt macht (repair_workshop: pools, W_repair Mechaniker oder Expertin)
ROLE_OF = {"W_intake": "clerk", "W_warranty_check": "clerk", "W_shipping": "clerk", "W_safety_check": "inspector",
           "W_quality_check": "inspector", "W_repair": "mechanic"}


def role(activity):
    """Die Rolle, die den Arbeitsschritt macht (ihr mittlerer Satz gilt, wenn der Mitarbeiter nicht bekannt ist); sonst None."""
    return ROLE_OF.get(activity)


def hours(values):
    """Zeitpunkte (Timestamp, datetime, Text oder Array davon; naive UTC oder mit Zeitzone) als Stunden seit EPOCH."""
    times = pd.to_datetime(pd.Series(values) if np.ndim(values) else pd.Series([values]))
    if times.dt.tz is not None:
        times = times.dt.tz_convert("UTC").dt.tz_localize(None)
    result = ((times - EPOCH).dt.total_seconds() / 3600).to_numpy()
    return result if np.ndim(values) else float(result[0])


class Plan:
    """Ein Tarif: je Mitarbeiter und Rolle der Satz je Tag ohne Stundenfaktor (daily) und die Kosten ganzer Tage aufsummiert (cum)."""

    def __init__(self, tariff):
        if tariff not in TARIFFS:
            raise ValueError(f"Unbekannter Tarif {tariff}, möglich: {', '.join(TARIFFS)}")
        rng = np.random.default_rng(TARIFFS[tariff])
        people = [person for members in ROLES.values() for person in members]
        base = np.array([MECHANIC_BASE[tariff][(int(p.rsplit("_", 1)[1]) - 1) % 3] if p.startswith("mechanic") else BASE[p.rsplit("_", 1)[0]]
                         for p in people])
        noise = rng.normal(0.0, SD, (len(people), DAYS))
        jumps = np.where(rng.random((len(people), DAYS)) < JUMP, np.log(rng.uniform(*JUMP_RANGE, (len(people), DAYS))), 0.0)
        level = np.zeros((len(people), DAYS))
        low, high = np.log(BOUNDS)
        for day in range(1, DAYS):
            level[:, day] = np.clip((1 - KEEP) * level[:, day - 1] + noise[:, day] + jumps[:, day], low, high)
        weekend = np.where(np.arange(DAYS) % 7 >= 5, WEEKEND, 1.0)  # EPOCH ist ein Montag
        daily = base[:, None] * np.exp(level) * weekend[None, :]
        # Rollen: das Mittel ihrer Mitarbeiter
        roles = [np.mean([daily[people.index(p)] for p in members], axis=0) for members in ROLES.values()]
        self.names = people + list(ROLES)
        self.index = {name: k for k, name in enumerate(self.names)}
        self.daily = np.vstack([daily] + roles)
        self.cum = np.concatenate([np.zeros((len(self.names), 1)), np.cumsum(self.daily * DAY_CUM[-1], axis=1)], axis=1)

    def rows(self, names):
        """Zeilen der Mitarbeiter bzw. Rollen (-1: unbekannt, z. B. system; kostet nichts)."""
        return np.array([self.index.get(name, -1) for name in np.atleast_1d(names)])

    def cost(self, rows, start, end):
        """Kosten je Intervall (rows aus self.rows, start und end in Stunden seit EPOCH, Arrays gleicher Länge) in €."""
        rows, start, end = np.atleast_1d(rows), np.atleast_1d(np.asarray(start, float)), np.atleast_1d(np.asarray(end, float))
        end = np.maximum(end, start)
        first, last = np.floor(start / 24).astype(int), np.floor(end / 24).astype(int)
        if len(rows) and (first.min() < 0 or last.max() >= DAYS):
            raise ValueError(f"Zeit außerhalb des Plans ({EPOCH:%d.%m.%Y} plus {DAYS} Tage)")
        known = rows >= 0
        r = np.where(known, rows, 0)
        before = self.daily[r, first] * np.interp(start - 24 * first, DAY_GRID, DAY_CUM)
        until = self.daily[r, last] * np.interp(end - 24 * last, DAY_GRID, DAY_CUM)
        return np.where(known, self.cum[r, last] - self.cum[r, first] + until - before, 0.0)

    def rate(self, rows, at):
        """Satz in € je Stunde zum Zeitpunkt at (Stunden seit EPOCH)."""
        rows, at = np.atleast_1d(rows), np.atleast_1d(np.asarray(at, float))
        day = np.floor(at / 24).astype(int)
        span = np.clip(np.searchsorted(DAY_GRID, at - 24 * day, side="right") - 1, 0, len(DAY_GRID) - 2)
        factor = (np.diff(DAY_CUM) / np.diff(DAY_GRID))[span]
        return np.where(rows >= 0, self.daily[np.where(rows >= 0, rows, 0), day] * factor, 0.0)


@functools.lru_cache(maxsize=None)
def plan(tariff="A"):
    return Plan(tariff)


def rate(resource, at, tariff="A"):
    """Stundensatz von resource (Mitarbeiter oder Rolle) zum Zeitpunkt at in €; Arrays gleicher Länge möglich."""
    p = plan(tariff)
    result = p.rate(p.rows(resource), hours(at))
    return result if np.ndim(at) or np.ndim(resource) else float(result[0])


def labor_cost(resource, start, end, tariff="A"):
    """Kosten von resource (Mitarbeiter oder Rolle) für die Arbeit von start bis end in €, Stunde für Stunde; Arrays gleicher Länge möglich."""
    p = plan(tariff)
    result = p.cost(p.rows(resource), hours(start), hours(end))
    return result if np.ndim(start) or np.ndim(resource) else float(result[0])
