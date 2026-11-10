"""
Validierung des Screenings im Simulator: trifft das Screening (AIPW auf einem beobachteten Log, evaluation.evaluate) die
wahre Wirkung einer Intervention? Im Simulator ist sie bekannt. Je Kandidat, der ein Wahlpunkt des Simulators ist
(zusammengelegte Punkte wie channel+angebot_nachlegen zählen für jeden ihrer Kandidaten):

- Simulator, beobachtet: der Simulator wie die Bank, darauf das Screening wie auf dem echten Log (Mitarbeiter mit Pool
  frei und den Merkmalen der Zuteilung, candidates.with_allocation)
- Simulator, wahr: derselbe Simulator mit denselben Ankünften, nur an diesem Wahlpunkt zufällig (sonst wie die Bank).
  Control Flow: Differenz der Outcome-Mittel mit und ohne Aktion am ersten Entscheidungspunkt des Falls (randomisiert).
  Mitarbeiter: je Mitarbeiter, wie viel besser ein Fall mit ihm ist als mit einem zufällig Gewählten aus dem Pool (die
  freien Kandidaten, die die Engine angeboten hat, hier mitgeschrieben): das Mittel der Fälle, in denen er gewählt wurde,
  gewichtet mit der Poolgröße (bekannte Propensity 1/Poolgröße, Hajek), minus das Mittel aller Fälle, in deren Pool er war.
  Das ist dieselbe Größe, die das Screening schätzt (evaluation.employee_effects).
- echt: das Screening auf dem echten Log.

Ergebnis je Kandidat: beobachtet gegen wahr (trifft das Screening die Wahrheit? Dann ist es auch auf dem echten Log
glaubwürdig, soweit die Konfundierung dort von derselben Art ist) und echt gegen wahr (gibt der Simulator wieder, was das
Screening im echten Log findet?). Mitarbeiter: die Streuung der Effekte (um das Rauschen bereinigt), ihre Reliabilität
(Anteil der Varianz einer typischen Schätzung, der nicht Rauschen ist) und die Korrelation der Effekte je Mitarbeiter,
nach Präzision gewichtet, roh und um das Rauschen korrigiert (roh geteilt durch die Wurzel der beiden Reliabilitäten,
nur wenn beide mindestens MIN_RELIABILITY). Die Streuung schätzt DerSimonian-Laird (präzise Mitarbeiter zählen mehr).

    from screening.validation import validate
    table, details = validate(simulator, real_log, n_cases=20000)
"""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from helpers.datasets import CASE, add_paths
from .candidates import build_candidate, candidate_tests
from .evaluation import between_spread, evaluate

N_CASES = 20000  # simulierte Fälle je Lauf (Bank und je Wahlpunkt randomisiert, dieselben Ankünfte)
FOLDS = 3  # Cross-Fitting im Screening (weniger als im Schritt screening: hier läuft es zweimal je Kandidat)
MIN_CASES = 200  # Mitarbeiter mit so vielen Fällen bekommen im Screening eine eigene Option (die übrigen: other)
MIN_TRUE = 30  # so oft gewählt, damit ein Mitarbeiter einen wahren Effekt bekommt
MIN_RELIABILITY = 0.3  # darunter keine um das Rauschen korrigierte Korrelation
POOL = "frei"
_SHARED = {}


class RecordingRandom:
    """Zufällige Wahl wie policies/rct_policy.py (ohne delta), schreibt je Wahl Fall, Wahlpunkt, Optionen, Wahl und Zeit mit."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.records = []

    def choose(self, point, case, options, time, engine):
        choice = options[int(self.rng.integers(len(options)))]
        self.records.append({"case": case.case_id, "point": point.name, "options": tuple(options), "choice": choice, "time": time})
        return choice


def _init_worker(dataset, log, intervention):
    add_paths()
    from simulation.dataset_simulator import Simulator
    _SHARED["simulator"] = Simulator(dataset, log, intervention)


def _simulate(job):
    """Ein Lauf: point None wie die Bank, sonst nur an diesem Wahlpunkt zufällig. Liefert (point, vorbereitetes Log, Wahlen)."""
    point, n_cases, seed, costs = job
    from preprocessing import chosen_outcome_spec
    from helpers.simulator_check import prepare_simulated
    simulator = _SHARED["simulator"]
    policy = RecordingRandom([seed, 11]) if point else None
    events, _ = simulator.run(n_cases, policy=policy, points=[point] if point else None, seed=seed)
    log = prepare_simulated(events, chosen_outcome_spec(simulator.dataset), costs)
    print(f"Validierung: Lauf {point or 'Bank'} fertig ({log[CASE].nunique():,} Fälle)", flush=True)
    return point, log, pd.DataFrame(policy.records if policy else [])


def point_of(name, points):
    """Der Wahlpunkt des Simulators zum Kandidaten name (auch Teil eines zusammengelegten Punkts), sonst None."""
    return next((point for point in points if name == point.name or name in point.name.split("+")), None)


def _control_truth(dataset, log, name):
    candidate = build_candidate(dataset, log, name)
    T, Y = candidate.T.to_numpy(dtype=int), candidate.Y.to_numpy(dtype=float)
    a, b = Y[T == 1], Y[T == 0]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, len(Y)
    return float(a.mean() - b.mean()), float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))), len(Y)


def _employee_truth(log, choices, point_name):
    """Wahrer Effekt je Mitarbeiter (siehe oben) an der ersten Wahl des Falls am Punkt, mit Standardfehler."""
    outcome = log.groupby(CASE)["outcome"].first()
    first = choices[choices["point"] == point_name].sort_values("time").groupby("case").head(1)
    first = first[first["options"].map(len) >= 2]
    first = first.assign(y=outcome.reindex(first["case"]).to_numpy(dtype=float), size=first["options"].map(len).astype(float))
    first = first[np.isfinite(first["y"])]
    members = sorted(set().union(*first["options"])) if len(first) else []
    rows = []
    for member in members:
        inside = first[first["options"].map(lambda options: member in options)]
        chosen = inside[inside["choice"] == member]
        if len(chosen) < MIN_TRUE:
            continue
        w, y = chosen["size"].to_numpy(), chosen["y"].to_numpy()
        mean = float((w * y).sum() / w.sum())
        se_mean = float(np.sqrt((w ** 2 * (y - mean) ** 2).sum()) / w.sum())
        base = inside["y"].to_numpy()
        rows.append({"Mitarbeiter": member, "gewählt": len(chosen), "im Pool": len(inside), "Effekt wahr": mean - float(base.mean()),
                     "Std.-Fehler wahr": float(np.hypot(se_mean, base.std(ddof=1) / np.sqrt(len(base))))})
    return pd.DataFrame(rows, columns=["Mitarbeiter", "gewählt", "im Pool", "Effekt wahr", "Std.-Fehler wahr"]).set_index("Mitarbeiter")


def _spread(effects, se):
    """Streuung der wahren Effekte (DerSimonian-Laird, evaluation.between_spread) und Reliabilität: Anteil der Varianz einer
    typischen Schätzung, der nicht Rauschen ist (Rauschen: harmonisches Mittel der quadrierten Standardfehler)."""
    spread = between_spread(effects, se)
    se = np.asarray(se, dtype=float)
    se = se[np.isfinite(se) & (se > 0)]
    if not np.isfinite(spread) or not len(se):
        return spread, np.nan
    noise = len(se) / np.sum(1 / se ** 2)
    return spread, spread ** 2 / (spread ** 2 + noise)


def _correlation(a, se_a, b, se_b):
    """
    Korrelation der Effekte zweier Schätzungen über die gemeinsamen Mitarbeiter, gewichtet mit 1 / (se_a² + se_b²) (präzise
    zählen mehr), roh und um das Rauschen korrigiert: geteilt durch die Wurzel der beiden Reliabilitäten mit denselben
    Gewichten (nur wenn beide mindestens MIN_RELIABILITY).
    """
    common = a.index.intersection(b.index)
    common = [m for m in common if np.isfinite(a[m]) and np.isfinite(b[m]) and se_a[m] > 0 and se_b[m] > 0]
    if len(common) < 5:
        return np.nan, np.nan, len(common)
    x, y, sx, sy = (np.asarray(s.loc[common], dtype=float) for s in (a, b, se_a, se_b))
    w = 1 / (sx ** 2 + sy ** 2)
    mx, my = (w * x).sum() / w.sum(), (w * y).sum() / w.sum()
    cov = (w * (x - mx) * (y - my)).sum() / w.sum()
    vx, vy = (w * (x - mx) ** 2).sum() / w.sum(), (w * (y - my) ** 2).sum() / w.sum()
    raw = float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else np.nan
    reliability = []
    for values, se in ((x, sx), (y, sy)):
        spread = between_spread(values, se)
        noise = (w * se ** 2).sum() / w.sum()
        reliability.append(spread ** 2 / (spread ** 2 + noise) if np.isfinite(spread) else np.nan)
    ok = np.isfinite(raw) and all(np.isfinite(r) and r >= MIN_RELIABILITY for r in reliability)
    return raw, (float(np.clip(raw / np.sqrt(reliability[0] * reliability[1]), -1, 1)) if ok else np.nan), len(common)


def _screen(dataset, log, name, kind, seed):
    """Das Screening eines Kandidaten (nur die Effekte): Zeile von evaluate, bei Mitarbeitern mit der Tabelle der Effekte."""
    candidate = build_candidate(dataset, log, name, seed, POOL if kind == "variable" else None)
    if candidate.skip or not len(candidate.X):
        return {"Ergebnis": f"nicht testbar: {candidate.skip}"}
    return evaluate(candidate, FOLDS, seed, min_cases=MIN_CASES, policy=False)


def validate(simulator, log, n_cases=N_CASES, seed=0, costs=None, workers=4, candidates=None):
    """
    Die Tabelle je Kandidat (siehe oben) und Details (je Mitarbeiter-Kandidat und Mitarbeiter: Effekt echt, beobachtet im
    Simulator und wahr). log: der echte Log nach dem Preprocessing; candidates: Namen (Standard alle des Katalogs, die ein
    Wahlpunkt des Simulators sind).
    """
    dataset = simulator.dataset
    tests = {**{name: "control_flow" for name in candidate_tests(dataset, "control_flow")}, **{name: "variable" for name in candidate_tests(dataset, "variable")}}
    chosen = {name: kind for name, kind in tests.items() if (candidates is None or name in candidates) and point_of(name, simulator.points)}
    points = sorted({point_of(name, simulator.points).name for name in chosen})
    print(f"Validierung: {len(chosen)} Kandidaten an {len(points)} Wahlpunkten, {n_cases:,} Fälle je Lauf ...", flush=True)
    jobs = [(None, n_cases, seed, costs)] + [(point, n_cases, seed, costs) for point in points]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker,
                             initargs=(dataset, simulator.log, simulator.intervention["key"])) as pool:
        runs = {point: (events, records) for point, events, records in pool.map(_simulate, jobs)}
    bank = runs.pop(None)[0]
    rows, details = [], []
    for name, kind in chosen.items():
        point = point_of(name, simulator.points)
        print(f"Validierung: {name} (Wahlpunkt {point.name}) ...", flush=True)
        randomized, records = runs[point.name]
        observed, real = _screen(dataset, bank, name, kind, seed), _screen(dataset, log, name, kind, seed)
        row = {"Kandidat": name, "Art": "Control Flow" if kind == "control_flow" else "Mitarbeiter", "Wahlpunkt": point.name,
               "Überlappung echt": real.get("Überlappung"), "Überlappung Simulator": observed.get("Überlappung"),
               "Placebo d echt": real.get("Placebo d"), "Placebo d Simulator": observed.get("Placebo d")}
        if kind == "control_flow":
            truth, truth_se, n = _control_truth(dataset, randomized, name)
            estimate = lambda result: (result.get("Effekt (AIPW)", np.nan), (result.get("Effekt KI oben", np.nan) - result.get("Effekt (AIPW)", np.nan)) / 1.96)
            (real_value, real_se), (observed_value, observed_se) = estimate(real), estimate(observed)
            row.update({"echt (AIPW)": real_value, "echt Std.-Fehler": real_se, "Simulator beobachtet (AIPW)": observed_value,
                        "Simulator beobachtet Std.-Fehler": observed_se, "Simulator wahr": truth, "Simulator wahr Std.-Fehler": truth_se,
                        "Fälle wahr": n,
                        "Screening trifft wahr (z)": (observed_value - truth) / np.hypot(observed_se, truth_se),
                        "echt gegen Simulator wahr (z)": (real_value - truth) / np.hypot(real_se, truth_se)})
        else:
            truth = _employee_truth(randomized, records, point.name)
            sets = {"echt": real.get("_effects"), "Simulator beobachtet": observed.get("_effects")}
            effects = {label: table["Effekt ggü. Pool (AIPW)"] for label, table in sets.items() if table is not None and len(table)}
            ses = {label: sets[label]["Std.-Fehler"] for label in effects}
            if len(truth):
                effects["Simulator wahr"], ses["Simulator wahr"] = truth["Effekt wahr"], truth["Std.-Fehler wahr"]
            stats = {label: _spread(effects[label].to_numpy(), ses[label].to_numpy()) for label in effects}
            for label in ("echt", "Simulator beobachtet", "Simulator wahr"):
                spread, reliability = stats.get(label, (np.nan, np.nan))
                row.update({f"Streuung {label}": spread, f"Reliabilität {label}": reliability, f"Mitarbeiter {label}": len(effects.get(label, []))})
            for (a, b), label in ((("Simulator beobachtet", "Simulator wahr"), "Screening trifft wahr"), (("echt", "Simulator wahr"), "echt gegen Simulator wahr")):
                if a in effects and b in effects:
                    raw, corrected, common = _correlation(effects[a], ses[a], effects[b], ses[b])
                    row.update({f"{label}: Korrelation": raw, f"{label}: Korrelation korrigiert": corrected, f"{label}: Mitarbeiter": common})
            members = sorted(set().union(*(effect.index for effect in effects.values())))
            for member in members:
                details.append({"Kandidat": name, "Mitarbeiter": member, **{f"Effekt {label}": float(effects[label].get(member, np.nan)) for label in effects},
                                **{f"Std.-Fehler {label}": float(ses[label].get(member, np.nan)) for label in ses}})
        rows.append(row)
    return pd.DataFrame(rows).set_index("Kandidat"), pd.DataFrame(details)
