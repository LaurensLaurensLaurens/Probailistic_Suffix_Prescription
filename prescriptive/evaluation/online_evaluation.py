"""
Online-Auswertung auf dem Simulator (BPIC17, simulation/dataset_simulator.py): Der Simulator gibt die Umgebung vor (alle
Fälle in einer Engine, Dienstplan, Berechtigungen, belegte Mitarbeiter), die Policy entscheidet live an den Wahlpunkten
(points: Namen, Standard alle des Simulators, z. B. nur channel oder nur "Mitarbeiter: W_Call incomplete files").

Policies:
    bank      keine Policy: das Modell zieht jedes Event wie die Bank, die Zuteilungsstrategie teilt die Mitarbeiter zu
    random    an jedem Wahlpunkt eine zufällige Option (policies/rct_policy.py ohne delta)
    xgb       die Option mit dem höchsten geschätzten Outcome nach XGBoost (policies/xgb_policy.py), das Modell des gewählten
              Outcomes (helpers/training.model_path: models/xgb_model.pkl bzw. z. B. xgb_model_gewinn.pkl; ein anderes über
              policy_params, z. B. {"xgb": {"model": "models/xgb_model_repair_kosten_satz.pkl"}})
    lstm      dasselbe mit dem LSTM (policies/lstm_policy.py, models/lstm_model.pkl bzw. z. B. lstm_model_gewinn.pkl)
    psp       dasselbe mit dem PSP, Outcome aus gezogenen Suffixen (policies/psp_policy.py, models/psp_model<outcome>.pkl)
    psp_lookahead
              Mitarbeiterwahl nach dem PSP mit Blick auf die anderen laufenden Fälle, darf warten (policies/psp_lookahead_policy.py,
              dasselbe Modell wie psp); nur an Mitarbeiter-Wahlpunkten, sonst wie die Bank
    psp_erm   wie psp, aber die Option mit dem kleinsten entropischen Risikomaß der gezogenen Durchlaufzeiten
              (policies/psp_erm_policy.py, dasselbe Modell wie psp; γ über policy_params, z. B. {"psp_erm": {"gamma": 0.1}})

Je Wiederholung (seed) läuft jede Policy einmal über n_cases Ankünfte mit demselben seed (gleiche Ankünfte und
Fallattribute, jeder Fall zieht aus eigenen Zufallszahlen). Die Policy wird nur für die Fälle mit Ankunftsnummer im
Fenster [warmup, n_cases - cooldown) gefragt und nur diese werden gewertet; die anderen laufen mit, damit die Auslastung
stimmt. Verglichen wird je seed gegen die Bank (gepaart): Uplift des Outcomes je Fall in % mit 95-%-Konfidenzintervall
über die Wiederholungen. Das Outcome je Fall wie beim Simulator-Check (helpers/simulator_check.prepare_simulated).

Dazu der Schwanz der Durchlaufzeit (TAIL), je Policy und gepaart gegen die Bank mit 95-%-KI: 90- und 95-%-Quantil,
CVaR 90 % (Mittel der längsten 10 % der Fälle) und der Anteil der Fälle über τ, dem TAIL_QUANTILE-Quantil der Bank in
derselben Wiederholung (bei der Bank also 1 − TAIL_QUANTILE). Je Mitarbeiter-Wahlpunkt die Wartezeit auf einen
Mitarbeiter dort (WAITING): je Fall, der ihn erreicht, wie oft gewartet wurde, 90-%-Quantil, Anteil "alle beschäftigt".

    from prescriptive.evaluation.online_evaluation import evaluate_online
    table, runs = evaluate_online(simulator, policies=("bank", "random", "xgb"), points=["channel"], seeds=5)
"""
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from helpers.datasets import ACTIVITY, CASE, LIFECYCLE, TIMESTAMP, add_paths
from helpers.parameters import SEED, WORKERS

POLICIES = ("bank", "random", "xgb", "lstm", "psp", "psp_lookahead", "psp_erm")
PSP_POLICIES = ("psp", "psp_lookahead", "psp_erm")  # alle mit dem Modell des PSP
N_CASES, WARMUP, COOLDOWN, SEEDS = 4000, 700, 1300, 5  # wie scripts/main.py --online-cases, --online-warmup, --online-cooldown, --online-seeds
TAIL_QUANTILE = 0.9  # τ: dieses Quantil der Durchlaufzeit unter der Bank, je Wiederholung
ABOVE = "Anteil über τ (Bank-P90)"
TAIL = ["Durchlaufzeit P90 (Tage)", "Durchlaufzeit P95 (Tage)", "CVaR 90 % (Tage)", ABOVE]
WAITING = ("Wartezeit je Fall dort (h)", "Anteil mit Wartezeit", "Wartezeit P90 (h)", "davon alle beschäftigt (Anteil)")  # je Wahlpunkt ": <Name>"
_SHARED = {}  # was die Arbeitsprozesse brauchen (_init_worker)


def evaluate_online(simulator, policies=POLICIES, points=None, n_cases=N_CASES, warmup=WARMUP, cooldown=COOLDOWN, seeds=SEEDS, seed=SEED,
                    workers=WORKERS, costs=None, policy_params=None):
    """
    Tabelle je Policy (Mittel über die Wiederholungen, Uplift gegen die Bank) und die Zeilen je Lauf. points: Namen der
    Wahlpunkte (None: alle des Simulators); seeds: Wiederholungen, Wiederholung r mit seed + r. policy_params: je Policy
    Argumente für ihre Klasse, z. B. {"psp_erm": {"gamma": 0.1}}.
    """
    unknown = set(policies) - set(POLICIES)
    if unknown:
        raise ValueError(f"Unbekannte Policies {', '.join(sorted(unknown))}, möglich: {', '.join(POLICIES)}")
    policies = ["bank"] + [name for name in policies if name != "bank"]
    window = (warmup, n_cases - cooldown)
    jobs = [(name, seed + r, n_cases, window, points) for r in range(seeds) for name in policies]
    # Die langen Läufe (Modelle) zuerst, damit die kurzen die Lücken füllen
    jobs.sort(key=lambda job: job[0] in ("bank", "random"))
    names = points or [point.name for point in simulator.points]
    print(f"Online {simulator.dataset}, Wahlpunkte {', '.join(names)}: {len(policies)} Policies x {seeds} Wiederholungen mit je {n_cases:,} Fällen, "
          f"gewertet {window[0]} bis {window[1]}, {min(workers, len(jobs))} Prozesse ...", flush=True)
    started = time.time()
    config = (simulator.dataset, simulator.log, simulator.intervention["key"], costs, policy_params)
    if min(workers, len(jobs)) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker,
                                 initargs=config) as pool:
            rows = list(pool.map(_run, jobs))
    else:
        _init_worker(*config)
        rows = [_run(job) for job in jobs]
    print(f"Fertig in {time.time() - started:.0f} s", flush=True)
    return compare(rows)


# Gepaarter Vergleich gegen die Bank je seed: Uplift des Outcomes je Fall in % und die Differenz der Schwanz-Kennzahlen
# je Wiederholung, dann Mittel und 95-%-Konfidenzintervall über die Wiederholungen
def compare(rows):
    from scipy import stats
    runs = pd.DataFrame(rows)
    bank = runs[runs["Policy"] == "bank"].set_index("seed")
    tau = bank["_days"].map(lambda days: float(np.quantile(days, TAIL_QUANTILE)))
    runs[ABOVE] = [float(np.mean(days > tau[seed])) for days, seed in zip(runs["_days"], runs["seed"])]
    runs = runs.drop(columns="_days")
    bank = runs[runs["Policy"] == "bank"].set_index("seed")
    reference = runs["seed"].map(bank["Outcome je Fall"])
    runs["ggü. Bank je Fall"] = runs["Outcome je Fall"] - reference
    runs["Uplift ggü. Bank (%)"] = runs["ggü. Bank je Fall"] / reference.abs() * 100
    for metric in TAIL:
        runs[f"{metric} ggü. Bank"] = runs[metric] - runs["seed"].map(bank[metric])
    table = runs.drop(columns="seed").groupby("Policy", sort=False).mean(numeric_only=True)
    for column, label in [("Uplift ggü. Bank (%)", "Uplift")] + [(f"{metric} ggü. Bank",) * 2 for metric in TAIL]:
        spread = runs.groupby("Policy", sort=False)[column]
        n = spread.count()
        half = spread.std() / np.sqrt(n) * stats.t.ppf(0.975, np.maximum(n - 1, 1))
        table[f"{label} 95-%-KI unten"] = table[column] - half
        table[f"{label} 95-%-KI oben"] = table[column] + half
    table["Wiederholungen"] = n
    return table, runs


# ------------------------------------------------------------------ Läufe

# Wird aufgerufen von evaluate_online -> ProcessPoolExecutor -> _init_worker (je Arbeitsprozess einmal): der Simulator
# des Datensatzes (das Modell ist schon trainiert)
def _init_worker(dataset, log, intervention, costs, policy_params=None):
    add_paths()
    from simulation.dataset_simulator import Simulator
    _SHARED.update(simulator=Simulator(dataset, log, intervention), costs=costs, policy_params=policy_params or {})


def _policy(name, seed):
    if name == "bank":
        return None
    if name == "random":
        from prescriptive.policies.rct_policy import RCTPolicy
        return RCTPolicy([seed, 7])
    from helpers.training import model_path
    path = model_path("psp" if name in PSP_POLICIES else name, _SHARED["simulator"].dataset)
    params = _SHARED["policy_params"].get(name) or {}
    if name == "psp_lookahead":
        from prescriptive.policies.psp_lookahead_policy import PSPLookaheadPolicy
        return PSPLookaheadPolicy(path, seed=seed, **params)
    if name == "psp_erm":
        from prescriptive.policies.psp_erm_policy import PSPERMPolicy
        return PSPERMPolicy(path, seed=seed, **params)
    if name == "lstm":
        from prescriptive.policies.lstm_policy import LSTMPolicy
        return LSTMPolicy(path, seed=seed)
    if name == "psp":
        from prescriptive.policies.psp_policy import PSPPolicy
        return PSPPolicy(path, seed=seed)
    from prescriptive.policies.xgb_policy import XGBPolicy
    return XGBPolicy(params.get("model") or path, seed=seed)  # model: ein anderes XGBoost (z. B. mit Stundensätzen)


# Ein Lauf einer Policy. job: (Name, seed, Fälle, Fenster, Wahlpunkte)
# Wird aufgerufen von evaluate_online -> _run (je Lauf, im Arbeitsprozess)
def _run(job):
    from preprocessing import chosen_outcome_spec
    from helpers.simulator_check import prepare_simulated
    name, seed, n_cases, window, points = job
    simulator = _SHARED["simulator"]
    started = time.time()
    policy = _policy(name, seed)
    events, choices = simulator.run(n_cases, policy=policy, points=points, window=window, seed=seed)
    log = prepare_simulated(events, chosen_outcome_spec(simulator.dataset), _SHARED["costs"])
    numbers = log[CASE].astype(str).str.rsplit("_", n=1).str[1].astype(int)
    log = log[(numbers >= window[0]) & (numbers < window[1])]
    cases = log.groupby(CASE)
    days = (cases[TIMESTAMP].max() - cases[TIMESTAMP].min()).dt.total_seconds() / 86400
    row = {"Policy": name, "seed": seed, "Fälle": log[CASE].nunique(), "Outcome je Fall": cases["outcome"].first().mean(),
           "Events je Fall": len(log) / max(log[CASE].nunique(), 1), "Durchlaufzeit (Tage)": days.mean(),
           "Wartezeit auf Mitarbeiter je Fall (h, alle Fälle)": simulator.waiting_hours / n_cases,
           "Erzwungene Zuteilungen je Fall (alle Fälle)": simulator.forced_assignments / n_cases}
    p90, p95 = np.quantile(days, [0.9, 0.95])
    # _days: die Durchlaufzeiten, für den Anteil über τ der Bank in compare (danach entfernt)
    row.update({TAIL[0]: p90, TAIL[1]: p95, TAIL[2]: days[days >= p90].mean(), "_days": days.to_numpy()})
    row.update(_point_rows(simulator, log, choices, points))
    row.update(getattr(policy, "stats", None) or {})  # was die Policy selbst zählt (psp_lookahead, psp_erm)
    print(f"seed {seed}, {name}: Outcome je Fall {row['Outcome je Fall']:,.1f} ({time.time() - started:.0f} s)", flush=True)
    return row


# Je Wahlpunkt: wie oft die Policy gewählt hat; beim Control Flow der Anteil der zweiten Option an allen
# Entscheidungspunkten (auch ohne Policy, aus dem Log), beim Mitarbeiter die freien Kandidaten je Wahl und die
# Wartezeit auf einen Mitarbeiter dort (simulator.waits, nur die gewerteten Fälle)
def _point_rows(simulator, log, choices, points):
    from interventions import WAIT, option_token
    rows = {}
    for point in simulator.points:
        if points is not None and point.name not in points:
            continue
        made = choices[(choices["point"] == point.name) & choices["choice"].notna()]
        rows[f"Wahlen der Policy: {point.name}"] = len(made)
        rows[f"davon warten: {point.name}"] = int((made["choice"] == WAIT).sum())
        if point.kind == "control_flow":
            ordered = log.sort_values([CASE, TIMESTAMP], kind="stable")
            options = [option_token(option) for option in point.options]
            following = (ordered[ACTIVITY].astype(str) + "|" + ordered[LIFECYCLE].astype(str)).groupby(ordered[CASE]).shift(-1)
            at = (ordered[ACTIVITY] == point.activity) & (ordered[LIFECYCLE] == point.lifecycle) & following.isin(options)
            rows[f"Anteil {point.options[1]}: {point.name}"] = (following[at] == options[1]).mean() if at.any() else np.nan
        else:
            rows[f"freie Kandidaten je Wahl: {point.name}"] = made["options"].mean() if len(made) else np.nan
            rows.update(_waiting_rows(simulator.waits, log, point))
    return rows


# Wartezeit auf einen Mitarbeiter an einem Mitarbeiter-Wahlpunkt: je Fall, der ihn erreicht (Stunden), Anteil der
# Beginne mit Wartezeit, 90-%-Quantil über alle Beginne (ohne Wartezeit 0) und Anteil "alle beschäftigt" der Wartezeit
def _waiting_rows(waits, log, point):
    from simulation.case_engine_lifecycle_aware import ALL_BUSY
    starts = log[(log[ACTIVITY] == point.activity) & (log[LIFECYCLE] == point.lifecycle)]
    waits = waits[(waits["activity"] == point.activity) & (waits["lifecycle"] == point.lifecycle) & waits["case"].isin(set(log[CASE]))]
    hours = np.r_[waits["seconds"].to_numpy() / 3600, np.zeros(max(len(starts) - len(waits), 0))]
    cases = starts[CASE].nunique()
    busy = waits.loc[waits["reason"] == ALL_BUSY, "seconds"].sum() / waits["seconds"].sum() if waits["seconds"].sum() > 0 else np.nan
    values = (waits["seconds"].sum() / 3600 / cases if cases else np.nan, len(waits) / len(starts) if len(starts) else np.nan,
              float(np.quantile(hours, 0.9)) if len(hours) else np.nan, busy)
    return {f"{name}: {point.name}": value for name, value in zip(WAITING, values)}
