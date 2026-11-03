"""
Erste Prüfung der Reparaturwerkstatt (process-simulation-engine/simulation/repair_engine.py), noch ohne PSP: Ist am Wahlpunkt etwas zu holen,
und schlägt der Blick auf die anderen Fälle die besten Handregeln? Vergleich mit policies/repair_rules.py.

Fair gemacht: Die Schwellen (limit der Regeln, queue_weight von Lookahead und Orakel) werden auf eigenen Seeds (TUNE) eingestellt und auf anderen
(EVAL) ausgewertet, je Policy gleich sorgfältig. Alle Policies sehen dieselben Fälle (gepaart je Seed).

Aufruf aus process-simulation-engine/ (Standard: alle RATES und DELAYS):
    .venv/bin/python -u ../prescriptive/scripts/repair_check.py --rates 1.3 1.4 --delays 0.2
Ergebnisse in prescriptive/scripts/results_repair/.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "process-simulation-engine"), ROOT]

from prescriptive.policies.repair_rules import CountRule, HoldAlways, OracleLookahead, TrueLookahead  # noqa: E402
from simulation.repair_engine import RepairEngine  # noqa: E402

OUT = os.path.join(ROOT, "prescriptive", "scripts", "results_repair")
CASES, WARMUP = 12000, 800
TUNE, EVAL = range(100, 104), range(0, 12)
RATES = (1.1, 1.2, 1.3)
DELAYS = (0.2, 0.0)
LIMITS = (1, 2, 3, 5, 8, 12, 20, 40, float("inf"))
WEIGHTS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 6.0, 12.0)
TUNED = {"Regel: Premium-Muster": ("limit", LIMITS), "Regel: Muster, bald erwartet": ("limit", LIMITS),
         "Lookahead wahr": ("weight", WEIGHTS), "Orakel": ("weight", WEIGHTS)}


def make(name, value, process):
    if name == "immer nehmen":
        return None
    if name == "immer freihalten":
        return HoldAlways()
    if name == "Regel: Premium-Muster":
        return CountRule(value, process=process)
    if name == "Regel: Muster, bald erwartet":
        return CountRule(value, imminent=True, process=process)
    return (TrueLookahead if name == "Lookahead wahr" else OracleLookahead)(value, process)


def simulate(job):
    rate, delay, name, value, seed = job
    process = {"delay": delay}
    engine = RepairEngine(rate, seed, process)
    policy = make(name, value, process)
    engine.set_policy(policy, [engine.point] if policy is not None else [], window=(WARMUP, CASES - WARMUP))
    engine.run(CASES)
    table = engine.case_table()
    table = table[table["number"].between(WARMUP, CASES - WARMUP - 1) & ~table["cancelled"]]
    made = pd.DataFrame(engine.choices, columns=["choice"])
    premium = table[table["premium"]]
    normal = table[~table["premium"]]
    return {"Rate": rate, "Verzögerung": delay, "Policy": name, "Einstellung": value, "seed": seed, "Wert je Fall": table["value"].mean(),
            "Premium pünktlich": premium["on_time"].mean(), "Normal pünktlich": normal["on_time"].mean(), "Tage je Fall": table["days"].mean(),
            "freigehalten": (made["choice"] == "warten").mean() if len(made) else 0.0, "Fälle gewertet": len(table)}


def run_all(jobs):
    with ProcessPoolExecutor(max_workers=12) as pool:
        return pd.DataFrame(list(pool.map(simulate, jobs, chunksize=4)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", nargs="+", type=float, default=list(RATES))
    parser.add_argument("--delays", nargs="+", type=float, default=list(DELAYS))
    args = parser.parse_args()
    rates, delays = args.rates, args.delays
    os.makedirs(OUT, exist_ok=True)
    tuned = run_all([(rate, delay, name, value, seed) for rate in rates for delay in delays for name, (_, values) in TUNED.items() for value in values for seed in TUNE])
    choice = tuned.groupby(["Rate", "Verzögerung", "Policy", "Einstellung"])["Wert je Fall"].mean().reset_index()
    best = choice.loc[choice.groupby(["Rate", "Verzögerung", "Policy"])["Wert je Fall"].idxmax()]
    tuned.to_csv(os.path.join(OUT, "einstellung.csv"), index=False)
    print("Gewählte Einstellungen (Tuning-Seeds):")
    print(best.pivot(index=["Rate", "Verzögerung"], columns="Policy", values="Einstellung").to_string(), flush=True)

    jobs = []
    for rate in rates:
        for delay in delays:
            for name in ("immer nehmen", "immer freihalten"):
                jobs += [(rate, delay, name, None, seed) for seed in EVAL]
            for row in best[(best["Rate"] == rate) & (best["Verzögerung"] == delay)].itertuples():
                jobs += [(rate, delay, row.Policy, row.Einstellung, seed) for seed in EVAL]
    runs = run_all(jobs)
    runs.to_csv(os.path.join(OUT, "laeufe.csv"), index=False)

    order = ["immer nehmen", "immer freihalten", "Regel: Premium-Muster", "Regel: Muster, bald erwartet", "Lookahead wahr", "Orakel"]
    rows = []
    for (rate, delay), part in runs.groupby(["Rate", "Verzögerung"]):
        wide = part.pivot(index="seed", columns="Policy", values="Wert je Fall")
        rules = wide[[name for name in order[:4]]].mean().idxmax()
        for name in order:
            one = part[part["Policy"] == name]
            gap = 100 * (wide[name] / wide[rules] - 1)
            half = 2.2 * gap.std(ddof=1) / np.sqrt(len(gap))  # 95-%-KI, t mit 11 Freiheitsgraden
            rows.append({"Rate": rate, "Verzögerung": delay, "Policy": name, "Wert je Fall": one["Wert je Fall"].mean(),
                         "ggü. immer nehmen %": 100 * (wide[name].mean() / wide["immer nehmen"].mean() - 1), "ggü. beste Regel %": gap.mean(), "KI ±": half,
                         "Premium pünktlich": one["Premium pünktlich"].mean(), "Normal pünktlich": one["Normal pünktlich"].mean(),
                         "Tage je Fall": one["Tage je Fall"].mean(), "freigehalten": one["freigehalten"].mean()})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "ergebnis.csv"), index=False)
    with pd.option_context("display.width", 190, "display.max_columns", 20, "display.max_rows", 100):
        print(table.round(3).to_string(index=False), flush=True)
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
