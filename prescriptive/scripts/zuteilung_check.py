"""
Zuteilung der Mitarbeiter: der echte Log gegen den Simulator wie die Bank, mit der Zuteilung und Berechtigung des Katalogs
(lauf.zuteilung, lauf.berechtigung; eine Kopie über PSP_CATALOG). Kennzahlen an jedem Beginn oder jeder Fortsetzung eines
Arbeitsschritts (start, resume; ohne Systemnutzer):
- Kontinuität: derselbe Mitarbeiter wie beim letzten Beginn oder der letzten Fortsetzung eines Arbeitsschritts im Fall
- Vier-Augen: bei W_Validate application start hatte der Mitarbeiter den Fall vorher schon (sollte selten sein)
- Spezialisierung: Anteil der 10 aktivsten Mitarbeiter an der Arbeit des Schritts; aktive Mitarbeiter je Schritt
- ohne Erfahrung: Anteil der Beginne durch Mitarbeiter, die den Schritt im echten Log weniger als 30-mal begonnen oder fortgesetzt haben
- Gini der Arbeit (Beginne und Fortsetzungen je Mitarbeiter), Wartezeit schedule → start (Median, Stunden), Durchlaufzeit

Aufruf (im Ordner process-simulation-engine): zuteilung_check.py echt|sim --out <csv> [--n-cases 10000]
"""
import argparse
import os
import sys

PRESCRIPTIVE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PRESCRIPTIVE)

import numpy as np
import pandas as pd

from helpers.datasets import ACTIVITY, CASE, LIFECYCLE, RESOURCE, TIMESTAMP, add_paths

MAIN = ["W_Complete application", "W_Validate application", "W_Call after offers", "W_Call incomplete files", "W_Handle leads"]


def gini(values):
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(values) / (n * values.sum())) if n and values.sum() else np.nan


def kpis(log, real_counts, system_users, label):
    log = log.sort_values([CASE, TIMESTAMP], kind="stable").reset_index(drop=True)
    resource = log[RESOURCE].astype(str)
    work = log[ACTIVITY].astype(str).str.startswith("W_")
    opening = work & log[LIFECYCLE].isin(["start", "resume"]) & ~resource.isin(system_users)
    rows = log[opening].copy()
    rows["res"] = resource[opening]
    # letzter Beginn oder letzte Fortsetzung eines Arbeitsschritts im Fall, und wer den Fall schon hatte
    rows["vorher"] = rows.groupby(CASE)["res"].shift(1)
    rows["hatte"] = [r in seen for r, seen in zip(rows["res"], rows.groupby(CASE)["res"].transform(lambda s: [set(s.iloc[:k]) for k in range(len(s))]))]
    out = {}
    for activity in MAIN:
        part = rows[rows[ACTIVITY] == activity]
        start = part[part[LIFECYCLE] == "start"]
        counts = part["res"].value_counts()
        out[f"{activity}: Kontinuität bei start"] = float((start["res"] == start["vorher"]).mean())
        out[f"{activity}: Top-10-Anteil"] = float(counts.head(10).sum() / counts.sum()) if counts.sum() else np.nan
        out[f"{activity}: aktive Mitarbeiter"] = int(len(counts))
        out[f"{activity}: Beginne ohne Erfahrung (<30)"] = float(np.mean([real_counts.get((r, activity), 0) < 30 for r in part["res"]]))
    validate = rows[(rows[ACTIVITY] == "W_Validate application") & (rows[LIFECYCLE] == "start")]
    out["W_Validate application: Validierer hatte den Fall schon"] = float(validate["hatte"].mean())
    out["Gini der Arbeit je Mitarbeiter"] = gini(rows["res"].value_counts().to_numpy())
    times = pd.to_datetime(log[TIMESTAMP])
    for activity in ["W_Complete application", "W_Validate application", "W_Call incomplete files"]:
        mask = log[ACTIVITY] == activity
        sched = log[mask & (log[LIFECYCLE] == "schedule")].groupby(CASE)[TIMESTAMP].min()
        first = log[mask & (log[LIFECYCLE] == "start")].groupby(CASE)[TIMESTAMP].min()
        gap = (pd.to_datetime(first) - pd.to_datetime(sched.reindex(first.index))).dt.total_seconds().dropna() / 3600
        out[f"{activity}: Wartezeit schedule → start, Median (h)"] = float(gap[gap >= 0].median())
    days = log.groupby(CASE)[TIMESTAMP].agg(lambda s: (pd.to_datetime(s).max() - pd.to_datetime(s).min()).total_seconds() / 86400)
    out["Durchlaufzeit Mittel (Tage)"], out["Durchlaufzeit P90 (Tage)"] = float(days.mean()), float(days.quantile(0.9))
    return pd.Series(out, name=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("echt", "sim"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-cases", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default=None)
    parser.add_argument("--first-cases", type=int, default=None, help="echt: nur die ersten so vielen Fälle (nach Beginn), vergleichbar mit der Simulation ab dem ersten Tag")
    args = parser.parse_args()
    add_paths()
    import interventions
    from helpers.datasets import load_dataset
    from preprocessing import preprocess
    from resources.case_allocation import work_counts
    from resources.datasets import system_users
    raw, _, _ = load_dataset("bpic17")
    log = preprocess("bpic17", raw)
    users = system_users("bpic17", log)
    real_counts = work_counts(log, "W_")
    if args.what == "echt":
        part = log
        if args.first_cases:
            first = log.groupby(CASE)[TIMESTAMP].min().sort_values().index[:args.first_cases]
            part = log[log[CASE].isin(set(first))]
        table = kpis(part, real_counts, users, args.label or "echt")
    else:
        from preprocessing import chosen_outcome_spec
        from helpers.simulator_check import prepare_simulated
        from simulation.dataset_simulator import Simulator
        simulator = Simulator("bpic17", log, None)
        label = args.label or f"Simulator ({interventions.run_permissions('bpic17')}, {interventions.run_allocation('bpic17')})"
        print(label, flush=True)
        events, _ = simulator.run(args.n_cases, seed=args.seed)
        table = kpis(prepare_simulated(events, chosen_outcome_spec("bpic17")), real_counts, users, label)
        table["Wartezeit auf Mitarbeiter je Fall (h, Engine)"] = simulator.waiting_hours / args.n_cases
    table.to_csv(args.out)
    print(table.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
