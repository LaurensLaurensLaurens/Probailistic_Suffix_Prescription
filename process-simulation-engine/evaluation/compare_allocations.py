"""
Compares the allocation policies of the case engine on the same cases: waiting for an employee by
reason, throughput time, handovers and how evenly the work is spread.

Usage (from process-simulation-engine/):
    python -m evaluation.compare_allocations --cases 2000
    python -m evaluation.compare_allocations --cases 1000 --allocations random pmsp
"""
import argparse
import contextlib
import io
import time

import numpy as np
import pandas as pd

from next_event_prediction.train import prepared_log
from resources.case_allocation import ALLOCATIONS
from simulation.case_engine_lifecycle_aware import build_engine

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-log", default="eventlog/eventlog.xes.gz")
    parser.add_argument("--cases", type=int, default=2000)
    parser.add_argument("--allocations", nargs="+", choices=ALLOCATIONS, default=list(ALLOCATIONS))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Der Log nach dem Preprocessing (prescriptive/helpers/preprocessing), wie für das Modell
    event_log = prepared_log("bpic17", args.event_log)
    rows = {}
    for name in args.allocations:
        with contextlib.redirect_stdout(io.StringIO()):
            engine = build_engine(event_log, args.cases, seed=args.seed, allocation=name)
        started = time.time()
        log = pd.DataFrame(engine.run(args.cases))
        seconds = time.time() - started
        cases = log.groupby("case:concept:name")
        throughput = (cases["time:timestamp"].max() - cases["time:timestamp"].min()).dt.total_seconds() / 3600
        handovers = (log["org:resource"] != cases["org:resource"].shift(1)).groupby(log["case:concept:name"]).sum() - 1
        workload = log[log["org:resource"] != "User_1"]["org:resource"].value_counts().sort_values().to_numpy()
        gini = (2 * np.arange(1, len(workload) + 1) - len(workload) - 1).dot(workload) / (len(workload) * workload.sum())
        rows[name] = {
            "Laufzeit (s)": seconds,
            **{f"Warten: {reason} (h/Fall)": total / 3600 / args.cases for reason, total in engine.waiting_seconds.items()},
            "Durchlaufzeit Mittel (h)": throughput.mean(),
            "Durchlaufzeit Median (h)": throughput.median(),
            "Übergaben pro Fall": handovers.mean(),
            "eingesetzte Mitarbeiter": len(workload),
            "Gini der Arbeitslast": gini,
            "bewilligt (%)": 100 * cases["concept:name"].apply(lambda activities: "A_Pending" in set(activities)).mean(),
        }
        print(f"{name}: {seconds:.0f} s", flush=True)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(2).to_string())
