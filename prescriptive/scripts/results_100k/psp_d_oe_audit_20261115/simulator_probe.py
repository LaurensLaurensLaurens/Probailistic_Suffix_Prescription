"""Reproduce both continuations of the recorded open-session decision, ID 102."""
import json
import os
from pathlib import Path
import random
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MPLCONFIGDIR"] = str(OUT / "matplotlib")
sys.path.insert(0, str(ROOT))
from prescriptive.scripts import diagnose_d_sampling_20261115 as old
import numpy as np
import pandas as pd
from next_event_prediction.event_time_model import EventTimeModel
from simulation.dataset_simulator import model_folder
from prescriptive.policies.ev_policy_event_costs.costs import EventCosts


def main():
    started = time.time()
    model = EventTimeModel.load(model_folder("bpic17"))
    costs = EventCosts(old.REFERENCE/"event_costs.yaml")
    snapshot_path = old.OUT/"state_102.pkl"
    saved = old.load_snapshot(snapshot_path, model)
    record = saved["record"]
    assert record["case"] == "Application_750"
    expected = pd.read_csv(old.OUT/"case750_branches.csv")
    results = []
    for option in record["options"]:
        saved = old.load_snapshot(snapshot_path, model)
        engine = saved["engine"]
        engine.policy = None
        random.setstate(saved["python_random"])
        case = engine.cases[record["case"]]
        original_sessions = dict(case.sessions)
        old.resume_forced(engine, record, option)
        scheduled_gap = case.pending[1]-record["now"]
        old.process_until_finished(engine, record["case"])
        trace = [event for event in engine.events if event[old.CASE] == record["case"]]
        sunk = costs.state(record["events"]).total(record["now"])
        remaining = costs.state(trace).total()-sunk
        historic = expected.loc[expected.option.eq(option)].iloc[0]
        # This is the original RNG branch; other stored repeats, if present, do not enter this check.
        for column in ("total_eur", "remaining_cost", "remaining_cost_eur"):
            if column in historic:
                assert np.isclose(remaining, float(historic[column]), atol=1e-6), (remaining, historic)
                break
        next_event = trace[len(record["events"])]
        row = {"option": option, "open_sessions_before": original_sessions,
            "scheduled_first_gap_hours": scheduled_gap/3600,
            "observed_first_gap_hours": (pd.Timestamp(next_event[old.TS]).timestamp()-record["now"])/3600,
            "remaining_cost_eur": remaining,
            "remaining_days": (pd.Timestamp(trace[-1][old.TS]).timestamp()-record["now"])/86400,
            "events": len(trace), "capped": record["case"] in engine.truncated_cases}
        results.append(row)
        pd.DataFrame(trace).to_csv(OUT/f"case750_reproduced_{len(results)-1}.csv", index=False)
        print(json.dumps(row), flush=True)
    (OUT/"simulator_probe.json").write_text(json.dumps({"status": "passed", "seconds": time.time()-started,
        "method": "existing exact engine snapshot; one forced option then bank continuation with original RNG",
        "results": results}, indent=2, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    main()
