"""Audit work sessions in the saved 200-case bank and D simulation logs."""
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))
from prescriptive.policies.ev_policy_event_costs.costs import EventCosts

REF = ROOT / "prescriptive/policies/ev_policy_event_costs/results/psp_d_ue_200_faelle_500_ziehungen_seed0_20261114"
BASE = OUT.parent / "d_sampling_diagnosis_20261115"
CASE, ACT, LC, TS = "case:concept:name", "concept:name", "lifecycle:transition", "time:timestamp"


def main():
    costs = EventCosts(REF/"event_costs.yaml")
    summaries, intervals = [], []
    for policy in ("bank", "psp"):
        frame = pd.read_csv(REF/f"online/{policy}_seed0/events_with_costs.csv")
        frame[TS] = pd.to_datetime(frame[TS], utc=True)
        frame = frame.sort_values([CASE, TS], kind="stable")
        bills = []
        missing, open_at_end, reopens = 0, 0, 0
        for case_id, events in frame.groupby(CASE, sort=True):
            records = events.to_dict("records")
            state = costs.state(records)
            bill = state.summary()
            bills.append(bill)
            open_at_end += bool(state.opened)
            reopens += bill["reopened_segments"]
            missing += not events[ACT].isin(["A_Pending", "A_Denied", "A_Cancelled"]).any()
            assert np.isclose(bill["total_eur"], float(events.case_event_cost_eur.iloc[0])), case_id
            opened = {}
            for event in records:
                activity, lifecycle = event[ACT], event[LC]
                at = event[TS].timestamp()
                kind = None
                if lifecycle in ("start", "resume"):
                    if activity in opened:
                        began, kind = opened[activity], "reopen"
                    opened[activity] = at
                elif lifecycle in ("suspend", "complete", "ate_abort", "withdraw") and activity in opened:
                    began, kind = opened.pop(activity), "close"
                if kind:
                    hours = (at-began)/3600
                    intervals.append({"policy": policy, "case": case_id, "activity": activity, "kind": kind,
                        "hours": hours, "time_cost_eur": hours*costs.tariff(activity)["eur_per_work_hour"]})
            end = records[-1][TS].timestamp()
            for activity, began in opened.items():
                hours = (end-began)/3600
                intervals.append({"policy": policy, "case": case_id, "activity": activity, "kind": "unclosed",
                    "hours": hours, "time_cost_eur": hours*costs.tariff(activity)["eur_per_work_hour"]})
        work = pd.DataFrame(intervals)
        work = work.loc[work.policy.eq(policy)]
        bill_frame = pd.DataFrame(bills)
        summaries.append({"policy": policy, "cases": len(bills), "events": len(frame),
            "mean_case_cost_eur": float(bill_frame.total_eur.mean()),
            "mean_active_hours": float(bill_frame.active_hours.mean()),
            "missing_business_end_cases": int(missing), "open_session_end_cases": int(open_at_end),
            "reopened_segments": int(reopens), "work_intervals": len(work),
            "intervals_over24h": int(work.hours.gt(24).sum()),
            "max_work_interval_hours": float(work.hours.max()),
            "all_saved_costs_reproduced": True})
    context = []
    for record in pickle.loads((BASE/"prefixes.pkl").read_bytes()):
        for option in record["options"]:
            activity, _, lifecycle = option.rpartition("|")
            if lifecycle in ("start", "resume") and activity in record["open_sim_sessions"]:
                context.append({"id": record["id"], "case": record["case"], "point": record["point"],
                    "option": option, "chosen": record["chosen"] == option})
    pd.DataFrame(intervals).to_csv(OUT/"simulator_work_intervals.csv", index=False)
    pd.DataFrame(context).to_csv(OUT/"already_open_options.csv", index=False)
    result = {"logs": summaries, "historical_decisions": len(pickle.loads((BASE/"prefixes.pkl").read_bytes())),
        "already_open_options_offered": len(context), "already_open_options_chosen": sum(r["chosen"] for r in context)}
    (OUT/"simulator_log_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
