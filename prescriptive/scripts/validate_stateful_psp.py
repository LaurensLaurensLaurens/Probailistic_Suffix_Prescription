"""Matched held-out suffix check for the prepared stateful D and OE copies."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from prescriptive.helpers.datasets import add_paths
add_paths()
import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefixes", type=int, default=100)
    parser.add_argument("--draws", type=int, default=30)
    args = parser.parse_args()
    out = args.out.resolve()
    os.environ["PSP_CATALOG"] = str(out/"catalog.yaml")
    from prescriptive.estimators.psp_model import PSPModel
    from prescriptive.estimators.conditional_gaps import SuffixSamplingError
    from prescriptive.policies.ev_policy_event_costs.costs import EventCosts
    from prescriptive.scripts.psp_suffixguete import event_scores, time_scores
    from interventions.work_state import WorkState
    torch.set_num_threads(1)
    costs = EventCosts(out/"event_costs.yaml")
    picks = pd.read_csv(ROOT/"prescriptive/scripts/results_100k/suffixguete/prefixe.csv")
    picks = picks[picks.Modell.eq("(d) wie (c) mit Zeitklassen")].iloc[:args.prefixes]
    raw = pd.read_pickle(ROOT/"prescriptive/data/cache/bpic17_raw.pkl")
    wanted = set(picks.Fall.astype(str))
    raw = raw.loc[raw["case:concept:name"].astype(str).isin(wanted)].copy()
    raw["concept:name"] = raw["concept:name"].astype(str).str.strip()
    raw = raw.sort_values(["case:concept:name", "time:timestamp"], kind="stable")
    cases = {str(case): part.reset_index(drop=True) for case, part in raw.groupby("case:concept:name", sort=False)}
    rows, failures = [], []
    for name in ("D", "OE"):
        model = PSPModel.load(out/"models"/f"{name}_stateful.pkl")
        for index, pick in picks.iterrows():
            assert str(pick.Fall) in set(map(str, model.test_cases))
            part, k = cases[str(pick.Fall)], int(pick["Prefixlänge"])
            prefix = part.iloc[:k].to_dict("records")
            true_tokens = (part["concept:name"].astype(str)+"|"+part["lifecycle:transition"].astype(str)).tolist()[k:]
            truth_days = (part["time:timestamp"].iloc[-1]-part["time:timestamp"].iloc[k-1]).total_seconds()/86400
            true_first = (part["time:timestamp"].iloc[k]-part["time:timestamp"].iloc[k-1]).total_seconds()
            elapsed = model._encode(prefix)
            state = WorkState.from_events(prefix)
            torch.manual_seed(20261115+index)
            try:
                drawn = model.draw_cases([elapsed[:4]], args.draws, work_states=[state], previous_tokens=[state.previous])[0]
            except SuffixSamplingError as error:
                failures.append({"model": name, "case": str(pick.Fall), "error": str(error)})
                continue
            predicted_costs = []
            at = part["time:timestamp"].iloc[k-1].timestamp()
            sunk = costs.state(prefix).total(at)
            for suffix in drawn:
                check = state.copy()
                bill = costs.state(prefix)
                time = at
                for token, _, gap in suffix:
                    assert check.permits(token)
                    check.push(token)
                    a, _, lc = token.rpartition("|")
                    time += gap
                    bill.push(a, lc, time)
                assert check.complete and not bill.opened
                predicted_costs.append(bill.total(time)-sunk)
            row = {"model": name, "case": str(pick.Fall), "prefix_length": k,
                **event_scores(true_tokens, [[s[0] for s in suffix] for suffix in drawn]),
                **time_scores(true_first, truth_days*86400, [s[0][2] if s else 0. for s in drawn],
                    [sum(s[2] for s in suffix) for suffix in drawn]),
                "remaining_cost_mae": abs(np.mean(predicted_costs)-(costs.state(part.to_dict("records")).total()-sunk))}
            rows.append(row)
            if len(rows) % 10 == 0:
                print("HELDOUT", name, len(rows), flush=True)
        pd.DataFrame(rows).to_csv(out/"heldout_prefixes.csv", index=False)
    frame = pd.DataFrame(rows)
    summary = frame.drop(columns=["case", "prefix_length"]).groupby("model").mean(numeric_only=True)
    summary.to_csv(out/"heldout_summary.csv")
    (out/"heldout_validation.json").write_text(json.dumps({"status": "complete", "requested_prefixes_per_model": len(picks),
        "draws_per_prefix": args.draws, "successful_prefixes": frame.groupby("model").size().to_dict(),
        "sampling_failures": failures, "invalid_returned_suffixes": 0,
        "selection": "first requested rows of original 1000 held-out prefixes, same for both models; no validation or test case used for calibration"}, indent=2)+"\n")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
