"""Isolated diagnosis of recorded D choices; never changes models or engine sources."""
import argparse
import copy
from datetime import datetime
import heapq
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths
add_paths()

import cloudpickle
import numpy as np
import pandas as pd

REFERENCE = ROOT / "prescriptive/policies/ev_policy_event_costs/results/psp_d_ue_200_faelle_500_ziehungen_seed0_20261114"
OUT = ROOT / "prescriptive/scripts/results_100k/d_sampling_diagnosis_20261115"
os.environ["PSP_CATALOG"] = str(REFERENCE / "catalog.yaml")
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "matplotlib"))
CASE, ACT, LC, TS = "case:concept:name", "concept:name", "lifecycle:transition", "time:timestamp"


def write(name, value):
    path = OUT / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temp.replace(path)


class SnapshotPickler(cloudpickle.CloudPickler):
    def persistent_id(self, obj):
        if obj is self.sim_model:
            return "shared_simulator_model"
        return None


class SnapshotUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        assert pid == "shared_simulator_model"
        return self.sim_model


def dump_snapshot(engine, record, path):
    policy = engine.policy
    try:
        engine.policy = None
        with path.open("wb") as stream:
            writer = SnapshotPickler(stream, protocol=pickle.HIGHEST_PROTOCOL)
            writer.sim_model = engine.model
            writer.dump({"engine": engine, "record": record, "python_random": random.getstate()})
    finally:
        engine.policy = policy


def load_snapshot(path, model):
    with path.open("rb") as stream:
        reader = SnapshotUnpickler(stream)
        reader.sim_model = model
        return reader.load()


class RecordedPolicy:
    def __init__(self, rows, start=0, targets=(), capture=False):
        self.rows, self.at, self.targets, self.capture = rows, start, set(targets), capture
        self.prefixes = []

    def choose(self, point, case, options, now, engine):
        row = self.rows[self.at]
        assert row["case"] == case.case_id, (self.at, row["case"], case.case_id)
        assert row["point"] == point.name
        assert abs(row["time"] - now) < 1e-5
        record = {"id": self.at, "case": case.case_id, "point": point.name, "options": list(options),
                  "chosen": row["choice"], "now": now, "events": copy.deepcopy(case.prefix),
                  "open_sim_sessions": dict(case.sessions), "busy_employees": len(engine.busy),
                  "waiting_tasks": len(engine.waiting)}
        if self.capture:
            self.prefixes.append(record)
            if self.at in self.targets:
                dump_snapshot(engine, record, OUT / f"state_{self.at:03d}.pkl")
                print("SNAPSHOT", self.at, case.case_id, point.name, flush=True)
        self.at += 1
        return row["choice"]


def setup():
    os.environ["PSP_CATALOG"] = str(REFERENCE / "catalog.yaml")
    from helpers.datasets import load_dataset
    from preprocessing import preprocess
    raw, _, _ = load_dataset("bpic17")
    return preprocess("bpic17", raw)


def replay():
    from simulation.dataset_simulator import Simulator
    from prescriptive.policies.ev_policy_event_costs.costs import EventCosts
    assert not (OUT / "replay_validation.json").exists()
    OUT.mkdir(parents=True, exist_ok=True)
    choices = pd.read_csv(REFERENCE / "online/psp_seed0/choices.csv")
    rng = np.random.default_rng(20261115)
    targets = []
    for _, group in choices.groupby("point", sort=True):
        unique = group.drop_duplicates("case")
        targets.extend(sorted(rng.choice(unique.index.to_numpy(), min(3, len(unique)), replace=False).tolist()))
    write("selected_decisions.json", sorted(targets))
    log = setup()
    sim = Simulator("bpic17", log)
    policy = RecordedPolicy(choices.to_dict("records"), targets=targets, capture=True)
    started = time.time()
    events, _ = sim.run(2200, policy=policy, window=(700, 900), seed=0)
    assert policy.at == len(choices)
    from preprocessing.steps import COMMON_STEPS
    for step in COMMON_STEPS:
        events = step(events)
    selected = events[CASE].str.rsplit("_", n=1).str[-1].astype(int).between(700, 899)
    actual = events.loc[selected].reset_index(drop=True)
    expected = pd.read_csv(REFERENCE / "online/psp_seed0/events_with_costs.csv")
    for col in (CASE, ACT, LC, "org:resource"):
        assert actual[col].astype(str).tolist() == expected[col].astype(str).tolist(), col
    assert np.allclose(pd.to_datetime(actual[TS], utc=True).dt.as_unit("ns").astype("int64"),
                       pd.to_datetime(expected[TS], utc=True).dt.as_unit("ns").astype("int64"), rtol=0, atol=1000)
    costs = EventCosts(REFERENCE / "event_costs.yaml")
    bill = costs.case_costs(actual).sort_index()
    old_bill = pd.read_csv(REFERENCE / "online/psp_seed0/case_costs.csv").set_index(CASE).sort_index()
    assert np.allclose(bill.to_numpy(), old_bill.to_numpy(), rtol=0, atol=1e-8)
    with (OUT / "prefixes.pkl").open("wb") as stream:
        pickle.dump(policy.prefixes, stream)
    actual.to_pickle(OUT / "replayed_events.pkl")
    write("replay_validation.json", {"status": "passed", "decisions": policy.at, "events": len(actual),
          "same_events_resources_timestamps_and_costs": True, "seconds": time.time()-started,
          "costs_per_case": bill.total_eur.mean(), "snapshots": targets})
    print("REPLAY PASSED", time.time()-started, flush=True)


def resume_forced(engine, record, option):
    """Continue _advance AFTER its model.step and bank-option draw already happened."""
    from simulation.case_engine_lifecycle_aware import EVENT, OPENING
    from next_event_prediction.data_prep import END, sample_seconds
    case, now = engine.cases[record["case"]], record["now"]
    token = engine.model.encoder.token_to_id[option if "|" in option else option + "|complete"]
    if token == END or case.n_events >= engine.model.max_length - 1:
        if token != END:
            engine.truncated_cases.add(case.case_id)
        engine._finish(case, now)
        return
    activity, lifecycle = engine._tokens[token]
    waiting = engine._is_work_item(activity) and lifecycle in OPENING
    performer = None if waiting else case.holder(activity)
    logits = engine.model.time_logits(case.model_state, np.array([token]),
                                      np.array([engine.model.encoder.resource_id(performer)]))
    time_bin = engine.model.draw(logits, case.rng, engine.min_probability)
    seconds = float(sample_seconds(time_bin, case.rng, engine.model.encoder.time_edges)[0])
    case.pending = (token, now + seconds)
    engine._schedule(now + seconds, EVENT, case.case_id)


def process_until_finished(engine, target):
    from simulation.case_engine_lifecycle_aware import ARRIVAL, EVENT, OBSERVE
    while engine.queue and target in engine.cases:
        at, _, kind, case_id = heapq.heappop(engine.queue)
        if kind == ARRIVAL:
            engine._arrive(case_id, at)
        elif kind == EVENT:
            engine._due(engine.cases[case_id], at)
        elif kind == OBSERVE:
            raise AssertionError("No observers in reference run")
        else:
            engine.retry_scheduled = False
            engine._assign(at)


def branches(repeats):
    from simulation.dataset_simulator import model_folder
    from next_event_prediction.event_time_model import EventTimeModel
    from prescriptive.policies.ev_policy_event_costs.costs import EventCosts
    os.environ["PSP_CATALOG"] = str(REFERENCE / "catalog.yaml")
    model = EventTimeModel.load(model_folder("bpic17"))
    costs = EventCosts(REFERENCE / "event_costs.yaml")
    choices = pd.read_csv(REFERENCE / "online/psp_seed0/choices.csv").to_dict("records")
    events = pd.read_pickle(OUT / "replayed_events.pkl")
    rows = []
    validations = []
    for path in sorted(OUT.glob("state_*.pkl")):
        snapshot = load_snapshot(path, model)
        record = snapshot["record"]
        engine = snapshot["engine"]
        engine.policy = RecordedPolicy(choices, start=record["id"]+1)
        random.setstate(snapshot["python_random"])
        resume_forced(engine, record, record["chosen"])
        process_until_finished(engine, record["case"])
        reproduced = [e for e in engine.events if e[CASE] == record["case"]]
        expected = events[events[CASE] == record["case"]]
        assert [e[ACT] for e in reproduced] == expected[ACT].tolist()
        assert [e[LC] for e in reproduced] == expected[LC].tolist()
        assert [e["org:resource"] for e in reproduced] == expected["org:resource"].tolist()
        assert np.allclose(pd.to_datetime([e[TS] for e in reproduced], utc=True).as_unit("ns").astype("int64"),
                           pd.to_datetime(expected[TS], utc=True).dt.as_unit("ns").astype("int64"), rtol=0, atol=1000)
        validations.append(record["id"])
        write("fork_validation.json", {"status": "in_progress", "exact_reproductions": validations})
        for repeat in range(repeats):
            for option in record["options"]:
                state = load_snapshot(path, model)
                engine = state["engine"]
                engine.policy = None  # one forced intervention, then bank continuation everywhere
                if repeat == 0:
                    random.setstate(state["python_random"])
                else:
                    random.seed(100000 + record["id"] * 100 + repeat)
                    for case_id, case in engine.cases.items():
                        number = int(case_id.rsplit("_", 1)[1])
                        case.rng = np.random.default_rng([20261115, number, repeat])
                resume_forced(engine, record, option)
                process_until_finished(engine, record["case"])
                trace = [e for e in engine.events if e[CASE] == record["case"]]
                observed = costs.state(record["events"])
                result = costs.state(trace).summary()
                sunk = observed.summary(record["now"])
                row = {"id": record["id"], "point": record["point"], "case": record["case"], "option": option,
                       "repeat": repeat, "first_gap_seconds": (pd.Timestamp(trace[len(record["events"])][TS]).timestamp()-record["now"])
                       if len(trace)>len(record["events"]) else 0,
                       "remaining_events": len(trace)-len(record["events"]),
                       "remaining_days": (pd.Timestamp(trace[-1][TS]).timestamp()-record["now"])/86400,
                       "capped": record["case"] in engine.truncated_cases}
                for key in ("base_eur", "lifecycle_eur", "time_eur", "active_hours", "total_eur",
                            "orphan_closures", "reopened_segments"):
                    row[key] = result[key] - sunk[key]
                rows.append(row)
                pd.DataFrame(rows).to_csv(OUT / "simulator_branches.csv", index=False)
                if repeat == 0:
                    pd.DataFrame(trace).to_csv(OUT / f"branch_{record['id']:03d}_{record['options'].index(option)}.csv", index=False)
            print("BRANCH", record["id"], repeat, flush=True)
    write("fork_validation.json", {"status": "passed", "exact_reproductions": validations,
          "repeats": repeats, "continuation": "one forced action then bank everywhere; future per-case RNG resampled for repeats > 0"})


def samples(n, repeats, all_prefixes=False):
    import torch
    from prescriptive.estimators.psp_model import PSPModel
    from prescriptive.policies.ev_policy_event_costs.costs import EventCosts
    from prescriptive.policies.ev_policy_event_costs.psp_policy import EventCostPSPModel
    torch.set_num_threads(1)
    model = PSPModel.load(ROOT / "prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_klassen_option_encoder_ue.pkl")
    model.set_threads(1)
    costs = EventCosts(REFERENCE / "event_costs.yaml")
    adapter = EventCostPSPModel(model, costs)
    prefixes = pickle.loads((OUT / "prefixes.pkl").read_bytes())
    targets = set(json.loads((OUT / "selected_decisions.json").read_text()))
    rows, details = [], []
    true_draw = model.option_suffixes
    drawn_buffer = {}
    def capture(*args, **kwargs):
        result = true_draw(*args, **kwargs)
        drawn_buffer["result"] = result
        return result
    model.option_suffixes = capture
    for record in prefixes:
        if not all_prefixes and record["id"] not in targets:
            continue
        point = model.points[record["point"]]
        observed = costs.state(record["events"])
        sunk = observed.summary(record["now"])
        for repeat in range(repeats):
            torch.manual_seed(200000 + record["id"] * 100 + repeat)
            scores = -adapter.outcome_samples(point, record["events"], record["options"], record["now"], n)
            drawn, elapsed = drawn_buffer["result"]
            firsts = [np.asarray([s[0][2] for s in suffixes]) for suffixes in drawn]
            shared_first = bool(np.array_equal(firsts[0], firsts[1]))
            paired_se = float((scores[0]-scores[1]).std(ddof=1)/np.sqrt(n))
            for option_index, (option, suffixes) in enumerate(zip(record["options"], drawn)):
                cap = 1 + max(model.encoder_decoder.window_size - model.encoder_decoder.min_suffix_size - len(record["events"]), 1)
                for k, suffix in enumerate(suffixes):
                    state = observed.copy()
                    at = max(pd.Timestamp(record["events"][0][TS]).timestamp()+elapsed, record["now"], state.last)
                    initial_orphans = state.orphan_closures
                    initial_reopened = state.reopened_segments
                    for token, _, gap in suffix:
                        activity, _, lifecycle = token.rpartition("|")
                        at += gap
                        state.push(activity, lifecycle, at)
                    result = state.summary(at)
                    cost = result["total_eur"] - sunk["total_eur"]
                    assert np.isclose(max(cost,0),scores[option_index,k])
                    row = {"id": record["id"], "point": record["point"], "case": record["case"],
                           "option": option, "repeat": repeat, "draw": k, "events": len(suffix),
                           "first_gap_seconds": suffix[0][2], "days": sum(s[2] for s in suffix)/86400,
                           "at_cap": len(suffix) >= cap,
                           "missing_business_end": not any(e[ACT] in ("A_Pending","A_Denied","A_Cancelled") for e in record["events"])
                              and not any(s[0].split("|")[0] in ("A_Pending","A_Denied","A_Cancelled") for s in suffix),
                           "new_orphans": state.orphan_closures-initial_orphans,
                           "new_reopens": state.reopened_segments-initial_reopened,
                           "open_at_end": len(state.opened)}
                    for key in ("base_eur", "lifecycle_eur", "time_eur", "active_hours", "total_eur"):
                        row[key] = result[key]-sunk[key]
                    details.append(row)
                rows.append({"id": record["id"], "point": record["point"], "case": record["case"], "option": option,
                             "repeat": repeat, "samples": n, "mean_cost": scores[option_index].mean(),
                             "median_cost": np.median(scores[option_index]), "sd_cost": scores[option_index].std(ddof=1),
                             "paired_se": paired_se, "same_first_gaps_for_both_options": shared_first,
                             "chosen": option == record["chosen"], "prefix_events": len(record["events"])})
                if repeat == 0:
                    with (OUT / f"suffixes_{record['id']:03d}_{option_index}.pkl").open("wb") as stream:
                        pickle.dump(suffixes, stream)
            pd.DataFrame(rows).to_csv(OUT / "psp_estimates.csv", index=False)
            pd.DataFrame(details).to_csv(OUT / "psp_draws.csv", index=False)
            print("SAMPLES", record["id"], repeat, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("replay", "branches", "samples"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()
    if args.phase == "replay":
        replay()
    elif args.phase == "branches":
        branches(args.repeats)
    else:
        samples(args.samples, args.repeats)
