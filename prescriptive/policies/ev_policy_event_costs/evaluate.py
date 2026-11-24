"""Independent online EV comparison using the shared event-cost function."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml

from .costs import ACTIVITY, CASE, DEFAULT_CONFIG, TIMESTAMP, EventCosts

PACKAGE = Path(__file__).resolve().parent
POLICIES = ("bank", "random", "psp", "s_learner", "x_learner", "causal_forest")
LEARNERS = {"s_learner": "s", "x_learner": "x", "causal_forest": "cf"}
_SHARED = {}


def _init_worker(dataset, log, intervention, config, policy_params, out):
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    from simulation.dataset_simulator import Simulator, is_trained, model_folder
    if not is_trained(model_folder(dataset)):
        raise FileNotFoundError("Train the simulator first; event-cost evaluation does not train it implicitly")
    _SHARED.update(simulator=Simulator(dataset, log, intervention), costs=EventCosts(config),
                   policy_params=policy_params, out=out)


def _policy(name, seed):
    if name == "bank":
        return None
    if name == "random":
        from prescriptive.policies.rct_policy import RCTPolicy
        return RCTPolicy([seed, 7])
    params = dict(_SHARED["policy_params"].get(name) or {})
    if "model_path" not in params:
        raise ValueError(f"{name}: model_path is required")
    if name == "psp":
        from .psp_policy import PSPEventCostPolicy
        return PSPEventCostPolicy(seed=seed, cost_config=_SHARED["costs"], **params)
    from .cate_policy import CATEEventCostPolicy
    policy = CATEEventCostPolicy(seed=seed, cost_config=_SHARED["costs"], **params)
    if policy.model.learner != LEARNERS[name]:
        raise ValueError(f"{name} requires a {LEARNERS[name]} learner")
    return policy


def _run(job):
    from prescriptive.evaluation.online_evaluation import TAIL, TARGET_EVENTS, _point_rows
    from preprocessing.steps import COMMON_STEPS
    name, seed, n_cases, window, points = job
    simulator, costs = _SHARED["simulator"], _SHARED["costs"]
    started = time.time()
    policy = _policy(name, seed)
    events, choices = simulator.run(n_cases, policy=policy, points=points, window=window, seed=seed)
    for step in COMMON_STEPS:
        events = step(events)
    numbers = events[CASE].astype(str).str.rsplit("_", n=1).str[1].astype(int)
    events = events[(numbers >= window[0]) & (numbers < window[1])]
    if events.empty:
        raise ValueError("No cases in the scoring window")
    bill = costs.case_costs(events)
    grouped = events.groupby(CASE)
    days = (grouped[TIMESTAMP].max() - grouped[TIMESTAMP].min()).dt.total_seconds() / 86400
    row = {"Policy": name, "seed": seed, "Fälle": len(bill), "Outcome je Fall": -bill["total_eur"].mean(),
           "Eventkosten je Fall (€)": bill["total_eur"].mean(), "Basiskosten je Fall (€)": bill["base_eur"].mean(),
           "Lifecycle-Kosten je Fall (€)": bill["lifecycle_eur"].mean(), "Zeitkosten je Fall (€)": bill["time_eur"].mean(),
           "Aktive Arbeitszeit je Fall (h)": bill["active_hours"].mean(), "Events je Fall": len(events) / len(bill),
           "Durchlaufzeit (Tage)": days.mean(), "Offene Abschnitte je Fall": bill["unclosed_segments"].mean(),
           "Enden ohne Beginn je Fall": bill["orphan_closures"].mean(),
           "Neubeginne ohne Ende je Fall": bill["reopened_segments"].mean(),
           "Events mit Standardtarif je Fall": bill["unconfigured_events"].mean(),
           "Wartezeit auf Mitarbeiter je Fall (h, alle Fälle)": simulator.waiting_hours / n_cases,
           "Erzwungene Zuteilungen je Fall (alle Fälle)": simulator.forced_assignments / n_cases,
           "Laufzeit (s)": time.time() - started}
    p90, p95 = np.quantile(days, [0.9, 0.95])
    row.update({TAIL[0]: p90, TAIL[1]: p95, TAIL[2]: days[days >= p90].mean(), "_days": days.to_numpy()})
    for target in TARGET_EVENTS.get(simulator.dataset, ()):
        row[f"Anteil {target}"] = float(events.loc[events[ACTIVITY] == target, CASE].nunique() / len(bill))
    row.update(_point_rows(simulator, events, choices, points))
    if _SHARED["out"] is not None:
        folder = Path(_SHARED["out"]) / f"{name}_seed{seed}"
        folder.mkdir(exist_ok=False)
        bill.to_csv(folder / "case_costs.csv")
        choices.to_csv(folder / "choices.csv", index=False)
        costs.label_log(events).to_csv(folder / "events_with_costs.csv", index=False)
    print(f"seed {seed}, {name}: {row['Eventkosten je Fall (€)']:.2f} EUR/case", flush=True)
    return row


def evaluate_event_costs(simulator, policies=("bank", "psp"), policy_params=None, cost_config=None,
                         n_cases=2200, warmup=700, cooldown=1300, seeds=1, seed=0, workers=1, points=None, out=None):
    """Same simulator, choice points and paired EV comparison; only the outcome changes."""
    from prescriptive.evaluation.online_evaluation import compare
    if set(policies) - set(POLICIES) or len(set(policies)) != len(policies):
        raise ValueError(f"Use distinct policies from {POLICIES}")
    if min(seeds, workers, n_cases) < 1 or min(warmup, cooldown) < 0 or warmup + cooldown >= n_cases:
        raise ValueError("Invalid run count or scoring window")
    policies = ["bank"] + [name for name in policies if name != "bank"]
    config = EventCosts(cost_config).to_dict()
    shared = (simulator.dataset, simulator.log, simulator.intervention["key"], config, policy_params or {}, out)
    jobs = [(name, seed + repeat, n_cases, (warmup, n_cases - cooldown), points)
            for repeat in range(seeds) for name in policies]
    if min(workers, len(jobs)) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker, initargs=shared) as pool:
            rows = list(pool.map(_run, jobs))
    else:
        _init_worker(*shared)
        rows = [_run(job) for job in jobs]
    return compare(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="bpic17")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=["bank", "psp"])
    parser.add_argument("--psp-model", type=Path)
    parser.add_argument("--cate-model", action="append", default=[], metavar="POLICY=PATH")
    parser.add_argument("--cost-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--scored-cases", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=700)
    parser.add_argument("--cooldown", type=int, default=1300)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.samples, args.scored_cases, args.seeds, args.workers) < 1 or min(args.warmup, args.cooldown) < 0:
        parser.error("Counts must be positive and warmup/cooldown nonnegative")
    out = args.out.resolve()
    costs = EventCosts(args.cost_config.resolve())
    if args.catalog:
        os.environ["PSP_CATALOG"] = str(args.catalog.resolve())
    params = {}
    if args.psp_model:
        params["psp"] = {"model_path": str(args.psp_model.resolve()), "samples": args.samples}
    for item in args.cate_model:
        name, separator, path = item.partition("=")
        if not separator or name not in LEARNERS or name in params:
            parser.error("--cate-model expects a unique s_learner, x_learner or causal_forest followed by =PATH")
        params[name] = {"model_path": str(Path(path).resolve())}
    for name in args.policies:
        if name not in ("bank", "random") and name not in params:
            parser.error(f"Model missing for {name}")
        if name in params and not Path(params[name]["model_path"]).is_file():
            parser.error(f"Model not found: {params[name]['model_path']}")
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    import interventions
    from helpers.datasets import load_dataset
    from preprocessing import preprocess
    from simulation.dataset_simulator import Simulator, is_trained, model_folder
    if not is_trained(model_folder(args.dataset)):
        parser.error("The simulator must already be trained")
    out.mkdir(parents=True, exist_ok=False)
    (out / "event_costs.yaml").write_text(yaml.safe_dump(costs.to_dict(), sort_keys=False, allow_unicode=True))
    (out / "catalog.yaml").write_bytes(Path(interventions.CANDIDATES_FILE).read_bytes())
    manifest = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                "objective": "negative_event_costs", "cost_fingerprint": costs.fingerprint,
                "dataset": args.dataset, "policies": args.policies, "policy_params": params,
                "samples": args.samples, "scored_cases": args.scored_cases, "warmup": args.warmup,
                "cooldown": args.cooldown, "seeds": args.seeds, "seed": args.seed}
    try:
        (out / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
        raw, _, _ = load_dataset(args.dataset)
        log = preprocess(args.dataset, raw)
        del raw
        simulator = Simulator(args.dataset, log)
        manifest["points"] = [point.name for point in simulator.points]
        table, runs = evaluate_event_costs(
            simulator, args.policies, params, costs, n_cases=args.warmup + args.scored_cases + args.cooldown,
            warmup=args.warmup, cooldown=args.cooldown, seeds=args.seeds, seed=args.seed, workers=args.workers, out=str(out))
        table.to_csv(out / "comparison.csv")
        runs.to_csv(out / "runs.csv", index=False)
        manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
        print(table[["Eventkosten je Fall (€)", "Uplift ggü. Bank (%)", "Durchlaufzeit (Tage)"]].to_string())
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        (out / "run.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
