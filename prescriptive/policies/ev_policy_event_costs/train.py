"""Train the standard EV S/X learners and causal forest on event costs.

From the project root:
    process-simulation-engine/.venv/bin/python -m prescriptive.policies.ev_policy_event_costs.train --tag real_alle
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .costs import CASE, DEFAULT_CONFIG, EventCosts

PACKAGE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="bpic17")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--learner", nargs="+", choices=("s", "x", "cf"), default=["s", "x", "cf"])
    parser.add_argument("--cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not args.tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.tag):
        parser.error("--tag must contain only letters, digits, underscores and hyphens")
    if args.cases < 0 or len(set(args.learner)) != len(args.learner):
        parser.error("--cases must be nonnegative and learners must not be repeated")
    out = (args.out or PACKAGE / "models" / args.tag).resolve()
    costs = EventCosts(args.cost_config.resolve())
    if args.catalog:
        os.environ["PSP_CATALOG"] = str(args.catalog.resolve())
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    import interventions
    from helpers.datasets import load_dataset
    from helpers.training import sample_cases, test_cases
    from preprocessing import preprocess
    from .cate_model import EventCostCATEModel

    out.mkdir(parents=True, exist_ok=False)
    (out / "event_costs.yaml").write_text(yaml.safe_dump(costs.to_dict(), sort_keys=False, allow_unicode=True))
    manifest = {"status": "training", "started_at": datetime.now(timezone.utc).isoformat(),
                "objective": "negative_event_costs", "dataset": args.dataset, "seed": args.seed,
                "requested_cases": args.cases, "learners": args.learner, "cost_fingerprint": costs.fingerprint,
                "catalog": str(interventions.CANDIDATES_FILE)}
    status = out / "training.json"
    def save():
        status.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    save()
    try:
        (out / "catalog.yaml").write_bytes(Path(interventions.CANDIDATES_FILE).read_bytes())
        raw, _, _ = load_dataset(args.dataset)
        log = preprocess(args.dataset, raw)
        del raw
        held_out = test_cases(log)
        held = log[CASE].astype(str).isin(held_out)
        train, test = log[~held], log[held]
        if args.cases:
            train = sample_cases(train, args.cases, args.seed)
        points = interventions.choice_points(args.dataset)
        first = EventCostCATEModel(args.dataset, points, seed=args.seed, learner=args.learner[0], cost_config=costs)
        samples, testing = first.samples(train), first.samples(test)
        manifest.update(train_cases=int(train[CASE].nunique()), test_cases=len(held_out),
                        points=[point.name for point in points], completed_learners=[])
        rows = []
        for learner in args.learner:
            print(f"Training {learner}: negative event costs in EUR", flush=True)
            model = EventCostCATEModel(args.dataset, points, seed=args.seed, learner=learner, cost_config=costs).fit(train, samples)
            model.test_cases = held_out
            model.trained_on = {"echter Log": args.dataset, "outcome": model.objective,
                                "cases": int(train[CASE].nunique()), "cost_fingerprint": costs.fingerprint}
            model.save(str(out / f"cate_model_{learner}_event_costs.pkl"))
            for point in points:
                if not model.knows(point) or not testing.get(point.name):
                    continue
                features = [row[1] for row in testing[point.name]]
                effects = model.effects(point, features)
                for k in model.models[point.name]:
                    effect = effects[:, k - 1]
                    rows.append({"learner": learner, "point": point.name, "action": point.options[k],
                                 "reference": point.options[0], "test_decisions": len(effect),
                                 "mean_saving_eur": float(effect.mean()), "std_saving_eur": float(effect.std()),
                                 "share_positive": float(np.mean(effect > 0))})
            manifest["completed_learners"].append(learner)
            manifest["current_learner"] = learner
            save()
        pd.DataFrame(rows).to_csv(out / "effects_test_cases.csv", index=False)
        manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        save()
    print(f"Models, tariff snapshot and diagnostics: {out}", flush=True)


if __name__ == "__main__":
    main()
