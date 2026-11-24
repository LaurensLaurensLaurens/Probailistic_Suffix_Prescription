"""Read existing online runs; compare fixed real-log prefixes and repeat PSP decisions.

The probe uses held-out real cases, not recovered prefixes of the running online
seeds. It never changes a model, policy, catalog, or an existing result directory.
Run from process-simulation-engine with its .venv Python.
"""
import argparse
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings("ignore", category=FutureWarning)
P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P))
sys.path.insert(0, str(P / "scripts"))

import numpy as np
import pandas as pd

from helpers.datasets import ACTIVITY, CASE, LIFECYCLE, TIMESTAMP, add_paths
from psp_diagnose import ORDER

ROOT = P / "scripts/results_100k"
TAG = "real_alle_s60_token_oe"
MODEL = P / f"models/psp_model_durchlaufzeit_{TAG}.pkl"
NOISE = ROOT / f"psp_rauschen_voll_{TAG}_{TAG}"
ENDS = {"A_Pending", "A_Denied", "A_Cancelled"}
DAY = 86400
_WORKER = {}


def online_snapshot(out):
    text = (P / f"data/PSP_online300_{TAG}.log").read_text()
    token = {int(s): -float(v) for s, v in re.findall(
        r"^seed (\d+), psp: Outcome je Fall (-?[\d.]+)", text, re.M)}
    paths = {
        "c100": ROOT / "neue_punkte_online_stufe2c_suffixe100/prescription/PSP/bpic17_online_laeufe.csv",
        "cf": ROOT / "neue_punkte_online_cate/prescription/Causal_Forest/bpic17_online_laeufe.csv",
        "regel": ROOT / "neue_punkte_online_cate/prescription/Regel/bpic17_online_laeufe.csv",
    }
    table = pd.DataFrame({"seed": sorted(token)})
    table["token_oe300_days_rounded"] = table.seed.map(token)
    for label, path in paths.items():
        data = pd.read_csv(path)
        policy = {"c100": "psp", "cf": "causal_forest", "regel": "regel"}[label]
        chosen = data.loc[data.Policy == policy].set_index("seed")
        table[f"{label}_days"] = table.seed.map(-chosen["Outcome je Fall"])
        if label == "c100":
            bank = data.loc[data.Policy == "bank"].set_index("seed")
            table["bank_days"] = table.seed.map(-bank["Outcome je Fall"])
        if label == "cf":
            keys = [c for c in chosen if c.startswith("Anteil ") and ": " in c]
            chosen.loc[table.seed, keys].to_csv(out / "cf_online_choice_rates_matched_seeds.csv")
    table["token_days_above_cf"] = table.token_oe300_days_rounded - table.cf_days
    table.to_csv(out / "online_matched_seeds.csv", index=False)
    print("ONLINE_SNAPSHOT", table.to_string(index=False), flush=True)


def features(model, events, now):
    from prescriptive.estimators.xgb_model import Prefix, _seconds
    prefix = Prefix(model._attributes(events[0]), model.event_numeric)
    for event, seconds in zip(events, _seconds([e[TIMESTAMP] for e in events])):
        prefix.add(f"{event[ACTIVITY]}|{event.get(LIFECYCLE, 'complete')}", seconds,
                   {name: event.get(name) for name in model.event_numeric})
    return prefix.row(now)


def init_worker(settings):
    add_paths()
    import torch
    from prescriptive.estimators.psp_model import PSPModel
    torch.set_num_threads(1)
    model = PSPModel.load(str(MODEL))
    model.set_threads(1)
    _WORKER.update(settings, model=model)


def probe_point(job):
    import torch
    name, decisions, ordinal = job
    model = _WORKER["model"]
    rows = []
    point = model.points[name]
    options = list(ORDER[name])
    allowed, ends = _WORKER["allowed"], _WORKER["ends"]
    started = time.time()
    for index, (case_id, events, now, cf_diff, old_diff) in enumerate(decisions):
        cap = 1 + max(model.encoder_decoder.window_size - model.encoder_decoder.min_suffix_size - len(events), 1)
        last_prefix = f"{events[-1][ACTIVITY]}|{events[-1][LIFECYCLE]}"
        for repeat in range(_WORKER["repeats"]):
            seed = 20261113 + ordinal * 100000 + index * 100 + repeat
            torch.manual_seed(seed)
            drawn, elapsed = model.option_suffixes(point, events, options, now, _WORKER["samples"])
            times = np.array([[-model.outcome(elapsed, s, "negative_cycle_days", gaps=True) - elapsed / DAY
                               for s in suffixes] for suffixes in drawn])
            diff = times[1] - times[0]
            row = {"Wahlpunkt": name, "Fall": case_id, "repeat": repeat, "rng_seed": seed,
                   "draws_per_option": _WORKER["samples"], "cf_diff_days": cf_diff,
                   "old_psp100_diff_days": old_diff, "psp300_diff_days": diff.mean(),
                   "paired_se_days": diff.std(ddof=1) / np.sqrt(len(diff)),
                   "agreement_cf": bool((diff.mean() > 0) == (cf_diff > 0))}
            for arm, suffixes in zip(("action", "reference"), drawn):
                incomplete, bad_path, impossible_early_end, short = [], [], [], []
                gaps, lengths = [], []
                for suffix in suffixes:
                    tokens = [s[0] for s in suffix]
                    incomplete.append(not any(t.split("|")[0] in ENDS for t in tokens))
                    sequence = [last_prefix, *tokens]
                    bad_path.append(any(pair not in allowed for pair in zip(sequence[:-1], sequence[1:])))
                    early = len(suffix) < cap
                    short.append(early)
                    impossible_early_end.append(early and bool(tokens) and tokens[-1] not in ends)
                    gaps.append(suffix[0][2] / DAY if suffix else 0)
                    lengths.append(len(suffix))
                row.update({f"{arm}_without_terminal_share": np.mean(incomplete),
                            f"{arm}_unseen_transition_share": np.mean(bad_path),
                            f"{arm}_impossible_early_end_share": np.mean(impossible_early_end),
                            f"{arm}_early_end_share": np.mean(short),
                            f"{arm}_first_gap_days": np.mean(gaps),
                            f"{arm}_suffix_length": np.mean(lengths)})
            row["first_gap_advantage_days"] = row["reference_first_gap_days"] - row["action_first_gap_days"]
            rows.append(row)
        if (index + 1) % 3 == 0:
            print(f"PROBE {name}: {index + 1}/{len(decisions)} prefixes, {time.time() - started:.0f}s", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(_WORKER["out"]) / f"probe300_{name}.csv", index=False)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefixes-per-point", type=int, default=12)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.prefixes_per_point < 2 or args.prefixes_per_point % 2:
        parser.error("--prefixes-per-point must be positive and even")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    add_paths()
    from helpers.datasets import load_dataset
    from preprocessing import preprocess
    from prescriptive.estimators.psp_model import PSPModel
    from prescriptive.estimators.cate_model import CATEModel
    online_snapshot(out)
    raw, _, _ = load_dataset("bpic17")
    log = preprocess("bpic17", raw).sort_values([CASE, TIMESTAMP], kind="stable")
    del raw
    model = PSPModel.load(str(MODEL))
    model.set_threads(1)
    test_cases = set(map(str, model.test_cases))
    held = log[CASE].astype(str).isin(test_cases)
    train = log.loc[~held, [CASE, ACTIVITY, LIFECYCLE]]
    tokens = (train[ACTIVITY].astype(str) + "|" + train[LIFECYCLE].astype(str)).to_numpy()
    cases = train[CASE].to_numpy()
    same = cases[:-1] == cases[1:]
    allowed = set(zip(tokens[:-1][same], tokens[1:][same]))
    ends = set(tokens[np.r_[~same, True]])
    test = log.loc[held].copy()
    del train, tokens, cases, log, held, same
    noise = pd.read_csv(NOISE / "entscheidungen.csv")
    assert not noise.duplicated(["Wahlpunkt", "Fall"]).any()
    wanted = set(zip(noise.Wahlpunkt, noise.Fall))
    contexts = {}
    for name, samples in model.samples(test).items():
        for case_id, (part, cut, now), chosen in samples:
            key = (name, str(case_id))
            if key in wanted and key not in contexts:
                contexts[key] = (part.iloc[:cut].to_dict("records"), now)
    assert len(contexts) == len(noise), (len(contexts), len(noise))
    del test, samples, part, model
    gc.collect()
    for learner in ("cf", "x", "s"):
        estimator = CATEModel.load(str(P / f"models/cate_model_{learner}_durchlaufzeit_real_alle.pkl"))
        estimator.set_threads(1)
        assert set(map(str, estimator.test_cases)) == test_cases
        for name, subset in noise.groupby("Wahlpunkt", sort=False):
            rows = [features(estimator, *contexts[(name, str(case))]) for case in subset.Fall]
            values = estimator.effects(estimator.points[name], rows)[:, 0]
            # Convert model's option order to diagnostic ORDER (action minus reference).
            direction = 1 if estimator.options[name][1] == ORDER[name][0] else -1
            noise.loc[subset.index, f"{learner}_diff_days"] = direction * values
        del estimator
    noise.to_csv(out / "offline800_same_prefix_effects.csv", index=False)
    summary = []
    for name, t in noise.groupby("Wahlpunkt", sort=False):
        summary.append({"Wahlpunkt": name, "prefixes": len(t),
                        "psp100_mean_days": t.Diff.mean(), "cf_mean_days": t.cf_diff_days.mean(),
                        "psp100_cf_agreement": ((t.Diff > 0) == (t.cf_diff_days > 0)).mean(),
                        "cf_action_share": (t.cf_diff_days > 0).mean(),
                        "psp100_action_share": (t.Diff > 0).mean(),
                        "psp100_paired_se_median_days": t["SE gepaart"].median(),
                        "projected_psp300_se_median_days": t["SE gepaart"].median() / np.sqrt(3)})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "offline800_summary.csv", index=False)
    print("OFFLINE800", summary.to_string(index=False), flush=True)
    jobs = []
    rng = np.random.default_rng(20261113)
    for ordinal, (name, t) in enumerate(noise.groupby("Wahlpunkt", sort=False)):
        selected = []
        for action in (True, False):
            arm = t.loc[(t["gewählt"] == ORDER[name][0]) == action]
            selected.extend(rng.choice(arm.index, size=args.prefixes_per_point // 2, replace=False))
        decisions = [(str(noise.loc[i, "Fall"]), *contexts[(name, str(noise.loc[i, "Fall"]))],
                      noise.loc[i, "cf_diff_days"], noise.loc[i, "Diff"]) for i in selected]
        jobs.append((name, decisions, ordinal))
    settings = {"allowed": allowed, "ends": ends, "samples": args.samples,
                "repeats": args.repeats, "out": str(out)}
    (out / "settings.json").write_text(json.dumps({**vars(args), "allowed_pairs": len(allowed),
                                                   "allowed_end_tokens": len(ends),
                                                   "probe_source": "held-out real-log first decisions; not online prefixes"}, indent=2))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=(settings,)) as pool:
        frames = [future.result() for future in as_completed([pool.submit(probe_point, job) for job in jobs])]
    probe = pd.concat(frames, ignore_index=True)
    assert len(probe) == len(ORDER) * args.prefixes_per_point * args.repeats
    probe.to_csv(out / "probe300_all.csv", index=False)
    stats = []
    for name, t in probe.groupby("Wahlpunkt", sort=False):
        repeated = t.pivot(index="Fall", columns="repeat", values="psp300_diff_days")
        stats.append({"Wahlpunkt": name, "prefixes": len(repeated),
                      "replicate_calls": len(t), "mean_psp300_effect_days": t.psp300_diff_days.mean(),
                      "cf_mean_days": t.groupby("Fall").cf_diff_days.first().mean(),
                      "agreement_cf": t.agreement_cf.mean(),
                      "action_share": (t.psp300_diff_days > 0).mean(),
                      "decision_flip_rate_two_repeats": ((repeated[0] > 0) != (repeated[1] > 0)).mean(),
                      "paired_se_median_days": t.paired_se_days.median(),
                      "clear_abs_z_gt_2_share": (abs(t.psp300_diff_days / t.paired_se_days) > 2).mean(),
                      "without_terminal_share": t[["action_without_terminal_share", "reference_without_terminal_share"]].to_numpy().mean(),
                      "unseen_transition_share": t[["action_unseen_transition_share", "reference_unseen_transition_share"]].to_numpy().mean(),
                      "impossible_early_end_share": t[["action_impossible_early_end_share", "reference_impossible_early_end_share"]].to_numpy().mean(),
                      "first_gap_advantage_days": t.first_gap_advantage_days.mean()})
    stats = pd.DataFrame(stats)
    stats.to_csv(out / "probe300_summary.csv", index=False)
    online_snapshot(out)
    print("PROBE300", stats.to_string(index=False), flush=True)
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
