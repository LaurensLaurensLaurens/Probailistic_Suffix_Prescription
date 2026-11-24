"""Generate 30,000 sequential-RCT cases and screen all 20 BPIC17 catalog points.

Run from the repository root with the simulator virtualenv. Results are atomic,
resumable checkpoints under screening/results/. No production catalog is changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "screening/results"
WORK = RESULTS / "vollstaendig_lauf"
RCT = RESULTS / "rct_30000"
REAL = RESULTS / "aipw_alle_real"
CATALOG = ROOT / "screening/configs/bpic17_alle.yaml"
SHARED = {}


def utc():
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def save_pickle(frame, path):
    temp = path.with_suffix(".tmp.pkl")
    frame.to_pickle(temp)
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure():
    import yaml
    if not CATALOG.exists():
        source = RESULTS / "provenienz/screening_alle/catalog_screening_alle.yaml"
        catalog = yaml.safe_load(source.read_text())
        entry = catalog["bpic17"]
        points = entry["interventionen"]
        entry["lauf"]["control_flow"] = [point["name"] for point in points if point["art"] == "control_flow" and point["name"] != entry["standard_intervention"]]
        entry["lauf"]["mitarbeiterwahl"] = [point["name"] for point in points if point["test"].get("kind") == "resource"]
        entry["lauf"]["zuteilung"] = "learned"
        entry["lauf"]["nur_mitarbeiter"] = False
        CATALOG.parent.mkdir(parents=True, exist_ok=True)
        CATALOG.write_text("# Complete 20-point screening/RCT study; production catalog is unchanged.\n" + yaml.safe_dump(catalog, allow_unicode=True, sort_keys=False))
    os.environ["PSP_CATALOG"] = str(CATALOG)
    os.environ["MPLCONFIGDIR"] = str(WORK / "matplotlib_cache")
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    import interventions
    tests = {point["name"]: point["test"] for point in interventions.candidates("bpic17")}
    if len(tests) != 20:
        raise ValueError(f"Expected all 20 catalog points, got {len(tests)}")
    return tests


def prepare(tests, args):
    import pandas as pd
    from helpers.datasets import load_dataset
    from preprocessing import preprocess
    from screening.rct_all_points import study_design
    WORK.mkdir(parents=True, exist_ok=True)
    (RCT / "daten").mkdir(parents=True, exist_ok=True)
    source_paths = [CATALOG, Path(__file__), ROOT / "screening/all_points.py", ROOT / "screening/rct_all_points.py",
                    ROOT / "screening/all_points_report.py", ROOT / "screening/candidates.py", ROOT / "screening/features.py",
                    ROOT / "screening/events.py", ROOT / "process-simulation-engine/simulation/case_engine_lifecycle_aware.py",
                    ROOT / "process-simulation-engine/models/allocation/learned_allocation_dienstplan.pkl",
                    ROOT / "prescriptive/data/cache/bpic17_raw.pkl"]
    source_paths += sorted((ROOT / "process-simulation-engine/case_attributes/BPIC2017").glob("attribute_*.pkl"))
    model = ROOT / "process-simulation-engine/models/next_event/bpic17"
    if model.exists():
        source_paths += sorted(p for p in model.iterdir() if p.is_file())
    spec = {"dataset": "bpic17", "n_cases": 30000, "runs": 10, "cases_per_run": 3000,
            "seed": args.seed, "folds": args.folds, "points": list(tests),
            "allocation_fallback": "learned", "delta": 0,
            "sources_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_paths}}
    manifest = WORK / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != spec:
        has_results = any((RCT / "daten/laeufe").glob("*/summary.json")) or any((REAL / "checkpoints").glob("point*.json"))
        if has_results:
            raise ValueError("Existing study has a different source/configuration manifest; preserve it before starting a new study")
    write_json(manifest, spec)
    real_path = WORK / "real.pkl"
    if not real_path.exists():
        raw, _, _ = load_dataset("bpic17")
        log = preprocess("bpic17", raw)
        save_pickle(log, real_path)
        del raw
    else:
        log = pd.read_pickle(real_path)
    design_path = RCT / "daten/design.json"
    if not design_path.exists():
        write_json(design_path, study_design(log, tests))
    print(f"{utc()} prepared {log['case:concept:name'].nunique()} real cases; {len(tests)} points", flush=True)
    return spec


def worker_init():
    configure()
    import pandas as pd
    SHARED["real"] = pd.read_pickle(WORK / "real.pkl")
    SHARED["design"] = json.loads((RCT / "daten/design.json").read_text())


def simulate_job(run, seed, n_cases=3000, pilot=False):
    from screening.rct_all_points import simulate_run
    started = time.monotonic()
    events, records, summary = simulate_run(SHARED["real"], SHARED["design"], n_cases, seed, run)
    out = WORK / "pilot" if pilot else RCT / "daten/laeufe" / f"run{run:02d}"
    out.mkdir(parents=True, exist_ok=True)
    save_pickle(events, out / "log.pkl")
    save_pickle(records, out / "entscheidungen.pkl")
    summary["elapsed_seconds"] = time.monotonic() - started
    write_json(out / "summary.json", summary)
    print(f"{utc()} RCT run {run}: {summary}", flush=True)
    return "simulation", run


def analyse_job(source, index, name, test, seed, folds, threads):
    import pandas as pd
    from screening.all_points import evaluate_point
    if source == "rct" and "rct" not in SHARED:
        SHARED["rct"] = pd.read_pickle(RCT / "daten/log.pkl")
        SHARED["records"] = pd.read_pickle(RCT / "daten/entscheidungen.pkl")
    log = SHARED["rct"] if source == "rct" else SHARED["real"]
    records = SHARED["records"] if source == "rct" else None
    started = time.monotonic()
    print(f"{utc()} AIPW {source} {name}: start", flush=True)
    summary, details = evaluate_point(name, test, log, records, seed=seed, folds=folds, threads=threads)
    summary["Laufzeit (s)"] = time.monotonic() - started
    out = (RCT if source == "rct" else REAL) / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    details.to_csv(out / f"point{index:02d}_optionen.csv", index=False)
    write_json(out / f"point{index:02d}.json", summary)
    print(f"{utc()} AIPW {source} {name}: {summary['Status']}; {len(details)} contrasts; {summary['Laufzeit (s)']:.1f}s", flush=True)
    return source, index


def combine_runs(spec):
    import pandas as pd
    logs, choices, summaries = [], [], []
    for run in range(10):
        folder = RCT / "daten/laeufe" / f"run{run:02d}"
        logs.append(pd.read_pickle(folder / "log.pkl"))
        choices.append(pd.read_pickle(folder / "entscheidungen.pkl"))
        summaries.append(json.loads((folder / "summary.json").read_text()))
    log = pd.concat(logs, ignore_index=True)
    records = pd.concat(choices, ignore_index=True)
    n = int(log["case:concept:name"].nunique())
    if n != 30000:
        raise ValueError(f"Expected exactly 30000 unique RCT cases, got {n}")
    save_pickle(log, RCT / "daten/log.pkl")
    save_pickle(records, RCT / "daten/entscheidungen.pkl")
    metadata = {**spec, "completed": utc(), "actual_cases": n, "events": len(log), "assignments": len(records),
                "log_sha256": sha(RCT / "daten/log.pkl"), "assignments_sha256": sha(RCT / "daten/entscheidungen.pkl"),
                "runs_summary": summaries,
                "response_time": "assigned deadline ITT, not a contrast of realized post-treatment lag bins",
                "staff": "uniform over recorded available employees; deterministic pools retained, contrasts require >=2 available employees"}
    write_json(RCT / "daten/rct.json", metadata)
    print(f"{utc()} combined exactly {n} randomized cases, {len(log)} events, {len(records)} assignments", flush=True)


def render_completed(tests, source):
    import pandas as pd
    from screening.all_points_report import render, publish_index
    out = RCT if source == "rct" else REAL
    rows, details = [], []
    for index, name in enumerate(tests):
        path = out / "checkpoints" / f"point{index:02d}.json"
        if not path.exists():
            return False
        rows.append(json.loads(path.read_text()))
        detail_path = path.with_name(f"point{index:02d}_optionen.csv")
        try:
            part = pd.read_csv(detail_path)
        except pd.errors.EmptyDataError:
            continue
        details.append(part)
    summary = pd.DataFrame(rows)
    all_details = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    render(out, summary, all_details, randomized=source == "rct")
    publish_index(RESULTS)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--pilot", type=int, default=0)
    args = parser.parse_args()
    tests = configure()
    spec = prepare(tests, args)
    if args.prepare_only:
        return
    if args.pilot:
        worker_init()
        simulate_job(0, args.seed + 1000, args.pilot, True)
        return
    simulation_done = {run for run in range(10) if (RCT / "daten/laeufe" / f"run{run:02d}/summary.json").exists()}
    completed = {source: {index for index in range(20) if ((RCT if source == "rct" else REAL) / "checkpoints" / f"point{index:02d}.json").exists()}
                 for source in ("real", "rct")}
    def status(state="running", errors=None):
        write_json(WORK / "status.json", {"state": state, "updated": utc(), "completed_simulation_runs": sorted(simulation_done),
                   "rct_cases_generated": len(simulation_done) * 3000,
                   "completed_real_points": len(completed["real"]), "completed_rct_points": len(completed["rct"]),
                   "points_per_dataset": 20, "errors": errors or []})
    status()
    errors = []
    submitted_rct = False
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=worker_init) as pool:
        pending = {}
        for run in range(10):
            if run not in simulation_done:
                pending[pool.submit(simulate_job, run, args.seed + 1000 + run)] = ("simulation", run)
        for index, (name, test) in enumerate(tests.items()):
            if index not in completed["real"]:
                pending[pool.submit(analyse_job, "real", index, name, test, args.seed, args.folds, args.threads)] = ("real", index)
        while pending or (not submitted_rct and len(simulation_done) == 10):
            if not submitted_rct and len(simulation_done) == 10:
                if not (RCT / "daten/rct.json").exists():
                    combine_runs(spec)
                for index, (name, test) in enumerate(tests.items()):
                    if index not in completed["rct"]:
                        pending[pool.submit(analyse_job, "rct", index, name, test, args.seed, args.folds, args.threads)] = ("rct", index)
                submitted_rct = True
            if not pending:
                break
            done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                try:
                    source, index = future.result()
                except Exception as error:
                    import traceback
                    traceback.print_exc()
                    errors.append({"job": job, "error": repr(error)})
                    continue
                if source == "simulation":
                    simulation_done.add(index)
                else:
                    completed[source].add(index)
                    if len(completed[source]) == 20:
                        render_completed(tests, source)
            status(errors=errors)
    if errors or len(simulation_done) != 10 or any(len(done) != 20 for done in completed.values()):
        status("failed", errors)
        raise RuntimeError(f"Study incomplete; {len(errors)} errors; see {WORK / 'status.json'}")
    for source in ("real", "rct"):
        render_completed(tests, source)
    status("complete")
    print(f"{utc()} COMPLETE: 30,000 RCT cases; all 20 real and all 20 RCT points screened", flush=True)


if __name__ == "__main__":
    main()
