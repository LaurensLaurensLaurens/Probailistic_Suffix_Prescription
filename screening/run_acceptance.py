"""AIPW for offer acceptance at all 20 existing BPIC17 intervention points.

Reuses the real log and exactly the same saved 30,000 sequential-RCT cases.
Primary target O_Accepted; supplementary target O_Returned. No simulation needed.
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

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "screening/results_accepted"
DEFAULT_SOURCE = ROOT / "screening/results_cycle_time"
CATALOG = ROOT / "screening/configs/bpic17_alle.yaml"
SHARED = {}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    import numpy as np
    def safe(v):
        if isinstance(v, dict):
            return {str(k): safe(x) for k, x in v.items()}
        if isinstance(v, (tuple, list)):
            return [safe(x) for x in v]
        if isinstance(v, np.generic):
            return safe(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def configure():
    os.environ["PSP_CATALOG"] = str(CATALOG)
    os.environ["MPLCONFIGDIR"] = str(OUT / ".matplotlib_cache")
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    import interventions
    tests = {point["name"]: point["test"] for point in interventions.candidates("bpic17")}
    if len(tests) != 20:
        raise ValueError(f"Expected the same 20 points, got {len(tests)}")
    return tests


def folder(endpoint, source):
    base = OUT if endpoint == "accepted" else OUT / endpoint
    return base / ("rct_30000" if source == "rct" else "aipw_alle_real")


def init_worker(source_root):
    configure()
    import pandas as pd
    source_root = Path(source_root)
    SHARED["source_root"] = source_root
    SHARED["real"] = pd.read_pickle(source_root / "vollstaendig_lauf/real.pkl")


def analyse_job(source, endpoint, index, name, test, seed, folds, threads):
    import pandas as pd
    from screening.acceptance import ENDPOINTS, contrast_pvalues
    from screening.all_points import evaluate_point
    if source == "rct" and "rct" not in SHARED:
        base = SHARED["source_root"] / "rct_30000/daten"
        SHARED["rct"] = pd.read_pickle(base / "log.pkl")
        SHARED["records"] = pd.read_pickle(base / "entscheidungen.pkl")
    log = SHARED[source]
    records = SHARED.get("records") if source == "rct" else None
    started = time.monotonic()
    print(f"{utc()} {source} {endpoint} {name}: start", flush=True)
    summary, details = evaluate_point(name, test, log, records, seed=seed, folds=folds, threads=threads,
                                      outcome_event=ENDPOINTS[endpoint]["event"])
    details = contrast_pvalues(details, source == "rct")
    summary["Laufzeit (s)"] = time.monotonic() - started
    out = folder(endpoint, source) / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    temp = out / f"point{index:02d}_optionen.csv.tmp"
    details.to_csv(temp, index=False)
    temp.replace(out / f"point{index:02d}_optionen.csv")
    write_json(out / f"point{index:02d}.json", summary)
    print(f"{utc()} {source} {endpoint} {name}: {summary['Status']}; n={summary['Entscheidungsfälle']}; "
          f"excluded={summary['Ziel bereits erreicht: ausgeschlossen']}; {len(details)} contrasts; "
          f"{summary['Laufzeit (s)']:.1f}s", flush=True)
    return source, endpoint, index


def prepare(source_root, tests, args):
    import pandas as pd
    from screening.acceptance import ENDPOINTS, SEMANTICS_URL, ACTIVITY_URL
    real = source_root / "vollstaendig_lauf/real.pkl"
    rct = source_root / "rct_30000/daten/log.pkl"
    records = source_root / "rct_30000/daten/entscheidungen.pkl"
    metadata = source_root / "rct_30000/daten/rct.json"
    for path in (real, rct, records, metadata, CATALOG):
        if not path.exists():
            raise FileNotFoundError(path)
    original = json.loads(metadata.read_text())
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in (real, rct, records, metadata)}
    if hashes[str(rct.relative_to(ROOT))] != original["log_sha256"]:
        raise ValueError("RCT data no longer matches the saved simulation checksum")
    if hashes[str(records.relative_to(ROOT))] != original["assignments_sha256"]:
        raise ValueError("RCT assignments no longer match the saved simulation checksum")
    if list(tests) != original["points"]:
        raise ValueError("Intervention points differ from the saved 30,000-case RCT")
    paths = [Path(__file__), ROOT / "screening/acceptance.py", ROOT / "screening/acceptance_report.py",
             ROOT / "screening/all_points.py", ROOT / "screening/all_points_report.py", ROOT / "screening/candidates.py",
             ROOT / "screening/events.py", ROOT / "screening/features.py", ROOT / "screening/causal.py", CATALOG]
    spec = {"outcomes": ENDPOINTS, "points": list(tests), "seed": args.seed, "folds": args.folds,
            "outcome_model": "XGBoost binary:logistic, constant fallback for single-class training arms",
            "risk_set": "first target event strictly after decision, or no target observed; already achieved targets excluded",
            "unit": "application; at least one future offer reaches target before recorded trace end",
            "multiplicity": "Holm over all estimable option contrasts per dataset and endpoint",
            "input_sha256": hashes, "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
            "rct_actual_cases": original["actual_cases"], "rct_regenerated": False,
            "semantic_sources": [SEMANTICS_URL, ACTIVITY_URL]}
    manifest = OUT / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != spec:
        raise ValueError("Acceptance run has a different source/data manifest; keep the existing results separate")
    write_json(manifest, spec)
    basics = {}
    for source, path in (("real", real), ("rct", rct)):
        log = pd.read_pickle(path)
        case, activity, life = "case:concept:name", "concept:name", "lifecycle:transition"
        cases = log[case].drop_duplicates()
        basic = {"cases": len(cases), "events": len(log), "targets": {}}
        rows = pd.DataFrame({"case": cases.to_numpy()}).set_index("case")
        for endpoint, value in ENDPOINTS.items():
            mask = log[activity].eq(value["event"]) & log[life].eq("complete")
            first = log.loc[mask].groupby(case)["time:timestamp"].min()
            rows[endpoint] = rows.index.isin(first.index).astype(int)
            rows[f"{endpoint}_first_time"] = first.reindex(rows.index)
            basic["targets"][endpoint] = {"event": value["event"], "cases": len(first), "observed_rate": len(first) / len(cases)}
        rows.to_csv(OUT / f"fall_outcomes_{source}.csv")
        basics[source] = basic
        del log
    write_json(OUT / "zielgroessen.json", basics)
    print(f"{utc()} prepared {basics['real']['cases']} real cases and {basics['rct']['cases']} saved RCT cases; "
          "20 points, 2 endpoints; input checksums verified", flush=True)


def render_family(tests, endpoint, source):
    import pandas as pd
    from screening.acceptance import annotate_results
    from screening.acceptance_report import render, publish
    out = folder(endpoint, source)
    rows, details = [], []
    for index, name in enumerate(tests):
        checkpoint = out / "checkpoints" / f"point{index:02d}.json"
        if not checkpoint.exists():
            return False
        rows.append(json.loads(checkpoint.read_text()))
        try:
            details.append(pd.read_csv(checkpoint.with_name(f"point{index:02d}_optionen.csv")))
        except pd.errors.EmptyDataError:
            pass
    combined = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    summary, combined = annotate_results(pd.DataFrame(rows), combined)
    render(out, summary, combined, endpoint, source)
    publish(OUT)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tests = configure()
    source_root = args.source_root.resolve()
    prepare(source_root, tests, args)
    if args.prepare_only:
        return
    families = [(endpoint, source) for endpoint in ("accepted", "returned") for source in ("real", "rct")]
    completed = {(endpoint, source): {i for i in range(20) if (folder(endpoint, source) / "checkpoints" / f"point{i:02d}.json").exists()}
                 for endpoint, source in families}
    errors = []
    def status(state="running"):
        write_json(OUT / "status.json", {"state": state, "updated": utc(), "points_per_family": 20,
                   "completed": {f"{endpoint}_{source}": len(done) for (endpoint, source), done in completed.items()},
                   "rct_cases_reused": 30000, "errors": errors})
    status()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=init_worker, initargs=(source_root,)) as pool:
        pending = {}
        for endpoint, source in families:
            for index, (name, test) in enumerate(tests.items()):
                if index not in completed[endpoint, source]:
                    future = pool.submit(analyse_job, source, endpoint, index, name, test, args.seed, args.folds, args.threads)
                    pending[future] = endpoint, source, index
        while pending:
            done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                try:
                    source, endpoint, index = future.result()
                except Exception as error:
                    import traceback
                    traceback.print_exc()
                    errors.append({"job": job, "error": repr(error)})
                    continue
                completed[endpoint, source].add(index)
                if len(completed[endpoint, source]) == 20:
                    render_family(tests, endpoint, source)
            status()
    if errors or any(len(done) != 20 for done in completed.values()):
        status("failed")
        raise RuntimeError(f"Acceptance run incomplete: {errors}")
    for endpoint, source in families:
        render_family(tests, endpoint, source)
    status("complete")
    print(f"{utc()} COMPLETE: all 20 points, real + saved RCT, O_Accepted + O_Returned", flush=True)


if __name__ == "__main__":
    main()
