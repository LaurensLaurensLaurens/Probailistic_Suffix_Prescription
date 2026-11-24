"""Diesen Lauf in kurzen Batches fortsetzen; jeder fertige Seed wird sofort atomar gespeichert.

    process-simulation-engine/.venv/bin/python .../resume_simulation.py --max-jobs 8

Gleiche Online-Auswertung und gleiche Parameter wie im ersten Versuch. Vorhandene Checkpoints werden
geprüft und übersprungen. Nach dem 40. Lauf werden die vollständigen Tabellen und der Bericht erstellt.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))
os.environ["PSP_CATALOG"] = str(OUT / "catalog.yaml")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(OUT / "matplotlib_cache")

import numpy as np

from prescriptive.helpers.datasets import add_paths
add_paths()

from prescriptive.evaluation.online_evaluation import _init_worker, _run, compare

POLICIES = ("bank", "s_learner_erm", "x_learner_erm", "causal_forest_erm")
FOLDERS = {"s_learner_erm": "S_Learner_ERM", "x_learner_erm": "X_Learner_ERM", "causal_forest_erm": "Causal_Forest_ERM"}
CHECKPOINTS = OUT / "checkpoints"


def default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, default=default, ensure_ascii=False) + "\n")
    temporary.replace(path)


def checkpoint(name, seed):
    return CHECKPOINTS / f"{name}_seed{seed}.json"


def completed(jobs):
    rows = []
    for name, seed, _, _, _ in jobs:
        path = checkpoint(name, seed)
        if path.exists():
            row = json.loads(path.read_text())
            assert row["Policy"] == name and row["seed"] == seed
            assert row["Fälle"] == 1000 and len(row["_days"]) == 1000
            row["_days"] = np.asarray(row["_days"], dtype=float)
            if name != "bank":
                assert row["ERM γ (1/Tag)"] == 0.1
            rows.append(row)
    return rows


def report(rows):
    from helpers.report import Report, set_outcome, uplift_chart
    table, runs = compare(rows)
    table.to_csv(OUT / "ergebnis_gesamt.csv")
    runs.to_csv(OUT / "laeufe_gesamt.csv", index=False)
    set_outcome("negative_cycle_days")
    document = Report(str(OUT), "BPIC17: drei CATE-ERM-Policies, gemeinsames γ=0,1")
    document.heading("Online-Auswertung: 10 Seeds, je 3.000 Ankünfte und 1.000 gewertete Fälle")
    for name, folder in FOLDERS.items():
        part = table.loc[["bank", name]]
        part_runs = runs[runs.Policy.isin(["bank", name])]
        sub = f"prescription/{folder}"
        document.figure("bpic17", "online_diagramm", uplift_chart(part, part_runs), f"{name}: online, Uplift gegen die Bank", sub=sub)
        document.table("bpic17", "online", part.round(3), f"{name}: Mittel je Policy", sub=sub)
        document.table("bpic17", "online_laeufe", part_runs.round(3), f"{name}: je Seed", sub=sub)
        # CSVs in voller Genauigkeit; der HTML-Bericht bleibt lesbar gerundet.
        part.to_csv(OUT / sub / "bpic17_online.csv")
        part_runs.to_csv(OUT / sub / "bpic17_online_laeufe.csv")
    print(table[["Durchlaufzeit (Tage)", "Durchlaufzeit P90 (Tage)", "CVaR 90 % (Tage)", "ERM γ=0.1 (Tage)"]].to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-jobs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    assert args.max_jobs > 0 and 0 < args.workers <= 4
    CHECKPOINTS.mkdir(exist_ok=True)
    manifest = json.loads((OUT / "laufparameter.json").read_text())
    policy_params = {name: {"model": manifest["models"][learner]["path"], "gamma": 0.1}
                     for name, learner in zip(POLICIES[1:], ("s", "x", "cf"))}
    configuration = {"dataset": "bpic17", "policies": POLICIES, "seeds": list(range(10)),
                     "arrivals": 3000, "window": [700, 1700], "points": None, "policy_params": policy_params,
                     "catalog_sha256": hashlib.sha256((OUT / "catalog.yaml").read_bytes()).hexdigest(),
                     "models_sha256": {key: hashlib.sha256(Path(value["path"]).read_bytes()).hexdigest()
                                       for key, value in manifest["models"].items()}}
    # JSON-Rundlauf normalisiert Tupel, damit identische Folgeaufrufe identisch verglichen werden.
    configuration = json.loads(json.dumps(configuration))
    settings_path = OUT / "checkpoint_settings.json"
    if settings_path.exists():
        assert json.loads(settings_path.read_text()) == configuration, "Konfiguration unterscheidet sich von den Checkpoints"
    else:
        assert not list(CHECKPOINTS.glob("*.json")), "Checkpoints ohne Konfiguration"
        save(settings_path, configuration)
    jobs = [(name, seed, 3000, (700, 1700), None) for seed in range(10) for name in POLICIES]
    rows = completed(jobs)
    print(f"Bereits gespeichert: {len(rows)}/40", flush=True)
    if len(rows) == len(jobs):
        report(rows)
        return
    pending = [job for job in jobs if not checkpoint(job[0], job[1]).exists()]
    # Jeweils die nächsten zwei Seeds, innerhalb des Batches die langen Modelle zuerst.
    pending = pending[:args.max_jobs]
    priority = {"x_learner_erm": 0, "causal_forest_erm": 1, "s_learner_erm": 2, "bank": 3}
    pending.sort(key=lambda job: priority[job[0]])
    from helpers.datasets import load_dataset
    from preprocessing import preprocess
    from simulation.dataset_simulator import Simulator, is_trained, model_folder
    assert is_trained(model_folder("bpic17")), "Der bestehende Simulator fehlt; kein automatisches Neutraining"
    raw, _, _ = load_dataset("bpic17")
    log = preprocess("bpic17", raw)
    del raw
    simulator = Simulator("bpic17", log)
    print(f"Batch: {len(pending)} Läufe, {min(args.workers, len(pending))} Prozesse; {simulator}", flush=True)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(pending)), mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker, initargs=("bpic17", log, simulator.intervention["key"], None, policy_params)) as pool:
        futures = {pool.submit(_run, job): job for job in pending}
        for future in as_completed(futures):
            row = future.result()
            save(checkpoint(row["Policy"], row["seed"]), row)
            print(f"Checkpoint gespeichert: {row['Policy']}, seed {row['seed']}", flush=True)
    rows = completed(jobs)
    print(f"Batch fertig in {time.monotonic() - started:.0f} s; gespeichert: {len(rows)}/40", flush=True)
    if len(rows) == len(jobs):
        report(rows)


if __name__ == "__main__":
    main()
