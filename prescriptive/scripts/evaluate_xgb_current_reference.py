"""Paired XGBoost time-policy reference for the current corrected simulator."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import warnings
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/"prescriptive/policies/ev_policy_event_costs/results/d_oe_zustand_20261115"
MODEL = ROOT/"prescriptive/models/xgb_model_durchlaufzeit_xgb_real_10k_neuepunkte.pkl"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PSP_CATALOG"] = str(SOURCE/"catalog.yaml")
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths
add_paths()
import pandas as pd


def now():
    return datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds")


def write(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str)+"\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def job(task):
    out, seed = task
    from prescriptive.helpers.datasets import load_dataset
    from preprocessing import preprocess
    from simulation.dataset_simulator import Simulator
    from prescriptive.policies.xgb_policy import XGBPolicy
    from prescriptive.policies.ev_policy_event_costs.costs import EventCosts
    folder = out/f"xgb_time_seed{seed}"
    folder.mkdir(exist_ok=False)
    started = time.time()
    write(folder/"status.json", {"status": "running", "seed": seed, "started_at": now()})
    raw, _, _ = load_dataset("bpic17")
    log = preprocess("bpic17", raw)
    sim = Simulator("bpic17", log)
    policy = XGBPolicy(model_path=str(MODEL), seed=seed)
    assert policy.model.dataset == "bpic17"
    assert policy.model.trained_on["outcome"] == "durchlaufzeit"
    events, choices = sim.run(2200, policy=policy, window=(700, 900), seed=seed)
    numbers = events["case:concept:name"].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
    events = events.loc[numbers.ge(700)&numbers.lt(900)]
    costs = EventCosts(SOURCE/"event_costs.yaml")
    bills = costs.case_costs(events)
    times = events.groupby("case:concept:name")["time:timestamp"]
    days = (times.max()-times.min()).dt.total_seconds()/86400
    bills["cycle_days"] = days
    bank_cases = pd.read_csv(SOURCE/f"bank_reference_seed{seed}/case_costs.csv", index_col=0)
    assert len(bills) == 200 and set(bills.index) == set(bank_cases.index)
    bank = json.loads((SOURCE/f"bank_reference_seed{seed}/result.json").read_text())
    result = {"model": "XGBoost", "objective": "time", "seed": seed, "cases": len(bills),
        "cycle_days": float(days.mean()), "cost_eur": float(bills.total_eur.mean()),
        "bank_cycle_days": bank["cycle_days"], "bank_cost_eur": bank["cost_eur"],
        "time_saving_days": bank["cycle_days"]-float(days.mean()),
        "time_saving_percent": 100*(1-float(days.mean())/bank["cycle_days"]),
        "active_hours": float(bills.active_hours.mean()), "seconds": time.time()-started}
    bills.to_csv(folder/"case_costs.csv")
    choices.to_csv(folder/"choices.csv", index=False)
    events.to_pickle(folder/"events.pkl")
    write(folder/"result.json", result)
    write(folder/"status.json", {"status": "complete", "seed": seed, "updated_at": now()})
    print("XGB COMPLETE", json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(out/"matplotlib")
    from prescriptive.estimators.xgb_model import XGBModel
    model = XGBModel.load(MODEL)
    files = [Path(__file__), MODEL, SOURCE/"catalog.yaml", SOURCE/"event_costs.yaml",
        ROOT/"prescriptive/policies/xgb_policy.py", ROOT/"prescriptive/estimators/xgb_model.py",
        ROOT/"process-simulation-engine/simulation/dataset_simulator.py",
        ROOT/"process-simulation-engine/simulation/case_engine_lifecycle_aware.py",
        ROOT/"process-simulation-engine/interventions/work_state.py"]
    hashes = {str(path): sha(path) for path in files}
    write(out/"manifest.json", {"prepared_at": now(), "seeds": [0,1,2], "scored_cases": 200,
        "warmup": 700, "cooldown": 1300, "allocation": "random", "model": str(MODEL),
        "trained_on": model.trained_on, "source_sha256": hashes,
        "baseline": "same three completed bank jobs as the current corrected D/OE comparison",
        "objective": "negative_cycle_days", "costs": "secondary measured outcome; this XGBoost is not trained to optimize event costs"})
    write(out/"status.json", {"status": "running", "updated_at": now(), "completed_seeds": []})
    rows = []
    try:
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(job, (out, seed)) for seed in [0,1,2]]
            for future in as_completed(futures):
                rows.append(future.result())
                pd.DataFrame(rows).sort_values("seed").to_csv(out/"runs.csv", index=False)
                write(out/"status.json", {"status": "running", "updated_at": now(), "completed_seeds": sorted(r["seed"] for r in rows)})
        assert all(sha(path)==expected for path, expected in hashes.items())
        table = pd.DataFrame(rows).sort_values("seed")
        lines = ["**XGBoost im aktuellen Simulatorvergleich**", "",
            "Vorhandenes auf Durchlaufzeit trainiertes XGBoost-Modell, 10.000 echte Trainingsfälle. "
            "Seeds 0–2, je 200 bewertete Fälle, 700 Warmup und 1.300 Cooldown. "
            "Dieselbe korrigierte Optionsprüfung, Zufallszuteilung, Fall-IDs, Tarife und Bankreferenz wie beim laufenden D/OE-Vergleich. "
            "XGBoost verwendet keine Suffixziehungen und wurde hier nicht neu trainiert.", "",
            "| Seed | Bank (Tage) | XGBoost (Tage) | Zeitersparnis (%) | XGBoost-Kosten/Fall (€) |",
            "|---|---:|---:|---:|---:|"]
        for row in table.itertuples():
            lines.append(f"| {row.seed} | {row.bank_cycle_days:.3f} | {row.cycle_days:.3f} | {row.time_saving_percent:+.2f} | {row.cost_eur:.2f} |")
        lines += ["", f"Mittel über die drei Seeds: Bank {table.bank_cycle_days.mean():.3f} Tage, "
            f"XGBoost {table.cycle_days.mean():.3f} Tage. Mittlerer gepaarter relativer Zeitgewinn: {table.time_saving_percent.mean():.2f} %.", "",
            "Die Kosten sind ein zusätzlich gemessener Ausgang dieser Zeitpolicy. Ein auf Eventkosten optimiertes XGBoost "
            "wurde mit dieser Referenz nicht getestet. Die vorhandenen XGBoost- und PSP-Modellstände unterscheiden sich "
            "in Trainingsumfang und Verfahren; der Vergleich isoliert keinen Architektureffekt."]
        (out/"bericht.md").write_text("\n".join(lines)+"\n")
        write(out/"status.json", {"status": "complete", "updated_at": now(), "completed_seeds": [0,1,2]})
    except BaseException as error:
        write(out/"status.json", {"status": "failed", "updated_at": now(), "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
