"""PSP (c) im selben korrigierten Simulator wie der D/OE-Vergleich vom 15.11.2026 (2026-11-16).

Die Tabelle aus policies/ev_policy_event_costs/results/d_oe_zustand_20261115 enthält D und OE, aber nicht (c).
Dieses Skript stellt (c) unter denselben Bedingungen daneben: Seeds 0-2, je 200 bewertete Fälle (700 Warmup,
1.300 Cooldown), derselbe Katalog (Zufallszuteilung), dieselben Fall-IDs, Tarife und Bankreferenzen.

Das Modell models/psp_model_durchlaufzeit_real_alle_s60_normal.pkl wird unverändert geladen (keine Datei wird
geschrieben): normale Abstände aus dem Decoder, Option 1 ms nach dem Entscheidungs-Event, keine Zustandsregeln,
kein zusätzliches empirisches Zeitmodell. Ziel: Durchlaufzeit. Drei Arme:
- fest100:  PSPPolicy mit festen 100 Ziehungen je Option (so wurde (c) bisher immer ausgewertet)
- adaptiv_mit:  dieselbe Regel wie "D, Durchlaufzeit, mit": 30 Ziehungen, bei unsicherer Differenz 100, danach
                bei weiterhin unsicherer Differenz (<= 2 gepaarte Standardfehler) Rückfall auf die Bank
- adaptiv_ohne: wie "D, Durchlaufzeit, ohne": statt des Rückfalls die Option mit dem besseren Mittel
Die adaptive Regel wird nur im Speicher eingeschaltet (params sampling_control), wie bei den D/OE-Kopien.

Aufruf im Ordner process-simulation-engine:
    .venv/bin/python -u <dieses Skript> --workers 6
"""
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

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/"prescriptive/policies/ev_policy_event_costs/results/d_oe_zustand_20261115"
MODEL = ROOT/"prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal.pkl"
OUT = Path(__file__).resolve().parent/"c_im_korrigierten_simulator"
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

SEEDS, SCORED = [0, 1, 2], 200
ARMS = ["fest100", "adaptiv_ohne", "adaptiv_mit"]


def now():
    return datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds")


def write(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str)+"\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def simulator():
    import interventions
    interventions.CANDIDATES_FILE = str(SOURCE/"catalog.yaml")
    from prescriptive.helpers.datasets import load_dataset
    from preprocessing import preprocess
    from simulation.dataset_simulator import Simulator
    raw, _, _ = load_dataset("bpic17")
    return Simulator("bpic17", preprocess("bpic17", raw))


def score(events):
    from prescriptive.policies.ev_policy_event_costs.costs import EventCosts
    numbers = events["case:concept:name"].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
    events = events.loc[numbers.ge(700) & numbers.lt(700+SCORED)]
    bills = EventCosts(SOURCE/"event_costs.yaml").case_costs(events)
    times = events.groupby("case:concept:name")["time:timestamp"]
    bills["cycle_days"] = (times.max()-times.min()).dt.total_seconds()/86400
    assert len(bills) == SCORED
    return events, bills


def bank_check(seed=0):
    """Derselbe Simulator wie am 15.11.? Die Bank von Seed 0 muss die gespeicherte Referenz je Fall reproduzieren."""
    events, _ = simulator().run(700+SCORED+1300, policy=None, window=(700, 700+SCORED), seed=seed)
    _, bills = score(events)
    reference = pd.read_csv(SOURCE/f"bank_reference_seed{seed}/case_costs.csv", index_col=0)
    difference = float((bills.cycle_days-reference.cycle_days.reindex(bills.index)).abs().max())
    return {"seed": seed, "bank_cycle_days": float(bills.cycle_days.mean()), "reference_cycle_days": float(reference.cycle_days.mean()),
            "max_abs_case_difference_days": difference, "reproduced": bool(difference < 1e-6)}


def job(task):
    arm, seed = task
    import torch
    from prescriptive.policies.ev_policy.psp_policy import PSPPolicy
    from prescriptive.policies.ev_policy.uncertainty_ablation import UncertaintyAblationPolicy
    folder = OUT/f"{arm}_seed{seed}"
    folder.mkdir(exist_ok=False)
    started = time.time()
    if arm == "fest100":
        policy = PSPPolicy(model_path=str(MODEL), seed=seed, samples=100)
        assert "sampling_control" not in policy.model.params
    else:
        policy = UncertaintyAblationPolicy(model_path=str(MODEL), seed=seed, samples=30, bank_on_uncertain=(arm == "adaptiv_mit"))
        policy.model.params["sampling_control"] = {"max_samples": 100, "z": 2.}  # nur im Speicher, wie bei D/OE
    torch.set_num_threads(1)
    model = policy.model
    assert model._outcome_kind() == "negative_cycle_days" and model.value_source() == "suffix"
    assert not getattr(model, "work_state_constraints", False) and getattr(model, "conditional_gaps", None) is None
    assert getattr(model, "transition_rules", None) is None and not model.option_in_encoder()
    progress = {"status": "running", "arm": arm, "seed": seed, "started_at": now(), "decisions": 0}
    write(folder/"progress.json", progress)
    original = policy.choose

    def choose(point, case, options, at, engine):
        selected = original(point, case, options, at, engine)
        progress["decisions"] += 1
        if progress["decisions"] == 1 or progress["decisions"] % 10 == 0:
            progress.update(updated_at=now(), elapsed_seconds=time.time()-started,
                            sampling_fallbacks=getattr(policy, "sampling_fallbacks", 0),
                            decision_audit=getattr(policy, "decision_audit", {}))
            write(folder/"progress.json", progress)
        return selected
    policy.choose = choose
    events, choices = simulator().run(700+SCORED+1300, policy=policy, window=(700, 700+SCORED), seed=seed)
    events, bills = score(events)
    bank = pd.read_csv(SOURCE/f"bank_reference_seed{seed}/case_costs.csv", index_col=0)
    assert set(bills.index) == set(bank.index)
    result = {"model": "(c)", "objective": "time", "arm": arm, "seed": seed, "cases": len(bills),
              "cycle_days": float(bills.cycle_days.mean()), "labor_eur": float(bills.time_eur.mean()),
              "cost_eur": float(bills.total_eur.mean()), "bank_cycle_days": float(bank.cycle_days.mean()),
              "bank_labor_eur": float(bank.time_eur.mean()), "bank_cost_eur": float(bank.total_eur.mean()),
              "time_saving_days": float(bank.cycle_days.mean()-bills.cycle_days.mean()),
              "decisions": progress["decisions"], "sampling_fallbacks": getattr(policy, "sampling_fallbacks", 0),
              "decision_audit": getattr(policy, "decision_audit", {}),
              "uncertainty_overrides": getattr(policy, "uncertainty_overrides", None), "seconds": time.time()-started}
    bills.to_csv(folder/"case_costs.csv")
    choices.to_csv(folder/"choices.csv", index=False)
    events.to_pickle(folder/"events.pkl")
    write(folder/"result.json", result)
    write(folder/"progress.json", {**progress, "status": "complete", "updated_at": now()})
    print("COMPLETE", json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(OUT/"matplotlib")
    reference = json.loads((SOURCE/"manifest.json").read_text())
    sources = {path: {"hash_15_11": expected, "unchanged": sha(path) == expected}
               for path, expected in reference["source_sha256"].items() if path.endswith(".py")}
    files = [Path(__file__), MODEL, SOURCE/"catalog.yaml", SOURCE/"event_costs.yaml",
             ROOT/"prescriptive/policies/ev_policy/psp_policy.py", ROOT/"prescriptive/policies/ev_policy/uncertainty_ablation.py",
             ROOT/"prescriptive/estimators/psp_model.py", ROOT/"process-simulation-engine/simulation/dataset_simulator.py",
             ROOT/"process-simulation-engine/simulation/case_engine_lifecycle_aware.py",
             ROOT/"process-simulation-engine/interventions/work_state.py"]
    hashes = {str(path): sha(path) for path in files}
    check = bank_check(0)
    print("BANK CHECK", json.dumps(check), flush=True)
    write(OUT/"manifest.json", {"prepared_at": now(), "model": str(MODEL), "seeds": SEEDS, "scored_cases": SCORED, "warmup": 700,
                                "cooldown": 1300, "allocation": "random", "arms": ARMS, "objective": "negative_cycle_days",
                                "sources_of_15_11_comparison": sources, "bank_check": check, "source_sha256": hashes})
    if not check["reproduced"]:
        raise RuntimeError("Die Bankreferenz vom 15.11. wird nicht reproduziert; kein gepaarter Vergleich möglich")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(job, (arm, seed)) for arm in ARMS for seed in SEEDS]
        for future in as_completed(futures):
            rows.append(future.result())
            pd.DataFrame(rows).sort_values(["arm", "seed"]).to_csv(OUT/"runs.csv", index=False)
    assert all(sha(path) == expected for path, expected in hashes.items()), "Quellen während des Laufs geändert"
    print("ALLES FERTIG", now(), flush=True)


if __name__ == "__main__":
    main()
