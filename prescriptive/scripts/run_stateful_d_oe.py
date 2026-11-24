"""Prepare, validate and run a paired D/OE comparison with work-state checks.

The frozen event networks get the same train-only conditional time calibration.
All seeds use the corrected simulator, the same tariffs and random allocation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths
add_paths()
import numpy as np
import pandas as pd

SOURCES = [Path(__file__), ROOT/"prescriptive/estimators/psp_model.py",
    ROOT/"prescriptive/estimators/conditional_gaps.py", ROOT/"prescriptive/policies/ev_policy/psp_policy.py",
    ROOT/"process-simulation-engine/interventions/work_state.py",
    ROOT/"process-simulation-engine/simulation/case_engine_lifecycle_aware.py",
    ROOT/"process-simulation-engine/simulation/dataset_simulator.py",
    ROOT/"prescriptive/policies/ev_policy_event_costs/costs.py",
    ROOT/"prescriptive/policies/ev_policy_event_costs/psp_policy.py"]
MODELS = {"D": ROOT/"prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_klassen_option_encoder_ue.pkl",
    "OE": ROOT/"prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_token_oe_ue.pkl"}


def now():
    return datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds")


def write(path, data):
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str, allow_nan=False)+"\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out, args):
    out.mkdir(parents=True, exist_ok=False)
    (out/"models").mkdir()
    (out/"catalog.yaml").write_bytes((ROOT/"prescriptive/data/catalog_neue_punkte_zuteilung_random.yaml").read_bytes())
    (out/"event_costs.yaml").write_bytes((ROOT/"prescriptive/policies/ev_policy_event_costs/event_costs.yaml").read_bytes())
    os.environ["PSP_CATALOG"] = str(out/"catalog.yaml")
    from prescriptive.estimators.psp_model import PSPModel, VALIDATION_SHARE
    from prescriptive.estimators.conditional_gaps import ConditionalGaps
    from prescriptive.scripts.psp_uebergaenge_kopie import rules_from
    from prescriptive.helpers.datasets import load_dataset
    from preprocessing import preprocess
    raw, _, _ = load_dataset("bpic17")
    log = preprocess("bpic17", raw).sort_values(["case:concept:name", "time:timestamp"], kind="stable")
    reference = PSPModel.load(MODELS["D"])
    outer = log.loc[~log["case:concept:name"].astype(str).isin(set(map(str, reference.test_cases)))]
    # Reproduce the legacy loader's shuffled case split, excluding validation too.
    cases = np.asarray(outer["case:concept:name"].astype(str).unique(), dtype=object)
    np.random.RandomState(reference.seed).shuffle(cases)
    validation = set(cases[:int(VALIDATION_SHARE*len(cases))])
    train = outer.loc[~outer["case:concept:name"].astype(str).isin(validation)]
    assert train["case:concept:name"].nunique() == reference.summary["alle Prefixe"]["Trainingsfälle"]
    calibration = ConditionalGaps().fit(train)
    assert not set(calibration.training_cases) & set(map(str, reference.test_cases))
    train_ids_path = out/"calibration_training_cases.json"
    write(train_ids_path, calibration.training_cases)
    provenance = {str(p): sha(p) for p in SOURCES+list(MODELS.values())}
    for name, source in MODELS.items():
        model = PSPModel.load(source)
        assert model.test_cases == reference.test_cases
        model.work_state_constraints = True
        model.conditional_gaps = calibration
        model.sampling_audit = {}
        model.params["sampling_control"] = {"max_samples": args.max_samples, "z": 2.}
        model.transition_rules = rules_from(train, True)
        model.save(out/"models"/f"{name}_stateful.pkl")
    manifest = {"prepared_at": now(), "seeds": list(range(args.seeds)), "scored_cases": args.scored_cases,
        "warmup": 700, "cooldown": 1300, "samples": args.samples, "max_samples": args.max_samples,
        "workers": args.workers, "calibration_training_cases": len(calibration.training_cases),
        "excluded_validation_cases": len(validation), "excluded_test_cases": len(reference.test_cases),
        "time_model": "common empirical distribution conditional on previous event, current event, own work state and any active work; backoff retains current event and work state",
        "fallback": "bank when a complete supported suffix cannot be drawn or the Monte Carlo difference is <=2 paired standard errors at max_samples",
        "source_sha256": provenance,
        "prepared_sha256": {str(p): sha(p) for p in [out/"catalog.yaml", out/"event_costs.yaml", train_ids_path,
            out/"models/D_stateful.pkl", out/"models/OE_stateful.pkl"]}}
    write(out/"manifest.json", manifest)
    write(out/"status.json", {"status": "prepared", "updated_at": now()})
    print(json.dumps({k:v for k,v in manifest.items() if not k.endswith("sha256")}, indent=2), flush=True)


def check(out):
    manifest = json.loads((out/"manifest.json").read_text())
    for group in ("source_sha256", "prepared_sha256"):
        for filename, expected in manifest[group].items():
            if sha(filename) != expected:
                raise RuntimeError(f"Input changed after preparation: {filename}")
    return manifest


def probe(out):
    import torch
    from interventions.work_state import WorkState
    from prescriptive.estimators.psp_model import PSPModel
    from prescriptive.estimators.conditional_gaps import SuffixSamplingError
    from prescriptive.policies.ev_policy_event_costs.costs import EventCosts
    from simulation.case_engine_lifecycle_aware import CaseEngine
    from types import SimpleNamespace
    check(out)
    torch.set_num_threads(1)
    costs = EventCosts(out/"event_costs.yaml")
    old = ROOT/"prescriptive/scripts/results_100k/d_sampling_diagnosis_20261115"
    targets = set(json.loads((old/"selected_decisions.json").read_text()))
    rows, failures, timings = [], [], []
    for name in MODELS:
        model = PSPModel.load(out/"models"/f"{name}_stateful.pkl")
        model.set_threads(1)
        for record in pickle.loads((old/"prefixes.pkl").read_bytes()):
            if record["id"] not in targets and record["id"] != 102:
                continue
            options = CaseEngine.control_options(SimpleNamespace(prefix=record["events"], sessions=record["open_sim_sessions"]), record["options"])
            if record["id"] == 102:
                assert options == ["A_Accepted"], options
            torch.manual_seed(300000+record["id"])
            started = time.time()
            try:
                drawn, elapsed = model.option_suffixes(model.points[record["point"]], record["events"], options, record["now"], 30)
            except SuffixSamplingError as error:
                failures.append({"model": name, "id": record["id"], "error": str(error)})
                print("PROBE FALLBACK", name, record["id"], flush=True)
                continue
            timings.append(time.time()-started)
            for option, suffixes in zip(options, drawn):
                for suffix in suffixes:
                    state = WorkState.from_events(record["events"])
                    bill = costs.state(record["events"])
                    sunk, at = bill.total(record["now"]), record["now"]
                    for token, _, gap in suffix:
                        assert state.permits(token), (name, record["id"], token)
                        state.push(token)
                        at += gap
                        a, _, lc = token.rpartition("|")
                        bill.push(a, lc, at)
                    assert state.complete and not bill.opened, (name, record["id"])
                    rows.append({"model": name, "id": record["id"], "option": option,
                        "remaining_cost": bill.total(at)-sunk, "days": sum(s[2] for s in suffix)/86400,
                        "events": len(suffix), "first_gap_hours": suffix[0][2]/3600})
            print("PROBE", name, record["id"], round(timings[-1], 2), flush=True)
    pd.DataFrame(rows).to_csv(out/"probe_draws.csv", index=False)
    result = {"status": "passed" if len(failures) <= 6 and len(rows) > 1000 else "failed",
        "valid_suffixes": len(rows), "invalid_returned_suffixes": 0, "fallback_calls": failures,
        "mean_call_seconds": float(np.mean(timings)) if timings else None,
        "case750_duplicate_start_filtered": True}
    write(out/"probe.json", result)
    if result["status"] != "passed":
        raise RuntimeError(result)


def job(task):
    out, name, objective, seed, scored, samples = task
    from prescriptive.helpers.datasets import load_dataset
    from preprocessing import preprocess
    from simulation.dataset_simulator import Simulator
    from prescriptive.policies.ev_policy.psp_policy import PSPPolicy
    from prescriptive.policies.ev_policy_event_costs.psp_policy import PSPEventCostPolicy
    from prescriptive.policies.ev_policy_event_costs.costs import EventCosts
    os.environ["PSP_CATALOG"] = str(out/"catalog.yaml")
    import interventions
    interventions.CANDIDATES_FILE = str(out/"catalog.yaml")
    folder = out/f"{name}_{objective}_seed{seed}"
    folder.mkdir(exist_ok=False)
    started = time.time()
    progress = {"status": "running", "started_at": now(), "decisions": 0, "model": name, "objective": objective, "seed": seed}
    write(folder/"progress.json", progress)
    raw, _, _ = load_dataset("bpic17")
    log = preprocess("bpic17", raw)
    sim = Simulator("bpic17", log)
    costs = EventCosts(out/"event_costs.yaml")
    policy = None
    if name != "bank":
        cls = PSPEventCostPolicy if objective == "cost" else PSPPolicy
        kwargs = {"cost_config": costs} if objective == "cost" else {}
        policy = cls(model_path=str(out/"models"/f"{name}_stateful.pkl"), seed=seed, samples=samples, **kwargs)
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
    events, choices = sim.run(700+scored+1300, policy=policy, window=(700, 700+scored), seed=seed)
    numbers = events["case:concept:name"].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
    events = events.loc[numbers.ge(700)&numbers.lt(700+scored)]
    bills = costs.case_costs(events)
    grouped = events.groupby("case:concept:name")["time:timestamp"]
    days = (grouped.max()-grouped.min()).dt.total_seconds()/86400
    assert len(bills) == scored
    bills["cycle_days"] = days
    bills.to_csv(folder/"case_costs.csv")
    choices.to_csv(folder/"choices.csv", index=False)
    events.to_pickle(folder/"events.pkl")
    result = {"model": name, "objective": objective, "seed": seed, "cases": len(bills),
        "cost_eur": float(bills.total_eur.mean()), "cycle_days": float(days.mean()),
        "active_hours": float(bills.active_hours.mean()), "seconds": time.time()-started,
        "sampling_fallbacks": getattr(policy, "sampling_fallbacks", 0),
        "decision_audit": getattr(policy, "decision_audit", {})}
    write(folder/"result.json", result)
    write(folder/"progress.json", {**progress, "status": "complete", "updated_at": now()})
    print("COMPLETE", json.dumps(result), flush=True)
    return result


def run(out):
    manifest = check(out)
    if json.loads((out/"probe.json").read_text())["status"] != "passed":
        raise RuntimeError("A passing real-model probe is required")
    write(out/"status.json", {"status": "running", "updated_at": now(), "completed_jobs": 0})
    tasks = [(out, "bank", "reference", seed, manifest["scored_cases"], manifest["samples"]) for seed in manifest["seeds"]]
    tasks += [(out, name, objective, seed, manifest["scored_cases"], manifest["samples"])
        for seed in manifest["seeds"] for objective in ("cost", "time") for name in MODELS]
    rows = []
    try:
        import multiprocessing
        with ProcessPoolExecutor(max_workers=manifest["workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
            pending = {pool.submit(job, task): task for task in tasks}
            for future in as_completed(pending):
                rows.append(future.result())
                pd.DataFrame(rows).to_csv(out/"runs.csv", index=False)
                write(out/"status.json", {"status": "running", "updated_at": now(), "completed_jobs": len(rows), "total_jobs": len(tasks)})
        check(out)
        frame = pd.DataFrame(rows)
        bank = frame[frame.model.eq("bank")].set_index("seed")
        compared = frame[~frame.model.eq("bank")].copy()
        compared["cost_saving_eur"] = compared.seed.map(bank.cost_eur)-compared.cost_eur
        compared["time_saving_days"] = compared.seed.map(bank.cycle_days)-compared.cycle_days
        compared.to_csv(out/"paired_results.csv", index=False)
        summary = compared.groupby(["model", "objective"])[["cost_eur", "cycle_days", "cost_saving_eur", "time_saving_days"]].mean()
        summary.to_csv(out/"comparison.csv")
        lines = ["**D und OE mit Zustandsprüfung und bedingter Zeitkalibrierung**", "",
            f"{len(manifest['seeds'])} gepaarte Seeds, je {manifest['scored_cases']} bewertete Fälle; 700 Warmup und 1.300 Cooldown. "
            f"Je Option zunächst {manifest['samples']}, bei unsicherer Differenz {manifest['max_samples']} Ziehungen. "
            "Rückfall auf Bankverhalten bei nicht ausreichend getrennter Monte-Carlo-Schätzung. "
            "Die vorhandenen Eventnetze wurden nicht neu trainiert. Beide Modelle verwenden dasselbe train-only Zeitmodell.", "",
            "| Modell | Ziel | Kosten/Fall (€) | Tage/Fall | Kostenersparnis ggü. Bank (€) | Zeitersparnis ggü. Bank (Tage) |",
            "|---|---|---:|---:|---:|---:|"]
        for (name, objective), row in summary.iterrows():
            lines.append(f"| {name} | {objective} | {row.cost_eur:.2f} | {row.cycle_days:.3f} | {row.cost_saving_eur:+.2f} | {row.time_saving_days:+.3f} |")
        lines += ["", "Alle Verfahren verwenden die korrigierte Optionsprüfung des Simulators. "
            "Die historische Bank ist deshalb keine unverändert übernommene Referenz. "
            "Seedweise Ergebnisse und alle Rückfälle stehen in paired_results.csv und den Fortschrittsdateien."]
        (out/"auswertung.md").write_text("\n".join(lines)+"\n")
        write(out/"status.json", {"status": "complete", "updated_at": now(), "completed_jobs": len(rows)})
    except BaseException as error:
        write(out/"status.json", {"status": "failed", "updated_at": now(), "error": repr(error), "completed_jobs": len(rows)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "probe", "run", "launch"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--scored-cases", type=int, default=200)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if min(args.seeds, args.scored_cases, args.samples, args.workers) < 1 or args.max_samples < args.samples:
        parser.error("Positive counts and max-samples >= samples are required")
    out = args.out.resolve()
    os.environ["MPLCONFIGDIR"] = str(out/"matplotlib")
    os.environ["PSP_CATALOG"] = str(out/"catalog.yaml")
    if args.phase == "prepare":
        prepare(out, args)
    elif args.phase == "probe":
        probe(out)
    elif args.phase == "run":
        run(out)
    else:
        check(out)
        with (out/"pipeline.log").open("x") as stream:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "run", "--out", str(out)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        (out/"pipeline.pid").write_text(str(process.pid)+"\n")
        print(json.dumps({"pid": process.pid, "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
