"""One token_oe online seed with optional transition rules and persistent progress.

Uses main.py and its normal PSP policy, one worker. Observation adds no model
queries or random draws. Defaults match the earlier 300-draw online run.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import runpy
import sys
import time
import warnings

P = Path(__file__).resolve().parents[1]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PSP_CATALOG"] = str(P / "data/catalog_neue_punkte_zuteilung_random.yaml")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(P))
from helpers.datasets import add_paths
add_paths()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=P / "models/psp_model_durchlaufzeit_real_alle_s60_token_oe_ue.pkl")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--scored-cases", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=700)
    parser.add_argument("--cooldown", type=int, default=1300)
    parser.add_argument("--without-transition-rules", action="store_true")
    parser.add_argument("--require-shared-option-gap", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.scored_cases < 1 or min(args.warmup, args.cooldown) < 0:
        parser.error("Samples and scored cases must be positive; warmup/cooldown nonnegative")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    from prescriptive.policies.ev_policy.psp_policy import PSPPolicy
    from prescriptive.evaluation import online_evaluation
    original_choose = PSPPolicy.choose
    original_run = online_evaluation._run
    started = time.time()
    counts = Counter()
    cases = set()
    n_cases = args.warmup + args.scored_cases + args.cooldown
    progress = {"status": "starting", "seed": args.seed, "samples_per_option": args.samples,
                "model": str(model_path), "catalog": os.environ["PSP_CATALOG"],
                "online_cases": n_cases, "scored_cases": args.scored_cases,
                "warmup": args.warmup, "cooldown": args.cooldown,
                "transition_rules_expected": not args.without_transition_rules,
                "shared_option_gap_expected": args.require_shared_option_gap,
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "decisions_completed": 0}

    def save_progress():
        progress.update(updated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                        elapsed_seconds=round(time.time() - started, 1),
                        cases_with_decisions=len(cases), choices_by_point=dict(counts))
        temporary = out / "progress.json.tmp"
        temporary.write_text(json.dumps(progress, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(out / "progress.json")

    save_progress()
    with (out / "choices.jsonl").open("x", buffering=1) as choices:
        def observed_choose(policy, point, case, options, now, engine):
            rules = getattr(policy.model, "transition_rules", None)
            shared_gap = bool(getattr(policy.model, "option_gap_distributions", None)) or bool(
                (policy.model.params or {}).get("shared_option_gap", False))
            if shared_gap != args.require_shared_option_gap:
                raise RuntimeError("Unexpected shared option-gap configuration")
            if args.without_transition_rules and rules is not None:
                raise RuntimeError("Expected unrestricted baseline model")
            if not args.without_transition_rules and (not rules or not rules.get("pairs") or not rules.get("ends")):
                raise RuntimeError("Expected active observed-transition rules")
            call_started = time.time()
            selected = original_choose(policy, point, case, options, now, engine)
            cases.add(case.case_id)
            counts[f"{point.name}: {selected}"] += 1
            progress.update(status="running", decisions_completed=progress["decisions_completed"] + 1,
                            last_case=case.case_id, last_point=point.name,
                            allowed_pairs=len(rules["pairs"]) if rules else None,
                            allowed_end_tokens=len(rules["ends"]) if rules else None)
            record = {"case": case.case_id, "point": point.name, "choice": selected,
                      "simulation_time": now, "seconds": time.time() - call_started}
            if args.require_shared_option_gap:
                record["option_gap_draws"] = policy.model.last_option_gap_draws
            choices.write(json.dumps(record, ensure_ascii=False) + "\n")
            if progress["decisions_completed"] == 1 or progress["decisions_completed"] % 25 == 0:
                save_progress()
            if progress["decisions_completed"] % 100 == 0:
                print(f"PROGRESS: {progress['decisions_completed']} Entscheidungen, "
                      f"{len(cases)} Fälle mit Entscheidungen, {time.time() - started:.0f} s", flush=True)
            return selected

        def observed_run(job):
            row = original_run(job)
            def encode(value):
                if hasattr(value, "tolist"):
                    return value.tolist()
                if hasattr(value, "item"):
                    return value.item()
                raise TypeError(type(value).__name__)
            # Save each policy immediately, including per-case durations, before report rendering.
            path = out / f"{job[0]}_seed{job[1]}_row.json"
            path.write_text(json.dumps(row, default=encode, ensure_ascii=False, indent=2) + "\n")
            return row

        PSPPolicy.choose = observed_choose
        online_evaluation._run = observed_run
        sys.argv = [str(P / "scripts/main.py"), "--delta", "0.5", "--rct-cases", "10000", "--rct-runs", "10",
                    "--steps", "online", "--online-policies", "psp", "--psp-model", str(model_path),
                    "--psp-samples", str(args.samples), "--online-seeds", "1", "--seed", str(args.seed),
                    "--online-cases", str(n_cases), "--online-warmup", str(args.warmup), "--online-cooldown", str(args.cooldown),
                    "--online-workers", "1", "--out", str(out)]
        try:
            runpy.run_path(str(P / "scripts/main.py"), run_name="__main__")
        except BaseException as error:
            progress.update(status="failed", error=repr(error))
            raise
        else:
            progress["status"] = "complete"
        finally:
            PSPPolicy.choose = original_choose
            online_evaluation._run = original_run
            save_progress()


if __name__ == "__main__":
    main()
