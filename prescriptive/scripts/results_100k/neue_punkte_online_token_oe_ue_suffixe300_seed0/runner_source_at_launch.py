"""One token_oe online seed with transition rules and persistent choice/progress logs.

Uses main.py and its normal PSP policy, one worker. Observation adds no model
queries or random draws. All settings match the earlier 300-draw online run.
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
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    model_path = P / "models/psp_model_durchlaufzeit_real_alle_s60_token_oe_ue.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    from prescriptive.policies.ev_policy.psp_policy import PSPPolicy
    original_choose = PSPPolicy.choose
    started = time.time()
    counts = Counter()
    cases = set()
    progress = {"status": "starting", "seed": args.seed, "samples_per_option": 300,
                "model": str(model_path), "catalog": os.environ["PSP_CATALOG"],
                "online_cases": 3000, "warmup": 700, "cooldown": 1300,
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
            if not rules or not rules.get("pairs") or not rules.get("ends"):
                raise RuntimeError("Expected active observed-transition rules")
            call_started = time.time()
            selected = original_choose(policy, point, case, options, now, engine)
            cases.add(case.case_id)
            counts[f"{point.name}: {selected}"] += 1
            progress.update(status="running", decisions_completed=progress["decisions_completed"] + 1,
                            last_case=case.case_id, last_point=point.name,
                            allowed_pairs=len(rules["pairs"]), allowed_end_tokens=len(rules["ends"]))
            choices.write(json.dumps({"case": case.case_id, "point": point.name, "choice": selected,
                                      "simulation_time": now, "seconds": time.time() - call_started},
                                     ensure_ascii=False) + "\n")
            if progress["decisions_completed"] == 1 or progress["decisions_completed"] % 25 == 0:
                save_progress()
            if progress["decisions_completed"] % 100 == 0:
                print(f"PROGRESS: {progress['decisions_completed']} Entscheidungen, "
                      f"{len(cases)} Fälle mit Entscheidungen, {time.time() - started:.0f} s", flush=True)
            return selected

        PSPPolicy.choose = observed_choose
        sys.argv = [str(P / "scripts/main.py"), "--delta", "0.5", "--rct-cases", "10000", "--rct-runs", "10",
                    "--steps", "online", "--online-policies", "psp", "--psp-model", str(model_path),
                    "--psp-samples", "300", "--online-seeds", "1", "--seed", str(args.seed),
                    "--online-cases", "3000", "--online-warmup", "700", "--online-cooldown", "1300",
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
            save_progress()


if __name__ == "__main__":
    main()
