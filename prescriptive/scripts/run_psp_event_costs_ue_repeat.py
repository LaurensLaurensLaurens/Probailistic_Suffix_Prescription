"""Repeat the validated constrained event-cost run with a different draw count."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

P = Path(__file__).resolve().parents[1]
ROOT = P.parent
sys.path.insert(0, str(ROOT))
from prescriptive.scripts import run_psp_event_costs_ue as base

REFERENCE = base.PACKAGE / "results/psp_d_ue_200_faelle_300_ziehungen_seed0_20261114"
read, write, sha, now = base.read, base.write, base.sha, base.now


def prepare(out, samples):
    original = read(REFERENCE / "manifest.json")
    base.check(original)
    assert read(REFERENCE / "status.json")["status"] == "complete"
    assert read(REFERENCE / "validation.json")["status"] == "passed"
    assert read(REFERENCE / "sampling_audit.json")["invalid_transitions"] == 0
    assert original["samples_per_option"] == 300
    assert original["scored_cases"] == 200 and original["seed"] == 0
    out.mkdir(parents=True, exist_ok=False)
    for filename in ("catalog.yaml", "event_costs.yaml", "transition_rules.json"):
        (out / filename).write_bytes((REFERENCE / filename).read_bytes())
    arguments = list(original["evaluate_args"])
    for option, value in (("--samples", str(samples)), ("--catalog", str(out / "catalog.yaml")),
                          ("--cost-config", str(out / "event_costs.yaml")), ("--out", str(out / "online"))):
        arguments[arguments.index(option) + 1] = value
    assert arguments[arguments.index("--psp-model") + 1] == original["model"]
    sources = [Path(name) for name in original["source_sha256"]]
    sources += [Path(__file__).resolve(), out / "catalog.yaml", out / "event_costs.yaml", out / "transition_rules.json",
                *[REFERENCE / name for name in ("manifest.json", "validation.json", "comparison.csv", "paired_case_costs.csv",
                                               "online/run.json", "online/bank_seed0/case_costs.csv", "online/psp_seed0/case_costs.csv")]]
    write(out / "manifest.json", {"prepared_at": now(), "objective": "negative_event_costs", "variant": "d_with_transition_rules",
          "model": original["model"], "samples_per_option": samples, "scored_cases": 200, "seed": 0,
          "warmup": 700, "cooldown": 1300, "reference_constrained_300": str(REFERENCE),
          "cost_fingerprint": original["cost_fingerprint"], "evaluate_args": arguments,
          "command": [sys.executable, "-u", str(Path(__file__).resolve()), "--evaluate", "--out", str(out)],
          "source_sha256": {str(path): sha(path) for path in sources}})
    write(out / "status.json", {"status": "prepared", "updated_at": now(), "samples_per_option": samples,
          "scored_cases": 200, "seed": 0, "transition_rules": True})


def evaluate(out):
    base.add_paths(out)
    from prescriptive.estimators.psp_model import PSPModel
    from prescriptive.policies.ev_policy_event_costs import evaluate as evaluator
    m = read(out / "manifest.json")
    base.check(m)
    pairs = set(map(tuple, read(out / "transition_rules.json")["pairs"]))
    original = PSPModel.option_suffixes
    counts = {"status": "running", "checked_transitions": 0, "invalid_transitions": 0,
              "checked_suffixes": 0, "decisions_completed": 0, "samples_per_option": m["samples_per_option"]}

    def save():
        counts["updated_at"] = now()
        write(out / "sampling_audit.json", counts)

    def checked(model, point, events, options, at, n):
        assert n == m["samples_per_option"] and getattr(model, "transition_rules", None)
        drawn, elapsed = original(model, point, events, options, at, n)
        assert len(drawn) == len(options) and all(len(suffixes) == n for suffixes in drawn)
        base.audit_suffixes(events, drawn, pairs, counts)
        counts["decisions_completed"] += 1
        if counts["decisions_completed"] == 1 or counts["decisions_completed"] % 25 == 0:
            save()
        return drawn, elapsed

    PSPModel.option_suffixes = checked
    sys.argv = [evaluator.__file__, *m["evaluate_args"]]
    save()
    try:
        evaluator.main()
        counts["status"] = "complete"
    except BaseException as error:
        counts.update(status="failed", error=repr(error))
        raise
    finally:
        PSPModel.option_suffixes = original
        save()


def report(out):
    import numpy as np
    import pandas as pd
    m = read(out / "manifest.json")
    samples = m["samples_per_option"]
    current, previous = read(out / "online/run.json"), read(REFERENCE / "online/run.json")
    assert current["status"] == previous["status"] == "complete"
    assert current["samples"] == samples and previous["samples"] == 300
    for key in ("scored_cases", "warmup", "cooldown", "seed", "seeds", "cost_fingerprint", "points"):
        assert current[key] == previous[key], key
    assert current["policy_params"]["psp"]["model_path"] == previous["policy_params"]["psp"]["model_path"]
    audit = read(out / "sampling_audit.json")
    assert audit["status"] == "complete" and audit["invalid_transitions"] == 0
    old_table = pd.read_csv(REFERENCE / "comparison.csv").set_index("Verfahren")
    current_table = pd.read_csv(out / "online/comparison.csv").set_index("Policy")
    bank_bill = pd.read_csv(out / "online/bank_seed0/case_costs.csv").set_index("case:concept:name").sort_index()
    prior_bank = pd.read_csv(REFERENCE / "online/bank_seed0/case_costs.csv").set_index("case:concept:name").sort_index()
    psp_bill = pd.read_csv(out / "online/psp_seed0/case_costs.csv").set_index("case:concept:name").sort_index()
    assert len(bank_bill) == len(psp_bill) == 200
    assert bank_bill.index.equals(prior_bank.index) and bank_bill.index.equals(psp_bill.index)
    assert np.allclose(bank_bill.to_numpy(), prior_bank.to_numpy(), rtol=0, atol=1e-10)
    label = f"PSP-D, {samples} mit Sperre"
    table = pd.concat([old_table, pd.DataFrame([current_table.loc["psp"]], index=pd.Index([label], name="Verfahren"))])
    bank = float(current_table.loc["bank", "Eventkosten je Fall (€)"])
    assert np.isclose(bank, old_table.loc["Bank", "Eventkosten je Fall (€)"])
    assert np.isclose(psp_bill.total_eur.mean(), table.loc[label, "Eventkosten je Fall (€)"])
    table["Ersparnis ggü. Bank (€)"] = bank - table["Eventkosten je Fall (€)"]
    table["Ersparnis ggü. Bank (%)"] = table["Ersparnis ggü. Bank (€)"] / bank * 100
    table.to_csv(out / "comparison.csv")
    paired = pd.read_csv(REFERENCE / "paired_case_costs.csv").set_index("case:concept:name").sort_index()
    assert paired.index.equals(bank_bill.index)
    paired[label] = psp_bill.total_eur
    paired.to_csv(out / "paired_case_costs.csv")
    delta = table.loc[label, "Eventkosten je Fall (€)"] - table.loc["PSP-D, 300 mit Sperre", "Eventkosten je Fall (€)"]
    lines = [f"# PSP-D mit Übergangssperre: {samples} Ziehungen", "",
             f"200 Fälle, Seed 0, {samples} Ziehungen je Option; 700 Warmup, 1.300 Cooldown. "
             "Derselbe Modellstand, dieselbe Übergangssperre, Tarife, Wahlpunkte und Ressourcenstrategie wie beim 300er-Lauf.", "",
             "| Verfahren | Kosten/Fall (€) ↓ | Ersparnis ggü. Bank (€) | Ersparnis (%) | Durchlaufzeit (Tage) |",
             "|---|---:|---:|---:|---:|"]
    for name, row in table.iterrows():
        lines.append(f"| {name} | {row['Eventkosten je Fall (€)']:.2f} | {row['Ersparnis ggü. Bank (€)']:+.2f} | "
                     f"{row['Ersparnis ggü. Bank (%)']:+.2f} | {row['Durchlaufzeit (Tage)']:.2f} |")
    lines += ["", f"{samples} minus 300 Ziehungen, jeweils mit Sperre: {delta:+.2f} €/Fall. "
              "Negative Werte bedeuten niedrigere Kosten.", "",
              f"Geprüft: {audit['checked_suffixes']:,} Suffixe, {audit['checked_transitions']:,} Übergänge, "
              f"{audit['invalid_transitions']} unbeobachtete Übergänge; einschließlich Prefix→Option.", "",
              "Bankkosten und Arbeitszeiten je Fall stimmen mit dem 300er-Lauf überein. "
              "Alle Verfahren haben dieselben 200 Fall-IDs. Ein Seed; kein Konfidenzintervall über mehrere Seeds."]
    (out / "auswertung.md").write_text("\n".join(lines) + "\n")
    write(out / "validation.json", {"status": "passed", "completed_at": now(), "same_200_case_ids": True,
          "bank_cost_components_identical": True, "same_model_rules_tariffs_and_points": True,
          "no_unobserved_sampled_transitions": True, "cost_delta_vs_constrained_300_eur": float(delta)})


def run(out):
    m = read(out / "manifest.json")
    if read(out / "status.json")["status"] != "prepared":
        raise RuntimeError("Run must be prepared")
    state = {"pid": os.getpid(), "started_at": now(), "samples_per_option": m["samples_per_option"],
             "scored_cases": 200, "seed": 0, "transition_rules": True}

    def update(phase, **extra):
        state.update(status=phase, updated_at=now(), **extra)
        write(out / "status.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    try:
        base.check(m)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", MPLBACKEND="Agg", MPLCONFIGDIR=str(out / "matplotlib"),
                   PYTHONWARNINGS="ignore::FutureWarning")
        with (out / "online.log").open("x") as log:
            process = subprocess.Popen(m["command"], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            update("online", evaluator_pid=process.pid)
            while process.poll() is None:
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    path = out / "sampling_audit.json"
                    audit = read(path) if path.exists() else {}
                    update("online", decisions_completed=audit.get("decisions_completed", 0),
                           checked_transitions=audit.get("checked_transitions", 0), invalid_transitions=audit.get("invalid_transitions", 0))
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, m["command"])
        base.check(m)
        update("reporting", evaluator_pid=None)
        report(out)
        audit = read(out / "sampling_audit.json")
        update("complete", decisions_completed=audit["decisions_completed"], checked_transitions=audit["checked_transitions"],
               invalid_transitions=audit["invalid_transitions"], report=str(out / "auswertung.md"))
    except BaseException as error:
        update("failed", error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--out", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "launch", "run", "evaluate"):
        mode.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("Draw count must be positive")
    out = (args.out or base.PACKAGE / f"results/psp_d_ue_200_faelle_{args.samples}_ziehungen_seed0_20261114").resolve()
    if args.prepare:
        prepare(out, args.samples)
        print(f"Prepared: {out}")
    elif args.launch:
        if read(out / "status.json")["status"] != "prepared" or (out / "pipeline.pid").exists():
            raise RuntimeError("Run must be prepared and not already launched")
        base.check(read(out / "manifest.json"))
        with (out / "pipeline.log").open("x") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run", "--out", str(out)],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        (out / "pipeline.pid").write_text(str(process.pid) + "\n")
        print(f"Started PID {process.pid}: {out}")
    elif args.evaluate:
        evaluate(out)
    else:
        run(out)


if __name__ == "__main__":
    main()
