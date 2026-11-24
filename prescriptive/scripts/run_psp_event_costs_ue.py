"""Repeat PSP-D's 300-draw event-cost evaluation with observed-transition rules."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

P = Path(__file__).resolve().parents[1]
ROOT = P.parent
PACKAGE = P / "policies/ev_policy_event_costs"
REFERENCE = PACKAGE / "results/psp_d_200_faelle_300_ziehungen_seed0_20261114_neustart"
REFERENCE_30 = PACKAGE / "results/vergleich_200_faelle_30_ziehungen_seed0_20261113"
DEFAULT_OUT = PACKAGE / "results/psp_d_ue_200_faelle_300_ziehungen_seed0_20261114"
SOURCE_MODEL = P / "models/psp_model_durchlaufzeit_real_alle_s60_klassen_option_encoder.pkl"
MODEL = P / "models/psp_model_durchlaufzeit_real_alle_s60_klassen_option_encoder_ue.pkl"


def now():
    return datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_paths(out):
    os.environ["PSP_CATALOG"] = str(out / "catalog.yaml")
    sys.path.insert(0, str(ROOT))
    from prescriptive.helpers.datasets import add_paths as initialize
    initialize()


def weights_hash(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.net.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def audit_suffixes(events, drawn, pairs, counts):
    previous = str(events[-1]["concept:name"]) + "|" + str(events[-1].get("lifecycle:transition", "complete"))
    for option_draws in drawn:
        for suffix in option_draws:
            last = previous
            for token, _, _ in suffix:
                counts["checked_transitions"] += 1
                if (last, str(token)) not in pairs:
                    counts["invalid_transitions"] += 1
                    raise RuntimeError(f"Forbidden sampled transition: {last!r} -> {token!r}")
                last = str(token)
            counts["checked_suffixes"] += 1


def prepare(out):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    original = read(REFERENCE / "manifest.json")
    for name, digest in original["sha256"].items():
        if sha(Path(name)) != digest:
            raise RuntimeError(f"Input differs from unrestricted reference: {name}")
    if MODEL.exists():
        raise FileExistsError(MODEL)
    out.mkdir(parents=True, exist_ok=False)
    for name in ("catalog.yaml", "event_costs.yaml"):
        (out / name).write_bytes((REFERENCE / name).read_bytes())
    add_paths(out)
    import numpy as np
    import torch
    from prescriptive.estimators.psp_model import PSPModel
    from prescriptive.helpers.datasets import load_dataset
    from prescriptive.policies.ev_policy_event_costs.costs import EventCosts
    from prescriptive.scripts.psp_uebergaenge_kopie import rules_from
    from preprocessing import preprocess

    torch.set_num_threads(1)
    model = PSPModel.load(SOURCE_MODEL)
    assert model.gap_classes() and not model.gap_tokens() and not model.gap_condition_on_event()
    assert model.option_in_encoder() and not model.params.get("shared_option_gap", False)
    assert getattr(model, "transition_rules", None) is None
    source_digest, weight_digest = sha(SOURCE_MODEL), weights_hash(model)
    original_params = dict(model.params)
    raw, _, _ = load_dataset("bpic17")
    log = preprocess("bpic17", raw).sort_values(["case:concept:name", "time:timestamp"], kind="stable")
    del raw
    held_out = set(map(str, model.test_cases))
    train = log[~log["case:concept:name"].astype(str).isin(held_out)]
    training_ids = set(train["case:concept:name"].astype(str))
    assert len(training_ids) == 25208 and len(held_out) == 6301 and training_ids.isdisjoint(held_out)
    model.transition_rules = rules_from(train, lifecycle_tokens=model.lifecycle_tokens)
    model.transition_rules.update(quelle="25208 Fälle des echten Logs, ohne 6301 äußere Testfälle",
                                  training_cases=len(training_ids), excluded_test_cases=len(held_out))
    write(out / "transition_rules.json", model.transition_rules)
    pairs = set(map(tuple, model.transition_rules["pairs"]))
    ends = set(model.transition_rules["ends"])
    assert pairs and ends
    # Verify EOS and successor masks independently of stochastic draws.
    evaluator = model._evaluator()
    masks = model._transition_masks(evaluator)
    for previous, mask in masks["by_token"].items():
        assert bool(mask[evaluator.eos_id]) == (previous in ends)
        for number in torch.nonzero(mask, as_tuple=True)[0].tolist():
            if number != evaluator.eos_id:
                assert (previous, str(evaluator.inverted_suffix_categories[0][number])) in pairs
    probe_ids = set()
    for point in model.points.values():
        mask = (train["concept:name"] == point.activity) & (train["lifecycle:transition"] == point.lifecycle)
        probe_ids.update(train.loc[mask, "case:concept:name"].astype(str).unique()[:3])
    samples = model.samples(train[train["case:concept:name"].astype(str).isin(probe_ids)])
    counts = {"checked_transitions": 0, "invalid_transitions": 0, "checked_suffixes": 0}
    checked_points = []
    for name, point in model.points.items():
        rows = samples[name]
        assert rows, name
        _, (part, cut, at), _ = rows[0]
        events = part.iloc[:cut].to_dict("records")
        options = list(point.options)
        torch.manual_seed(0)
        drawn, _ = model.option_suffixes(point, events, options, at, 300)
        assert len(drawn) == len(options) and all(len(suffixes) == 300 for suffixes in drawn)
        audit_suffixes(events, drawn, pairs, counts)
        assert all(np.isfinite(gap) and gap >= 0 for suffixes in drawn for suffix in suffixes for _, _, gap in suffix)
        checked_points.append(name)
    assert weights_hash(model) == weight_digest and model.params == original_params
    model.__dict__.pop("_transition_cache", None)
    model.save(MODEL)
    copied = PSPModel.load(MODEL)
    assert weights_hash(copied) == weight_digest and copied.params == original_params
    assert copied.test_cases == model.test_cases and copied.points == model.points
    assert sha(SOURCE_MODEL) == source_digest
    config = EventCosts(out / "event_costs.yaml")
    assert config.fingerprint == original["cost_fingerprint"]
    write(out / "sampling_validation.json", {"status": "passed", "checked_at": now(),
          "weights_unchanged": True, "original_model_unchanged": True, "parameters_unchanged": True,
          "model_weights_sha256": weight_digest, "training_cases": len(training_ids), "excluded_test_cases": len(held_out),
          "test_cases_excluded_from_transition_rules": True, "allowed_pairs": len(pairs), "allowed_end_tokens": len(ends),
          "checked_points": checked_points, "samples_per_option": 300, **counts})
    arguments = ["--policies", "bank", "psp", "--psp-model", str(MODEL), "--samples", "300",
                 "--scored-cases", "200", "--seed", "0", "--seeds", "1", "--warmup", "700", "--cooldown", "1300",
                 "--workers", "1", "--catalog", str(out / "catalog.yaml"),
                 "--cost-config", str(out / "event_costs.yaml"), "--out", str(out / "online")]
    sources = [Path(name) for name in original["sha256"]]
    sources += [Path(__file__).resolve(), MODEL, P / "scripts/psp_uebergaenge_kopie.py", out / "transition_rules.json",
                out / "catalog.yaml", out / "event_costs.yaml", out / "sampling_validation.json",
                REFERENCE / "online/comparison.csv", REFERENCE / "online/run.json",
                REFERENCE / "online/bank_seed0/case_costs.csv", REFERENCE / "online/psp_seed0/case_costs.csv"]
    write(out / "manifest.json", {"prepared_at": now(), "objective": "negative_event_costs", "variant": "d_with_transition_rules",
          "model": str(MODEL), "reference_model": str(SOURCE_MODEL), "reference_300": str(REFERENCE),
          "samples_per_option": 300, "scored_cases": 200, "seed": 0, "warmup": 700, "cooldown": 1300,
          "cost_fingerprint": config.fingerprint, "evaluate_args": arguments,
          "command": [sys.executable, "-u", str(Path(__file__).resolve()), "--evaluate", "--out", str(out)],
          "source_sha256": {str(path): sha(path) for path in sources}})
    write(out / "status.json", {"status": "prepared", "updated_at": now(), "model": str(MODEL),
          "samples_per_option": 300, "scored_cases": 200, "seed": 0})


def check(manifest):
    for name, digest in manifest["source_sha256"].items():
        if sha(Path(name)) != digest:
            raise RuntimeError(f"Prepared input changed: {name}")


def evaluate(out):
    add_paths(out)
    from prescriptive.estimators.psp_model import PSPModel
    from prescriptive.policies.ev_policy_event_costs import evaluate as evaluator
    m = read(out / "manifest.json")
    check(m)
    pairs = set(map(tuple, read(out / "transition_rules.json")["pairs"]))
    original = PSPModel.option_suffixes
    counts = {"status": "running", "checked_transitions": 0, "invalid_transitions": 0,
              "checked_suffixes": 0, "decisions_completed": 0}

    def save():
        counts["updated_at"] = now()
        write(out / "sampling_audit.json", counts)

    def checked(model, point, events, options, at, n):
        assert n == 300 and getattr(model, "transition_rules", None)
        drawn, elapsed = original(model, point, events, options, at, n)
        audit_suffixes(events, drawn, pairs, counts)
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
    current = read(out / "online/run.json")
    previous = read(REFERENCE / "online/run.json")
    assert current["status"] == previous["status"] == "complete"
    for key in ("samples", "scored_cases", "warmup", "cooldown", "seed", "seeds", "cost_fingerprint", "points"):
        assert current[key] == previous[key], key
    audit = read(out / "sampling_audit.json")
    assert audit["status"] == "complete" and audit["invalid_transitions"] == 0
    tables = {"current": pd.read_csv(out / "online/comparison.csv").set_index("Policy"),
              "previous": pd.read_csv(REFERENCE / "online/comparison.csv").set_index("Policy"),
              "old": pd.read_csv(REFERENCE_30 / "online/comparison.csv").set_index("Policy")}
    locations = {"Bank": out / "online/bank_seed0", "Bank vorher": REFERENCE / "online/bank_seed0",
                 "PSP-D, 30 ohne Sperre": REFERENCE_30 / "online/psp_seed0",
                 "PSP-D, 300 ohne Sperre": REFERENCE / "online/psp_seed0",
                 "PSP-D, 300 mit Sperre": out / "online/psp_seed0"}
    bills = {name: pd.read_csv(folder / "case_costs.csv").set_index("case:concept:name").sort_index()
             for name, folder in locations.items()}
    for bill in bills.values():
        assert len(bill) == 200 and bill.index.equals(bills["Bank"].index)
    assert np.allclose(bills["Bank"].to_numpy(), bills["Bank vorher"].to_numpy(), rtol=0, atol=1e-10)
    labels = ["Bank", "PSP-D, 30 ohne Sperre", "PSP-D, 300 ohne Sperre", "PSP-D, 300 mit Sperre"]
    table = pd.DataFrame([tables["current"].loc["bank"], tables["old"].loc["psp"],
                          tables["previous"].loc["psp"], tables["current"].loc["psp"]], index=pd.Index(labels, name="Verfahren"))
    bank = table.loc["Bank", "Eventkosten je Fall (€)"]
    table["Ersparnis ggü. Bank (€)"] = bank - table["Eventkosten je Fall (€)"]
    table["Ersparnis ggü. Bank (%)"] = table["Ersparnis ggü. Bank (€)"] / bank * 100
    for label, row in table.iterrows():
        assert np.isclose(bills[label]["total_eur"].mean(), row["Eventkosten je Fall (€)"])
    table.to_csv(out / "comparison.csv")
    pd.DataFrame({label: bills[label]["total_eur"] for label in labels}).to_csv(out / "paired_case_costs.csv")
    lines = ["# PSP-D: Eventkosten mit Übergangssperre", "",
             "200 Fälle, Seed 0, 300 Ziehungen je Option; 700 Warmup und 1.300 Cooldown. "
             "Modellgewichte, Tarife, Wahlpunkte und Ressourcenstrategie entsprechen dem unbeschränkten 300er-Lauf. "
             "Die Sperre erlaubt ausschließlich direkt beobachtete Aktivität/Lifecycle-Paare und beobachtete Fallenden "
             "aus den 25.208 echten Fällen außerhalb der 6.301 Testfälle.", "",
             "| Verfahren | Kosten/Fall (€) ↓ | Ersparnis ggü. Bank (€) | Ersparnis (%) | Durchlaufzeit (Tage) |",
             "|---|---:|---:|---:|---:|"]
    for label, row in table.iterrows():
        lines.append(f"| {label} | {row['Eventkosten je Fall (€)']:.2f} | {row['Ersparnis ggü. Bank (€)']:+.2f} | "
                     f"{row['Ersparnis ggü. Bank (%)']:+.2f} | {row['Durchlaufzeit (Tage)']:.2f} |")
    delta = table.loc[labels[-1], "Eventkosten je Fall (€)"] - table.loc[labels[-2], "Eventkosten je Fall (€)"]
    lines += ["", f"Mit minus ohne Sperre bei 300 Ziehungen: {delta:+.2f} €/Fall. Negative Werte bedeuten niedrigere Kosten.",
              "", f"Laufende Prüfung: {audit['checked_suffixes']:,} Suffixe und {audit['checked_transitions']:,} Übergänge; "
              f"{audit['invalid_transitions']} unbeobachtete Übergänge. Auch Prefix→Options-Event wird geprüft. "
              "Die Masken erlauben EOS nur nach beobachteten Endtokens; das unveränderte Längenlimit bleibt bestehen.",
              "", "Bankkosten und Arbeitszeiten je Fall stimmen mit dem unbeschränkten Lauf überein. "
              "Alle verglichenen Verfahren haben dieselben 200 Fall-IDs. Ein Seed; kein Konfidenzintervall über mehrere Seeds."]
    (out / "auswertung.md").write_text("\n".join(lines) + "\n")
    write(out / "validation.json", {"status": "passed", "completed_at": now(), "same_200_case_ids": True,
          "bank_case_cost_components_identical": True, "no_unobserved_sampled_transitions": True,
          "same_weights_tariffs_points_and_window": True, "cost_delta_constrained_minus_unrestricted_eur": float(delta)})


def run(out):
    m = read(out / "manifest.json")
    if read(out / "status.json")["status"] != "prepared":
        raise RuntimeError("Run must be prepared")
    state = {"pid": os.getpid(), "started_at": now(), "samples_per_option": 300, "scored_cases": 200,
             "seed": 0, "transition_rules": True}

    def update(phase, **extra):
        state.update(status=phase, updated_at=now(), **extra)
        write(out / "status.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    try:
        check(m)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", MPLBACKEND="Agg", MPLCONFIGDIR=str(out / "matplotlib"))
        with (out / "online.log").open("x") as log:
            process = subprocess.Popen(m["command"], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            update("online", evaluator_pid=process.pid)
            while process.poll() is None:
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    audit_file = out / "sampling_audit.json"
                    audit = read(audit_file) if audit_file.exists() else {}
                    update("online", evaluator_pid=process.pid, decisions_completed=audit.get("decisions_completed", 0),
                           checked_transitions=audit.get("checked_transitions", 0), invalid_transitions=audit.get("invalid_transitions", 0))
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, m["command"])
        check(m)
        update("reporting")
        report(out)
        update("complete", evaluator_pid=None, report=str(out / "auswertung.md"))
    except BaseException as error:
        update("failed", error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "launch", "run", "evaluate"):
        mode.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.prepare:
        prepare(out)
        print(f"Prepared: {out}")
    elif args.launch:
        if read(out / "status.json")["status"] != "prepared" or (out / "pipeline.pid").exists():
            raise RuntimeError("Run must be prepared and not already launched")
        check(read(out / "manifest.json"))
        with (out / "pipeline.log").open("x") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run", "--out", str(out)],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        (out / "pipeline.pid").write_text(str(process.pid) + "\n")
        print(json.dumps({"pid": process.pid, "out": str(out)}, ensure_ascii=False))
    elif args.evaluate:
        evaluate(out)
    else:
        run(out)


if __name__ == "__main__":
    main()
