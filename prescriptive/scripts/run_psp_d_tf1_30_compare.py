"""Wait for the validated D-TF1 model, then run its paired 200-case evaluation."""
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
RESULTS = P / "scripts/results_100k"
TRAINING = RESULTS / "d_tf1_training_20261114"
MODEL = P / "models/psp_model_durchlaufzeit_real_alle_s60_klassen_tf1_option_encoder.pkl"
DEFAULT_OUT = RESULTS / "d_tf1_suffixe30_cases200_seed0_20261114"
REFERENCES = {
    "D bisher": RESULTS / "d_option_encoder_suffixe30_cases200_seed0_20261113/d_korrigiert",
    "E": RESULTS / "e_suffixe30_cases200_seed0_20261113/e",
    "C, 1 ms": RESULTS / "c_shared_gap_suffixe30_paired200_seed0/original_1ms",
    "OE": RESULTS / "token_oe_suffixe30_vergleich_c_seed0_20261113/oe_ohne_sperre",
}


def now():
    return datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds")


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(out):
    training = read(TRAINING / "run_manifest.json")
    assert training["model"] == str(MODEL) and training["teacher_forcing"] == 1.0
    assert training["epochs"] == 15
    if read(TRAINING / "status.json")["status"] in {"failed", "interrupted"}:
        raise RuntimeError("The training failed; inspect it before queuing evaluation")
    sources = [Path(__file__).resolve(), TRAINING / "run_manifest.json", P / "scripts/run_token_oe_ue_seed.py",
               P / "scripts/main.py", P / "estimators/psp_model.py", P / "policies/ev_policy/psp_policy.py",
               P / "evaluation/online_evaluation.py", P / "data/catalog_neue_punkte_zuteilung_random.yaml",
               ROOT / "process-simulation-engine/simulation/case_engine_lifecycle_aware.py",
               ROOT / "process-simulation-engine/resources/case_allocation.py"]
    for folder in REFERENCES.values():
        progress = read(folder / "progress.json")
        assert progress["status"] == "complete" and progress["samples_per_option"] == 30
        assert progress["seed"] == 0 and progress["scored_cases"] == 200
        assert not progress["transition_rules_expected"]
        sources += [folder / name for name in ("progress.json", "psp_seed0_row.json", "bank_seed0_row.json")]
    catalog = P / "data/catalog_neue_punkte_zuteilung_random.yaml"
    d_manifest = read(RESULTS / "d_option_encoder_suffixe30_cases200_seed0_20261113/run_manifest.json")
    assert sha(catalog) == d_manifest["sha256"][str(catalog)]
    out.mkdir(parents=True, exist_ok=False)
    (out / "catalog_snapshot.yaml").write_bytes(catalog.read_bytes())
    command = [sys.executable, "-u", str(P / "scripts/run_token_oe_ue_seed.py"),
               "--model", str(MODEL), "--seed", "0", "--samples", "30", "--scored-cases", "200",
               "--warmup", "700", "--cooldown", "1300", "--without-transition-rules", "--out", str(out / "d_tf1")]
    write(out / "run_manifest.json", {
        "prepared_at": now(), "variant": "d_tf1", "objective": "negative_cycle_days", "model": str(MODEL),
        "training_folder": str(TRAINING), "training_teacher_forcing": 1.0, "seed": 0,
        "samples_per_option": 30, "scored_cases": 200, "warmup": 700, "cooldown": 1300,
        "arrivals": 2200, "allocation": "random", "transition_rules": False, "shared_option_gap": False,
        "online_command": command, "references": {key: str(value) for key, value in REFERENCES.items()},
        "sha256": {str(path): sha(path) for path in sources}})
    write(out / "status.json", {"status": "prepared", "updated_at": now(), "model": str(MODEL),
          "seed": 0, "samples_per_option": 30, "scored_cases": 200})


def check_sources(manifest):
    for name, digest in manifest["sha256"].items():
        if sha(Path(name)) != digest:
            raise RuntimeError(f"Prepared input changed: {name}")


def compare(out):
    folders = {"D, TF 1": out / "d_tf1", **REFERENCES}
    rows, banks = {}, {}
    for label, folder in folders.items():
        progress = read(folder / "progress.json")
        assert progress["status"] == "complete"
        assert progress["samples_per_option"] == 30 and progress["seed"] == 0 and progress["scored_cases"] == 200
        rows[label] = read(folder / "psp_seed0_row.json")
        banks[label] = read(folder / "bank_seed0_row.json")
        assert rows[label]["Fälle"] == banks[label]["Fälle"] == 200
    same_banks = all(bank["_days"] == banks["D, TF 1"]["_days"] for bank in banks.values())
    bank_days = banks["D, TF 1"]["Durchlaufzeit (Tage)"]
    days = {label: row["Durchlaufzeit (Tage)"] for label, row in rows.items()}
    write(out / "vergleich.json", {"completed_at": now(), "seed": 0, "samples_per_option": 30,
          "scored_cases": 200, "bank_case_durations_identical": same_banks, "bank_days": bank_days,
          "policy_days": days, "d_tf1_minus_previous_d_days": days["D, TF 1"] - days["D bisher"],
          "d_tf1_improvement_vs_bank_percent": (bank_days - days["D, TF 1"]) / bank_days * 100})
    lines = ["# D mit Teacher Forcing 1: Online-Vergleich", "",
             "200 bewertete Fälle, 30 freie Suffixziehungen je Option, Seed 0. "
             "700 Warmup und 1.300 Cooldown, zufällige Ressourcenvergabe, vier bisherige Wahlpunkte. "
             "Option im Encoder, Optionsabstand je Suffix; Ziel: minimale mittlere Durchlaufzeit.", "",
             "| Variante | Durchlaufzeit (Tage) ↓ | Verbesserung ggü. Bank |",
             "|---|---:|---:|"]
    for label, value in days.items():
        lines.append(f"| {label} | {value:.3f} | {(bank_days-value)/bank_days*100:+.2f} % |")
    lines += [f"| Bank | {bank_days:.3f} | 0,00 % |", "",
              f"D mit TF 1 minus bisheriges D: {days['D, TF 1']-days['D bisher']:+.3f} Tage. "
              "Negative Werte bedeuten kürzere Durchlaufzeit.", "",
              f"Bank-Falldauern aller Vergleichsläufe identisch: {'ja' if same_banks else 'NEIN'}. "
              "Die Auswertung verwendet das fertige, geprüfte Modell nach 15 Epochen. "
              "Teacher Forcing 1 betrifft das Training; die Auswertung zieht die Fortsetzungen frei.", "",
              "Ein Seed mit 200 Fällen; kein Konfidenzintervall über mehrere Seeds. "
              "Einzelne Laufwerte und Falldauern stehen unter d_tf1/."]
    (out / "auswertung.md").write_text("\n".join(lines) + "\n")
    if not same_banks:
        raise RuntimeError("Bank cases differ; comparison is not paired, see vergleich.json")


def run(out):
    manifest = read(out / "run_manifest.json")
    if read(out / "status.json")["status"] != "prepared":
        raise RuntimeError("Only a prepared evaluation may be started")
    state = {"pid": os.getpid(), "started_at": now(), "model": str(MODEL),
             "seed": 0, "samples_per_option": 30, "scored_cases": 200}

    def update(phase, **extra):
        state.update(status=phase, updated_at=now(), **extra)
        write(out / "status.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    try:
        check_sources(manifest)
        while True:
            training = read(TRAINING / "status.json")
            if training["status"] in {"failed", "interrupted"}:
                raise RuntimeError(f"Training stopped: {training}")
            validation = TRAINING / "model_validation.json"
            if training["status"] in {"diagnosis", "complete"} and validation.is_file() and MODEL.is_file():
                validated = read(validation)
                assert validated["status"] == "passed"
                assert validated["all_15_logged_teacher_forcing_ratios_equal_1"]
                assert validated["same_d_parameters_except_teacher_forcing_and_checkpoint_path"]
                assert sha(MODEL) == validated["model_sha256"]
                manifest["model_sha256"] = validated["model_sha256"]
                manifest["validation_sha256"] = sha(validation)
                write(out / "run_manifest.json", manifest)
                break
            pid = training.get("pid")
            if not pid:
                raise RuntimeError("Training has no supervisor PID and is not ready")
            try:
                os.kill(pid, 0)
            except ProcessLookupError as error:
                raise RuntimeError("Training process disappeared before the model was ready") from error
            update("waiting_for_training", training_status=training["status"],
                   training_pid=pid, training_updated_at=training.get("updated_at"),
                   training_epochs_logged=training.get("epochs_logged", 0))
            time.sleep(60)
        check_sources(manifest)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        process = subprocess.Popen(manifest["online_command"], cwd=ROOT / "process-simulation-engine",
                                   env=env, stdin=subprocess.DEVNULL)
        update("online", child_pid=process.pid)
        while process.poll() is None:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                progress_file = out / "d_tf1/progress.json"
                progress = read(progress_file) if progress_file.is_file() else {}
                update("online", child_pid=process.pid, decisions_completed=progress.get("decisions_completed", 0),
                       cases_with_decisions=progress.get("cases_with_decisions", 0))
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, manifest["online_command"])
        check_sources(manifest)
        assert sha(MODEL) == manifest["model_sha256"]
        update("comparing", child_pid=None)
        compare(out)
        update("complete", report=str(out / "auswertung.md"))
    except BaseException as error:
        update("failed", error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.prepare:
        prepare(out)
        print(f"Prepared: {out}")
    elif args.launch:
        if read(out / "status.json")["status"] != "prepared" or (out / "pipeline.pid").exists():
            raise RuntimeError("Evaluation must be prepared and not already launched")
        with (out / "pipeline.log").open("x") as stream:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run", "--out", str(out)],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        (out / "pipeline.pid").write_text(str(process.pid) + "\n")
        print(json.dumps({"pid": process.pid, "out": str(out)}, ensure_ascii=False))
    else:
        run(out)


if __name__ == "__main__":
    main()
