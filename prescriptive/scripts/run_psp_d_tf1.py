"""Train the corrected D variant with constant teacher forcing 1, in isolation."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

P = Path(__file__).resolve().parents[1]
ROOT = P.parent
TAG = "real_alle_s60_klassen_tf1_option_encoder"
MODEL = P / f"models/psp_model_durchlaufzeit_{TAG}.pkl"
CHECKPOINTS = P / f"models/psp_training_{TAG}"
REFERENCE = P / "models/psp_model_durchlaufzeit_real_alle_s60_klassen_option_encoder.pkl"
DEFAULT_OUT = P / "scripts/results_100k/d_tf1_training_20261114"


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
    if MODEL.exists() or CHECKPOINTS.exists():
        raise FileExistsError("Target model or checkpoints already exist; inspect before restarting")
    if not REFERENCE.is_file():
        raise FileNotFoundError(REFERENCE)
    out.mkdir(parents=True, exist_ok=False)
    catalog = out / "training_catalog.yaml"
    catalog.write_bytes((P / "data/catalog_neue_punkte.yaml").read_bytes())
    command = [sys.executable, "-u", str(P / "scripts/train_psp_real.py"),
               "--tag", TAG, "--cases", "0", "--epochs", "15", "--learning-rate", "1e-4",
               "--point-weight", "3", "--catalog-points", "--suffix-size", "60",
               "--statics-without-future", "--time-from", "gaps", "--gap-classes",
               "--teacher-forcing", "1.0", "--seed", "42", "--option-in-encoder-after-training"]
    sources = [Path(__file__).resolve(), P / "scripts/train_psp_real.py", P / "scripts/psp_diagnose.py",
               P / "estimators/psp_model.py", P / "helpers/training.py", P / "helpers/datasets.py",
               ROOT / "probabilistic_suffix_prediction_U-ED-LSTM/src/trainer/trainer.py",
               ROOT / "probabilistic_suffix_prediction_U-ED-LSTM/src/model/dropout_uncertainty_enc_dec_LSTM/dropout_uncertainty_model.py",
               ROOT / "probabilistic_suffix_prediction_U-ED-LSTM/src/model/dropout_uncertainty_enc_dec_LSTM/dropout_uncertainty_decoder.py",
               catalog, REFERENCE]
    manifest = {"prepared_at": now(), "variant": "d_tf1", "model": str(MODEL),
                "reference_model": str(REFERENCE), "checkpoints": str(CHECKPOINTS),
                "training_seed": 42, "epochs": 15, "teacher_forcing": 1.0, "suffix_size": 60,
                "learning_rate": 1e-4, "point_weight": 3, "training_cases": "all outside the same 6301 held-out cases",
                "options": "encoder, enabled after training to preserve D sample weighting",
                "option_gap": "drawn separately for every suffix", "event_conditioned_time_class": False,
                "transition_rules": False, "training_catalog": str(catalog), "training_command": command,
                "validation_command": [sys.executable, "-u", str(Path(__file__).resolve()), "--validate", "--out", str(out)],
                "diagnosis_command": [sys.executable, "-u", str(P / "scripts/psp_diagnose.py"),
                    "--model", str(MODEL), "--out", str(out / "diagnose"), "--decisions", "100", "--samples", "30", "--workers", "4"],
                "sha256": {str(path): sha(path) for path in sources}}
    write(out / "run_manifest.json", manifest)
    write(out / "status.json", {"status": "prepared", "updated_at": now(), "epochs": 15,
                                "teacher_forcing": 1.0, "model": str(MODEL)})
    (out / "auftrag.md").write_text(
        "# D mit durchgehendem Teacher Forcing 1\n\n"
        "Neues Training der korrigierten D-Variante: getrennte Event- und Zeitklassen-Köpfe, "
        "Option im Encoder, Optionsabstand je Suffixziehung. Teacher Forcing bleibt in allen 15 Epochen bei 1,0.\n\n"
        "Alle 25.208 Fälle außerhalb des bisherigen Testsplits werden für Training/Validierung verwendet "
        "(20.482 Training, 4.726 Validierung; 6.301 Testfälle ausgeschlossen). "
        "Seed 42, Decoderlänge 60, Lernrate 0,0001, Wahlpunktgewicht 3, Batchgröße 256. "
        "Statische Merkmale ohne Zukunftswissen; Zeit aus der Summe der Eventabstände.\n\n"
        "Der Optionsmodus wird nach dem Training gesetzt, damit die Probengewichtung der bisherigen D-Variante entspricht. "
        "Danach werden Parameter, Fall-Split, Teacher-Forcing-Protokoll und Optionsziehung geprüft. "
        "Es folgt die übliche Diagnose auf echten Testentscheidungen: 100 je Wahlpunkt, 30 Ziehungen je Option.\n\n"
        f"Neues Modell: `{MODEL}`. Status: `status.json`; Fortschritt: `pipeline.log`; "
        "nach Abschluss: `model_validation.json`, `diagnose/diagnose.csv`, `auswertung.md`.\n")


def check_sources(manifest):
    for name, digest in manifest["sha256"].items():
        if sha(Path(name)) != digest:
            raise RuntimeError(f"Prepared source or reference changed: {name}")


def validate(out):
    sys.path.insert(0, str(ROOT))
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    import numpy as np
    import pickle
    import torch
    from prescriptive.estimators.psp_model import PSPModel

    torch.set_num_threads(1)
    reference, model = PSPModel.load(REFERENCE), PSPModel.load(MODEL)
    ignored = {"teacher_forcing", "checkpoints"}
    assert {k: v for k, v in model.params.items() if k not in ignored} == {
        k: v for k, v in reference.params.items() if k not in ignored}
    assert model.params["teacher_forcing"] == 1.0
    assert model.seed == reference.seed == 42
    assert model.test_cases == reference.test_cases and len(model.test_cases) == 6301
    assert model.points == reference.points
    assert model.gap_classes() and not model.gap_condition_on_event() and not model.gap_tokens()
    assert getattr(model.net, "categorical_conditioning", None) is None
    assert model.option_in_encoder() and not model.option_in_decoder()
    assert not model.params.get("shared_option_gap", False)
    assert getattr(model, "option_gap_distributions", None) is None
    assert getattr(model, "transition_rules", None) is None
    summary, previous = model.summary["alle Prefixe"], reference.summary["alle Prefixe"]
    for key in ("Trainingsfälle", "Validierungsfälle", "Proben", "Proben Validierung", "Proben am Wahlpunkt", "Gewicht am Wahlpunkt", "Epochen", "window_size"):
        assert summary[key] == previous[key], key
    epochs = re.findall(r"Epoch \[(\d+)/15\].*Teacher forcing ratio: ([0-9.eE+-]+)", (out / "pipeline.log").read_text())
    assert [int(epoch) for epoch, _ in epochs] == list(range(1, 16))
    assert all(float(ratio) == 1.0 for _, ratio in epochs)
    fixture = P / "scripts/results_100k/token_oe_suffixe30_paired200_seed0/failed_prefix.pkl"
    probe = pickle.loads(fixture.read_bytes())
    torch.manual_seed(123)
    drawn, elapsed = model.option_suffixes_many(model.points[probe["point"].name], [probe["events"]],
                                               probe["options"], probe["now"], 30)
    first_gaps = {}
    for option, cases in zip(probe["options"], drawn):
        assert len(cases[0]) == 30
        for suffix in cases[0]:
            assert suffix[0][0] == (option if "|" in option else option + "|complete")
            assert np.isfinite(suffix[0][2]) and suffix[0][2] >= 0
            assert np.isclose(suffix[0][1] - elapsed[0], suffix[0][2])
        first_gaps[str(option)] = [suffix[0][2] for suffix in cases[0]]
    write(out / "model_validation.json", {"status": "passed", "completed_at": now(),
          "model_sha256": sha(MODEL), "same_d_parameters_except_teacher_forcing_and_checkpoint_path": True,
          "same_sample_counts_and_test_cases": True, "same_choice_points": True,
          "all_15_logged_teacher_forcing_ratios_equal_1": True, "separate_event_and_gap_class_heads": True,
          "option_in_encoder_with_individual_suffix_gaps": True, "first_gaps_seconds_on_probe": first_gaps,
          "training_summary": summary})


def run(out):
    manifest = read(out / "run_manifest.json")
    if read(out / "status.json")["status"] != "prepared":
        raise RuntimeError("Only a prepared training can be started")
    state = {"pid": os.getpid(), "started_at": now(), "epochs": 15, "teacher_forcing": 1.0,
             "model": str(MODEL)}

    def update(phase, **extra):
        state.update(status=phase, updated_at=now(), **extra)
        write(out / "status.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    def command(args, phase, env):
        process = subprocess.Popen(args, cwd=ROOT / "process-simulation-engine", env=env,
                                   stdin=subprocess.DEVNULL)
        started = time.monotonic()
        update(phase, child_pid=process.pid)
        while process.poll() is None:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                epochs = re.findall(r"Epoch \[(\d+)/15\].*Teacher forcing ratio: ([0-9.eE+-]+)",
                                    (out / "pipeline.log").read_text())
                extra = {"child_pid": process.pid, "phase_elapsed_seconds": round(time.monotonic() - started),
                         "epochs_logged": max([int(epoch) for epoch, _ in epochs], default=0)}
                if epochs:
                    extra["last_logged_teacher_forcing"] = float(epochs[-1][1])
                update(phase, **extra)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, args)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PSP_CATALOG=manifest["training_catalog"],
               MPLBACKEND="Agg", MPLCONFIGDIR=str(out / "matplotlib"))
    try:
        check_sources(manifest)
        update("checking_gpu")
        subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA GPU required'; print(torch.cuda.get_device_name(0), flush=True)"],
                       env=env, check=True)
        while True:
            query = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
            free = int(query.splitlines()[0].strip())
            if free >= 7000:
                break
            update("waiting_for_gpu", free_mib=free, required_mib=7000)
            time.sleep(60)
        command(manifest["training_command"], "training", env)
        check_sources(manifest)
        cpu = dict(env, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        command(manifest["validation_command"], "validating_model", cpu)
        command(manifest["diagnosis_command"], "diagnosis", cpu)
        check_sources(manifest)
        validation = read(out / "model_validation.json")
        assert validation["status"] == "passed"
        (out / "auswertung.md").write_text(
            "# D mit Teacher Forcing 1: Training abgeschlossen\n\n"
            "15 Epochen; in jeder Epoche Teacher Forcing 1,0. Trainingsparameter, Probenanzahlen, "
            "Testfälle und Wahlpunkte entsprechen der korrigierten D-Referenz. "
            "Option im Encoder; individueller Optionsabstand je Suffix.\n\n"
            f"Modell: `{MODEL}`. Modell- und Ziehungsprüfung: `model_validation.json`. "
            "Diagnose auf echten Testentscheidungen: `diagnose/diagnose.csv`. "
            "Ein neuer Online-Vergleich ist noch nicht Bestandteil dieses Trainingslaufs.\n")
        update("complete", epochs_logged=15, report=str(out / "auswertung.md"), child_pid=None)
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
    mode.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.prepare:
        prepare(out)
        print(f"Prepared: {out}")
    elif args.launch:
        if read(out / "status.json")["status"] != "prepared" or (out / "pipeline.pid").exists():
            raise RuntimeError("Run must be prepared and not already launched")
        with (out / "pipeline.log").open("x") as stream:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run", "--out", str(out)],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        (out / "pipeline.pid").write_text(str(process.pid) + "\n")
        print(json.dumps({"pid": process.pid, "out": str(out)}, ensure_ascii=False))
    elif args.validate:
        validate(out)
    else:
        run(out)


if __name__ == "__main__":
    main()
