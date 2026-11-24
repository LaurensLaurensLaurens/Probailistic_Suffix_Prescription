"""Durable, isolated 100-epoch D/TF1 training after the current real-log run."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
from zoneinfo import ZoneInfo

P = Path(__file__).resolve().parents[1]
ROOT = P.parent
PREVIOUS = P / "scripts/results_100k/d_tf1_training_20261114"
OUT = P / "scripts/results_100k/d_tf1_100_training_20261115"
TAG = "real_alle_s60_klassen_tf1_e100_option_encoder_20261115"
MODEL = P / f"models/psp_model_durchlaufzeit_{TAG}.pkl"
BEST_MODEL = MODEL.with_name(MODEL.stem + "_best_val.pkl")
CHECKPOINTS = P / f"models/psp_training_{TAG}"


def now():
    return datetime.now(ZoneInfo("Europe/Vienna")).isoformat(timespec="seconds")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sources(manifest):
    changed = [path for path, digest in manifest["source_sha256"].items() if sha(path) != digest]
    if changed:
        raise RuntimeError(f"Prepared training inputs changed: {changed}")


def dependency_ready(folder):
    state = read(folder / "status.json")
    if state["status"] in {"failed", "interrupted", "cancelled"}:
        raise RuntimeError(f"Predecessor stopped: {state}")
    if state["status"] != "complete":
        pid = state.get("pid")
        if not pid:
            raise RuntimeError("No predecessor supervisor PID")
        os.kill(pid, 0)
        return False
    validation = read(folder / "model_validation.json")
    if validation.get("status") != "passed" or not validation.get("all_15_logged_teacher_forcing_ratios_equal_1"):
        raise RuntimeError("The predecessor did not pass D/TF1 validation")
    model = Path(read(folder / "run_manifest.json")["model"])
    if sha(model) != validation["model_sha256"]:
        raise RuntimeError("Predecessor model no longer matches its validation")
    return True


def prepare(out):
    previous = read(PREVIOUS / "run_manifest.json")
    if previous["epochs"] != 15 or previous["teacher_forcing"] != 1.0:
        raise RuntimeError("Expected the authorized 15-epoch D/TF1 predecessor")
    for path in (out, MODEL, BEST_MODEL, CHECKPOINTS):
        if path.exists():
            raise FileExistsError(path)
    if read(PREVIOUS / "status.json")["status"] in {"failed", "interrupted", "cancelled"}:
        raise RuntimeError("Inspect the stopped predecessor before queuing this run")
    out.mkdir(parents=True)
    catalog = out / "training_catalog.yaml"
    catalog.write_bytes((PREVIOUS / "training_catalog.yaml").read_bytes())
    command = list(previous["training_command"])
    command[0] = sys.executable
    command[command.index("--tag")+1] = TAG
    command[command.index("--epochs")+1] = "100"
    sources = [Path(name) for name in previous["sha256"]]
    sources += [Path(__file__).resolve(), catalog, PREVIOUS / "run_manifest.json"]
    manifest = {"prepared_at": now(), "variant": "d_tf1_100_epochs", "dataset": "real_bpic17",
        "predecessor": str(PREVIOUS), "start_condition": "predecessor complete and validated, then GPU/RAM available",
        "epochs": 100, "fresh_training": True, "teacher_forcing": 1.0, "seed": 42,
        "training_cases": 20482, "validation_cases": 4726, "held_out_cases": 6301,
        "learning_rate": 1e-4, "batch_size": 256, "suffix_size": 60, "point_weight": 3,
        "option_in_encoder_after_training": True, "individual_option_gaps": True,
        "transition_rules_during_training": False, "event_conditioned_gap_class": False,
        "training_catalog": str(catalog), "model": str(MODEL), "best_model": str(BEST_MODEL),
        "checkpoints": str(CHECKPOINTS), "reference_model": previous["reference_model"],
        "training_command": command, "minimum_free_gpu_mib": 7000, "minimum_available_ram_gib": 18,
        "checkpoint_selection": "minimum Standard Validation Total; keep final epoch separately; no early stopping",
        "source_sha256": {str(path): sha(path) for path in sources}}
    write(out / "run_manifest.json", manifest)
    write(out / "status.json", {"status": "prepared", "updated_at": now(), "epochs": 100,
                               "teacher_forcing": 1.0, "model": str(MODEL)})
    (out / "auftrag.md").write_text(
        "# D mit TF 1: Anschlusslauf über 100 Epochen\n\n"
        "Neues Training von Anfang an mit 100 Epochen auf dem echten BPIC17-Log. "
        "Gleiche D-Konfiguration wie im vorangehenden 15-Epochen-Lauf: TF durchgehend 1; "
        "20.482 Trainingsfälle, 4.726 Validierungsfälle, 6.301 unveränderte Testfälle außerhalb des Trainings; "
        "Seed 42, Start-Lernrate 0,0001, Batchgröße 256, Decoderlänge 60, Wahlpunktgewicht 3. "
        "Option im Encoder nach dem Training, Optionsabstand je Suffix.\n\n"
        "Start erst nach vollständigem und geprüftem Abschluss des bisherigen Echtlog-Laufs. "
        "Der RCT-Lauf wird weiterlaufen gelassen; zusätzlich mindestens 7.000 MiB freier GPU-Speicher "
        "und 18 GiB verfügbarer Arbeitsspeicher. Eigenes Modell und eigene Checkpoints.\n\n"
        "100 Epochen werden vollständig trainiert. Zusätzlich zum letzten Stand wird die Epoche mit "
        "kleinstem Standard-Validierungsverlust gespeichert. Das ist die beste Epoche nach diesem "
        "Validierungskriterium, keine Zusage des besten Online-Outcomes. Der Trainingsalgorithmus "
        "und die bestehenden Quelldateien werden nicht geändert. Anschließend Parameter-/Ziehungsprüfung "
        "und die übliche Diagnose mit 100 Entscheidungen je Wahlpunkt und 30 Ziehungen je Option "
        "auf dem besten Validierungscheckpoint.\n")


class CheckpointRecorder:
    def __init__(self, out, checkpoints):
        self.out, self.checkpoints = out, checkpoints
        self.rows, self.best = [], float("inf")

    def record(self, trainer, result):
        standard, uncertainty = float(result[-2]), float(result[-1])
        if not math.isfinite(standard) or not math.isfinite(uncertainty):
            raise RuntimeError("Nonfinite validation loss")
        epoch = len(self.rows)+1
        row = {"epoch": epoch, "completed_at": now(), "standard_validation_loss": standard,
               "uncertainty_validation_loss": uncertainty,
               "teacher_forcing": float(trainer.teacher_forcing_ratio),
               "learning_rate": float(trainer.optimizer.param_groups[0]["lr"])}
        if standard < self.best:
            temp = self.checkpoints / "best_val.pt.tmp"
            trainer.model.save(str(temp))
            temp.replace(self.checkpoints / "best_val.pt")
            self.best = standard
            write(self.out / "best_checkpoint.json", row)
        self.rows.append(row)
        write(self.out / "epoch_history.json", self.rows)
        print("EPOCH_TRACKING", json.dumps(row), flush=True)


def worker(out):
    sys.path.insert(0, str(ROOT))
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    from prescriptive.estimators.psp_model import PSPModel  # registers the PSP source path
    from trainer.trainer import Trainer
    manifest = read(out / "run_manifest.json")
    check_sources(manifest)
    recorder = CheckpointRecorder(out, Path(manifest["checkpoints"]))
    original = Trainer.validation_epoch

    def recorded_validation(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        recorder.record(self, result)
        return result

    Trainer.validation_epoch = recorded_validation
    command = manifest["training_command"]
    sys.argv = command[2:]
    try:
        runpy.run_path(command[2], run_name="__main__")
    finally:
        Trainer.validation_epoch = original
    if len(recorder.rows) != 100 or any(row["teacher_forcing"] != 1.0 for row in recorder.rows):
        raise RuntimeError("Expected exactly 100 completed epochs with TF 1")
    # Saving an extra checkpoint has no effect on optimization, RNG or model mode.
    from prescriptive.estimators.psp_model import PSPModel
    model = PSPModel.load(manifest["model"])
    model.net = PSPModel._load_net(str(Path(manifest["checkpoints"]) / "best_val.pt"))
    model.net.eval()
    best = read(out / "best_checkpoint.json")
    model.summary["Checkpointauswahl"] = {"Kriterium": "Standard Validation Total", "Epoche": best["epoch"],
        "Validierungsverlust": best["standard_validation_loss"], "Trainierte Epochen insgesamt": 100}
    model.save(manifest["best_model"])


def validate(out):
    sys.path.insert(0, str(ROOT))
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    import pickle
    import numpy as np
    import torch
    from prescriptive.estimators.psp_model import PSPModel
    torch.set_num_threads(1)
    m = read(out / "run_manifest.json")
    reference = PSPModel.load(m["reference_model"])
    history = read(out / "epoch_history.json")
    assert [row["epoch"] for row in history] == list(range(1, 101))
    assert all(row["teacher_forcing"] == 1.0 for row in history)
    best = read(out / "best_checkpoint.json")
    assert best["standard_validation_loss"] == min(row["standard_validation_loss"] for row in history)
    ignored = {"epochs", "teacher_forcing", "checkpoints"}
    results = {}
    for label, filename in (("final", m["model"]), ("best_validation", m["best_model"])):
        model = PSPModel.load(filename)
        assert {k: v for k, v in model.params.items() if k not in ignored} == {
            k: v for k, v in reference.params.items() if k not in ignored}
        assert model.params["epochs"] == 100 and model.params["teacher_forcing"] == 1.0
        assert model.seed == reference.seed == 42 and model.test_cases == reference.test_cases
        assert len(model.test_cases) == 6301 and model.points == reference.points
        assert model.gap_classes() and not model.gap_condition_on_event() and not model.gap_tokens()
        assert model.option_in_encoder() and not model.params.get("shared_option_gap", False)
        assert not getattr(model, "transition_rules", None)
        summary = model.summary["alle Prefixe"]
        for key in ("Trainingsfälle", "Validierungsfälle", "Proben", "Proben Validierung", "Proben am Wahlpunkt",
                    "Gewicht am Wahlpunkt", "window_size"):
            assert summary[key] == reference.summary["alle Prefixe"][key], key
        assert summary["Epochen"] == 100
        probe = pickle.loads((P / "scripts/results_100k/token_oe_suffixe30_paired200_seed0/failed_prefix.pkl").read_bytes())
        model.set_threads(1)
        torch.manual_seed(123)
        drawn, elapsed = model.option_suffixes(model.points[probe["point"].name], probe["events"],
                                              probe["options"], probe["now"], 30)
        for option, suffixes in zip(probe["options"], drawn):
            assert len(suffixes) == 30
            for suffix in suffixes:
                assert suffix[0][0] == (option if "|" in option else option+"|complete")
                assert np.isfinite(suffix[0][2]) and suffix[0][2] >= 0
                assert np.isclose(suffix[0][1]-elapsed, suffix[0][2])
        results[label] = {"model": filename, "sha256": sha(filename), "training_summary": summary}
    write(out / "model_validation.json", {"status": "passed", "completed_at": now(), "best_epoch": best["epoch"],
        "all_100_epochs_tf1": True, "same_split_samples_architecture_and_other_parameters": True, "models": results})


def resources_ready(manifest):
    output = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    free = int(output.strip())
    memory = {line.split(':', 1)[0]: line.split(':', 1)[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
    ram = int(memory['MemAvailable'].split()[0])/1024**2
    return free >= manifest["minimum_free_gpu_mib"] and ram >= manifest["minimum_available_ram_gib"], free, ram


def run(out):
    m = read(out / "run_manifest.json")
    if read(out / "status.json")["status"] != "prepared":
        raise RuntimeError("Run is not prepared or was already started")
    fd = os.open(out / "supervisor.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    state = {"pid": os.getpid(), "started_at": now(), "epochs": 100, "teacher_forcing": 1.0, "model": m["model"]}

    def update(phase, **extra):
        state.update(status=phase, updated_at=now(), **extra)
        write(out / "status.json", state)
        print(json.dumps(state), flush=True)

    def command(args, phase, env):
        process = subprocess.Popen(args, cwd=ROOT / "process-simulation-engine", env=env, stdin=subprocess.DEVNULL)
        update(phase, child_pid=process.pid)
        while process.poll() is None:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                history = read(out / "epoch_history.json") if (out / "epoch_history.json").exists() else []
                update(phase, child_pid=process.pid, epochs_completed=len(history),
                       last_epoch=history[-1] if history else None)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, args)

    try:
        check_sources(m)
        while not dependency_ready(Path(m["predecessor"])):
            update("waiting_for_predecessor", predecessor_status=read(PREVIOUS / "status.json")["status"])
            time.sleep(60)
        while True:
            ready, gpu, ram = resources_ready(m)
            if ready:
                break
            update("waiting_for_resources", gpu_free_mib=gpu, ram_available_gib=round(ram, 2))
            time.sleep(60)
        check_sources(m)
        if any(Path(m[key]).exists() for key in ("model", "best_model", "checkpoints")):
            raise FileExistsError("New model/checkpoint paths already exist")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PSP_CATALOG=m["training_catalog"],
                   MPLBACKEND="Agg", MPLCONFIGDIR=str(out / "matplotlib"))
        subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA GPU required'"], env=env, check=True)
        command([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--out", str(out)], "training", env)
        check_sources(m)
        cpu = dict(env, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        command([sys.executable, "-u", str(Path(__file__).resolve()), "--validate", "--out", str(out)], "validating", cpu)
        command([sys.executable, "-u", str(P / "scripts/psp_diagnose.py"), "--model", m["best_model"],
                 "--out", str(out / "diagnose"), "--decisions", "100", "--samples", "30", "--workers", "4"], "diagnosis", cpu)
        check_sources(m)
        best = read(out / "best_checkpoint.json")
        (out / "auswertung.md").write_text(
            f"# D mit TF 1: 100 Epochen abgeschlossen\n\nBestes Standard-Validierungsergebnis: Epoche {best['epoch']}, "
            f"Verlust {best['standard_validation_loss']:.6g}.\n\nBestes Modell: `{m['best_model']}`. "
            f"Letzte Epoche: `{m['model']}`. Beide wurden geprüft; Diagnose auf dem besten Validierungscheckpoint "
            "unter `diagnose/`. Ein Online-Benchmark wurde nicht automatisch gestartet.\n")
        update("complete", child_pid=None, epochs_completed=100, best_epoch=best["epoch"])
    except BaseException as error:
        update("failed", error=repr(error))
        raise


def self_check(out):
    from types import SimpleNamespace
    sys.path.insert(0, str(ROOT))
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    from prescriptive.estimators.psp_model import PSPModel
    from trainer.trainer import Trainer
    assert callable(Trainer.validation_epoch) and callable(PSPModel._load_net)
    with tempfile.TemporaryDirectory(prefix="d100_check_") as name:
        folder = Path(name)
        class FakeModel:
            def save(self, path):
                Path(path).write_text(str(len(recorder.rows)+1))
        trainer = SimpleNamespace(model=FakeModel(), teacher_forcing_ratio=1.0,
                                  optimizer=SimpleNamespace(param_groups=[{"lr": 1e-4}]))
        recorder = CheckpointRecorder(folder, folder)
        for loss in (3., 2., 4.):
            recorder.record(trainer, (None, None, None, None, loss, loss+1))
        assert read(folder / "best_checkpoint.json")["epoch"] == 2
        assert (folder / "best_val.pt").read_text() == "2"
        assert len(read(folder / "epoch_history.json")) == 3
        write(folder / "status.json", {"status": "training", "pid": os.getpid()})
        assert not dependency_ready(folder)
        fake_model = folder / "model.pkl"
        fake_model.write_text("completed model")
        write(folder / "run_manifest.json", {"model": str(fake_model)})
        write(folder / "model_validation.json", {"status": "passed", "all_15_logged_teacher_forcing_ratios_equal_1": True,
                                                   "model_sha256": sha(fake_model)})
        write(folder / "status.json", {"status": "complete"})
        assert dependency_ready(folder)
        write(folder / "status.json", {"status": "failed"})
        try:
            dependency_ready(folder)
        except RuntimeError:
            pass
        else:
            raise AssertionError("A failed predecessor must block training")
    manifest = read(out / "run_manifest.json")
    previous = read(PREVIOUS / "run_manifest.json")
    expected = list(previous["training_command"])
    expected[0] = sys.executable
    expected[expected.index("--tag")+1] = TAG
    expected[expected.index("--epochs")+1] = "100"
    assert manifest["training_command"] == expected
    assert sha(manifest["training_catalog"]) == sha(PREVIOUS / "training_catalog.yaml")
    check_sources(manifest)
    write(out / "scheduler_validation.json", {"status": "passed", "checked_at": now(),
         "same_training_command_except_epoch_count_and_new_tag": True,
         "best_checkpoint_preserved_across_worse_later_epochs": True,
         "worker_model_and_trainer_imports_available": True,
         "waits_for_predecessor_and_rejects_failed_predecessor": True})
    print("SCHEDULER VALIDATION PASSED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    modes = parser.add_mutually_exclusive_group(required=True)
    for option in ("prepare", "launch", "run", "worker", "validate", "self-check"):
        modes.add_argument("--"+option, action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.prepare:
        prepare(out)
    elif args.self_check:
        self_check(out)
    elif args.launch:
        if read(out / "status.json")["status"] != "prepared" or (out / "pipeline.pid").exists():
            raise RuntimeError("Already launched or not prepared")
        if read(out / "scheduler_validation.json")["status"] != "passed":
            raise RuntimeError("Validate the queue first")
        with (out / "pipeline.log").open("x") as stream:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run", "--out", str(out)],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        (out / "pipeline.pid").write_text(str(process.pid)+"\n")
        print(json.dumps({"pid": process.pid, "out": str(out)}), flush=True)
    elif args.run:
        run(out)
    elif args.worker:
        worker(out)
    else:
        validate(out)


if __name__ == "__main__":
    main()
