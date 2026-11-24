"""Fresh OE/TF1 training: the stopped D-100 settings with joint event/time tokens."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prescriptive.scripts import queue_psp_d_tf1_100 as common

P = ROOT / "prescriptive"
D_RUN = P / "scripts/results_100k/d_tf1_100_training_20261115"
OUT = P / "scripts/results_100k/oe_tf1_100_training_20261115"
TAG = "real_alle_s60_token_oe_tf1_e100_20261115"
MODEL = P / f"models/psp_model_durchlaufzeit_{TAG}.pkl"
BEST_MODEL = MODEL.with_name(MODEL.stem+"_best_val.pkl")
CHECKPOINTS = P / f"models/psp_training_{TAG}"
OE_REFERENCE = P / "models/psp_model_durchlaufzeit_real_alle_s60_token_oe.pkl"
now, read, write, sha, check_sources = common.now, common.read, common.write, common.sha, common.check_sources


def expected_command():
    command = list(read(D_RUN / "run_manifest.json")["training_command"])
    command[0] = sys.executable
    command[command.index("--tag")+1] = TAG
    command[command.index("--gap-classes")] = "--gap-tokens"
    return command


def prepare(out):
    reference = read(D_RUN / "run_manifest.json")
    assert reference["epochs"] == 100 and reference["teacher_forcing"] == 1.0
    for path in (out, MODEL, BEST_MODEL, CHECKPOINTS):
        if path.exists():
            raise FileExistsError(path)
    out.mkdir(parents=True)
    catalog = out / "training_catalog.yaml"
    catalog.write_bytes(Path(reference["training_catalog"]).read_bytes())
    sources = [Path(__file__).resolve(), Path(common.__file__).resolve(),
        P / "scripts/train_psp_real.py", P / "scripts/psp_diagnose.py", P / "estimators/psp_model.py",
        P / "helpers/datasets.py", P / "helpers/training.py",
        ROOT / "probabilistic_suffix_prediction_U-ED-LSTM/src/trainer/trainer.py",
        ROOT / "probabilistic_suffix_prediction_U-ED-LSTM/src/model/dropout_uncertainty_enc_dec_LSTM/dropout_uncertainty_model.py",
        ROOT / "probabilistic_suffix_prediction_U-ED-LSTM/src/model/dropout_uncertainty_enc_dec_LSTM/dropout_uncertainty_decoder.py",
        D_RUN / "run_manifest.json", Path(reference["training_catalog"]), Path(reference["reference_model"]),
        OE_REFERENCE, catalog]
    manifest = {"prepared_at": now(), "variant": "oe_tf1_100_epochs", "dataset": "real_bpic17",
        "settings_reference": str(D_RUN), "epochs": 100, "fresh_training": True, "teacher_forcing": 1.0,
        "seed": 42, "training_cases": 20482, "validation_cases": 4726, "held_out_cases": 6301,
        "learning_rate": 1e-4, "batch_size": 256, "suffix_size": 60, "point_weight": 3,
        "gap_tokens": True, "gap_classes": False, "event_conditioned_gap_class": False,
        "option_in_encoder_after_training": True, "individual_option_gaps": True,
        "transition_rules_during_training": False, "training_catalog": str(catalog),
        "model": str(MODEL), "best_model": str(BEST_MODEL), "checkpoints": str(CHECKPOINTS),
        "reference_model": reference["reference_model"], "architecture_reference": str(OE_REFERENCE),
        "training_command": expected_command(), "minimum_free_gpu_mib": 7000, "minimum_available_ram_gib": 18,
        "start_condition": "start now when GPU and RAM are available; no predecessor dependency",
        "checkpoint_selection": "minimum Standard Validation Total, plus final epoch; all 100 epochs trained",
        "sample_weighting": "same as stopped D-100: enable encoder option mode after fit, unlike older OE",
        "source_sha256": {str(path): sha(path) for path in sources}}
    write(out / "run_manifest.json", manifest)
    write(out / "status.json", {"status": "prepared", "updated_at": now(), "epochs": 100,
                                "teacher_forcing": 1.0, "model": str(MODEL)})
    (out / "auftrag.md").write_text(
        "# OE: 100 Epochen mit durchgehend TF 1\n\n"
        "Neues Training von Anfang an auf dem echten BPIC17-Log. Einstellungen wie im gestoppten "
        "D-Lauf: 100 Epochen, Teacher Forcing 1, Start-Lernrate 0,0001, Batch 256, Decoderlänge 60, "
        "Wahlpunktgewicht 3, Seed 42, 20.482 Trainingsfälle, 4.726 Validierungsfälle und dieselben "
        "6.301 ausgeschlossenen Testfälle. Statische Merkmale ohne Zukunftswissen, Zeit aus Eventabständen.\n\n"
        "Architekturänderung: gemeinsames Token für Aktivität, Lifecycle und Zeitklasse (OE), statt "
        "getrennter D-Köpfe. Option im Encoder nach dem Training; individueller Optionsabstand je Suffix. "
        "Die Trainingsproben werden wie bei D gewichtet. Das ältere OE gewichtete zusätzliche "
        "Entscheidungs-Prefixe; diese zusätzliche Änderung wird hier nicht übernommen.\n\n"
        "Start jetzt bei ausreichenden Ressourcen. D und RCT werden nicht neu gestartet. "
        "Eigenes Modell und eigene Checkpoints. Alle 100 Epochen laufen; beste Standard-Validierungsepoche "
        "und letzter Stand werden getrennt gespeichert. Anschließend Prüfung beider Modelle und "
        "die übliche Diagnose des besten Validierungsstands mit 100 Entscheidungen je Wahlpunkt und "
        "30 Ziehungen je Option. Sitzungsregeln und Simulatorcode werden mit diesem Training nicht geändert.\n")


def validate(out):
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    import pickle
    import numpy as np
    import torch
    from prescriptive.estimators.psp_model import PSPModel
    torch.set_num_threads(1)
    m = read(out / "run_manifest.json")
    reference = PSPModel.load(m["reference_model"])
    expected = dict(reference.params)
    expected.pop("gap_classes")
    expected["gap_tokens"] = True
    ignored = {"epochs", "teacher_forcing", "checkpoints"}
    history = read(out / "epoch_history.json")
    assert [r["epoch"] for r in history] == list(range(1, 101))
    assert all(r["teacher_forcing"] == 1.0 for r in history)
    best = read(out / "best_checkpoint.json")
    assert best["standard_validation_loss"] == min(r["standard_validation_loss"] for r in history)
    results = {}
    for label, filename in (("final", m["model"]), ("best_validation", m["best_model"])):
        model = PSPModel.load(filename)
        assert {k: v for k, v in model.params.items() if k not in ignored} == {
            k: v for k, v in expected.items() if k not in ignored}
        assert model.params["epochs"] == 100 and model.params["teacher_forcing"] == 1.0
        assert model.gap_tokens() and not model.gap_classes() and not model.gap_condition_on_event()
        assert model.option_in_encoder() and not model.option_in_decoder()
        assert not model.params.get("shared_option_gap", False)
        assert not getattr(model, "transition_rules", None)
        assert model.seed == reference.seed == 42 and model.test_cases == reference.test_cases
        assert len(model.test_cases) == 6301 and model.points == reference.points
        summary = model.summary["alle Prefixe"]
        for key in ("Trainingsfälle", "Validierungsfälle", "Proben", "Proben Validierung",
                    "Proben am Wahlpunkt", "Gewicht am Wahlpunkt", "window_size"):
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
        "all_100_epochs_tf1": True, "joint_event_lifecycle_time_class_tokens": True,
        "same_split_and_training_sample_weighting_as_d": True, "models": results})


def self_check(out):
    from types import SimpleNamespace
    m = read(out / "run_manifest.json")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PSP_CATALOG"] = m["training_catalog"]
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    import torch
    from prescriptive.estimators.psp_model import PSPModel
    from trainer.trainer import Trainer
    torch.set_num_threads(1)
    assert callable(Trainer.validation_epoch)
    assert m["training_command"] == expected_command()
    assert sha(m["training_catalog"]) == sha(read(D_RUN / "run_manifest.json")["training_catalog"])
    reference = PSPModel.load(m["reference_model"])
    params = dict(reference.params)
    params.pop("gap_classes")
    params.pop("option_as")
    params.update(epochs=100, teacher_forcing=1.0, gap_tokens=True, checkpoints=m["checkpoints"])
    configured = PSPModel("bpic17", list(reference.points.values()), seed=42, params=params)
    assert configured.gap_tokens() and not configured.gap_classes()
    assert not configured.option_in_encoder()  # set only after fit to preserve D sample weighting
    assert configured.params["teacher_forcing"] == 1.0 and configured.params["epochs"] == 100
    architecture = PSPModel.load(m["architecture_reference"])
    with tempfile.TemporaryDirectory(prefix="oe100_check_") as name:
        folder = Path(name)
        recorder = common.CheckpointRecorder(folder, folder)
        trainer = SimpleNamespace(model=architecture.net, teacher_forcing_ratio=1.0,
                                  optimizer=SimpleNamespace(param_groups=[{"lr": 1e-4}]))
        parameter = next(architecture.net.parameters())
        for epoch, loss in enumerate((3., 2., 4.), start=1):
            with torch.no_grad():
                parameter.reshape(-1)[0] = float(epoch)
            recorder.record(trainer, (None, None, None, None, loss, loss+1))
        restored = PSPModel._load_net(str(folder / "best_val.pt"))
        assert float(next(restored.parameters()).reshape(-1)[0]) == 2.
        assert read(folder / "best_checkpoint.json")["epoch"] == 2
        assert len(read(folder / "epoch_history.json")) == 3
    check_sources(m)
    write(out / "scheduler_validation.json", {"status": "passed", "checked_at": now(),
        "same_d100_command_except_new_tag_and_joint_token_flag": True,
        "worker_model_and_trainer_imports_available": True,
        "oe_best_checkpoint_restorable_and_kept_across_worse_later_epoch": True,
        "option_in_encoder_only_after_fit_preserves_d_weighting": True})
    print("OE TRAINING PREFLIGHT PASSED", flush=True)


def run(out):
    import time
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
        while True:
            ready, gpu, ram = common.resources_ready(m)
            if ready:
                break
            update("waiting_for_resources", gpu_free_mib=gpu, ram_available_gib=round(ram, 2))
            time.sleep(60)
        if any(Path(m[key]).exists() for key in ("model", "best_model", "checkpoints")):
            raise FileExistsError("New model/checkpoint paths already exist")
        check_sources(m)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PSP_CATALOG=m["training_catalog"],
                   MPLBACKEND="Agg", MPLCONFIGDIR=str(out / "matplotlib"))
        subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA GPU required'; print('GPU:', torch.cuda.get_device_name(0), flush=True)"], env=env, check=True)
        command([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--out", str(out)], "training", env)
        check_sources(m)
        cpu = dict(env, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        command([sys.executable, "-u", str(Path(__file__).resolve()), "--validate", "--out", str(out)], "validating", cpu)
        command([sys.executable, "-u", str(P / "scripts/psp_diagnose.py"), "--model", m["best_model"],
                 "--out", str(out / "diagnose"), "--decisions", "100", "--samples", "30", "--workers", "4"], "diagnosis", cpu)
        check_sources(m)
        best = read(out / "best_checkpoint.json")
        (out / "auswertung.md").write_text(
            f"# OE mit TF 1: 100 Epochen abgeschlossen\n\nBestes Standard-Validierungsergebnis: "
            f"Epoche {best['epoch']}, Verlust {best['standard_validation_loss']:.6g}.\n\n"
            f"Bestes Modell: `{m['best_model']}`. Letzte Epoche: `{m['model']}`. "
            "Beide geprüft; Diagnose auf dem besten Validierungscheckpoint unter `diagnose/`. "
            "Ein Online-Benchmark wurde nicht automatisch gestartet.\n")
        update("complete", child_pid=None, epochs_completed=100, best_epoch=best["epoch"])
    except BaseException as error:
        update("failed", error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare", "self-check", "launch", "run", "worker", "validate"):
        modes.add_argument("--"+mode, action="store_true")
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
            raise RuntimeError("Validate the launcher first")
        with (out / "pipeline.log").open("x") as stream:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run", "--out", str(out)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        (out / "pipeline.pid").write_text(str(process.pid)+"\n")
        print(json.dumps({"pid": process.pid, "out": str(out)}), flush=True)
    elif args.run:
        run(out)
    elif args.worker:
        common.worker(out)
    else:
        validate(out)


if __name__ == "__main__":
    main()
