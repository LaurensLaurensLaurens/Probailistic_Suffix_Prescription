"""Train D-TF1 on fresh RCT data at the same four intervention points.

Reuse helpers.rct.simulate_rct and helpers.training.train_estimators. Keep the
real-log D training untouched, with separate catalog, data, model and logs.
"""
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
from types import SimpleNamespace
from zoneinfo import ZoneInfo

P = Path(__file__).resolve().parents[1]
ROOT = P.parent
TAG = "rct_s60_klassen_tf1_option_encoder_20261114"
MODEL = P / f"models/psp_model_durchlaufzeit_{TAG}.pkl"
CHECKPOINTS = P / f"models/psp_training_{TAG}"
REFERENCE = P / "models/psp_model_durchlaufzeit_real_alle_s60_klassen_option_encoder.pkl"
REAL_RUN = P / "scripts/results_100k/d_tf1_training_20261114"
DEFAULT_OUT = P / "scripts/results_100k/d_rct_tf1_training_20261114"
POINTS = ["angebot_nachlegen", "annehmen_bei_konzept", "annehmen_nach_vervollstaendigung", "handle_leads"]
CASE = "case:concept:name"


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


def canonical(value):
    return json.dumps(value, sort_keys=True)


def add_paths():
    sys.path.insert(0, str(ROOT))
    from prescriptive.helpers.datasets import add_paths as initialize
    initialize()


def prepare(out):
    if MODEL.exists() or CHECKPOINTS.exists():
        raise FileExistsError("RCT target model or checkpoints already exist")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    add_paths()
    from prescriptive.estimators.psp_model import PSPModel
    reference = PSPModel.load(REFERENCE)
    assert set(reference.points) == set(POINTS)
    assert reference.gap_classes() and not reference.gap_condition_on_event() and not reference.gap_tokens()
    params = dict(reference.params, teacher_forcing=1.0, checkpoints=str(CHECKPOINTS))
    assert params.pop("option_as") == "encoder"
    assert params["train_cases"] == 0 and params["epochs"] == 15 and params["learning_rate"] == 1e-4
    assert params["suffix_size"] == 60 and params["point_weight"] == 3
    real = read(REAL_RUN / "run_manifest.json")
    assert real["teacher_forcing"] == 1.0 and real["epochs"] == 15
    out.mkdir(parents=True, exist_ok=False)
    catalog = out / "catalog.yaml"
    catalog.write_bytes((REAL_RUN / "training_catalog.yaml").read_bytes())
    sources = [Path(__file__).resolve(), REFERENCE, catalog, REAL_RUN / "run_manifest.json",
               P / "helpers/rct.py", P / "helpers/training.py", P / "helpers/datasets.py",
               P / "estimators/psp_model.py", P / "policies/rct_policy.py", P / "enriched_log.py",
               ROOT / "process-simulation-engine/simulation/dataset_simulator.py",
               ROOT / "process-simulation-engine/simulation/case_engine_lifecycle_aware.py",
               ROOT / "process-simulation-engine/resources/case_allocation.py",
               ROOT / "probabilistic_suffix_prediction_U-ED-LSTM/src/trainer/trainer.py",
               ROOT / "probabilistic_suffix_prediction_U-ED-LSTM/src/event_log_loader/new_event_log_loader_v2.py",
               ROOT / "probabilistic_suffix_prediction_U-ED-LSTM/src/model/dropout_uncertainty_enc_dec_LSTM/dropout_uncertainty_model.py",
               ROOT / "probabilistic_suffix_prediction_U-ED-LSTM/src/model/dropout_uncertainty_enc_dec_LSTM/dropout_uncertainty_decoder.py"]
    simulator = ROOT / "process-simulation-engine/models/next_event/bpic17"
    if not simulator.is_dir():
        raise FileNotFoundError("The simulator must already be trained")
    sources += [path for path in simulator.rglob("*") if path.is_file()]
    manifest = {"prepared_at": now(), "variant": "d_rct_tf1", "model": str(MODEL), "reference_model": str(REFERENCE),
                "catalog": str(catalog), "training_seed": 42, "epochs": 15, "teacher_forcing": 1.0,
                "learning_rate": 1e-4, "suffix_size": 60, "point_weight": 3, "params_before_fit": params,
                "points": POINTS, "rct": {"cases_per_run": 3151, "runs": 10, "seed": 42, "delta": 0.5, "workers": 2},
                "total_selected_cases": 31509, "expected_training_cases": 20482, "expected_validation_cases": 4726,
                "expected_test_cases": 6301, "test_seed": 17, "case_selection_seed": 42,
                "option_in_encoder_after_training": True, "option_gap": "individual per suffix",
                "randomization": "same historical protocol: per-case bank probability delta=0.5, otherwise random options at all four points",
                "source_sha256": {str(path): sha(path) for path in sources}}
    for stage in ("generate-data", "train", "validate"):
        manifest[stage + "_command"] = [sys.executable, "-u", str(Path(__file__).resolve()), "--" + stage, "--out", str(out)]
    write(out / "manifest.json", manifest)
    write(out / "status.json", {"status": "prepared", "updated_at": now(), "model": str(MODEL)})
    (out / "auftrag.md").write_text(
        "# D mit TF 1 auf RCT-Daten\n\n"
        "Zusätzlicher Trainingslauf mit denselben D-TF1-Einstellungen wie beim echten Log: "
        "15 Epochen, Teacher Forcing durchgehend 1,0, Lernrate 0,0001, Decoderlänge 60, Batchgröße 256, "
        "Wahlpunktgewicht 3, Seed 42, statische Merkmale ohne Zukunftswissen. "
        "Event- und Zeitklasse getrennt; Optionsmodus Encoder erst nach fit setzen; Abstand je Suffix.\n\n"
        "Neue RCT-Daten werden mit dem vorhandenen Simulator an angebot_nachlegen, annehmen_bei_konzept, "
        "annehmen_nach_vervollstaendigung und handle_leads erzeugt. Katalog und Ressourcenstrategie "
        "entsprechen der eingefrorenen Trainingskonfiguration des echten D-TF1. "
        "Wie in der früheren RCT-Pipeline bleibt ein Fall mit Wahrscheinlichkeit 0,5 bei der Bank; "
        "sonst werden seine Optionen an den vier Punkten randomisiert.\n\n"
        "10 unabhängige Simulationsläufe zu je 3.151 Fällen ergeben 31.510 Fälle. Davon werden mit Seed 42 "
        "31.509 ausgewählt, damit der Umfang dem echten Log entspricht: "
        "20.482 Training, 4.726 Validierung, 6.301 Test. Der Split erfolgt vollständig nach Fällen.\n\n"
        "RCT-Erzeugung läuft auf der CPU; danach startet das GPU-Training bei ausreichendem Speicher automatisch. "
        "Vorhandene Standardmodelle, Kataloge und der laufende Echtlog-Trainingslauf werden nicht überschrieben.\n\n"
        "Status: status.json; Protokoll: pipeline.log; Datenprüfung: data_validation.json; "
        f"Modell: {MODEL}; Abschlussprüfung: model_validation.json.\n")


def check_sources(manifest):
    for name, digest in manifest["source_sha256"].items():
        if sha(Path(name)) != digest:
            raise RuntimeError(f"Prepared source/reference changed: {name}")


def generate_data(out):
    add_paths()
    import numpy as np
    import interventions
    from prescriptive.helpers.datasets import load_dataset
    from prescriptive.helpers.rct import rct_folder, simulate_rct
    from prescriptive.helpers.training import sample_cases, test_cases
    from preprocessing import preprocess
    from simulation.dataset_simulator import Simulator, is_trained, model_folder
    m = read(out / "manifest.json")
    assert is_trained(model_folder("bpic17"))
    raw, _, _ = load_dataset("bpic17")
    log = preprocess("bpic17", raw)
    del raw
    simulator = Simulator("bpic17", log)
    assert [point.name for point in simulator.points] == POINTS
    print(simulator, flush=True)
    spec = m["rct"]
    data = simulate_rct(simulator, n_cases=spec["cases_per_run"], runs=spec["runs"], seed=spec["seed"],
                        delta=spec["delta"], workers=spec["workers"], data_folder=str(out / "rct_data"))
    generated_cases = int(data[CASE].nunique())
    assert generated_cases == spec["cases_per_run"] * spec["runs"]
    data = sample_cases(data, m["total_selected_cases"], seed=m["case_selection_seed"])
    data[CASE] = data[CASE].astype(str)
    assert data[CASE].nunique() == m["total_selected_cases"]
    assert data[CASE].str.startswith("R").all()
    required = [CASE, "concept:name", "lifecycle:transition", "org:resource", "time:timestamp"]
    assert set(required) <= set(data)
    assert not data[[CASE, "concept:name", "time:timestamp"]].isna().any().any()
    assert "rct" in data and data["rct"].any() and (~data["rct"].astype(bool)).any()
    token = data["concept:name"].astype(str) + "|" + data["lifecycle:transition"].astype(str)
    following = token.groupby(data[CASE], sort=False).shift(-1)
    coverage = {}
    for point in simulator.points:
        mask = ((data["concept:name"] == point.activity) & (data["lifecycle:transition"] == point.lifecycle)
                & data["rct"].astype(bool))
        counts = following[mask].value_counts()
        coverage[point.name] = {str(option): int(counts.get(interventions.option_token(option), 0)) for option in point.options}
        assert all(value > 0 for value in coverage[point.name].values()), (point.name, coverage[point.name])
    sizes = data.groupby(CASE).size()
    held_out = test_cases(data)
    assert len(held_out) == m["expected_test_cases"]
    selected = out / "rct_data/training_pool.pkl"
    data.to_pickle(selected)
    write(out / "split.json", {"selected_case_ids": sorted(data[CASE].unique()), "test_case_ids": sorted(held_out),
                              "selection_seed": 42, "test_seed": 17})
    source = Path(rct_folder(simulator, spec["delta"], str(out / "rct_data"))) / "log.pkl"
    assert source.is_file()
    write(out / "data_validation.json", {"status": "passed", "completed_at": now(),
          "generated_cases": generated_cases, "selected_cases": int(data[CASE].nunique()), "events": len(data),
          "test_cases": len(held_out), "training_validation_pool": int(data[CASE].nunique()) - len(held_out),
          "points": POINTS, "allocation": interventions.run_allocation("bpic17"), "delta": spec["delta"],
          "randomized_cases": int(data.groupby(CASE)["rct"].first().sum()),
          "randomized_option_counts": coverage, "median_events_per_case": float(sizes.median()),
          "p985_events_per_case": float(sizes.quantile(.985)), "max_events_per_case": int(sizes.max()),
          "source_log": str(source), "source_sha256": sha(source),
          "selected_log": str(selected), "selected_sha256": sha(selected)})
    print("RCT data validation passed", flush=True)


def train(out):
    add_paths()
    import pandas as pd
    import interventions
    from prescriptive.helpers.training import train_estimators
    from prescriptive.estimators.psp_model import PSPModel
    m = read(out / "manifest.json")
    validated = read(out / "data_validation.json")
    assert validated["status"] == "passed" and sha(Path(validated["selected_log"])) == validated["selected_sha256"]
    if MODEL.exists() or CHECKPOINTS.exists():
        raise FileExistsError("RCT model or checkpoints already exist")
    data = pd.read_pickle(validated["selected_log"])
    intervention = interventions.get(interventions.catalog()["bpic17"]["standard_intervention"])
    context = SimpleNamespace(dataset="bpic17", intervention=intervention,
                              points=interventions.choice_points("bpic17", intervention))
    assert [point.name for point in context.points] == POINTS
    params = {**m["params_before_fit"], "option_in_encoder_after_training": True}
    print("Training parameters:", json.dumps(params, ensure_ascii=False), flush=True)
    table = train_estimators(context, data, estimators=("psp",), delta=m["rct"]["delta"], seed=42,
                             params={"psp": params}, model_paths={"psp": str(MODEL)})
    table.to_csv(out / "training.csv")
    model = PSPModel.load(MODEL)
    model.trained_on.update(data_source="RCT", rct_source_log=validated["source_log"],
                            rct_source_sha256=validated["source_sha256"],
                            selected_log=validated["selected_log"], selected_sha256=validated["selected_sha256"],
                            real_log_used_for_suffix_fit=False)
    model.save(MODEL)


def validate(out):
    add_paths()
    import numpy as np
    import pandas as pd
    import torch
    from prescriptive.estimators.psp_model import PSPModel
    m = read(out / "manifest.json")
    torch.set_num_threads(1)
    model = PSPModel.load(MODEL)
    reference = PSPModel.load(REFERENCE)
    expected = {**m["params_before_fit"], "option_as": "encoder"}
    assert canonical(model.params) == canonical(expected)
    ignored = {"checkpoints", "teacher_forcing"}
    assert canonical({k: v for k, v in model.params.items() if k not in ignored}) == canonical({
        k: v for k, v in reference.params.items() if k not in ignored})
    assert model.seed == reference.seed == 42 and model.points == reference.points
    assert model.gap_classes() and not model.gap_tokens() and not model.gap_condition_on_event()
    assert getattr(model.net, "categorical_conditioning", None) is None
    assert model.option_in_encoder() and not model.params.get("shared_option_gap", False)
    assert getattr(model, "transition_rules", None) is None and getattr(model, "option_gap_distributions", None) is None
    split = read(out / "split.json")
    assert model.test_cases == set(split["test_case_ids"])
    assert len(model.test_cases) == 6301 and model.test_cases.isdisjoint(reference.test_cases)
    assert model.trained_on["data_source"] == "RCT" and model.trained_on["real_log_used_for_suffix_fit"] is False
    summary = model.summary["alle Prefixe"]
    assert summary["Trainingsfälle"] == 20482 and summary["Validierungsfälle"] == 4726 and summary["Epochen"] == 15
    epochs = re.findall(r"Epoch \[(\d+)/15\].*Teacher forcing ratio: ([0-9.eE+-]+)", (out / "pipeline.log").read_text())
    assert [int(epoch) for epoch, _ in epochs] == list(range(1, 16)) and all(float(ratio) == 1.0 for _, ratio in epochs)
    data = pd.read_pickle(out / "rct_data/training_pool.pkl")
    some = set(sorted(model.test_cases)[:30])
    samples = model.samples(data[data[CASE].isin(some)])
    name, rows = next((name, rows) for name, rows in samples.items() if rows)
    _, (part, cut, at), _ = rows[0]
    options = list(model.points[name].options)
    torch.manual_seed(123)
    drawn, elapsed = model.option_suffixes_many(model.points[name], [part.iloc[:cut].to_dict("records")], options, at, 30)
    for option, cases in zip(options, drawn):
        assert len(cases[0]) == 30
        for suffix in cases[0]:
            assert suffix[0][0] == (option if "|" in option else option + "|complete")
            assert np.isfinite(suffix[0][2]) and suffix[0][2] >= 0
            assert np.isclose(suffix[0][1] - elapsed[0], suffix[0][2])
    write(out / "model_validation.json", {"status": "passed", "completed_at": now(), "model_sha256": sha(MODEL),
          "same_d_tf1_parameters_except_checkpoint_path": True, "same_training_validation_test_counts": True,
          "all_training_cases_from_rct": True, "test_cases_excluded_from_fit": True,
          "all_15_teacher_forcing_ratios_equal_1": True, "option_sampling_passed": True, "summary": summary})


def capacity():
    fields = {line.split(":", 1)[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line}
    free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()[0])
    return free, round(fields["MemAvailable"] / 1024 / 1024, 1)


def run(out):
    m = read(out / "manifest.json")
    if read(out / "status.json")["status"] != "prepared":
        raise RuntimeError("Only a prepared run may be started")
    state = {"pid": os.getpid(), "started_at": now(), "epochs": 15, "teacher_forcing": 1.0, "model": str(MODEL)}

    def update(phase, **extra):
        state.update(status=phase, updated_at=now(), **extra)
        write(out / "status.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    def command(stage, phase, env):
        process = subprocess.Popen(m[stage + "_command"], cwd=ROOT, env=env, stdin=subprocess.DEVNULL)
        began = time.monotonic()
        update(phase, child_pid=process.pid)
        while process.poll() is None:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                log = (out / "pipeline.log").read_text()
                epochs = re.findall(r"Epoch \[(\d+)/15\].*Teacher forcing ratio: ([0-9.eE+-]+)", log)
                update(phase, child_pid=process.pid, phase_elapsed_seconds=round(time.monotonic()-began),
                       rct_runs_completed=len(set(re.findall(r"RCT-Lauf (\d+):", log))),
                       epochs_logged=max([int(epoch) for epoch, _ in epochs], default=0))
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, m[stage + "_command"])

    env = dict(os.environ, PSP_CATALOG=m["catalog"], MPLBACKEND="Agg", MPLCONFIGDIR=str(out / "matplotlib"))
    env.pop("PRESCRIPTIVE_OUTCOME_BPIC17", None)
    cpu = dict(env, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1")
    gpu = dict(env, CUDA_VISIBLE_DEVICES="0")
    try:
        check_sources(m)
        command("generate-data", "generating_rct", cpu)
        check_sources(m)
        while True:
            free, ram = capacity()
            if free >= 7000 and ram >= 30:
                break
            update("waiting_for_training_capacity", gpu_free_mib=free, ram_available_gib=ram,
                   required_gpu_mib=7000, required_ram_gib=30)
            time.sleep(60)
        subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), flush=True)"], env=gpu, check=True)
        update("ready_to_train", gpu_free_mib=free, ram_available_gib=ram)
        command("train", "training", gpu)
        check_sources(m)
        command("validate", "validating_model", cpu)
        check_sources(m)
        (out / "auswertung.md").write_text(
            "# D-TF1 auf RCT-Daten: Training abgeschlossen\n\n"
            "15 Epochen, durchgehend Teacher Forcing 1. Alle übrigen D-Parameter entsprechen der Echtlog-Referenz. "
            "20.482 RCT-Trainingsfälle, 4.726 Validierungsfälle und 6.301 ausgeschlossene RCT-Testfälle. "
            "Die vier Interventionspunkte sind unverändert. Modell- und Datenprüfung bestanden.\n\n"
            f"Modell: `{MODEL}`. Details: model_validation.json, data_validation.json und training.csv.\n")
        update("complete", child_pid=None, epochs_logged=15, report=str(out / "auswertung.md"))
    except BaseException as error:
        update("failed", error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "launch", "run", "generate-data", "train", "validate"):
        mode.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.prepare:
        prepare(out)
        print(f"Prepared: {out}")
    elif args.launch:
        if read(out / "status.json")["status"] != "prepared" or (out / "pipeline.pid").exists():
            raise RuntimeError("Run must be prepared and not already launched")
        check_sources(read(out / "manifest.json"))
        with (out / "pipeline.log").open("x") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run", "--out", str(out)],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        (out / "pipeline.pid").write_text(str(process.pid) + "\n")
        print(json.dumps({"pid": process.pid, "out": str(out)}, ensure_ascii=False))
    elif args.generate_data:
        generate_data(out)
    elif args.train:
        train(out)
    elif args.validate:
        validate(out)
    else:
        run(out)


if __name__ == "__main__":
    main()
