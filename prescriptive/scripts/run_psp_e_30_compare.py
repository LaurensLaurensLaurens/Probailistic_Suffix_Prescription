"""Train variant E, then run the same 30-draw/200-case online comparison as D.

--prepare writes the concrete run configuration. --launch starts the prepared
pipeline in the background, with its own log, PID and persistent status.
"""
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
TAG = "real_alle_s60_klassen_event_oe"
MODEL = P / f"models/psp_model_durchlaufzeit_{TAG}.pkl"
D = RESULTS / "d_option_encoder_suffixe30_cases200_seed0_20261113"
REFERENCES = {
    "d": D / "d_korrigiert",
    "c_original": RESULTS / "c_shared_gap_suffixe30_paired200_seed0/original_1ms",
    "c_ein_abstand": RESULTS / "c_shared_gap_suffixe30_paired200_seed0/normal_einmal_pro_option",
    "oe": RESULTS / "token_oe_suffixe30_vergleich_c_seed0_20261113/oe_ohne_sperre",
}


def timestamp():
    return datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(out):
    for folder in REFERENCES.values():
        progress = read(folder / "progress.json")
        assert progress["status"] == "complete" and progress["samples_per_option"] == 30
        assert progress["seed"] == 0 and progress["scored_cases"] == 200
        assert not progress["transition_rules_expected"]
    catalog = P / "data/catalog_neue_punkte_zuteilung_random.yaml"
    assert sha(catalog) == read(D / "run_manifest.json")["sha256"][str(catalog)]
    if MODEL.exists() or (P / f"models/psp_training_{TAG}").exists():
        raise FileExistsError("E model or checkpoints already exist; inspect them before starting another training")
    out.mkdir(parents=True, exist_ok=False)
    training_catalog = out / "training_catalog.yaml"
    training_catalog.write_bytes((P / "data/catalog_neue_punkte.yaml").read_bytes())
    command = [sys.executable, "-u", str(P / "scripts/train_psp_real.py"), "--tag", TAG,
               "--cases", "0", "--epochs", "15", "--learning-rate", "1e-4", "--point-weight", "3",
               "--seed", "42", "--catalog-points", "--suffix-size", "60", "--statics-without-future",
               "--time-from", "gaps", "--gap-condition-on-event", "--option-in-encoder-after-training"]
    online = [sys.executable, "-u", str(P / "scripts/run_token_oe_ue_seed.py"), "--model", str(MODEL),
              "--seed", "0", "--samples", "30", "--scored-cases", "200", "--warmup", "700",
              "--cooldown", "1300", "--without-transition-rules", "--out", str(out / "e")]
    sources = [Path(__file__), P / "estimators/psp_model.py", P / "scripts/train_psp_real.py",
               P / "scripts/run_token_oe_ue_seed.py", training_catalog, catalog,
               ROOT / "probabilistic_suffix_prediction_U-ED-LSTM/src/model/dropout_uncertainty_enc_dec_LSTM/dropout_uncertainty_model.py",
               ROOT / "probabilistic_suffix_prediction_U-ED-LSTM/src/model/dropout_uncertainty_enc_dec_LSTM/dropout_uncertainty_decoder.py",
               ROOT / "process-simulation-engine/simulation/case_engine_lifecycle_aware.py",
               ROOT / "process-simulation-engine/resources/case_allocation.py"]
    write(out / "run_manifest.json", {
        "prepared_at": timestamp(), "variant": "e", "model": str(MODEL), "training_seed": 42,
        "seed": 0, "samples_per_option": 30, "scored_cases": 200, "warmup": 700, "cooldown": 1300,
        "allocation": "random", "transition_rules": False, "shared_option_gap": False,
        "training_catalog": str(training_catalog), "online_catalog": str(catalog),
        "training_command": command, "online_command": online,
        "references": {key: str(path) for key, path in REFERENCES.items()},
        "sha256": {str(path): sha(path) for path in sources}})
    write(out / "status.json", {"status": "prepared", "updated_at": timestamp(), "model": str(MODEL)})


def validate_model(out):
    import pickle
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    sys.path.insert(0, str(ROOT))
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    import numpy as np
    import torch
    from prescriptive.estimators.psp_model import ACTIVITY, GAP_CLASS, PSPModel

    model = PSPModel.load(MODEL)
    reference = PSPModel.load(P / "models/psp_model_durchlaufzeit_real_alle_s60_klassen_option_encoder.pkl")
    assert model.gap_condition_on_event() and model.gap_classes() and not model.gap_tokens()
    assert model.net.categorical_conditioning == (GAP_CLASS, ACTIVITY)
    assert model.option_in_encoder() and model.params.get("time_from") == "gaps"
    assert not model.params.get("shared_option_gap", False)
    assert getattr(model, "transition_rules", None) is None
    assert getattr(model, "option_gap_distributions", None) is None
    assert model.seed == reference.seed and model.test_cases == reference.test_cases
    assert model.points == reference.points
    assert model.summary["alle Prefixe"]["Epochen"] == 15
    ignored = {"checkpoints", "gap_condition_on_event"}
    assert {k: v for k, v in model.params.items() if k not in ignored} == {
        k: v for k, v in reference.params.items() if k not in ignored}
    probe = pickle.loads((RESULTS / "token_oe_suffixe30_paired200_seed0/failed_prefix.pkl").read_bytes())
    model.set_threads(1)
    torch.manual_seed(123)
    drawn, elapsed = model.option_suffixes_many(model.points[probe["point"].name], [probe["events"]],
                                               probe["options"], probe["now"], 30)
    gaps = {}
    for option, cases in zip(probe["options"], drawn):
        assert len(cases[0]) == 30
        for suffix in cases[0]:
            assert suffix[0][0] == (option if "|" in option else option + "|complete")
            assert np.isfinite(suffix[0][2]) and suffix[0][2] >= 0
            assert np.isclose(suffix[0][1] - elapsed[0], suffix[0][2])
        gaps[str(option)] = [suffix[0][2] for suffix in cases[0]]
    write(out / "sampling_validation.json", {"status": "passed", "model_sha256": sha(MODEL),
          "same_training_parameters_as_d_except_conditioning": True, "same_test_cases_and_choice_points": True,
          "event_conditioned_gap_class": True, "samples_per_option": 30, "first_gaps_seconds": gaps})


def compare(out):
    folders = {"e": out / "e", **REFERENCES}
    rows, banks = {}, {}
    for key, folder in folders.items():
        progress = read(folder / "progress.json")
        assert progress["status"] == "complete" and progress["samples_per_option"] == 30
        assert progress["seed"] == 0 and progress["scored_cases"] == 200
        rows[key] = read(folder / "psp_seed0_row.json")
        banks[key] = read(folder / "bank_seed0_row.json")
        assert rows[key]["Fälle"] == 200
    paired = all(bank["_days"] == banks["e"]["_days"] for bank in banks.values())
    days = {key: row["Durchlaufzeit (Tage)"] for key, row in rows.items()}
    bank_days = banks["e"]["Durchlaufzeit (Tage)"]
    write(out / "vergleich.json", {"seed": 0, "samples_per_option": 30, "scored_cases": 200,
          "bank_case_durations_identical": paired, "bank_days": bank_days, "policy_days": days,
          "e_minus_d_days": days["e"] - days["d"], "completed_at": timestamp()})
    lines = ["**PSP (e): Event → bedingte Zeitklasse → Abstand**", "",
             "Seed 0; 30 Ziehungen je Option; 200 gewertete Fälle; 700 Warmup, 1.300 Cooldown. "
             "Zufällige Mitarbeiterzuteilung, keine Übergangssperre.", "",
             "| Variante | Durchlaufzeit (Tage) |", "|---|---:|",
             *[f"| {key} | {value:.3f} |" for key, value in days.items()], f"| Bank | {bank_days:.3f} |", "",
             f"(e) minus (d): {days['e'] - days['d']:+.3f} Tage. Positive Werte bedeuten längere Durchlaufzeit.", "",
             f"Bank-Falldauern aller Vergleichsläufe identisch: {'ja' if paired else 'NEIN; Vergleich nicht gepaart'}.", "",
             "Ein Seed mit 200 Fällen ist ein Mini-Test. (e) wurde neu trainiert; der Zeitklassen-Kopf "
             "erhält das aktuelle Event. Trainingsparameter und Fall-Split entsprechen (d)."]
    (out / "auswertung.md").write_text("\n".join(lines) + "\n")
    if not paired:
        raise RuntimeError("Bank cases differ from the historical references; see vergleich.json")


def run(out):
    manifest = read(out / "run_manifest.json")
    status = {"pid": os.getpid(), "started_at": timestamp()}
    def update(phase, **extra):
        status.update(status=phase, updated_at=timestamp(), **extra)
        write(out / "status.json", status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
    try:
        for name, digest in manifest["sha256"].items():
            assert sha(Path(name)) == digest, f"Prepared source changed: {name}"
        update("checking_gpu")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU is accessible; full training was not started")
        while True:
            query = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
            free = int(query.splitlines()[0].strip())
            if free >= 7000:
                break
            update("waiting_for_gpu", free_mib=free, required_mib=7000)
            time.sleep(60)
        env = dict(os.environ, PSP_CATALOG=manifest["training_catalog"], MPLBACKEND="Agg")
        update("training", gpu=torch.cuda.get_device_name(0), free_mib=free, epochs=15)
        subprocess.run(manifest["training_command"], cwd=ROOT / "process-simulation-engine", env=env, check=True)
        update("validating_model")
        validate_model(out)
        update("online", samples_per_option=30, scored_cases=200)
        subprocess.run(manifest["online_command"], cwd=ROOT / "process-simulation-engine",
                       env=dict(env, CUDA_VISIBLE_DEVICES=""), check=True)
        update("comparing")
        compare(out)
        update("complete")
    except BaseException as error:
        update("failed", error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS / "e_suffixe30_cases200_seed0_20261113")
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
        if read(out / "status.json")["status"] != "prepared":
            raise RuntimeError("Only a prepared run can be launched")
        with (out / "pipeline.log").open("x") as stream:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run", "--out", str(out)],
                                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        (out / "pipeline.pid").write_text(str(process.pid) + "\n")
        print(json.dumps({"pid": process.pid, "out": str(out), "log": str(out / "pipeline.log")}))
    else:
        run(out)


if __name__ == "__main__":
    main()
