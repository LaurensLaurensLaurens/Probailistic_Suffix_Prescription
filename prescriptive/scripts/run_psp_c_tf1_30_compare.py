"""Evaluate the existing C-TF1 model on the paired 200-case/30-draw mini-test."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

P = Path(__file__).resolve().parents[1]
ROOT = P.parent
RESULTS = P / "scripts/results_100k"
MODEL = P / "models/psp_model_durchlaufzeit_real_alle_s60_normal_tf1.pkl"
BASE_MODEL = P / "models/psp_model_durchlaufzeit_real_alle_s60_normal.pkl"
DEFAULT_OUT = RESULTS / "c_tf1_suffixe30_cases200_seed0_20261114"
WRONG_QUEUE = RESULTS / "d_tf1_suffixe30_cases200_seed0_20261114"
REFERENCES = {
    "C bisher": RESULTS / "c_shared_gap_suffixe30_paired200_seed0/original_1ms",
    "D": RESULTS / "d_option_encoder_suffixe30_cases200_seed0_20261113/d_korrigiert",
    "E": RESULTS / "e_suffixe30_cases200_seed0_20261113/e",
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
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(ROOT))
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    from prescriptive.estimators.psp_model import PSPModel
    model, base = PSPModel.load(MODEL), PSPModel.load(BASE_MODEL)
    assert model.params["teacher_forcing"] == 1.0
    ignored = {"teacher_forcing", "checkpoints"}
    assert {k: v for k, v in model.params.items() if k not in ignored} == {
        k: v for k, v in base.params.items() if k not in ignored}
    assert model.seed == base.seed == 42 and model.points == base.points
    assert model.test_cases == base.test_cases and len(model.test_cases) == 6301
    assert not model.gap_classes() and not model.gap_tokens() and not model.gap_condition_on_event()
    assert not model.option_in_encoder() and not model.option_in_decoder()
    assert not model.params.get("shared_option_gap", False)
    assert getattr(model, "option_gap_distributions", None) is None
    assert getattr(model, "transition_rules", None) is None
    summary, prior = model.summary["alle Prefixe"], base.summary["alle Prefixe"]
    for key in ("Trainingsfälle", "Validierungsfälle", "Proben", "Proben Validierung", "Epochen", "Gewicht am Wahlpunkt"):
        assert summary[key] == prior[key], key
    assert summary["Epochen"] == 15
    sources = [Path(__file__).resolve(), MODEL, BASE_MODEL, P / "scripts/run_token_oe_ue_seed.py",
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
    previous = read(RESULTS / "d_option_encoder_suffixe30_cases200_seed0_20261113/run_manifest.json")
    assert sha(catalog) == previous["sha256"][str(catalog)]
    out.mkdir(parents=True, exist_ok=False)
    (out / "catalog_snapshot.yaml").write_bytes(catalog.read_bytes())
    command = [sys.executable, "-u", str(P / "scripts/run_token_oe_ue_seed.py"),
               "--model", str(MODEL), "--seed", "0", "--samples", "30", "--scored-cases", "200",
               "--warmup", "700", "--cooldown", "1300", "--without-transition-rules", "--out", str(out / "c_tf1")]
    write(out / "run_manifest.json", {"prepared_at": now(), "variant": "c_tf1", "objective": "negative_cycle_days",
          "model": str(MODEL), "training_teacher_forcing": 1.0, "seed": 0, "samples_per_option": 30,
          "scored_cases": 200, "warmup": 700, "cooldown": 1300, "arrivals": 2200, "allocation": "random",
          "transition_rules": False, "shared_option_gap": False, "option_mode": "original C, 1 ms",
          "online_command": command, "references": {key: str(value) for key, value in REFERENCES.items()},
          "sha256": {str(path): sha(path) for path in sources}})
    write(out / "model_validation.json", {"status": "passed", "checked_at": now(),
          "model_sha256": sha(MODEL), "same_c_parameters_except_teacher_forcing_and_checkpoint_path": True,
          "same_sample_counts_and_test_cases": True, "same_choice_points": True,
          "training_teacher_forcing": 1.0, "training_summary": summary})
    write(out / "status.json", {"status": "prepared", "updated_at": now(), "model": str(MODEL),
          "seed": 0, "samples_per_option": 30, "scored_cases": 200})


def check_sources(manifest):
    for name, digest in manifest["sha256"].items():
        if sha(Path(name)) != digest:
            raise RuntimeError(f"Prepared input changed: {name}")


def cancel_wrong_queue(out):
    path = WRONG_QUEUE / "status.json"
    if not path.exists():
        return
    state = read(path)
    if state["status"] == "cancelled":
        return
    if state["status"] not in {"prepared", "waiting_for_training"}:
        raise RuntimeError(f"D evaluation is no longer just queued; inspect before replacing: {state}")
    pid = state.get("pid")
    if pid:
        command_path = Path(f"/proc/{int(pid)}/cmdline")
        try:
            argv = command_path.read_bytes().decode().split("\0")
        except FileNotFoundError:
            argv = []
        if argv:
            assert str(P / "scripts/run_psp_d_tf1_30_compare.py") in argv and "--run" in argv
            assert str(WRONG_QUEUE) in argv
            os.kill(int(pid), signal.SIGTERM)
            for _ in range(50):
                stat = Path(f"/proc/{int(pid)}/stat")
                try:
                    if stat.read_text().split()[2] == "Z":
                        break
                except FileNotFoundError:
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("Queued D supervisor did not terminate")
    state.update(status="cancelled", updated_at=now(), replaced_by=str(out),
                 reason="User clarified the evaluation should use the already trained C-TF1 model.")
    write(path, state)


def compare(out):
    folders = {"C, TF 1": out / "c_tf1", **REFERENCES}
    rows, banks = {}, {}
    for label, folder in folders.items():
        progress = read(folder / "progress.json")
        assert progress["status"] == "complete"
        assert progress["samples_per_option"] == 30 and progress["seed"] == 0 and progress["scored_cases"] == 200
        rows[label] = read(folder / "psp_seed0_row.json")
        banks[label] = read(folder / "bank_seed0_row.json")
        assert rows[label]["Fälle"] == banks[label]["Fälle"] == 200
    same_banks = all(bank["_days"] == banks["C, TF 1"]["_days"] for bank in banks.values())
    bank_days = banks["C, TF 1"]["Durchlaufzeit (Tage)"]
    days = {label: row["Durchlaufzeit (Tage)"] for label, row in rows.items()}
    write(out / "vergleich.json", {"completed_at": now(), "seed": 0, "samples_per_option": 30,
          "scored_cases": 200, "bank_case_durations_identical": same_banks, "bank_days": bank_days,
          "policy_days": days, "c_tf1_minus_previous_c_days": days["C, TF 1"] - days["C bisher"],
          "c_tf1_improvement_vs_bank_percent": (bank_days - days["C, TF 1"]) / bank_days * 100})
    lines = ["# C mit Teacher Forcing 1: Online-Vergleich", "",
             "200 bewertete Fälle, 30 freie Suffixziehungen je Option, Seed 0. "
             "700 Warmup und 1.300 Cooldown, zufällige Ressourcenvergabe, vier bisherige Wahlpunkte. "
             "Das gespeicherte C-TF1-Modell wird unverändert verwendet, mit dem ursprünglichen Optionsabstand von 1 ms. "
             "Ziel: minimale mittlere Durchlaufzeit.", "",
             "| Variante | Durchlaufzeit (Tage) ↓ | Verbesserung ggü. Bank |", "|---|---:|---:|"]
    for label, value in days.items():
        lines.append(f"| {label} | {value:.3f} | {(bank_days-value)/bank_days*100:+.2f} % |")
    lines += [f"| Bank | {bank_days:.3f} | 0,00 % |", "",
              f"C mit TF 1 minus bisheriges C: {days['C, TF 1']-days['C bisher']:+.3f} Tage. "
              "Negative Werte bedeuten kürzere Durchlaufzeit.", "",
              f"Bank-Falldauern aller Vergleichsläufe identisch: {'ja' if same_banks else 'NEIN'}. "
              "Teacher Forcing 1 betrifft das Training; die Auswertung zieht die Fortsetzungen frei.", "",
              "Ein Seed mit 200 Fällen; kein Konfidenzintervall über mehrere Seeds. "
              "Einzelne Laufwerte und Falldauern stehen unter c_tf1/."]
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
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        process = subprocess.Popen(manifest["online_command"], cwd=ROOT / "process-simulation-engine",
                                   env=env, stdin=subprocess.DEVNULL)
        update("online", child_pid=process.pid)
        while process.poll() is None:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                progress_file = out / "c_tf1/progress.json"
                progress = read(progress_file) if progress_file.is_file() else {}
                update("online", child_pid=process.pid, decisions_completed=progress.get("decisions_completed", 0),
                       cases_with_decisions=progress.get("cases_with_decisions", 0))
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, manifest["online_command"])
        check_sources(manifest)
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
        check_sources(read(out / "run_manifest.json"))
        cancel_wrong_queue(out)
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
