"""ERM-Vergleiche unter der gelernten Bank-Basiswahl neu auswerten.

Zuerst Bank + drei CATE-ERM-Policies (Checkpoints, kurze Batches), anschließend PSP-ERM Variante c.
Vorhandene Modelle werden verwendet. Historische random-Ergebnisse bleiben unverändert.
Ein erneuter Aufruf setzt fehlende Läufe fort; alte Bank-Checkpoints werden nicht übernommen.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "prescriptive"
PREVIOUS = P / "scripts/results_100k/neue_punkte_online_cate_erm_gamma0.1"
CATE = P / "scripts/results_100k/neue_punkte_online_cate_erm_gamma0.1_bank"
PSP = P / "scripts/results_100k/neue_punkte_online_psp_erm_gamma0.1_suffixe100_bank"
STATUS = P / "data/erm_bank_status.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state(stage, **details):
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps({"stage": stage, "updated": now(), "cate_results": str(CATE),
                                     "psp_results": str(PSP), "allocation": "learned", **details}, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(STATUS)


def prepare():
    CATE.mkdir(parents=True, exist_ok=True)
    previous = json.loads((PREVIOUS / "laufparameter.json").read_text())
    if not (CATE / "catalog.yaml").exists():
        shutil.copy2(P / "data/catalog_neue_punkte.yaml", CATE / "catalog.yaml")
    import yaml
    assert yaml.safe_load((CATE / "catalog.yaml").read_text())["bpic17"]["lauf"]["zuteilung"] == "learned"
    for name in ("resume_simulation.py", "auswertung.py"):
        destination = CATE / name
        if not destination.exists():
            text = (PREVIOUS / name).read_text()
            text = text.replace("Mitarbeiterzuteilung random.", "Mitarbeiterzuteilung nach gelerntem Bankverhalten (learned).")
            text = text.replace('manifest["initial_attempt_exit_code"] = 143\n', '')
            text = text.replace("- online.log: erster, mit SIGTERM beendeter Versuch; seine nicht gespeicherten Teilergebnisse wurden nicht verwendet.",
                                "- online_resume.log: alle neu berechneten Einzelläufe unter gelernter Bank-Zuteilung.")
            destination.write_text(text)
    bank_files = ["process-simulation-engine/resources/case_allocation.py",
                  "process-simulation-engine/resources/BPIC2017/allocation/learned_allocation.py",
                  "process-simulation-engine/interventions/__init__.py",
                  "process-simulation-engine/models/allocation/learned_allocation_dienstplan.pkl"]
    manifest_path = CATE / "laufparameter.json"
    if not manifest_path.exists():
        metadata = previous.copy()
        for key in ("historical_ev_bank_replication_matches", "initial_attempt_exit_code", "resumption_script", "resumption_script_sha256", "checkpoint_settings_sha256"):
            metadata.pop(key, None)
        metadata.update(allocation="learned", catalog_sha256=digest(CATE / "catalog.yaml"),
                        bank_source_sha256={path: digest(ROOT / path) for path in bank_files},
                        source_sha256={path: digest(ROOT / path) for path in previous["source_sha256"]},
                        run_script=str(Path(__file__).resolve()),
                        baseline_note="Neue Bank-Läufe, keine Übernahme alter random-Referenzen")
        manifest_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    metadata = json.loads(manifest_path.read_text())
    assert metadata["allocation"] == "learned" and digest(CATE / "catalog.yaml") == metadata["catalog_sha256"]
    for path, checksum in metadata["bank_source_sha256"].items():
        assert digest(ROOT / path) == checksum, path
    for name in ("training_diagnostik.csv", "aktionsanteile_testfaelle.csv", "aktionsanteile_testfaelle.png"):
        target = CATE / name
        if not target.exists():
            shutil.copy2(PREVIOUS / name, target)
    for model in metadata["models"].values():
        assert digest(model["path"]) == model["sha256"]
    return metadata


def run(command, log):
    print(f"{now()}: {' '.join(map(str, command))}", flush=True)
    with log.open("ab") as stream:
        subprocess.run(list(map(str, command)), stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                       cwd=str(ROOT), check=True)


def main():
    prepare()
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(CATE / "matplotlib_cache")
    state("cate_erm")
    checkpoints = CATE / "checkpoints"
    while len(list(checkpoints.glob("*.json"))) < 40:
        run([sys.executable, "-u", CATE / "resume_simulation.py", "--max-jobs", "8", "--workers", "4"], CATE / "online_resume.log")
        state("cate_erm", completed_runs=len(list(checkpoints.glob("*.json"))))
    run([sys.executable, CATE / "auswertung.py"], CATE / "zusammenfassung.log")
    state("psp_erm", completed_cate_runs=40)
    PSP.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-u", P / "scripts/evaluate_psp_erm_real.py", "--reference", CATE, "--out", PSP, "--workers", "6"], PSP / "supervisor.log")
    results = P / "policies/erm_policy/results"
    results.mkdir(exist_ok=True)
    shutil.copy2(PSP / "vergleich_alle_erm.png", results / "online_vergleich_bank.png")
    shutil.copy2(PSP / "zusammenfassung.csv", results / "online_vergleich_bank.csv")
    state("complete", completed_cate_runs=40, completed_psp_seeds=10)
    print(f"{now()}: neue Bank-, CATE-ERM- und PSP-ERM-Auswertung vollständig", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        state("failed", error=traceback.format_exc())
        raise
