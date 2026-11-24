"""Paired short PSP tests: OE transition rules or (c) with one option-gap draw."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parents[1]
ROOT = P / "scripts/results_100k"


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def summarize(out, seed, experiment="oe_transitions"):
    is_c = experiment == "c_shared_gap"
    labels = ("original_1ms", "normal_einmal_pro_option") if is_c else ("ohne_sperre", "mit_sperre")
    descriptions = ("(c), ursprüngliche 1 ms", "(c), ein Normal-Abstand je Option") if is_c else ("Ohne Übergangssperre", "Mit Übergangssperre")
    rows, banks = {}, {}
    for label in labels:
        rows[label] = json.loads((out / label / f"psp_seed{seed}_row.json").read_text())
        banks[label] = json.loads((out / label / f"bank_seed{seed}_row.json").read_text())
    if banks[labels[0]]["_days"] != banks[labels[1]]["_days"]:
        raise RuntimeError("Bank case durations differ; inspect pairing before interpreting results")
    before, after = rows[labels[0]], rows[labels[1]]
    benefit = before["Durchlaufzeit (Tage)"] - after["Durchlaufzeit (Tage)"]
    summary = {"seed": seed, "experiment": experiment, "samples_per_option": 30, "scored_cases": 200,
               "bank_case_durations_identical": True, "completed_at": stamp(),
               "bank_days": banks[labels[0]]["Durchlaufzeit (Tage)"],
               "baseline_days": before["Durchlaufzeit (Tage)"], "variant_days": after["Durchlaufzeit (Tage)"],
               "improvement_days": benefit, "improvement_percent": 100 * benefit / before["Durchlaufzeit (Tage)"],
               "baseline_choices": sum(v for k, v in before.items() if k.startswith("Wahlen der Policy:")),
               "variant_choices": sum(v for k, v in after.items() if k.startswith("Wahlen der Policy:"))}
    if not is_c:
        summary.update(without_rules_days=summary["baseline_days"], with_rules_days=summary["variant_days"],
                       without_rules_choices=summary["baseline_choices"], with_rules_choices=summary["variant_choices"])
    write_json(out / "vergleich.json", summary)
    lines = ["# " + ("(c): ein geteilter Normal-Abstand je Option" if is_c else "token_oe: Test der Übergangssperre"), "",
             "Seed 0, je 30 Ziehungen pro Option, 700 Warmup-, 200 gewertete und 1.300 Cooldown-Fälle.", "",
             "Beide Läufe verwenden dieselben trainierten Gewichte, vier Wahlpunkte, zufällige Zuteilung und den gleichen Seed. Die Bank-Durchlaufzeiten aller 200 Fälle stimmen exakt überein.", "",
             "| Variante | Mittlere Durchlaufzeit (Tage) |", "|---|---:|",
             f"| Bank | {summary['bank_days']:.3f} |", f"| {descriptions[0]} | {summary['baseline_days']:.3f} |",
             f"| {descriptions[1]} | {summary['variant_days']:.3f} |", "",
             f"Verbesserung durch die Änderung: {benefit:+.3f} Tage ({summary['improvement_percent']:+.2f} %). Positiv bedeutet kürzer."]
    if is_c:
        lines.extend(["", "Je Entscheidung wird für jede Option unabhängig ein Abstand aus ihrer bei 0 trunkierten Normalverteilung gezogen. Alle 30 Suffixe dieser Option teilen diesen Abstand und dasselbe Encoder-Prefix. Die Normalparameter sind die empirischen Mittelwerte und Standardabweichungen der beobachteten Abstände im äußeren Trainingssplit; durch die Trunkierung ist der tatsächliche Erwartungswert größer als der rohe Lageparameter. Beide Varianten laufen ohne Übergangssperre. Die Erstabstandsunsicherheit wird bei dieser Variante nicht über 30 Zeitziehungen gemittelt."])
    lines.extend(["", "## Wahlen an den vier Punkten", "", f"| Kennzahl | {descriptions[0]} | {descriptions[1]} |", "|---|---:|---:|"])
    for key in before:
        if key.startswith("Wahlen der Policy:") or (key.startswith("Anteil ") and ": " in key):
            lines.append(f"| {key} | {before[key]:.4f} | {after[key]:.4f} |")
    lines.extend(["", "Ein einzelner Seed mit 200 gewerteten Fällen und 30 Ziehungen ist ein Vortest. Daraus folgt keine abgesicherte Aussage für andere Seeds oder Ziehungszahlen.", "",
                  "Die JSON-Dateien enthalten vollständige Kennzahlen und einzelne Falldauern. `choices.jsonl` protokolliert Entscheidungen; `progress.json` den Verlauf. In `run_manifest.json` stehen Parameter und Dateifingerabdrücke."])
    (out / "auswertung.md").write_text("\n".join(lines) + "\n")
    print("ERGEBNIS", json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "token_oe_suffixe30_paired200_seed0")
    parser.add_argument("--reuse-without", type=Path, help="Reuse an existing unrestricted run (may still be running)")
    parser.add_argument("--experiment", choices=("oe_transitions", "c_shared_gap"), default="oe_transitions")
    args = parser.parse_args()
    is_c = args.experiment == "c_shared_gap"
    if is_c and args.reuse_without:
        parser.error("--reuse-without is only available for the OE comparison")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.reuse_without:
        (out / "ohne_sperre").symlink_to(args.reuse_without.resolve(), target_is_directory=True)
    catalog = P / "data/catalog_neue_punkte_zuteilung_random.yaml"
    models = ({"original_1ms": P / "models/psp_model_durchlaufzeit_real_alle_s60_normal.pkl",
               "normal_einmal_pro_option": P / "models/psp_model_durchlaufzeit_real_alle_s60_normal_shared_gap.pkl"} if is_c else
              {"ohne_sperre": P / "models/psp_model_durchlaufzeit_real_alle_s60_token_oe.pkl",
               "mit_sperre": P / "models/psp_model_durchlaufzeit_real_alle_s60_token_oe_ue.pkl"})
    paths = [*models.values(), catalog, P / "estimators/psp_model.py", P / "scripts/run_token_oe_ue_seed.py"]
    manifest = {"seed": 0, "experiment": args.experiment, "samples_per_option": 30, "scored_cases": 200,
                "warmup": 700, "cooldown": 1300, "arrivals": 2200, "allocation": "random",
                "models": {label: str(path) for label, path in models.items()},
                "started_at": stamp(), "sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}}
    if is_c:
        manifest["option_gap_calibration"] = json.loads((P / "data/PSP_c_shared_option_gap_distributions.json").read_text())
    if args.reuse_without:
        manifest["reused_without_rules_run"] = str(args.reuse_without.resolve())
    write_json(out / "run_manifest.json", manifest)
    processes = {}
    logs = []
    state = {"status": "running", "started_at": stamp(), "runs": {}}
    try:
        for label, model in models.items():
            if label == "ohne_sperre" and args.reuse_without:
                state["runs"][label] = {"reused_from": str(args.reuse_without.resolve()), "exit_code": None}
                continue
            log = (out / f"{label}.log").open("x", buffering=1)
            logs.append(log)
            cmd = [sys.executable, "-u", str(P / "scripts/run_token_oe_ue_seed.py"), "--seed", "0",
                   "--samples", "30", "--scored-cases", "200", "--model", str(model), "--out", str(out / label)]
            if label == "ohne_sperre" or is_c:
                cmd.append("--without-transition-rules")
            if label == "normal_einmal_pro_option":
                cmd.append("--require-shared-option-gap")
            process = subprocess.Popen(cmd, cwd=P.parent / "process-simulation-engine", stdout=log, stderr=subprocess.STDOUT)
            processes[label] = process
            state["runs"][label] = {"pid": process.pid, "command": cmd}
        while True:
            for label, process in processes.items():
                state["runs"][label]["exit_code"] = process.poll()
                progress = out / label / "progress.json"
                if progress.exists():
                    state["runs"][label]["progress"] = json.loads(progress.read_text())
            if args.reuse_without:
                progress = out / "ohne_sperre/progress.json"
                if progress.exists():
                    reused = json.loads(progress.read_text())
                    state["runs"]["ohne_sperre"].update(progress=reused, exit_code=0 if reused["status"] == "complete" else
                                                        1 if reused["status"] == "failed" else None)
            state["updated_at"] = stamp()
            write_json(out / "status.json", state)
            if all(run["exit_code"] is not None for run in state["runs"].values()):
                break
            time.sleep(5)
        if any(run["exit_code"] != 0 for run in state["runs"].values()):
            raise RuntimeError(f"Run failed: { {label: run['exit_code'] for label, run in state['runs'].items()} }")
        summarize(out, 0, args.experiment)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        state["updated_at"] = stamp()
        write_json(out / "status.json", state)
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
