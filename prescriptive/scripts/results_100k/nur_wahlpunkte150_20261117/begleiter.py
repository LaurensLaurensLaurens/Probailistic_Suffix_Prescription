"""Begleiter des Trainings ZZ-TF1-WP150 = ZZ-TF1 nur auf Wahlpunkt-Proben (Training und Validierung), 150 Epochen (2026-11-17, 23 Uhr).

- sichert nach jeder fertigen Epoche last.pt als checkpoints/ep<k>.pt (der Trainer überschreibt last.pt)
- nach Epoche 3, 9 und am Ende (fertiges Modell): auswertbares Modell bauen (Netz des Checkpoints, Testfälle von (c),
  option_as "encoder" wie nach dem Training) und die Prüfreihe vom 16.11. laufen lassen:
  psp_rauschen (Optionssignal), punkte_alle (Vergleich mit der Simulator-Wahrheit), nachlegen_erklaerung (zweites
  Angebot verschickt?), angebote_einfuegen (Gedächtnis für Angebote), leads_gedaechtnis
- schreibt fortschritt.log und je Prüfung eine Kurzfassung pruefung_<stand>.md
Liest nur aus dem Training; ändert nichts daran. Aufruf im Ordner process-simulation-engine.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[4]
P = ROOT/"prescriptive"
TAG = "real_alle_s60_normal_zaehler_zeit_tf1_wp150"
LOG = P/f"data/PSP_{TAG}.log"
FOLDER = P/f"models/psp_training_{TAG}"
FINAL = P/f"models/psp_model_durchlaufzeit_{TAG}.pkl"
OUT = Path(__file__).resolve().parent
VERGLEICH = P/"scripts/results_100k/modellvergleich_20261116"
REFERENCE = P/"models/psp_model_durchlaufzeit_real_alle_s60_normal.pkl"
CHECKS = (15, 45, 90)
EPOCHS = 150
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PSP_CATALOG": str(P/"data/catalog_neue_punkte_zuteilung_random.yaml")}
PY = str(ROOT/"process-simulation-engine/.venv/bin/python")


def note(text):
    with open(OUT/"fortschritt.log", "a") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M')} {text}\n")


def epochs_done():
    if not LOG.exists():
        return 0
    return len(re.findall(r"Validation Loss for Scheduler", LOG.read_text(errors="ignore")))


def build(checkpoint, label):
    """Auswertbares Modell aus model.pkl des Trainings und einem gesicherten Checkpoint (wie oe_tf1_schnappschuss.py)."""
    target = OUT/f"modelle/psp_wp150_{label}.pkl"
    target.parent.mkdir(exist_ok=True)
    code = f"""
import sys; sys.path.insert(0, {str(ROOT)!r})
from prescriptive.helpers.datasets import add_paths; add_paths()
from prescriptive.estimators.psp_model import PSPModel
model = PSPModel.load({str(FOLDER/'model.pkl')!r})
model.net = PSPModel._load_net({str(checkpoint)!r}); model.net.eval()
model.params["option_as"] = "encoder"
reference = PSPModel.load({str(REFERENCE)!r})
model.test_cases, model.trained_on = reference.test_cases, {{"echter Log": "bpic17", "outcome": "durchlaufzeit"}}
assert model.counters() and model.time_on_event() and len(model.test_cases) == 6301
model.save({str(target)!r})
"""
    subprocess.run([PY, "-c", code], cwd=ROOT/"process-simulation-engine", env=ENV, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return target


def battery(model, label):
    env = {**ENV, "PSP_VERGLEICH_TAG": f"_wp150_{label}", "PSP_VERGLEICH_MODELS": json.dumps({f"ZZ-TF1-WP150 {label}": str(model)})}
    cwd = ROOT/"process-simulation-engine"
    logs = OUT/"logs"
    logs.mkdir(exist_ok=True)
    steps = [("rauschen", [PY, "-u", str(P/"scripts/psp_rauschen.py"), "--model", str(model), "--workers", "4", "--threads", "2",
                           "--out", str(OUT/f"rauschen_{label}")])]
    steps += [(name, [PY, "-u", str(VERGLEICH/f"{name}.py")]) for name in ("punkte_alle", "nachlegen_erklaerung", "angebote_einfuegen", "leads_gedaechtnis")]
    steps.insert(3, ("nachlegen_auswerten", [PY, "-u", str(VERGLEICH/"nachlegen_auswerten.py"), f"nachlegen_erklaerung_wp150_{label}"]))
    for name, command in steps:
        with open(logs/f"{name}_{label}.log", "w") as handle:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
        note(f"Prüfung {label}: {name} {'fertig' if result.returncode == 0 else f'FEHLER (Code {result.returncode})'}")
    summary(label)


def tail(path, marker, lines):
    try:
        text = path.read_text(errors="ignore").splitlines()
    except FileNotFoundError:
        return ["(fehlt)"]
    index = max((i for i, line in enumerate(text) if marker in line), default=None)
    return text[index:index+lines] if index is not None else text[-lines:]


def summary(label):
    logs = OUT/"logs"
    parts = [f"# Prüfung {label} ({time.strftime('%Y-%m-%d %H:%M')})", "", "## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)", "```"]
    try:
        table = (OUT/f"rauschen_{label}/rauschen.csv").read_text().splitlines()
        parts += [line for line in table if line.split(",")[0] in ("", "Diff Mittel (Tage)", "Vorzeichen wie AIPW (alle)", "SE gepaart Median",
                                                                   "Korrelation der Optionen Median", "Durchlaufzeit gezogen (Mittel)")]
    except FileNotFoundError:
        parts.append("(fehlt)")
    parts += ["```", "", "## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)", "```"]
    parts += tail(logs/f"punkte_alle_{label}.log", "Gruppe", 12)
    parts += ["```", "", "## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)", "```"]
    parts += tail(logs/f"nachlegen_auswerten_{label}.log", "Zusätzliches Angebot minus senden", 16)
    parts += ["```", "", "## Gedächtnis für Angebote / Leads", "```"]
    parts += tail(logs/f"angebote_einfuegen_{label}.log", "Mit eingefügtem", 4) + tail(logs/f"leads_gedaechtnis_{label}.log", "Restzeit ab erstem O_Sent", 6)
    parts += ["```"]
    (OUT/f"pruefung_{label}.md").write_text("\n".join(parts)+"\n")
    note(f"Kurzfassung pruefung_{label}.md geschrieben")


def main():
    (OUT/"checkpoints").mkdir(exist_ok=True)
    saved, checked, threads = 0, set(), []
    note("Begleiter gestartet")
    while True:
        done = epochs_done()
        last = FOLDER/"last.pt"
        if done > saved and last.exists():
            time.sleep(30)  # last.pt ist nach der Validierung geschrieben; kurz warten, falls der Trainer noch schreibt
            for epoch in range(saved+1, done+1):
                shutil.copy2(last, OUT/f"checkpoints/ep{epoch}.pt")
            saved = done
            note(f"Epoche {done} gesichert")
            for epoch in CHECKS:
                if done >= epoch and epoch not in checked:
                    checked.add(epoch)
                    note(f"Prüfung nach Epoche {epoch} beginnt")
                    # im Hintergrund, damit die Sicherung der nächsten Epochen weiterläuft
                    threads.append(threading.Thread(target=lambda e=epoch: battery(build(OUT/f"checkpoints/ep{e}.pt", f"ep{e}"), f"ep{e}")))
                    threads[-1].start()
        if FINAL.exists() and "final" not in checked and epochs_done() >= EPOCHS:
            checked.add("final")
            time.sleep(60)
            note("Fertiges Modell gefunden, Prüfung beginnt")
            battery(FINAL, "final")
            for thread in threads:
                thread.join()
            note("Begleiter fertig")
            return
        if "Abbruch im Training" in (LOG.read_text(errors="ignore") if LOG.exists() else ""):
            note("Training abgebrochen, Begleiter endet")
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
