"""Begleiter des β-NLL-Nachtrainings (2026-11-18): nach jeder fertigen Epoche last.pt als checkpoints/ep<k>.pt sichern, daraus ein
auswertbares Modell bauen (Testfälle von (c), option_as "encoder" wie nach dem Training) und die Zeitkopf-Sonde klein laufen lassen
(300 Fälle Teacher Forcing, 100 Prefixe × 10 Ziehungen; zeitkopf_sonde_20261118/sonde.py → sonde_ep<k>/). Kernzahlen in fortschritt.log.
Liest nur aus dem Training. Aufruf: <venv>/python begleiter.py TAG EPOCHEN (im Ordner process-simulation-engine)."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
P = ROOT/"prescriptive"
TAG, EPOCHS = sys.argv[1], int(sys.argv[2])
LOG = P/f"data/PSP_{TAG}.log"
FOLDER = P/f"models/psp_training_{TAG}"
REFERENCE = P/"models/psp_model_durchlaufzeit_real_alle_s60_normal.pkl"
SONDE = P/"scripts/results_100k/zeitkopf_sonde_20261118/sonde.py"
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PSP_CATALOG": str(P/"data/catalog_neue_punkte_zuteilung_random.yaml")}
PY = str(ROOT/"process-simulation-engine/.venv/bin/python")


def note(text):
    with open(OUT/"fortschritt.log", "a") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M')} {text}\n")


def epochs_done():
    return len(re.findall(r"Validation Loss for Scheduler", LOG.read_text(errors="ignore"))) if LOG.exists() else 0


def build(checkpoint, label):
    target = OUT/f"modelle/psp_bnll_{label}.pkl"
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
assert model.counters() and model.time_on_event() and model.clocks() and model.params.get("beta_nll") and len(model.test_cases) == 6301
model.save({str(target)!r})
"""
    subprocess.run([PY, "-c", code], cwd=ROOT/"process-simulation-engine", env=ENV, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return target


def probe(model, label):
    folder = OUT/f"sonde_{label}"
    with open(OUT/f"logs/sonde_{label}.log", "w") as handle:
        result = subprocess.run([PY, "-u", str(SONDE), "300", "100", "10", f"β-NLL {label}={model}"], cwd=ROOT/"process-simulation-engine",
                                env={**ENV, "SONDE_OUT": str(folder)}, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        note(f"Sonde {label}: FEHLER (Code {result.returncode})")
        return
    t = pd.read_csv(folder/"sonde.csv")
    get = lambda teil, methode, spalte: t[(t.teil == teil) & (t.methode == methode)][spalte].iloc[0]
    try:
        note(f"Sonde {label}: A_Cancelled Kopf-Abstand {get('A_Cancelled je Schritt', 'kopf', 'Abstand Mittel'):.1f} d "
             f"(echt {get('A_Cancelled je Schritt', 'abstand', 'Abstand Mittel'):.1f}), MAE {get('A_Cancelled je Schritt', 'kopf', 'MAE'):.1f}; "
             f"Summe O_Sent→Storno Kopf Median {get('Summe O_Sent → storno (1 O_Sent)', 'kopf', 'seit O_Sent Median'):.1f} d; "
             f"σ Kopf A_Cancelled {get('σ des Abstandskopfs (Tage, Median)', 'kopf', 'A_Cancelled'):.1f} d; "
             f"gezogen seit O_Sent (Abstandssumme) Median {get('gezogen: seit O_Sent bis A_Cancelled', 'abstandssumme', 'seit O_Sent Median'):.1f} d, "
             f"Anteil 29–32 d {get('gezogen: seit O_Sent bis A_Cancelled', 'abstandssumme', 'Anteil 29–32 d'):.2f}; "
             f"MAE alle Schritte {get('alle Schritte', 'kopf', 'MAE'):.2f} d")
    except (IndexError, KeyError) as error:
        note(f"Sonde {label}: Kurzfassung unvollständig ({error})")


def main():
    (OUT/"checkpoints").mkdir(exist_ok=True)
    (OUT/"logs").mkdir(exist_ok=True)
    saved = 0
    note(f"Begleiter gestartet ({TAG}, {EPOCHS} Epochen)")
    while saved < EPOCHS:
        done = epochs_done()
        if done > saved and (FOLDER/"last.pt").exists():
            time.sleep(30)  # last.pt wird nach der Validierung geschrieben
            for epoch in range(saved + 1, done + 1):
                shutil.copy2(FOLDER/"last.pt", OUT/f"checkpoints/ep{epoch}.pt")
            saved = done
            note(f"Epoche {done} gesichert, Sonde beginnt")
            probe(build(OUT/f"checkpoints/ep{done}.pt", f"ep{done}"), f"ep{done}")
            continue
        if "Abbruch im Training" in (OUT/"fortschritt.log").read_text(errors="ignore"):
            note("Training abgebrochen, Begleiter endet")
            return
        time.sleep(60)
    note("Begleiter fertig")


if __name__ == "__main__":
    main()
