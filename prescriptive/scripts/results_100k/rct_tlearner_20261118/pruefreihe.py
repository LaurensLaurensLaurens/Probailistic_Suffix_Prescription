"""Prüfreihe vom 16.11. für ZZ-TF1 und T-Learner trainiert auf Simulator-RCT-Daten (2026-11-18); Kopie von t_learner_20261117/pruefreihe.py.
Je Stand: psp_rauschen, punkte_alle, nachlegen_erklaerung (+ auswerten), angebote_einfuegen, leads_gedaechtnis. Schreibt
staende.json (Tag → Beschriftung, Modell) für register_erstellen.py. Aufruf: <venv>/python pruefreihe.py tag=Beschriftung=modell.pkl ..."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
P = ROOT/"prescriptive"
VERGLEICH = P/"scripts/results_100k/modellvergleich_20261116"
PY = str(ROOT/"process-simulation-engine/.venv/bin/python")
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PSP_CATALOG": str(P/"data/catalog_neue_punkte_zuteilung_random.yaml")}


def note(text):
    with open(OUT/"fortschritt.log", "a") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M')} {text}\n")


def run(tag, label, model):
    env = {**ENV, "PSP_VERGLEICH_TAG": f"_{tag}", "PSP_VERGLEICH_MODELS": json.dumps({label: str(model)})}
    cwd = ROOT/"process-simulation-engine"
    (OUT/"logs").mkdir(exist_ok=True)
    steps = [("rauschen", [PY, "-u", str(P/"scripts/psp_rauschen.py"), "--model", str(model), "--workers", "4", "--threads", "2",
                           "--out", str(OUT/f"rauschen_{tag}")])]
    steps += [(name, [PY, "-u", str(VERGLEICH/f"{name}.py")]) for name in ("punkte_alle", "nachlegen_erklaerung", "angebote_einfuegen",
                                                                              "leads_gedaechtnis")]
    steps.insert(3, ("nachlegen_auswerten", [PY, "-u", str(VERGLEICH/"nachlegen_auswerten.py"), f"nachlegen_erklaerung_{tag}"]))
    for name, command in steps:
        with open(OUT/f"logs/{name}_{tag}.log", "w") as handle:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
        note(f"{label}: {name} {'fertig' if result.returncode == 0 else f'FEHLER (Code {result.returncode})'}")


if __name__ == "__main__":
    registry = OUT/"staende.json"
    known = json.loads(registry.read_text()) if registry.exists() else {}
    for spec in sys.argv[1:]:
        tag, label, model = spec.split("=", 2)
        known[tag] = {"label": label, "modell": str(Path(model).resolve())}
        registry.write_text(json.dumps(known, indent=1, ensure_ascii=False))
        note(f"{label}: Prüfreihe beginnt")
        run(tag, label, Path(model).resolve())
    note("Prüfreihen fertig")
