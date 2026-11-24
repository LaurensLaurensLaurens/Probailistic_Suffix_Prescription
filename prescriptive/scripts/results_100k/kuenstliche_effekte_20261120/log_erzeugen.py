"""Trainings-Log aus dem Simulator mit künstlichen Effekten (2026-11-20, Nutzer: „okay mach das“ – Bank-Log v3 erzeugen, PSP darauf
trainieren, Conformance online auswerten). Gleicher Simulatorstand wie die Online-Auswertung F (Katalog d_oe_zustand_20261115, zufällige
Mitarbeiterzuteilung, 4 Wahlpunkte), Effekte aus kuenstliche_effekte/<Effektdatei> (Standard effekte_v3.yaml), Policy None = Bank
(historisches Verhalten, beobachtende Daten wie der echte Log). RUNS Läufe mit je CASES Fällen parallel (Saat 2000 + r, Fall-IDs mit
Präfix K<r>_), zusammen etwa so viele Fälle wie der echte Log.
Ausgabe in log_<Effektname>/: log_raw.pkl (Events wie Simulator.run, Eingabe für train_psp_real.py --log-pickle), effekte.csv (effect_log
je ausgeführter Option: Teilgruppe, Tage, Ablauf, Abweichung gezogen / angewandt), info.json.
Aufruf im Ordner process-simulation-engine: .venv/bin/python -u <dieses Skript> [Läufe] [Fälle je Lauf] [Effektdatei] [Namenszusatz]
"""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/"prescriptive/policies/ev_policy_event_costs/results/d_oe_zustand_20261115"
HERE = Path(__file__).resolve().parent
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
CASES = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
EFFEKTE = sys.argv[3] if len(sys.argv) > 3 else "kuenstliche_effekte/effekte_v3.yaml"
NAME = sys.argv[4] if len(sys.argv) > 4 else ""  # Zusatz im Ordnernamen, z. B. _x4 (2026-11-21)
OUT = HERE/f"log_{Path(EFFEKTE).stem}{NAME}"
WORKERS = min(RUNS, 16)  # Speicher: höchstens 16 Läufe gleichzeitig
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PSP_CATALOG"] = str(SOURCE/"catalog.yaml")
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths  # noqa: E402
add_paths()
import pandas as pd  # noqa: E402


def simulator():
    import interventions
    interventions.CANDIDATES_FILE = str(SOURCE/"catalog.yaml")
    from prescriptive.helpers.datasets import load_dataset
    from preprocessing import preprocess
    from simulation.dataset_simulator import Simulator
    raw, _, _ = load_dataset("bpic17")
    return Simulator("bpic17", preprocess("bpic17", raw, verbose=False))


def run(r):
    import kuenstliche_effekte as ke
    ke.aktivieren(ke.laden(EFFEKTE))
    started = time.time()
    events, _ = simulator().run(CASES, policy=None, seed=2000 + r)
    effekte = pd.DataFrame(ke.LETZTE_ENGINE.effect_log)
    events["case:concept:name"] = f"K{r}_" + events["case:concept:name"].astype(str)
    effekte["case"] = f"K{r}_" + effekte["case"].astype(str)
    print(f"Lauf {r}: {events['case:concept:name'].nunique():,} Fälle, {len(events):,} Events, {len(effekte):,} Optionen, "
          f"{time.time() - started:.0f} s", flush=True)
    return events, effekte


def main():
    OUT.mkdir(exist_ok=True)
    target = OUT/"log_raw.pkl"
    if target.exists():
        sys.exit(f"{target} gibt es schon")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        parts = list(pool.map(run, range(RUNS)))
    events = pd.concat([p[0] for p in parts], ignore_index=True)
    effekte = pd.concat([p[1] for p in parts], ignore_index=True)
    events.to_pickle(target)
    effekte.to_csv(OUT/"effekte.csv", index=False)
    info = {"runs": RUNS, "cases_per_run": CASES, "seeds": [2000 + r for r in range(RUNS)], "effekte": EFFEKTE,
            "catalog": str(SOURCE/"catalog.yaml"), "policy": "Bank (None)", "cases": int(events["case:concept:name"].nunique()),
            "events": int(len(events)), "optionen": effekte.groupby(["point", "option"]).size().rename("n").reset_index().to_dict("records")}
    (OUT/"info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
