"""RCT-Daten aus dem korrigierten Simulator (Aufbau F) für das Training von ZZ-TF1 + T-Learner auf Simulatordaten (2026-11-18, Wunsch des
Nutzers). Gleicher Simulatorstand wie die Online-Auswertung F: Katalog d_oe_zustand_20261115/catalog.yaml (zufällige Mitarbeiterzuteilung,
4 Wahlpunkte: annehmen_bei_konzept, annehmen_nach_vervollstaendigung, angebot_nachlegen, handle_leads).

Policy: an jedem Wahlpunkt mit 2 Optionen eine zufällige Option (50/50, Positivity); an angebot_nachlegen nur beim 1. Besuch im Fall
(danach Bank, wie first_only online) – sonst entstünden bei 50 % je Besuch endlose Angebotsschleifen. Alle Fälle randomisiert (delta 0).
RUNS Läufe mit je CASES Fällen parallel (eigene Saat 1000 + r, Fall-IDs mit Präfix R<r>_), zusammen etwa so viele Fälle wie der echte Log
(31.509). Ausgabe: rct_log_raw.pkl (Events wie der Simulator sie liefert, Eingabe für preprocess wie der echte Log), choices.csv,
rct.json. Aufruf im Ordner process-simulation-engine: .venv/bin/python -u <dieses Skript> [Läufe] [Fälle je Lauf]
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
OUT = Path(__file__).resolve().parent
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
CASES = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PSP_CATALOG"] = str(SOURCE/"catalog.yaml")
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths  # noqa: E402
add_paths()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIRST_ONLY = "angebot_nachlegen"


class RCTErsterBesuch:
    """Zufällige Option je Wahlpunkt; an angebot_nachlegen nur beim ersten Besuch, danach None (Bank)."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def choose(self, point, case, options, time, engine):
        if not options:
            return None
        if len(options) == 1:
            return options[0]
        if FIRST_ONLY in point.name.split("+"):
            visits = sum(1 for e in case.prefix if e.get("concept:name") == point.activity and e.get("lifecycle:transition", "complete") == point.lifecycle)
            if visits > 1:
                return None
        return options[int(self.rng.integers(len(options)))]


def simulator():
    import interventions
    interventions.CANDIDATES_FILE = str(SOURCE/"catalog.yaml")
    from prescriptive.helpers.datasets import load_dataset
    from preprocessing import preprocess
    from simulation.dataset_simulator import Simulator
    raw, _, _ = load_dataset("bpic17")
    return Simulator("bpic17", preprocess("bpic17", raw, verbose=False))


def run(r):
    started = time.time()
    events, choices = simulator().run(CASES, policy=RCTErsterBesuch(1000 + r), seed=1000 + r)
    for frame in (events, choices):
        column = "case:concept:name" if "case:concept:name" in frame.columns else "case"
        frame[column] = f"R{r}_" + frame[column].astype(str)
    print(f"Lauf {r}: {events['case:concept:name'].nunique():,} Fälle, {len(events):,} Events, {len(choices):,} Wahlen, "
          f"{time.time() - started:.0f} s", flush=True)
    return events, choices


def main():
    target = OUT/"rct_log_raw.pkl"
    if target.exists():
        sys.exit(f"{target} gibt es schon")
    with ProcessPoolExecutor(max_workers=RUNS, mp_context=multiprocessing.get_context("spawn")) as pool:
        parts = list(pool.map(run, range(RUNS)))
    events = pd.concat([p[0] for p in parts], ignore_index=True)
    choices = pd.concat([p[1] for p in parts], ignore_index=True)
    events.to_pickle(target)
    choices.to_csv(OUT/"choices.csv", index=False)
    two = choices[choices.options == 2] if "options" in choices.columns else choices
    shares = two.groupby(["point", "choice"], dropna=False).size().rename("n").reset_index() if len(two) else pd.DataFrame()
    (OUT/"rct.json").write_text(json.dumps({"runs": RUNS, "cases_per_run": CASES, "seeds": [1000 + r for r in range(RUNS)],
                                            "catalog": str(SOURCE/"catalog.yaml"), "first_only": FIRST_ONLY, "cases": int(events["case:concept:name"].nunique()),
                                            "events": int(len(events))}, indent=2))
    print(shares.to_string(index=False))
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
