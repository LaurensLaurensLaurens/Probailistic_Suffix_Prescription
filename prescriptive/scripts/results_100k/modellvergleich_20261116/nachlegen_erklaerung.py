"""Warum unterschätzen die Modelle den Effekt beim ersten Nachlegen? (2026-11-16)

1. Wahrheit mit Bankfortsetzung: eigener Simulatorlauf (korrigierter Simulator, Katalog des D/OE-Vergleichs, zufällige
   Zuteilung). Eine Policy randomisiert 50/50 nur die erste Entscheidung je Fall an angebot_nachlegen; alle späteren
   Besuche und alle anderen Punkte entscheidet die Bank (None). 4 Seeds x 3.000 Ankünfte, gewertet Fälle 500 bis 2.499.
2. Gezogene Verläufe: dieselben 60 Zustände "Nachlegen 1. Besuch" wie punkte_alle.py, je Option 100 Ziehungen mit OE,
   D-Netz und (c) (unverändert, Decoder-Zeiten). Gespeichert werden vollständige Token- und Abstandsfolgen ab der
   Entscheidung, damit die Abschnitte frei ausgewertet werden können.
Aufruf im Ordner process-simulation-engine.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d_ohne_ziehungen as base
from leads_oe import MODELS, TAG
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent/("nachlegen_erklaerung"+TAG)
DRAWS, SEEDS, ARRIVALS, WINDOW = 100, (0, 1, 2, 3), 3000, (500, 2500)


class FirstVisitRandom:
    """Nur der erste Besuch je Fall an angebot_nachlegen wird 50/50 gelost; sonst entscheidet die Bank."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(1000+seed)
        self.drawn = {}

    def choose(self, point, case, options, time, engine):
        if point.name != "angebot_nachlegen" or len(options) != 2:
            return None
        visits = sum(1 for e in case.prefix if str(e.get(base.ACT)).strip() == "O_Created")
        if visits != 1 or case.case_id in self.drawn:
            return None
        choice = options[int(self.rng.integers(2))]
        self.drawn[case.case_id] = choice
        return choice


def simulate(seed):
    import interventions
    interventions.CANDIDATES_FILE = str(base.SOURCE/"catalog.yaml")
    from prescriptive.helpers.datasets import load_dataset
    from preprocessing import preprocess
    from simulation.dataset_simulator import Simulator
    raw, _, _ = load_dataset("bpic17")
    sim = Simulator("bpic17", preprocess("bpic17", raw))
    policy = FirstVisitRandom(seed)
    events, _ = sim.run(ARRIVALS, policy=policy, window=WINDOW, seed=seed)
    numbers = events[base.CASE].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
    events = events.loc[numbers.ge(WINDOW[0]) & numbers.lt(WINDOW[1])].sort_values([base.CASE, base.TIME], kind="stable")
    rows = []
    for case, part in events.groupby(base.CASE, sort=False):
        if case not in policy.drawn:
            continue
        tokens = (part[base.ACT].astype(str).str.strip()+"|"+part[base.LIFE].astype(str)).tolist()
        i = tokens.index("O_Created|complete")
        times = part[base.TIME].to_numpy()
        gaps = list(np.diff(times[i:]).astype("timedelta64[ms]").astype(float)/86400000)
        rows.append({"quelle": "Simulator (Bank danach)", "seed": seed, "case": case, "option": policy.drawn[case],
                     "tokens": tokens[i+1:], "gaps": gaps})
    return pd.DataFrame(rows)


def draw(task):
    number, states = task
    import torch
    from prescriptive.estimators.psp_model import PSPModel
    torch.set_num_threads(1)
    models = {label: PSPModel.load(path) for label, path in MODELS.items()}
    events = {seed: pd.read_pickle(base.SOURCE/base.RUN.format(seed)/"events.pkl") for seed in sorted({s["seed"] for s in states})}
    rows = []
    for index, s in enumerate(states):
        part = events[s["seed"]]
        part = part.loc[part[base.CASE].astype(str).eq(s["case"])].sort_values(base.TIME, kind="stable").reset_index(drop=True)
        stamps = part[base.TIME].map(lambda t: pd.Timestamp(t).timestamp()).to_numpy()
        prefix = part.iloc[:int(np.searchsorted(stamps, s["time"]+1e-6, side="right"))].to_dict("records")
        for label, model in models.items():
            model.set_threads(1)
            torch.manual_seed(19000+97*number+index)
            drawn, _ = model.option_suffixes(model.points[s["point"]], prefix, list(s["options"]), s["time"], DRAWS)
            for option, suffixes in zip(s["options"], drawn):
                for suffix in suffixes:
                    tokens, gaps = [n for n, _, _ in suffix], [g/base.DAY for _, _, g in suffix]
                    if not model.option_in_encoder():
                        tokens, gaps = [option+"|complete"]+tokens, [0.0]+gaps
                    rows.append({"quelle": label, "seed": s["seed"], "case": s["case"], "option": option, "tokens": tokens, "gaps": gaps})
    return pd.DataFrame(rows)


def run(job):
    kind, payload = job
    return simulate(payload) if kind == "sim" else draw(payload)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(OUT/"matplotlib")
    states = base.pick()
    states = states[(states.point == "angebot_nachlegen") & (states.visit == 1)].to_dict("records")
    jobs = [("sim", seed) for seed in SEEDS] + [("draw", (w, states[w::5])) for w in range(5)]
    with ProcessPoolExecutor(len(jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        frame = pd.concat(list(pool.map(run, jobs)), ignore_index=True)
    frame.to_pickle(OUT/"verlaeufe.pkl")
    print(frame.groupby(["quelle", "option"]).size().to_string())
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
