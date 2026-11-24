"""Optionsunterschied an allen Wahlpunkten gegen die Simulator-Wahrheit (2026-11-16).

Dieselben Zustände wie d_ohne_ziehungen.py (je Gruppe 60: Konzept, nach Vervollständigung, Nachlegen 1. Besuch,
Nachlegen ab 2. Besuch, Leads), je Option 100 Ziehungen mit den unveränderten Modellen OE, D-Netz und (c)
(Zeiten aus dem Decoder). Wahrheit: randomisierte Entscheidungen der Simulator-RCT-Daten vom 14.11.
Gespeichert je Ziehung: Restzeit, Endzustand, Events, Zeit bis A_Concept / erstes O_Sent / W_Validate start.
Aufruf im Ordner process-simulation-engine.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d_ohne_ziehungen as base
from leads_oe import MODELS, TAG
from leads_meilensteine import milestones
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent/("punkte_alle"+TAG)
DRAWS = 100


def group(point, visit):
    return ("nachlegen 1. Besuch" if visit == 1 else "nachlegen ab 2. Besuch") if point == "angebot_nachlegen" else point


def work(task):
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
            torch.manual_seed(17000+97*number+index)
            drawn, _ = model.option_suffixes(model.points[s["point"]], prefix, list(s["options"]), s["time"], DRAWS)
            for option, suffixes in zip(s["options"], drawn):
                for k, suffix in enumerate(suffixes):
                    tokens, gaps = [n for n, _, _ in suffix], [g/base.DAY for _, _, g in suffix]
                    if not model.option_in_encoder():
                        tokens, gaps = [option]+tokens, [0.0]+gaps
                    rows.append({"modell": label, "gruppe": group(s["point"], s["visit"]), "seed": s["seed"], "case": s["case"],
                                 "visit": s["visit"], "rule": option == base.RULE[s["point"]], "k": k, "events": len(tokens),
                                 "end": base.end_state(tokens), "offers": sum(t.startswith("O_Create Offer") for t in tokens[1:]),
                                 **milestones(tokens, gaps)})
    return pd.DataFrame(rows)


def truth():
    rct = pd.read_pickle(base.ROOT/"prescriptive/scripts/results_100k/d_rct_tf1_training_20261114/rct_data/rct_bpic17_angebot_nachlegen_delta50/log.pkl")
    rct = rct[rct["rct"].astype(bool)].sort_values([base.CASE, base.TIME], kind="stable")
    decisions = {"A_Concept|complete": "annehmen_bei_konzept", "W_Complete application|suspend": "annehmen_nach_vervollstaendigung",
                 "O_Created|complete": "angebot_nachlegen", "W_Handle leads|schedule": "handle_leads"}
    other = {"annehmen_bei_konzept": "W_Complete application|start", "annehmen_nach_vervollstaendigung": "W_Complete application|resume",
             "angebot_nachlegen": "O_Create Offer|complete", "handle_leads": "W_Handle leads|start"}
    rows = []
    for _, part in rct.groupby(base.CASE, sort=False):
        tokens = (part[base.ACT].astype(str).str.strip()+"|"+part[base.LIFE].astype(str)).tolist()
        times = part[base.TIME].to_numpy()
        visits = {}
        for i in range(len(tokens)-1):
            point = decisions.get(tokens[i])
            if point is None:
                continue
            rule = base.RULE[point] if "|" in base.RULE[point] else base.RULE[point]+"|complete"
            if tokens[i+1] not in (rule, other[point]):
                continue
            visits[point] = visits.get(point, 0)+1
            if point != "angebot_nachlegen" and visits[point] > 1:
                continue
            gaps = np.diff(times[i:]).astype("timedelta64[ms]").astype(float)/86400000
            rows.append({"gruppe": group(point, visits[point]), "rule": tokens[i+1] == rule, "ENDE": gaps.sum(),
                         "events": len(tokens)-i-1, "end": base.end_state(tokens[i+1:])})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    states = base.pick().to_dict("records")
    order = np.random.default_rng(2).permutation(len(states))
    with ProcessPoolExecutor(6, mp_context=multiprocessing.get_context("spawn")) as pool:
        drawn = pd.concat(list(pool.map(work, [(w, [states[i] for i in order[w::6]]) for w in range(6)])), ignore_index=True)
    drawn.to_pickle(OUT/"ziehungen.pkl")
    real = truth()
    real.to_pickle(OUT/"simulator_rct.pkl")
    order_groups = ["annehmen_bei_konzept", "annehmen_nach_vervollstaendigung", "nachlegen 1. Besuch", "nachlegen ab 2. Besuch", "handle_leads"]
    rows = []
    for g in order_groups:
        r = real[real.gruppe == g]
        a, b = r[r.rule], r[~r.rule]
        rows.append({"Gruppe": g, "Quelle": "Simulator", "Vorteil Regel-Option": b.ENDE.mean()-a.ENDE.mean(),
                     "SE": np.sqrt(a.ENDE.var()/len(a)+b.ENDE.var()/len(b)),
                     "Events mehr (andere)": b.events.mean()-a.events.mean(),
                     "Cancelled Regel/andere": f"{(a.end == 'A_Cancelled').mean():.2f}/{(b.end == 'A_Cancelled').mean():.2f}"})
        for label, part in drawn[drawn.gruppe == g].groupby("modell", sort=False):
            wide = part.pivot_table(index=["seed", "case", "visit", "k"], columns="rule", values="ENDE")
            per = wide.groupby(level=[0, 1, 2])
            diff = per.apply(lambda t: (t[False]-t[True]).mean())
            se = per.apply(lambda t: (t[False]-t[True]).std(ddof=1)/np.sqrt(len(t)))
            a, b = part[part.rule], part[~part.rule]
            rows.append({"Gruppe": g, "Quelle": label, "Vorteil Regel-Option": diff.mean(), "SE": diff.std(ddof=1)/np.sqrt(len(diff)),
                         "Anteil Zustände richtig (100 Ziehungen)": (diff > 0).mean(), "MC-SE je Entscheidung": se.median(),
                         "Events mehr (andere)": b.events.mean()-a.events.mean(),
                         "Cancelled Regel/andere": f"{(a.end == 'A_Cancelled').mean():.2f}/{(b.end == 'A_Cancelled').mean():.2f}",
                         "Angebote Regel/andere": f"{a.offers.mean():.2f}/{b.offers.mean():.2f}"})
    table = pd.DataFrame(rows)
    table.to_csv(OUT/"vergleich.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(table.round(2).to_string(index=False))
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
