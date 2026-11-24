"""Leads: was passiert zwischen A_Concept und dem ersten O_Sent? (2026-11-16)

Dieselben 60 Leads-Zustände wie leads_oe.py, je Option 100 Ziehungen mit OE und dem D-Netz, dazu die randomisierten
Simulatorfälle. Je Verlauf im Abschnitt von A_Concept bis zum ersten O_Sent: Zahl der Events, Zahl der Pausen und
Wiederaufnahmen der Antragsvervollständigung, Zeit in Abständen ab 1 Tag, und welches Event auf die langen Abstände
folgt. Aufruf im Ordner process-simulation-engine.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d_ohne_ziehungen as base
from leads_oe import MODELS
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent/"leads_abschnitt"
DRAWS = 100
CONCEPT, SENT = "A_Concept|complete", "O_Sent (mail and online)|complete"


def phase(tokens, gaps):
    """Kennzahlen zwischen dem ersten A_Concept und dem ersten O_Sent (beide eingeschlossen als Grenzen)."""
    if CONCEPT not in tokens or SENT not in tokens:
        return None
    i, j = tokens.index(CONCEPT), tokens.index(SENT)
    if j < i:
        return None
    window, waits = tokens[i+1:j+1], gaps[i+1:j+1]
    long = Counter()
    for token, gap in zip(window, waits):
        if gap >= 1.0:
            long[token] += gap
    return {"events": len(window), "tage": float(sum(waits)), "lange_tage": float(sum(g for g in waits if g >= 1.0)),
            "suspend": sum(t == "W_Complete application|suspend" for t in window),
            "resume": sum(t == "W_Complete application|resume" for t in window),
            "angebote": sum(t == "O_Create Offer|complete" for t in window), "long_by": dict(long)}


def work(task):
    number, states = task
    import torch
    from prescriptive.estimators.psp_model import PSPModel
    torch.set_num_threads(1)
    models = {label: PSPModel.load(MODELS[label]) for label in ("OE", "D-Netz")}
    events = {seed: pd.read_pickle(base.SOURCE/base.RUN.format(seed)/"events.pkl") for seed in sorted({s["seed"] for s in states})}
    rows = []
    for index, s in enumerate(states):
        part = events[s["seed"]]
        part = part.loc[part[base.CASE].astype(str).eq(s["case"])].sort_values(base.TIME, kind="stable").reset_index(drop=True)
        stamps = part[base.TIME].map(lambda t: pd.Timestamp(t).timestamp()).to_numpy()
        prefix = part.iloc[:int(np.searchsorted(stamps, s["time"]+1e-6, side="right"))].to_dict("records")
        for label, model in models.items():
            model.set_threads(1)
            torch.manual_seed(13000+97*number+index)
            drawn, _ = model.option_suffixes(model.points[s["point"]], prefix, list(s["options"]), s["time"], DRAWS)
            for option, suffixes in zip(s["options"], drawn):
                for suffix in suffixes:
                    result = phase([n for n, _, _ in suffix], [g/base.DAY for _, _, g in suffix])
                    rows.append({"quelle": label, "option": option.split("|")[1], "erreicht": result is not None, **(result or {})})
    return pd.DataFrame(rows)


def simulator():
    rct = pd.read_pickle(base.ROOT/"prescriptive/scripts/results_100k/d_rct_tf1_training_20261114/rct_data/rct_bpic17_angebot_nachlegen_delta50/log.pkl")
    rct = rct[rct["rct"].astype(bool)].sort_values([base.CASE, base.TIME], kind="stable")
    rows = []
    for _, part in rct.groupby(base.CASE, sort=False):
        tokens = (part[base.ACT].astype(str).str.strip()+"|"+part[base.LIFE].astype(str)).tolist()
        for i in range(len(tokens)-1):
            if tokens[i] == "W_Handle leads|schedule" and tokens[i+1] in ("W_Handle leads|start", "W_Handle leads|withdraw"):
                gaps = np.r_[0.0, np.diff(part[base.TIME].to_numpy()[i+1:]).astype("timedelta64[ms]").astype(float)/86400000]
                result = phase(tokens[i+1:], list(gaps))
                rows.append({"quelle": "Simulator", "option": tokens[i+1].split("|")[1], "erreicht": result is not None, **(result or {})})
                break
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    states = base.pick()
    states = states[states.point == "handle_leads"].to_dict("records")
    with ProcessPoolExecutor(6, mp_context=multiprocessing.get_context("spawn")) as pool:
        drawn = pd.concat(list(pool.map(work, [(w, states[w::6]) for w in range(6)])), ignore_index=True)
    frame = pd.concat([simulator(), drawn], ignore_index=True)
    frame.to_pickle(OUT/"abschnitte.pkl")
    pd.set_option("display.width", 250)
    ok = frame[frame.erreicht]
    table = ok.groupby(["quelle", "option"])[["events", "tage", "lange_tage", "suspend", "resume", "angebote"]].mean()
    table["erreicht"] = frame.groupby(["quelle", "option"]).erreicht.mean()
    print("Abschnitt A_Concept bis erstes O_Sent, Mittel je Verlauf:\n" + table.round(2).to_string())
    print("\nstart minus withdraw:\n" + (table.xs("start", level=1)-table.xs("withdraw", level=1)).round(2).to_string())
    print("\nTage in langen Abständen (>= 1 d) nach dem Event, das danach kommt, je Verlauf:")
    for (source, option), group in ok.groupby(["quelle", "option"]):
        total = Counter()
        for entry in group.long_by:
            total.update(entry)
        print(f"  {source:<10} {option:<9}", "; ".join(f"{k} {v/len(group):.2f}" for k, v in total.most_common(4)))
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
