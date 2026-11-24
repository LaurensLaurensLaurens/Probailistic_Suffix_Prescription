"""Warum hat OE am Leads-Punkt das falsche Vorzeichen? (2026-11-16)

Dieselben 60 Leads-Zustände wie leads_meilensteine.py, je Option 200 Ziehungen, drei unveränderte Modelle mit
Decoder-Zeiten: OE (token_oe), das D-Netz (klassen_option_encoder) und (c) (normal, Option 1 ms nach der Entscheidung).
Gespeichert je Ziehung: die ersten 12 Token mit Abstand, Zeit bis zu Meilensteinen, kumulierte Zeit nach k Events.
Dazu die beobachteten Abstände im echten Log (Trainingsfälle) direkt nach W_Handle leads|schedule.
Aufruf im Ordner process-simulation-engine.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d_ohne_ziehungen as base
from leads_meilensteine import MILESTONES, milestones, truth
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent/"leads_oe"
M = base.ROOT/"prescriptive/models"
MODELS = {"OE": M/"psp_model_durchlaufzeit_real_alle_s60_token_oe.pkl",
          "D-Netz": M/"psp_model_durchlaufzeit_real_alle_s60_klassen_option_encoder.pkl",
          "(c)": M/"psp_model_durchlaufzeit_real_alle_s60_normal.pkl"}
# Andere Modelle für alle Auswertungen dieses Ordners: PSP_VERGLEICH_MODELS='{"Name": "Pfad", ...}', Ergebnisordner mit
# Endung PSP_VERGLEICH_TAG (die Kindprozesse erben beides)
if os.environ.get("PSP_VERGLEICH_MODELS"):
    MODELS = {label: Path(path) for label, path in json.loads(os.environ["PSP_VERGLEICH_MODELS"]).items()}
TAG = os.environ.get("PSP_VERGLEICH_TAG", "")
DRAWS, STEPS = 200, 12


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
            torch.manual_seed(9000+97*number+index)
            drawn, elapsed = model.option_suffixes(model.points[s["point"]], prefix, list(s["options"]), s["time"], DRAWS)
            option_first = model.option_in_encoder()
            for option, suffixes in zip(s["options"], drawn):
                for k, suffix in enumerate(suffixes):
                    tokens = [name for name, _, _ in suffix]
                    gaps = [gap/base.DAY for _, _, gap in suffix]
                    if not option_first:  # (c): die Option ist nicht Teil des Suffix, Abstand 1 ms
                        tokens, gaps = [option] + tokens, [0.0] + gaps
                    cumulative = np.cumsum(gaps)
                    row = {"modell": label, "case": s["case"], "option": option, "k": k, "events": len(tokens),
                           "tokens": tokens[:STEPS], "gaps": gaps[:STEPS], **milestones(tokens, gaps)}
                    row.update({f"t{j}": cumulative[j-1] if len(cumulative) >= j else cumulative[-1] for j in range(1, STEPS+1)})
                    rows.append(row)
    return pd.DataFrame(rows)


def real_gaps():
    raw = pd.read_pickle(base.ROOT/"prescriptive/data/cache/bpic17_raw.pkl")
    test = set(map(str, pd.read_pickle(MODELS["OE"]).test_cases)) if False else None
    raw = raw.sort_values([base.CASE, base.TIME], kind="stable")
    tok = raw[base.ACT].astype(str).str.strip()+"|"+raw[base.LIFE].astype(str)
    nxt = tok.groupby(raw[base.CASE]).shift(-1)
    gap = (raw[base.TIME].groupby(raw[base.CASE]).shift(-1)-raw[base.TIME]).dt.total_seconds()/base.DAY
    frame = pd.DataFrame({"tok": tok, "next": nxt, "gap": gap})
    frame = frame[(frame.tok == "W_Handle leads|schedule") & frame.next.isin(["W_Handle leads|start", "W_Handle leads|withdraw"])]
    return frame.groupby("next").gap.describe(percentiles=[0.1, 0.5, 0.9])


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    states = base.pick()
    states = states[states.point == "handle_leads"].to_dict("records")
    with ProcessPoolExecutor(6, mp_context=multiprocessing.get_context("spawn")) as pool:
        drawn = pd.concat(list(pool.map(work, [(w, states[w::6]) for w in range(6)])), ignore_index=True)
    drawn.to_pickle(OUT/"ziehungen.pkl")
    real = truth()
    real.to_pickle(OUT/"simulator_rct.pkl")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("Echter Log: Abstand (Tage) W_Handle leads|schedule -> Option")
    print(real_gaps().round(3).to_string())
    print("\nstart minus withdraw, Tage bis Meilenstein (positiv = start langsamer):")
    rows = {"Simulator": {m: real[real.option.str.endswith("start")][m].mean()-real[real.option.str.endswith("withdraw")][m].mean() for m in MILESTONES+["events"]}}
    for label, part in drawn.groupby("modell", sort=False):
        a, b = part[part.option.str.endswith("start")], part[part.option.str.endswith("withdraw")]
        rows[label] = {m: a[m].mean()-b[m].mean() for m in MILESTONES+["events"]}
    print(pd.DataFrame(rows).round(2).to_string())
    print("\nstart minus withdraw, kumulierte Zeit nach k Events ab Entscheidung (Tage):")
    cum = {}
    for label, part in drawn.groupby("modell", sort=False):
        a, b = part[part.option.str.endswith("start")], part[part.option.str.endswith("withdraw")]
        cum[label] = {f"nach {j}": a[f"t{j}"].mean()-b[f"t{j}"].mean() for j in range(1, STEPS+1)}
    print(pd.DataFrame(cum).round(2).to_string())
    for label, part in drawn.groupby("modell", sort=False):
        print(f"\n{label}: häufigste Anfänge (erste 5 Token) je Option, Anteil und mittlere Zeit bis Token 5 / bis Ende")
        part = part.assign(start5=part.tokens.map(lambda t: " > ".join(x.replace("W_", "").replace("application", "app") for x in t[1:5])))
        for option, group in part.groupby("option"):
            top = group.groupby("start5").agg(n=("k", "size"), t5=("t5", "mean"), ende=("ENDE", "mean")).sort_values("n", ascending=False).head(5)
            top["n"] = top.n/len(group)
            print(f"  {option}: Restzeit {group.ENDE.mean():.2f}, erster Abstand {group.gaps.map(lambda g: g[0]).mean():.3f} d")
            print(top.round(3).to_string())
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
