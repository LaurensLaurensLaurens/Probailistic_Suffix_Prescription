"""Berücksichtigen die Modelle in der Nachfassphase, wie viele Angebote der Fall hat? (2026-11-16)

Echte Testfälle, Prefix bis zum ersten W_Call after offers|start (alle Angebote der ersten Runde sind verschickt, jetzt
wartet die Bank auf den Kunden). Gruppen: genau 1 Angebot vorher und 2 oder mehr. Je Prefix 50 gezogene Suffixe mit
OE, D-Netz und (c); verglichen wird die Differenz der Gruppen in der Restzeit mit dem echten Log (beobachtet, für alle
Modelle dieselbe Zielgröße). Aufruf im Ordner process-simulation-engine.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d_ohne_ziehungen as base
from leads_oe import MODELS, TAG
from leads_gedaechtnis import work
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent/("angebote_gedaechtnis"+TAG)
PER_GROUP = 250
CALL = "W_Call after offers|start"


def prefixes():
    from prescriptive.estimators.psp_model import PSPModel
    test = set(map(str, PSPModel.load(next(iter(MODELS.values()))).test_cases))
    raw = pd.read_pickle(base.ROOT/"prescriptive/data/cache/bpic17_raw.pkl")
    raw = raw[raw[base.CASE].astype(str).isin(test)].copy()
    raw[base.ACT] = raw[base.ACT].astype(str).str.strip()
    raw = raw.sort_values([base.CASE, base.TIME], kind="stable")
    rows = []
    for case, part in raw.groupby(base.CASE, sort=False):
        part = part.reset_index(drop=True)
        tok = (part[base.ACT]+"|"+part[base.LIFE].astype(str)).tolist()
        if CALL not in tok:
            continue
        cut = tok.index(CALL)+1
        offers = sum(t == "O_Create Offer|complete" for t in tok[:cut])
        if offers == 0:
            continue
        stamps = part[base.TIME]
        rest_events = tok[cut:]
        rows.append({"case": str(case), "gruppe": "1 Angebot" if offers == 1 else "2+ Angebote", "events_vorher": cut,
                     "real_rest": (stamps.iloc[-1]-stamps.iloc[cut-1]).total_seconds()/base.DAY,
                     "real_offers_later": sum(t == "O_Create Offer|complete" for t in rest_events),
                     "real_cancelled_offers": sum(t == "O_Cancelled|complete" for t in rest_events),
                     "real_pending": "A_Pending|complete" in rest_events, "prefix": part.iloc[:cut].to_dict("records")})
    frame = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    return pd.concat([g.iloc[rng.choice(len(g), min(PER_GROUP, len(g)), replace=False)] for _, g in frame.groupby("gruppe")]), frame


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    picked, everything = prefixes()
    items = picked.to_dict("records")
    with ProcessPoolExecutor(6, mp_context=multiprocessing.get_context("spawn")) as pool:
        drawn = pd.concat(list(pool.map(work, [(w, items[w::6]) for w in range(6)])), ignore_index=True)
    drawn.to_csv(OUT/"ziehungen.csv", index=False)
    pd.set_option("display.width", 250)
    print("Alle Testfälle mit Nachfassen:\n" + everything.groupby("gruppe").agg(n=("case", "size"), rest=("real_rest", "mean"),
          spaetere_angebote=("real_offers_later", "mean"), stornierte_angebote=("real_cancelled_offers", "mean"), pending=("real_pending", "mean")).round(2).to_string())
    rows = {"echter Log": picked.groupby("gruppe").real_rest.mean()}
    for label, part in drawn.groupby("modell"):
        rows[label] = part.groupby("gruppe").rest.mean()
    diff = pd.DataFrame(rows).T
    diff["2+ minus 1"] = diff["2+ Angebote"]-diff["1 Angebot"]
    a, b = picked[picked.gruppe == "2+ Angebote"].real_rest, picked[picked.gruppe == "1 Angebot"].real_rest
    print("\nRestzeit ab erstem Nachfassen (Tage), ausgewählte Prefixe:\n" + diff.round(2).to_string())
    print(f"SE der echten Differenz: {np.sqrt(a.var()/len(a)+b.var()/len(b)):.2f}")
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
