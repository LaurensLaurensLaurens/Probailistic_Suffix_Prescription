"""Wo fehlt dem PSP die Zeit, die ein Zusatzangebot beim 1. Nachlegen kostet? (2026-11-19)

Vergleicht ab der Entscheidung (O_Created → O_Sent oder O_Create Offer) je Quelle und Option:
Zahl O_Sent vor W_Validate application|start, Zeit bis zum 2. und bis zum letzten O_Sent davor, Kundenantwort (letztes O_Sent bis
Validierung) und Nachfass-Start bis Validierung. Quellen: Simulator mit Bank-Fortsetzung und PSP-Ziehungen aus nachlegen_erklaerung
(60 Zustände × 100 Ziehungen je Option), dazu der echte Log (1. O_Created je Fall, beobachtet, nicht randomisiert).
Ausgabe nachlegen_wartezeiten.csv / .md. Aufruf: process-simulation-engine/.venv/bin/python nachlegen_wartezeiten.py [weitere Ordner]
(weitere nachlegen_erklaerung_<tag>-Ordner: deren PSP-Quelle wird zusätzlich ausgewertet)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
VAL = "W_Validate application|start"
QUELLEN = [("nachlegen_erklaerung_t_kopf_steigung", None), ("nachlegen_erklaerung_t_zztf1", "T-Learner (ZZ-TF1)"),
           ("nachlegen_erklaerung", "(c)")]


def metrics(tokens, gaps):
    cum = np.cumsum(gaps)
    stop = next((j for j, t in enumerate(tokens) if t == VAL), None)
    sends = [j for j in range(len(tokens) if stop is None else stop) if tokens[j].startswith("O_Sent")]
    call = next((j for j, t in enumerate(tokens) if t == "W_Call after offers|start"), None)
    return {"O_Sent vor Validierung": len(sends),
            "bis 2. O_Sent": cum[sends[1]] if len(sends) > 1 else np.nan,
            "bis letztes O_Sent": cum[sends[-1]] if sends else np.nan,
            "letztes O_Sent bis Validierung": cum[stop] - cum[sends[-1]] if stop is not None and sends else np.nan,
            "Nachfass-Start bis Validierung": cum[stop] - cum[call] if call is not None and stop is not None and stop > call else np.nan}


def drawn():
    frames, seen = [], set()
    for folder, only in QUELLEN + [(extra, None) for extra in sys.argv[1:]]:
        v = pd.read_pickle(OUT/folder/"verlaeufe.pkl")
        v = v[~v.quelle.isin(seen)] if only is None else v[v.quelle == only]
        seen |= set(v.quelle)
        m = pd.DataFrame([metrics(list(t), list(g)) for t, g in zip(v.tokens, v.gaps)])
        m["quelle"], m["opt"] = v.quelle.values, np.where(v.option.str.startswith("O_Create"), "extra", "senden")
        frames.append(m)
    return frames


def real():
    d = pd.read_pickle(ROOT/"prescriptive/data/cache/bpic17_raw.pkl")
    d["concept:name"] = d["concept:name"].str.strip()
    d = d.sort_values(["case:concept:name", "time:timestamp"], kind="stable")
    d["tok"] = d["concept:name"] + "|" + d["lifecycle:transition"]
    rows = []
    for _, g in d.groupby("case:concept:name", sort=False):
        names = g["concept:name"].tolist()
        if "O_Created" not in names:
            continue
        i = names.index("O_Created")
        if i + 1 >= len(names) or names[i + 1] not in ("O_Sent (mail and online)", "O_Create Offer"):
            continue
        days = g["time:timestamp"].astype("int64").to_numpy() / 86400e9
        r = metrics(g.tok.tolist()[i + 1:], list(np.diff(days[i:])))
        r["quelle"], r["opt"] = "echter Log (beobachtet)", "extra" if names[i + 1] == "O_Create Offer" else "senden"
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    table = pd.concat(drawn() + [real()]).groupby(["quelle", "opt"]).mean()
    table["n"] = pd.concat(drawn() + [real()]).groupby(["quelle", "opt"]).size()
    diff = table.xs("extra", level="opt") - table.xs("senden", level="opt")
    table.to_csv(OUT/"nachlegen_wartezeiten.csv")
    text = ["# Wartezeiten nach dem 1. Nachlegen (Tage ab Entscheidung, Mittel; 2026-11-19)", "", table.round(2).to_string(), "",
            "## extra − senden", "", diff.drop(columns="n").round(2).to_string(), ""]
    (OUT/"nachlegen_wartezeiten.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
