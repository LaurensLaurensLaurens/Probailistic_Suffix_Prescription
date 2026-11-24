"""Wann fällt (c) im Aufbau F auf die Bank zurück? (2026-11-17)
Quelle: c_im_korrigierten_simulator/adaptiv_mit_seed*/choices.csv (leere Wahl = Rückfall, weil |Differenz| <= 2 gepaarte SE nach
30 und 100 Ziehungen; sampling_fallbacks = 0) + events.pkl (Prefix bis zum Entscheidungszeitpunkt). Nur Wahlen mit 2 Optionen.
Vergleich mit der festen Regel (cf_im_korrigierten_simulator/regel_seed*/choices.csv).
Aufruf: process-simulation-engine/.venv/bin/python c_rueckfall_analyse.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SRC = OUT/"c_im_korrigierten_simulator"
REGEL = OUT/"cf_im_korrigierten_simulator"


def merkmale(seed):
    choices = pd.read_csv(SRC/f"adaptiv_mit_seed{seed}/choices.csv")
    choices = choices[choices.options == 2].copy()
    events = pd.read_pickle(SRC/f"adaptiv_mit_seed{seed}/events.pkl")
    events["t"] = events["time:timestamp"].astype("int64") / 1e9
    grouped = {c: g.sort_values("t", kind="stable") for c, g in events.groupby("case:concept:name", sort=False)}
    rows = []
    for r in choices.itertuples():
        case = grouped.get(r.case)
        prefix = case[case.t <= r.time + 1e-6] if case is not None else None
        row = {"seed": seed, "case": r.case, "point": r.point, "choice": r.choice, "rueckfall": pd.isna(r.choice)}
        if prefix is not None and len(prefix):
            names = prefix["concept:name"]
            row.update(angebote=int((names == "O_Created").sum()), ereignisse=len(prefix),
                       alter_tage=(r.time - prefix.t.iloc[0]) / 86400, ziel=prefix["case:LoanGoal"].iloc[0],
                       typ=prefix["case:ApplicationType"].iloc[0], betrag=prefix["case:RequestedAmount"].iloc[0],
                       incomplete=int((names == "W_Call incomplete files").any()),
                       validierung=int((names == "W_Validate application").any()))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["wiederholung"] = frame.groupby(["case", "point"]).cumcount() + 1
    return frame


def main():
    frame = pd.concat([merkmale(s) for s in range(3)], ignore_index=True)
    frame.to_csv(OUT/"c_rueckfall_analyse.csv", index=False)
    lines = ["# Rückfall von (c) auf die Bank im Aufbau F (adaptiv mit Rückfall, 3 Seeds × 200 Fälle)", ""]
    def table(title, by):
        g = frame.groupby(by).agg(n=("rueckfall", "size"), rueckfall=("rueckfall", "mean"))
        lines.extend([f"## {title}", "", "```", g.round(3).to_string(), "```", ""])
        print(title); print(g.round(3).to_string()); print()
    table("je Punkt", "point")
    table("je Punkt und Wiederholung im Fall", ["point", frame.wiederholung.clip(upper=4)])
    table("je Punkt und Angebotszahl bisher", ["point", frame.angebote.clip(upper=4)])
    frame["alter_klasse"] = pd.cut(frame.alter_tage, [-1, 1, 7, 14, 30, 1000], labels=["<1 d", "1-7", "7-14", "14-30", ">30"])
    table("je Punkt und Fallalter", ["point", "alter_klasse"])
    table("je Punkt und Antragsart", ["point", "typ"])
    frame["betrag_klasse"] = pd.qcut(frame.betrag, 4, duplicates="drop")
    table("je Punkt und beantragtem Betrag (Quartile)", ["point", "betrag_klasse"])
    decided = frame[~frame.rueckfall]
    g = decided.groupby(["point", "choice"]).size().rename("n")
    lines.extend(["## Wahl, wenn (c) sicher ist", "", "```", g.to_frame().to_string(), "```", ""])
    print(g.to_string()); print()
    regel = pd.concat([pd.read_csv(REGEL/f"regel_seed{s}/choices.csv").assign(seed=s) for s in range(3)])
    regel = regel[regel.options == 2].groupby(["point", "choice"]).size().rename("n")
    lines.extend(["## Wahl der festen Regel (gleicher Simulator)", "", "```", regel.to_frame().to_string(), "```", ""])
    print(regel.to_string())
    (OUT/"c_rueckfall_analyse.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
