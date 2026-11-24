"""Auswertung (2026-11-17): (c) in Aufbau F mit first_only = angebot_nachlegen gegen den Lauf ohne first_only (16.11.), die feste
Regel und die Bank. Gepaart je Seed (gleiche Fälle, gleiche Bankreferenz). Je Arm: Durchlaufzeit, Gewinn gegen Bank ± SE, Abstand
zur Regel ± SE, Angebote je Fall, Anteil Fälle mit ≥ 4 Angeboten, Rückfallanteil der Wahlen mit 2 Optionen je Punkt.
Ausgabe: c_first_only_auswertung.md/.csv. Aufruf: process-simulation-engine/.venv/bin/python c_first_only_auswerten.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
LAEUFE = {"ohne first_only (16.11.)": OUT/"c_im_korrigierten_simulator", "first_only Nachlegen (17.11.)": OUT/"c_first_only_im_korrigierten_simulator",
          "first_only Nachlegen, 300 Ziehungen (17.11.)": OUT/"c_first_only_300_im_korrigierten_simulator"}
REGEL = OUT/"cf_im_korrigierten_simulator"


def angebote(folder):
    cases = pd.read_csv(folder/"case_costs.csv", index_col=0)
    events = pd.read_pickle(folder/"events.pkl")
    events = events[events["case:concept:name"].isin(cases.index)]
    counts = events[events["concept:name"] == "O_Created"].groupby("case:concept:name").size().reindex(cases.index).fillna(0)
    return counts


def main():
    regel = pd.read_csv(REGEL/"runs.csv")
    regel = regel[regel.arm == "regel"].set_index("seed").cycle_days
    rows, punkte = [], []
    for lauf, folder in LAEUFE.items():
        if not (folder/"runs.csv").exists():
            continue
        runs = pd.read_csv(folder/"runs.csv")
        for arm, part in runs.groupby("arm"):
            part = part.set_index("seed").sort_index()
            gain = part.bank_cycle_days - part.cycle_days
            gap = part.cycle_days - regel.reindex(part.index)
            counts = pd.concat([angebote(folder/f"{arm}_seed{s}") for s in part.index])
            rows.append({"lauf": lauf, "arm": arm, "seeds": len(part), "dlz_tage": part.cycle_days.mean(), "bank_tage": part.bank_cycle_days.mean(),
                         "gewinn_tage": gain.mean(), "gewinn_se": gain.std() / np.sqrt(len(gain)) if len(gain) > 1 else np.nan,
                         "gewinn_je_seed": " / ".join(f"{g:+.2f}" for g in gain), "rueckstand_regel_tage": gap.mean(),
                         "rueckstand_regel_se": gap.std() / np.sqrt(len(gap)) if len(gap) > 1 else np.nan,
                         "angebote_je_fall": counts.mean(), "anteil_ge4_angebote": (counts >= 4).mean(), "max_angebote": int(counts.max()),
                         "kosten_eur": part.cost_eur.mean()})
            for s in part.index:
                choices = pd.read_csv(folder/f"{arm}_seed{s}/choices.csv")
                choices = choices[choices.options == 2]
                for point, g in choices.groupby("point"):
                    punkte.append({"lauf": lauf, "arm": arm, "seed": s, "punkt": point, "wahlen": len(g), "bank": float(g.choice.isna().mean())})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/"c_first_only_auswertung.csv", index=False)
    lines = ["# (c) in Aufbau F: mit und ohne first_only = angebot_nachlegen", "",
             f"Bank {frame.bank_tage.iloc[0]:.2f} d, feste Regel {regel.mean():.2f} d (3 Seeds × 200 Fälle, gepaart je Seed).", "",
             "```", frame.round(3).to_string(index=False), "```", ""]
    if punkte:
        table = pd.DataFrame(punkte).groupby(["lauf", "arm", "punkt"]).apply(
            lambda g: pd.Series({"wahlen": g.wahlen.sum(), "anteil_bank": (g.wahlen * g.bank).sum() / g.wahlen.sum()}), include_groups=False)
        lines += ["## Anteil der Wahlen mit 2 Optionen, bei denen die Bank entscheidet (Rückfall oder first_only)", "", "```",
                  table.round(3).to_string(), "```", ""]
    (OUT/"c_first_only_auswertung.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
