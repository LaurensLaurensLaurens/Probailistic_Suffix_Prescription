"""In wie vielen Entscheidungen kann das PSP die Optionen nicht unterscheiden? (2026-11-23, Nutzer: „in wie vielen der Fälle die Optionen
nicht unterschieden werden konnten?“; Hintergrund: Rückfall auf einen Causal Forest, wenn der Unterschied im Rauschen untergeht.)

Daten: die Ziehungen der Wirkungsdiagnosen (gleiche 300 Simulatorzustände aus Aufbau F, je Option 100 Suffixe, Bedingung event = keyed_noise
wie online). Je Zustand: Δ = Mittel Restzeit andere − Regel-Option (positiv = Regel-Option schneller), gepaart über den Ziehungsindex k
(gleiche Zufallszahlen), SE = SD(Differenz) / √n.
  - nicht unterscheidbar bei 100 Ziehungen: |Δ| < 1,96 · SE(100)
  - bei 300 Ziehungen (wie online): SE(300) = SE(100) / √3; Δ aus den 100 Ziehungen als Schätzung des Modellmittels (etwas zu weit von 0,
    daher eher zu wenige nicht unterscheidbare)
  - unterscheidbar, aber andere Option geschätzt besser (Δ < −1,96 · SE(300)) – im Simulator ist die Regel-Option je Fall die bessere
Gewichtet nach Wahlen je Fall online (tlcrn: konzept 0,570, nach V. 0,220, nachlegen 1. Besuch 0,958, leads 0,608; spätere Besuche von
nachlegen entscheidet online die Bank). Ausgabe unterscheidbarkeit.md / .csv hier. Aufruf: python unterscheidbarkeit.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODELLE = {"tlsvk (T-Learner + Suffix-Verlust + Kalibrierung)": HERE/"wirkung_kalibrierung/ziehungen.pkl",
           "olsv (8 Netze)": HERE/"wirkung_option_learner/ziehungen.pkl",
           "tlsv (T-Learner + Suffix-Verlust)": HERE/"wirkung/ziehungen.pkl",
           "plsv (S-Learner je Wahlpunkt)": HERE/"wirkung_punkt_learner/ziehungen.pkl",
           "T-Learner (ZZ-TF1, ohne Suffix-Verlust)": HERE.parent/"modellvergleich_20261116/crn_schluessel/ziehungen.pkl"}
GEWICHT = {"annehmen_bei_konzept": 0.570, "annehmen_nach_vervollstaendigung": 0.220, "nachlegen 1. Besuch": 0.958, "handle_leads": 0.608}
Z = 1.96

zeilen = []
for name, pfad in MODELLE.items():
    d = pd.read_pickle(pfad)
    d = d[d.bedingung == "event"]
    if "modell" in d.columns and d.modell.nunique() > 1:  # crn_schluessel: mehrere Modelle in einer Datei
        d = d[d.modell.str.contains("T-Learner")]
    breit = d.pivot_table(index=["gruppe", "seed", "case", "visit", "k"], columns="rule", values="rest")
    diff = (breit[False] - breit[True]).dropna()
    je = diff.groupby(level=["gruppe", "seed", "case", "visit"]).agg(["mean", "std", "size"])
    je["se100"] = je["std"] / np.sqrt(je["size"])
    je["se300"] = je["se100"] / np.sqrt(3)
    for gruppe, t in je.groupby(level="gruppe"):
        zeilen.append({"Modell": name, "Gruppe": gruppe, "Zustände": len(t), "Ziehungen": int(t["size"].median()),
                       "nicht unterscheidbar (100)": (t["mean"].abs() < Z * t.se100).mean(),
                       "nicht unterscheidbar (300)": (t["mean"].abs() < Z * t.se300).mean(),
                       "sicher andere besser (300)": (t["mean"] < -Z * t.se300).mean(),
                       "sicher Regel besser (300)": (t["mean"] > Z * t.se300).mean(),
                       "Δ Median (d)": t["mean"].median(), "SE(300) Median (d)": t.se300.median()})
tab = pd.DataFrame(zeilen)
gesamt = []
for name, t in tab.groupby("Modell", sort=False):
    t = t.set_index("Gruppe")
    w = pd.Series(GEWICHT)
    w = w[w.index.isin(t.index)]
    gesamt.append({"Modell": name, **{s: float((t.loc[w.index, s] * w).sum() / w.sum()) for s in
                                      ("nicht unterscheidbar (100)", "nicht unterscheidbar (300)", "sicher andere besser (300)", "sicher Regel besser (300)")}})
gesamt = pd.DataFrame(gesamt)
tab.to_csv(HERE/"unterscheidbarkeit.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 20):
    text = (f"# Unterscheidbarkeit der Optionen je Entscheidung (2026-11-23)\n\nAnteil der Zustände; nicht unterscheidbar = |Δ| < 1,96 · SE.\n\n"
            f"## Je Wahlpunkt\n\n```\n{tab.round(3).to_string(index=False)}\n```\n\n"
            f"## Gewichtet nach Online-Wahlen je Fall (ohne spätere nachlegen-Besuche)\n\n```\n{gesamt.round(3).to_string(index=False)}\n```\n")
(HERE/"unterscheidbarkeit.md").write_text(text)
print(text)
