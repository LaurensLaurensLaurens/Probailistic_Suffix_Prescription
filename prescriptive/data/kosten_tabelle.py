"""
Tabelle EXTRAZWISCHENERGEBNISSE/repair/kosten_echte_werkstatt_repair.png: das Kostenexperiment der Werkstatt in der echten Werkstatt
(repair_kosten.sh, repair_kosten_satz.sh), dazu der Funktionstest der neuen Vorausschau (data/lookahead_test.py, eine Zeile mit *).
Aufruf aus process-simulation-engine/: .venv/bin/python ../prescriptive/data/kosten_tabelle.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")]
from zwischenergebnisse import render

S = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts") + "/"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "EXTRAZWISCHENERGEBNISSE", "repair", "kosten_echte_werkstatt_repair.png")
# Funktionstest der neuen psp_lookahead (data/lookahead_test.log, 03.11.2026): 1 Seed, 600 Ankünfte, gewertet Fälle 100-399, Tarif A
TEST = {"psp": 12076.0, "psp_lookahead": 11849.0}


def runs(folder, policy):
    r = pd.read_csv(S + folder + "/wahrheit/laeufe.csv")
    return r[r.Policy == policy].set_index("seed").sort_index()["Kosten (€)"]


def euro(x, sign=False):
    return (f"{x:+,.0f}" if sign else f"{x:,.0f}").replace(",", ".").replace("-", "−") + " €"


base = {"A": "results_repair_kosten", "B": "results_repair_kosten_b"}
lines = [("Hausregel", "bank", base), ("Zufall", "random", base), ("XGB ohne Sätze, mit A trainiert", "xgb", base),
         ("XGB ohne Sätze, neu auf B trainiert", "xgb", {"B": "results_repair_kosten_b_xgbneu"}),
         ("XGB + Satz, mit A trainiert", "xgb", {"A": "results_repair_kosten_xgbsatz", "B": "results_repair_kosten_b_xgbsatz"}),
         ("XGB + Satz + 48 h, mit A trainiert", "xgb", {"A": "results_repair_kosten_xgbsatz48", "B": "results_repair_kosten_b_xgbsatz48"}),
         ("PSP (nicht neu trainiert)", "psp", base), ("Spezialist (wahre Fähigkeiten)", "spezialist", base)]
rows, highlight = [], []
for k, (label, policy, folders) in enumerate(lines):
    row = [label]
    for tarif in ("A", "B"):
        if tarif not in folders:
            row += ["–", "–", "–"]
            continue
        mine, bank, psp = runs(folders[tarif], policy), runs(base[tarif], "bank"), runs(base[tarif], "psp")
        assert np.allclose(runs(folders[tarif], "bank"), bank)  # gleiche Hausregel-Läufe, also gepaart
        row += [euro(mine.mean()), "–" if policy == "bank" else euro((mine - bank).mean(), True)]
        if policy in ("psp", "bank"):
            row.append("–")
        else:
            d = psp - mine
            row.append(f"{euro(-d.mean(), True)} (± {euro(2.262 * d.std(ddof=1) / np.sqrt(len(d)))[:-2]})")
    rows.append(row)
    if policy == "psp":
        highlight.append(k)
        # direkt unter dem PSP: der Funktionstest der neuen Vorausschau
        rows.append(["PSP mit Vorausschau (neu)*", euro(TEST["psp_lookahead"]) + "*", "–", euro(TEST["psp_lookahead"] - TEST["psp"], True) + "*", "–", "–", "–"])
        highlight.append(k + 1)
header = ["Policy", "Tarif A:\nKosten je Fall", "Tarif A: gegen\nHausregel", "Tarif A: PSP\ngünstiger um", "Tarif B:\nKosten je Fall",
          "Tarif B: gegen\nHausregel", "Tarif B: PSP\ngünstiger um"]
note = ("Die echte Werkstatt (bekannte Wahrheit), 10 Seeds, je 1.000 gewertete Fälle, gepaart je Seed. Kosten je Fall = Lohnkosten nach Stundensatz "
        "(je Mitarbeiter, Verlauf über die Tage, Uhrzeit, Wochenende; prescriptive/costs/repair) + 400 € je Tag Durchlaufzeit. Tarif B: Grundsätze "
        "der Mechaniker vertauscht, Verlauf neu gelost. Alle Modelle auf denselben 5.000 Fällen; XGB mit den Kosten nach Tarif A trainiert, außer "
        "„neu auf B trainiert“. Satz: Stundensatz der Option bei der Wahl als Eingabe von XGB, 48 h: dazu sein Mittel über die nächsten 48 Stunden "
        "(nach dem gültigen Tarif). PSP: das PSP mit gewichteten Wahlpunkten, rechnet die Kosten aus den gezogenen Suffixen mit dem gültigen Tarif. "
        "PSP günstiger um: Kosten dieser Policy minus PSP je Fall, mit 95-%-KI (negativ: diese Policy ist günstiger). "
        f"* Funktionstest der neuen Vorausschau (psp_lookahead: Umverteilen und Warten), nur Tarif A, 1 Seed, 300 gewertete Fälle, andere Fälle als "
        f"die übrigen Zeilen: ihre Kosten sind nicht mit ihnen vergleichbar, nur der Abstand zum PSP auf denselben 300 Fällen (PSP dort "
        f"{euro(TEST['psp'])}), ohne KI; Durchlaufzeit dabei +0,54 Tage. Die volle Auswertung steht aus. Stand 03.11.2026.")
render("Kosten in der echten Werkstatt: Tarif A und Tarifwechsel auf B", header, rows, OUT, note, highlight=highlight)
print("gespeichert:", OUT)
