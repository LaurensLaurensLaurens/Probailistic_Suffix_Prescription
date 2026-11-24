"""Bild EXTRAZWISCHENERGEBNISSE/2017/online_auswertung_bpic17.png (Stand 2026-11-17): alle Online-Läufe im Simulator, je Aufbau ein
Abschnitt (zeitlich geordnet), aus online.csv (online_sammeln.py). Gewinn = Durchlaufzeit Bank − Policy im selben Aufbau, gepaart je Seed.
Aufruf: <venv>/python online_tabelle_bild.py   (nach register_erstellen.py)
"""
import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
R = OUT.parent
P = R.parents[1]
ZIEL = P.parent/"EXTRAZWISCHENERGEBNISSE/2017/online_auswertung_bpic17.png"
sys.path.insert(0, str(P/"scripts"))
from zwischenergebnisse import number, render  # noqa: E402

ORDNUNG = ["K", "I", "J", "H", "A", "E", "B", "C", "D", "G", "F", "L"]  # zeitlich: 27.10. … 16.11.
KURZ = {"K": "Mitarbeiterwahl, Simulator-RCT-Modell (27.10.), 5 × 2.000 Fälle",
        "I": "Kanal + 2 Mitarbeiterpunkte (28.10.), 5 × 2.000 Fälle",
        "J": "nur Mitarbeiterwahl W_Call incomplete files, 5 × 2.000 Fälle",
        "H": "12 Interventionspunkte (Kontrollfluss + 8 Mitarbeiter), 10 × 1.000 Fälle",
        "A": "4 Wahlpunkte, älterer Simulator, 10 × 1.000 Fälle",
        "E": "4 Wahlpunkte, älterer Simulator, 300 Ziehungen, fertige Seeds × 1.000 Fälle",
        "B": "4 Wahlpunkte, gelernte Zuteilung, ERM-Ziel γ = 0,1, 10 × 1.000 Fälle",
        "C": "4 Wahlpunkte, 1 Seed × 200 Fälle, 30 Ziehungen",
        "D": "4 Wahlpunkte, 1 Seed × 1.000 Fälle, 300 Ziehungen",
        "G": "4 Wahlpunkte, Eventkosten-Ziel, 1 Seed × 200 Fälle",
        "F": "4 Wahlpunkte, korrigierter Simulator (16./17.11.), 3 × 200 Fälle; first_only: Nachlegen nur beim 1. Besuch, danach Bank",
        "L": "wie F, 12 Seeds × 200 Fälle (17.11.), alle PSP mit first_only; genauerer gepaarter Vergleich"}


def kurz(variant, label):
    """Kurze Variante: Modellname vorn weg, lange Zusätze knapp."""
    ersetzen = {"(zustandsbewusste Kopie: Arbeitszustand + empirische Abstandstabelle)": "(zustandsbewusst)",
                "(Original-Netz, Decoder-Zeiten)": "", "Unsicherheitsrückfall": "Rückfall", "direkte Restzeitbewertung":
                "Restzeitkopf (nicht mehr erlaubt)", "mit Restzeitkopf, Suffixbewertung": "mit Restzeitkopf, Suffixe",
                "Zeit ohne": "ohne Rückfall", "Zeit mit": "mit Rückfall", "adaptiv ohne": "adaptiv, ohne Rückfall",
                "adaptiv mit": "adaptiv, mit Rückfall", "fest 100": "100 Ziehungen fest", "OE mit Rückfall (token_oe_ue)": "mit Rückfall",
                "nur 1. Besuch Nachlegen (first_only), ": "first_only, "}
    codes = {"regel": "feste AIPW-Regel", "causal_forest": "–", "x_learner": "–", "s_learner": "–", "xgb": "–", "psp (c)": "100 Ziehungen",
             "causal_forest_erm": "ERM", "x_learner_erm": "ERM", "s_learner_erm": "ERM", "psp_erm": "ERM, 100 Ziehungen",
             "c_tf1": "Teacher Forcing 1", "e": "Zeitklasse je Event"}
    if variant in codes:
        return codes[variant]
    for old, new in ersetzen.items():
        variant = variant.replace(old, new)
    for prefix in ("psp, ", "(a), ", "(b), ", "(c), ", "(d), ", "OE, ", "s40, ", "OE "):
        if variant.startswith(prefix):
            variant = variant[len(prefix):]
    return variant.strip() or "–"


def paare_l():
    """Gepaarte Differenzen in Aufbau L (12 Seeds) aus f_zwoelf_paare.csv als Satz für die Fußnote."""
    path = R/"modellvergleich_20261116/f_zwoelf_paare.csv"
    if not path.exists():
        return ""
    t = pd.read_csv(path)
    names = {"regel": "Regel", "c_fest100": "(c) 100", "c_fest300": "(c) 300", "zztf1_fest100": "ZZ-TF1", "tl_fest300": "T-Learner 300",
             "tl_fest500": "T-Learner 500"}
    parts = []
    for r in t.itertuples():
        a, b = r.vergleich.split(" schneller als ")
        parts.append(f"{names.get(a, a)} − {names.get(b, b)} {number(r.tage, 2, sign=True)} ± {number(r.se, 2)}")
    return "L, gepaart je Seed (Tage Vorsprung der ersten): " + "; ".join(parts) + ". "


def main():
    online = pd.read_csv(OUT/"online.csv")
    online = online[online.label != "(Fehler)"].copy()
    online["kurz"] = online.aufbau.str[0]
    header = ["Aufbau / Policy", "Variante", "Zieh-\nungen", "Durchlauf-\nzeit (d)", "Bank\n(d)", "Gewinn\n(d)", "± SE\n(d)", "Uplift",
              "Angebote\nje Fall"]
    rows, highlight = [], []
    for letter in ORDNUNG + sorted(set(online.kurz) - set(ORDNUNG)):
        part = online[online.kurz == letter]
        if part.empty:
            continue
        highlight.append(len(rows))
        rows.append([f"{letter}: {KURZ.get(letter, part.aufbau.iloc[0][3:])}"] + [""] * (len(header) - 1))  # Abschnitt
        merged = part.groupby("aufbau", sort=False)
        for _, sub in merged:
            for r in sub.sort_values("gewinn_tage", ascending=False).itertuples():
                rows.append([r.label, kurz(str(r.variante), r.label), "–" if pd.isna(r.ziehungen) else f"{r.ziehungen:.0f}",
                             number(r.dlz_tage, 2), number(r.bank_tage, 2), number(r.gewinn_tage, 2, sign=True),
                             "–" if pd.isna(r.se_tage) else number(r.se_tage, 2), number(100 * r.gewinn_tage / r.bank_tage, 1, sign=True) + " %",
                             "–" if pd.isna(getattr(r, "angebote_je_fall", np.nan)) else number(r.angebote_je_fall, 2)])
    note = ("Gewinn = Durchlaufzeit der Bank minus Policy im selben Aufbau (positiv = schneller), ± Standardfehler über die Seeds (gepaart), "
            "wo mehrere Seeds vorliegen. Aufbauten unterscheiden sich in Simulatorstand, Zuteilung, Wahlpunkten, Fallzahl und Ziel und sind nur in "
            "sich vergleichbar. PSP-Modelle: (c) normale Abstände, Decoder 60; D Zeitklassen, Option im Encoder; OE Token mit Zeitklasse; "
            "s40 Decoder 40; 10k Original Decoder 5; Regel = feste AIPW-Regel je Punkt. F: D/OE als zustandsbewusste Kopien (Arbeitszustand + "
            "empirische Abstandstabelle). E: 300er-Lauf nach 5 bzw. 7 Seeds abgebrochen, Werte im Log auf 0,1 Tage gerundet. "
            "H, Verteilungsenden gegen Bank (früheres Bild vom 01.11.; P90 / CVaR 90 %, Tage, positiv = kürzer): Zufall +0,6 / +1,6, "
            "PSP Nachtlauf 0,0 / +0,1, XGB −4,7 / −7,6. "
            "F, first_only (17.11.): weiterhin alle 4 Wahlpunkte; an angebot_nachlegen entscheidet die Policy nur beim 1. Besuch im Fall, bei "
            "jedem weiteren Besuch die Bank. Grund: ohne first_only wählte (c) ab dem 2. Besuch fast immer ein weiteres Angebot (Simulator-Wahrheit: "
            "senden +0,4 d) und erzeugte eine Angebotsschleife (bis 31 Besuche, 2,4 Angebote je Fall); der 1. Besuch bleibt, weil senden dort "
            "+2,1 d bringt und (c) dort meist richtig wählt. Gepaart bringt first_only bei 100 Ziehungen fest +1,7 d. Angebote je Fall: Bank 1,31. "
            "Rückfall = bei |Differenz| ≤ 2 gepaarte SE entscheidet die Bank (schadet: unsichere Entscheidungen zeigen trotzdem zu 78–88 % in "
            "Regel-Richtung). "
            + paare_l() +
            "Quelle: prescriptive/scripts/results_100k/psp_register/online.csv (je Zeile mit Quelldatei). "
            f"Stand {datetime.date.today():%d.%m.%Y}.")
    render("Online-Auswertung BPIC17 im Simulator: alle Läufe", header, rows, str(ZIEL), note, highlight=highlight, span=highlight)
    online.drop(columns=["kurz"]).to_csv(ZIEL.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
