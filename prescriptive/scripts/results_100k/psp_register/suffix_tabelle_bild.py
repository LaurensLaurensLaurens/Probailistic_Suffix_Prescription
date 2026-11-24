"""Bild EXTRAZWISCHENERGEBNISSE/2017/suffix_vorhersage_bpic17.png (Stand 2026-11-17): alle PSP-Läufe mit Suffix- und Zeitmessung in
einer Tabelle (das frühere suffix_vorhersage_bpic17_testfaelle.png ist darin aufgegangen), im Stil von scripts/zwischenergebnisse.py (render). Zwei Messaufbauten, getrennt:
  A  1.000 Testprefixe aus den äußeren Testfällen, 100 Ziehungen je Prefix (psp_suffixguete.py; Werte aus register.csv bzw. der
     Gesamtübersicht vom 14.11. für die Semi-Markov-Referenz), dazu Zeitmaße und – wo gemessen – ungültige Übergänge
     (uebergaenge_pruefen.py, Verläufe ab dem 1. Besuch Nachlegen)
  B  200 Prefixe aus dem ganzen Log, 50 Ziehungen (ältere Läufe bis 09.11.; Werte aus suffix_aeltere_200_prefixe_stand_08-11.csv
     = früheres Bild vom 08.11., und der Gesamtübersicht)
Aufruf: <venv>/python suffix_tabelle_bild.py   (vorher register_erstellen.py, damit register.csv aktuell ist)
"""
import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
R = OUT.parent
P = R.parents[1]
ZIEL = P.parent/"EXTRAZWISCHENERGEBNISSE/2017/suffix_vorhersage_bpic17.png"
sys.path.insert(0, str(P/"scripts"))
from zwischenergebnisse import number, render  # noqa: E402

STEPS = (1, 2, 3, 5, 10, 20)
# Beschriftung im Bild je Register-Eintrag (Reihenfolge = Reihenfolge im Bild)
NAMEN = {
    "s40": "s40: 10k Fälle, Decoder 40",
    "(a)": "(a): Decoder 60, Log-Abstände",
    "(c)": "(c): Decoder 60, normale Abstände",
    "(d) 1 ms": "(d): wie (c), Zeitklassen",
    "OE": "OE: Token mit Zeitklasse",
    "(e)": "(e): Zeitklasse je Event",
    "(c)-TF1": "(c)-TF1: wie (c), TF 1",
    "D-TF1": "D-TF1: wie (d), TF 1",
    "OE-TF1 Ep. 1": "OE-TF1 Ep. 1 (bester Val.-Verlust)",
    "OE-TF1 Ep. 26": "OE-TF1 Ep. 26: wie OE, TF 1",
    "ZZ ep3": "ZZ Ep. 3 (TF 0,73): Zähler+Zeit",
    "ZZ ep7": "ZZ Ep. 7 (TF 0,20)",
    "ZZ ep9": "ZZ Ep. 9 (TF 0, 1. freie Epoche)",
    "ZZ ep11": "ZZ Ep. 11 (TF 0)",
    "ZZ ep13": "ZZ Ep. 13 (TF 0)",
    "ZZ final": "ZZ Ep. 15, fertig (TF 0)",
    "ZZ-TF1 ep3": "ZZ-TF1 Ep. 3: wie ZZ, TF 1",
    "ZZ-TF1 ep6": "ZZ-TF1 Ep. 6",
    "ZZ-TF1 ep9": "ZZ-TF1 Ep. 9",
    "ZZ-TF1 ep12": "ZZ-TF1 Ep. 12",
    "ZZ-TF1 final": "ZZ-TF1 Ep. 15, fertig",
    "ZZU-TF1 ep3": "ZZU-TF1 Ep. 3: ZZ-TF1 + Uhren",
    "ZZU-TF1 ep6": "ZZU-TF1 Ep. 6",
    "ZZU-TF1 ep9": "ZZU-TF1 Ep. 9",
    "ZZU-TF1 ep12": "ZZU-TF1 Ep. 12",
    "ZZU-TF1 final": "ZZU-TF1 Ep. 15, fertig",
    "ZZ-TF1-WP ep5": "ZZ-TF1-WP Ep. 5: nur Wahlpunkte",
    "ZZ-TF1-WP ep10": "ZZ-TF1-WP Ep. 10",
    "ZZ-TF1-WP ep20": "ZZ-TF1-WP Ep. 20",
    "ZZ-TF1-WP final": "ZZ-TF1-WP Ep. 30, fertig",
    "ZZ-TF1-WP150 ep15": "ZZ-TF1-WP150 Ep. 15",
    "ZZ-TF1-WP150 ep45": "ZZ-TF1-WP150 Ep. 45",
    "ZZ-TF1-WP150 ep90": "ZZ-TF1-WP150 Ep. 90",
    "ZZ-TF1-WP150 final": "ZZ-TF1-WP150 Ep. 150, fertig",
    "Pilot k_tf0": "Pilot: Zeitklassen, TF 0",
    "Pilot k_tf1": "Pilot: Zeitklassen, TF 1",
    "Pilot t_tf1_od": "Pilot: Token mit Zeitklasse, TF 1",
    "Pilot 2_t_tf0": "Pilot: Token mit Zeitklasse, TF 0",
    "Pilot 2_t_tf05": "Pilot: Token mit Zeitklasse, TF 0,5",
    "Pilot 3_t_tf05_kx": "Pilot: Token + Kontext, TF 0,5",
    "Pilot 3_t_tf1_kx": "Pilot: Token + Kontext, TF 1",
    "Pilot n_tf1": "Pilot: normale Abstände, TF 1",
}


def percent(value):
    return "–" if pd.isna(value) else f"{100 * value:.0f} %"


def num(value, digits):
    return "–" if pd.isna(value) else number(value, digits)


NAMES = {f"suffix_s{k}": f"Schritt {k}" for k in STEPS}
NAMES.update({"suffix_dls": "DLS", "suffix_laenge_mae": "Suffixlänge-MAE", "suffix_erster_abstand_mae_h": "Abstand Schritt 1 MAE (h)",
              "restzeit_mae": "Restzeit MAE (Tage)", "restzeit_bias": "Restzeit Fehler (Tage)", "restzeit_crps": "Restzeit CRPS (Tage)",
              "abdeckung80": "Abdeckung 80 %"})


STORNO = R/"zaehler_uhren_tf1_20261117/storno_abstand.csv"  # storno_abstand.py: Stornierung ab letztem O_Sent in gezogenen Suffixen
T_LEARNER = R/"t_learner_20261117"
ZEITKOPF = R/"zeitkopf_je_event_20261118"  # T-Learner-Arme mit Zeitkopf je Event (nur Zeitköpfe nachtrainiert)


def storno_werte():
    try:
        table = pd.read_csv(STORNO, index_col=0)
    except FileNotFoundError:
        return {}
    return {label: (row["Median"], row["Anteil 29–32 d"]) for label, row in table.iterrows()}


def zeile(name, r, storno=None):
    invalid = r.get("ungueltig_zwischen_je100", np.nan) + r.get("ungueltig_innerhalb_je100", np.nan)
    bias = r.get("Restzeit Fehler (Tage)")
    median, share = storno if storno else (np.nan, np.nan)
    return [name, *[percent(r.get(f"Schritt {k}")) for k in STEPS], num(r.get("DLS"), 2), num(r.get("Suffixlänge-MAE"), 1),
            num(r.get("Abstand Schritt 1 MAE (h)"), 1), num(r.get("Restzeit MAE (Tage)"), 1),
            "–" if pd.isna(bias) else number(bias, 1, sign=True), num(r.get("Restzeit CRPS (Tage)"), 1), percent(r.get("Abdeckung 80 %")),
            num(invalid, 1), num(median, 1), percent(share)]


def main():
    """Ein Bild für alle Suffix-Messungen (seit 17.11. zusammengelegt mit dem früheren suffix_vorhersage_bpic17_testfaelle.png)."""
    register = pd.read_csv(OUT/"register.csv").set_index("label")
    rows, data, highlight = [], [], []
    header = ["Modell", *[f"Schritt\n{k}" for k in STEPS], "DLS", "Länge\nMAE", "1. Abst.\nMAE (h)", "Restzeit\nMAE (d)",
              "Restzeit\nBias (d)", "Restzeit\nCRPS (d)", "Abdeck.\n80 %", "ungültig\nje 100", "Storno\n(d)", "Storno\n29–32"]
    storno = storno_werte()

    def abschnitt(text):
        highlight.append(len(rows))
        rows.append([text] + [""] * (len(header) - 1))

    abschnitt("A: 1.000 Testprefixe, 100 Ziehungen")
    reference = pd.read_csv(R/"psp_gesamtuebersicht_20261114/suffix_vollmodelle_1000_testprefixe.csv").set_index("Variante").loc["Semi-Markov (Referenz)"]
    rows.append(zeile("Semi-Markov-Kette (Referenz)", reference))
    data.append({"Aufbau": "A", "Modell": "Semi-Markov-Kette (Referenz)", **reference.to_dict()})
    if "echter Log" in storno:
        rows.append(zeile("echter Log (nur Storno-Zeitpunkt)", pd.Series(dtype=float), storno["echter Log"]))
        data.append({"Aufbau": "A", "Modell": "echter Log", "Storno Median (d)": storno["echter Log"][0], "Storno 29–32 d": storno["echter Log"][1]})
    for label, name in NAMEN.items():
        if label not in register.index or pd.isna(register.at[label, "suffix_s1"]):
            continue
        r = register.loc[label].rename(NAMES)
        rows.append(zeile(name, r, storno.get(label)))
        data.append({"Aufbau": "A", "Modell": name, **{k: r.get(k) for k in [*NAMES.values(), "ungueltig_zwischen_je100", "ungueltig_innerhalb_je100"]},
                     "Storno Median (d)": storno.get(label, (np.nan, np.nan))[0], "Storno 29–32 d": storno.get(label, (np.nan, np.nan))[1]})
    extra = [(T_LEARNER/"suffixguete_aktion/suffixguete.csv", "T-Learner aktion", "T-Learner: Aktionsmodell", "T-Learner Aktion", "T-Learner (ZZ-TF1)"),
             (T_LEARNER/"suffixguete_referenz/suffixguete.csv", "T-Learner referenz", "T-Learner: Referenzmodell", "T-Learner Referenz", "T-Learner (ZZ-TF1)"),
             (ZEITKOPF/"suffixguete_steigung/suffixguete.csv", "T-Learner Aktion Kopf je Event (Steigung)", "T-Learner + Zeitkopf je Event: Aktion",
              "T-Learner Aktion (steigung)", "T-Learner Kopf je Event (Steigung)"),
             (ZEITKOPF/"suffixguete_steigung/suffixguete.csv", "T-Learner Referenz Kopf je Event (Steigung)", "T-Learner + Zeitkopf je Event: Referenz",
              "T-Learner Referenz (steigung)", "T-Learner Kopf je Event (Steigung)"),
             (ZEITKOPF/"suffixguete_steigung_bnll1/suffixguete.csv", "T-Learner Aktion Kopf je Event (Steigung, β-NLL 1)",
              "T-Learner + Zeitkopf je Event, β-NLL 1: Aktion", "T-Learner Aktion (steigung_bnll1)", "T-Learner Kopf je Event (Steigung, β-NLL 1)"),
             (ZEITKOPF/"suffixguete_steigung_bnll1/suffixguete.csv", "T-Learner Referenz Kopf je Event (Steigung, β-NLL 1)",
              "T-Learner + Zeitkopf je Event, β-NLL 1: Referenz", "T-Learner Referenz (steigung_bnll1)", "T-Learner Kopf je Event (Steigung, β-NLL 1)"),
             (R/"restzeit_hilfsziel_20261118/suffixguete/suffixguete.csv", "ZZ-TF1 + Restzeit-Hilfsziel", "ZZ-TF1 + Restzeit-Hilfsziel (nur Training)",
              "ZZ-TF1 + Restzeit-Hilfsziel", "ZZ-TF1 + Restzeit-Hilfsziel"),
             (R/"tf_boden05_20261118/suffixguete/suffixguete.csv", "ZZ-TF0,5", "ZZ-TF0,5 (warm vom ZZ-TF1, TF-Boden 0,5)", "ZZ-TF0,5", "ZZ-TF0,5"),
             (R/"beta_nll_20261118/suffixguete/suffixguete.csv", "β-NLL 0,5", "β-NLL 0,5 (warm vom ZZU-TF1)", "β-NLL 0,5", "β-NLL 0,5")]
    for path, row_label, name, key, invalid_from in extra:  # 18.11.: Modelle außerhalb der Begleiter-Ordner
        if not path.exists():
            continue
        table = pd.read_csv(path).set_index("Modell")
        if row_label not in table.index:
            continue
        r = table.loc[row_label].copy()
        if invalid_from in register.index:  # ungültige Übergänge aus der Prüfreihe (T-Learner: beide Modelle zusammen)
            r["ungueltig_zwischen_je100"] = register.at[invalid_from, "ungueltig_zwischen_je100"]
            r["ungueltig_innerhalb_je100"] = register.at[invalid_from, "ungueltig_innerhalb_je100"]
        rows.append(zeile(name, r, storno.get(key)))
        data.append({"Aufbau": "A", "Modell": name, **r.to_dict(), "Storno Median (d)": storno.get(key, (np.nan, np.nan))[0],
                     "Storno 29–32 d": storno.get(key, (np.nan, np.nan))[1]})
    abschnitt("B: 200 Prefixe, 50 Ziehungen (ältere Läufe)")
    old = pd.read_csv(OUT/"suffix_aeltere_200_prefixe_stand_08-11.csv").set_index("Modell")  # Messung vom 08.11. (früheres Bild)
    older = pd.read_csv(R/"psp_gesamtuebersicht_20261114/suffix_aeltere_200_prefixe.csv").set_index("Variante")
    for label in old.index:
        name = {"Markov-Kette": "Markov-Kette (Referenz)",
                "PSP echter Log (10.000 Fälle, 1e-5, neue Punkte 3-fach)": "10k neue Punkte: Decoder 5, 1e-5",
                "PSP-Variante (Decoder 40, ohne Zukunftswissen, 1e-4)": "s40: Decoder 40, ohne Zukunftswissen"}.get(label, label)
        rows.append(zeile(name, old.loc[label]))
        data.append({"Aufbau": "B", "Modell": name, **old.loc[label].to_dict()})
    for label in older.index:
        if label.startswith("Stufe 2"):
            rows.append(zeile(label.replace("Stufe 2 ", "").replace("log-normale Abstände", "Log-Abstände"), older.loc[label]))
            data.append({"Aufbau": "B", "Modell": label.replace("Stufe 2 ", ""), **older.loc[label].drop("Modell").to_dict()})
    sections = list(highlight)  # Abschnittszeilen A und B: über die ganze Breite
    highlight += [1, highlight[1] + 1]  # die beiden Markov-Referenzen
    pd.DataFrame(data).to_csv(ZIEL.with_suffix(".csv"), index=False)
    note = ("A: 1.000 zufällige Prefixe aus den 6.301 äußeren Testfällen (gleiche Prefixe für alle Modelle), 100 Ziehungen je Prefix. "
            "B: 200 Prefixe aus dem ganzen Log, 50 Ziehungen (Läufe bis 09.11., gemessen 08.11.); A und B nicht direkt vergleichbar. "
            "Treffer: häufigstes gezogenes Aktivität|Lifecycle-Token an dieser Position gleich dem echten (nur Prefixe mit echtem Event dort). "
            "DLS: mittlere normierte Damerau-Levenshtein-Ähnlichkeit der Einzelziehungen. Länge-MAE: Fehler der Zahl restlicher Events. "
            "1. Abstand: Fehler der Zeit bis zum nächsten Event. Restzeit: vom letzten Prefix-Event bis Fallende, Mittel der Ziehungen "
            "(Bias = gezogen − echt; echte Restzeit im Mittel 14,3 Tage, naive Vorhersage mit dem Median MAE 10,8 Tage). Abdeckung: Anteil "
            "echter Restzeiten im 10.–90.-Perzentil der Ziehungen (ideal 80 %). Ungültig: nie beobachtete Übergänge zwischen Events plus "
            "Lifecycle-Verstöße innerhalb einer Aktivität je 100 gezogene Events, gemessen an den Verläufen ab dem 1. Besuch Nachlegen "
            "(echter Log und Simulator ≈ 0). Piloten: 3.000 Fälle, 6 Epochen; sonst alle Trainingsfälle, 15 Epochen, wenn nicht anders "
            "angegeben. TF = Teacher Forcing im Training (bei Zwischenständen der Wert der Epoche). Semi-Markov-Kette aus den Trainingsfällen. "
            "Storno: in 300 Testfällen (Schnitt nach dem ersten O_Sent, 20 Ziehungen bis Fallende) Tage vom letzten O_Sent bis A_Cancelled, "
            "Spalten Storno (d) = Median, Storno 29–32 = Anteil im Fenster 29–32 Tage (echter Log: 30,7 d, 73 % – feste 30-Tage-Frist; storno_abstand.py). WP/WP150: nur "
            "Wahlpunkt-Proben im Training (Suffixgüte auf zufälligen Prefixen daher schwach). T-Learner: zwei Modelle (Aktion / Referenz), "
            "warm vom ZZ-TF1, nachtrainiert nur auf ihrer Gruppe; Suffixgüte je Modell, ungültige Übergänge an den Verläufen beider zusammen. "
            f"Quelle: prescriptive/scripts/results_100k/psp_register (register.csv). Stand {datetime.date.today():%d.%m.%Y}.")
    render("Suffix- und Zeitvorhersage BPIC17: alle Läufe", header, rows, str(ZIEL), note, highlight=highlight, span=sections)


if __name__ == "__main__":
    main()
