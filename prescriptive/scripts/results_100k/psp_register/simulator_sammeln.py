"""Treue des Simulators gegenüber dem echten BPIC17-Log, für register_erstellen.py (Langformat simulator.csv + Markdown).

Ziel: wissen, welche Schwächen am Simulator liegen und welche an den Modellen. Quellen (Stand je Datei in der Spalte stand):
- neue_punkte_kpis (main.py --steps kpis, 10.000 simulierte Fälle): allgemeine Kennzahlen, Endaktivitäten, Kennzahlen je Wahlpunkt
  (Anteile, Unterscheidbarkeit der Entscheidungszustände echt gegen simuliert, Wirkungen im/außerhalb des Trägers)
- neue_punkte_validierung: AIPW auf dem echten Log vs. AIPW auf beobachtet simulierten Daten vs. wahrer randomisierter Effekt
- modellvergleich_20261116/punkte_alle/simulator_rct.pkl: RCT im Simulator (Stand 14.11.), Effekt der Regel-Option und seine
  Zerlegung in Endzustand (die Option ändert, wie der Fall endet) und Dauer (bei gleichem Ende), wie psp_zerlegung.py
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
R = OUT.parent
P = R.parents[1]
K = R/"neue_punkte_kpis/simulator"
V = R/"modellvergleich_20261116"


def _stand(path):
    return time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime))


def _quelle(path):
    return str(path.relative_to(R))


def allgemein(rows):
    path = K/"bpic17_simulator_kpis.csv"
    for r in pd.read_csv(path).itertuples():
        dev = (r.simuliert-r.echt)/r.echt*100 if r.echt else np.nan
        rows.append(dict(teil="1 allgemein", bereich=r.Bereich, kennzahl=r.Kennzahl, echt=r.echt, simulator=r.simuliert,
                         abweichung=dev, abweichung_als="% von echt", stand=_stand(path), quelle=_quelle(path)))
    path = K/"kpis_diagramme/detail_Endaktivitäten.csv"
    t = pd.read_csv(path, index_col=0)
    for name, r in t[(t["Anteil echt (%)"] >= 0.3) | (t["Anteil simuliert (%)"] >= 0.3)].iterrows():
        rows.append(dict(teil="1 allgemein", bereich="Endaktivität (Anteil der Fälle, %)", kennzahl=name, echt=r["Anteil echt (%)"],
                         simulator=r["Anteil simuliert (%)"], abweichung=r["Differenz"], abweichung_als="Prozentpunkte",
                         stand=_stand(path), quelle=_quelle(path)))


def wahlpunkte(rows):
    path = K/"bpic17_prescriptive_kpis.csv"
    for r in pd.read_csv(path).itertuples():
        rows.append(dict(teil="2 Wahlpunkte", bereich=r.Bereich, kennzahl=r.Kennzahl, echt=r.echt, simulator=r[4], simulator_zufall=r[5],
                         abweichung=r.Abweichung, se_abweichung=r[7], abweichung_als=r[8], bewertung=r.Bewertung, stand=_stand(path),
                         quelle=_quelle(path)))


def effekte(rows):
    path = R/"neue_punkte_validierung/validierung.csv"
    for r in pd.read_csv(path).itertuples():
        rows.append(dict(teil="3 Effekte", bereich=r.Kandidat, kennzahl="Effekt der Aktion (Tage, positiv = Aktion kürzer)",
                         echt=r[8], echt_se=r[9], simulator=r[10], simulator_se=r[11], simulator_zufall=r[12], simulator_zufall_se=r[13],
                         abweichung=r[12]/r[8] if r[8] else np.nan, abweichung_als="Simulator wahr / echt (Anteil der echten Effektgröße)",
                         z_echt_gegen_wahr=r[16], z_screening_trifft_wahr=r[15], ueberlappung_echt=r[4], ueberlappung_simulator=r[5],
                         stand=_stand(path), quelle=_quelle(path)))
    # RCT 14.11.: Vorteil der Regel-Option und Zerlegung Endzustand / Dauer
    path = V/"punkte_alle/simulator_rct.pkl"
    t = pd.read_pickle(path)
    t["end"] = t.end.fillna("ohne")
    for group, part in t.groupby("gruppe"):
        rule, other = part[part.rule], part[~part.rule]
        pa, pr = rule.end.value_counts(normalize=True), other.end.value_counts(normalize=True)
        ta, tr = rule.groupby("end").ENDE.mean(), other.groupby("end").ENDE.mean()
        ends = [e for e in set(ta.index) & set(tr.index)]
        path_part = -sum((pa.get(e, 0)-pr.get(e, 0))*(ta[e]+tr[e])/2 for e in ends)
        time_part = -sum((pa.get(e, 0)+pr.get(e, 0))/2*(ta[e]-tr[e]) for e in ends)
        total = other.ENDE.mean()-rule.ENDE.mean()
        se = np.sqrt(rule.ENDE.var()/len(rule)+other.ENDE.var()/len(other))
        rows.append(dict(teil="3 Effekte", bereich=group, kennzahl="RCT 14.11.: Vorteil Regel-Option (Tage)", simulator_zufall=total,
                         simulator_zufall_se=se, endzustand_anteil=path_part, dauer_anteil=time_part,
                         ende_regel=", ".join(f"{e} {pa.get(e, 0):.0%}" for e in ("A_Pending", "A_Cancelled", "A_Denied", "ohne")),
                         ende_andere=", ".join(f"{e} {pr.get(e, 0):.0%}" for e in ("A_Pending", "A_Cancelled", "A_Denied", "ohne")),
                         stand=_stand(path), quelle=_quelle(path)))


def simulator_kennzahlen():
    rows = []
    for function in (allgemein, wahlpunkte, effekte):
        try:
            function(rows)
        except Exception as error:
            rows.append(dict(teil="Fehler", bereich=function.__name__, kennzahl=str(error)))
    return pd.DataFrame(rows)


def markdown(frame, tabelle):
    md = ["## Simulator: Treue gegenüber dem echten Log (Langformat in simulator.csv)", "",
          "Wichtig für die Deutung aller Online-Ergebnisse. Die Messungen vom 06.11. stammen **vor** der Umstellung auf gelernte "
          "Zuteilung (08.11.) und der korrigierten Optionsprüfung (15.11.); die RCT-Wahrheit ist vom 14.11. Nach Simulatoränderungen "
          "neu messen (main.py --steps kpis; scripts/validate_screening.py).", ""]
    part = frame[frame.teil == "1 allgemein"]
    md += ["### Allgemein (10.000 simulierte Fälle gegen 31.509 echte)", "",
           tabelle(part, ["bereich", "kennzahl", "echt", "simulator", "abweichung", "abweichung_als", "stand"], {}), ""]
    part = frame[frame.teil == "3 Effekte"]
    md += ["### Effekte an den Wahlpunkten: echter Log gegen Simulator", "",
           "echt = AIPW auf dem echten Log; simulator = AIPW auf beobachtet simulierten Daten (Bank-Politik); simulator_zufall = wahrer "
           "Effekt bei zufälliger Wahl; abweichung = Anteil der echten Effektgröße, den der Simulator hat; z_echt_gegen_wahr = "
           "Unterschied echt − Simulator in Standardfehlern.", "",
           tabelle(part[part.kennzahl.str.startswith("Effekt")], ["bereich", "echt", "echt_se", "simulator", "simulator_zufall",
                                                                  "simulator_zufall_se", "abweichung", "z_echt_gegen_wahr",
                                                                  "ueberlappung_echt", "ueberlappung_simulator", "stand"], {}), "",
           "RCT im Simulator (14.11.): Vorteil der Regel-Option und Zerlegung in Endzustand (die Option ändert, wie der Fall endet: "
           "Pending/Cancelled/Denied) und Dauer (bei gleichem Ende); Formel wie psp_zerlegung.py.", "",
           tabelle(part[part.kennzahl.str.startswith("RCT")], ["bereich", "simulator_zufall", "simulator_zufall_se", "endzustand_anteil",
                                                               "dauer_anteil", "ende_regel", "ende_andere"], {}), ""]
    part = frame[frame.teil == "2 Wahlpunkte"]
    md += ["### Je Wahlpunkt (echt gegen Simulator mit Bank-Politik und mit Zufallswahl)", "",
           "„Lage unterscheidbar“: AUC eines Klassifikators echt gegen simuliert auf den Zuständen am Wahlpunkt (0,5 = nicht "
           "unterscheidbar); genannt ist das wichtigste Merkmal. „Wirkung … im/außerhalb des Trägers“: Effekt dort, wo die Wahl auch "
           "im echten Log vorkommt, bzw. außerhalb.", "",
           tabelle(part, ["bereich", "kennzahl", "echt", "simulator", "simulator_zufall", "abweichung", "abweichung_als", "bewertung"], {}), ""]
    return md


if __name__ == "__main__":
    print(simulator_kennzahlen().drop(columns=["quelle"]).to_string())
