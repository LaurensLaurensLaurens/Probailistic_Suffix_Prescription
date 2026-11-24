"""Wahrheit der Conformance-Effekte direkt in Aufbau F (2026-11-23, Nutzer: „ja, starte das“). Anlass: Das Orakel (Wahrheit aus
wahrheit.py, Option für ALLE Fälle eines Laufs erzwungen) schlägt die Regel online nur um 0,009 Abweichungen je Fall statt der ≈ 0,16, die
seine Tabelle verspricht.

Gerüst wie online_konformanz.py (Effekte v3, Katalog d_oe_zustand, 700 Einlauf-, 200 bewertete, 1.300 Auslauffälle, Seeds 0–11). Je
Wahlpunkt und Option ein Lauf: die Policy erzwingt nur an diesem Punkt und nur für die bewerteten Fälle die Option (nachlegen nur beim
1. Besuch, sonst bei jedem Besuch, wie das Orakel), an den anderen Punkten entscheidet die Bank. Beim 1. Besuch wird die Teilgruppe des
Falls festgehalten (Orakel.gruppe). 4 Punkte × 2 Optionen × 12 Seeds = 96 Läufe.

Auswertung je Punkt, gepaart je Seed und Fall (gleiche Fall-IDs): andere − Regel-Option in Abweichungen, konform, Durchlaufzeit
  - je Teilgruppe (Gruppe aus dem Regel-Lauf; Fälle, die den Punkt erreichen), Mittel über Fälle, SE über die 12 Seed-Mittel
  - über alle 200 bewerteten Fälle (enthält Fälle ohne Besuch des Punkts: Rückwirkungen und Paarungsrauschen)
dazu der daraus erwartete Vorsprung eines Orakels vor der Regel (Σ Entscheidungen × min(0, Effekt) je Fall) neben der alten Tabelle.
Ausgabe wahrheit_f/<punkt>_<regel|andere>_seed<s>/faelle.csv, wahrheit_f.md / .csv hier.
Aufruf im Ordner process-simulation-engine: .venv/bin/python -u <dieses Skript> [Seeds 0-11] [Prozesse 12]
(online_konformanz liest sys.argv beim Import; die Argumente gehen über Umgebungsvariablen an die Arbeitsprozesse.)
"""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import warnings

HERE = Path(__file__).resolve().parent
if len(sys.argv) > 1:
    os.environ["WAHRHEIT_F_SEEDS"] = sys.argv[1]
    os.environ["WAHRHEIT_F_PROZESSE"] = sys.argv[2] if len(sys.argv) > 2 else "12"
    del sys.argv[1:]
warnings.filterwarnings("ignore")
sys.path.insert(0, str(HERE))
import online_konformanz as ok  # noqa: E402  (Pfade, Simulator, bewerten, Orakel)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_s = os.environ.get("WAHRHEIT_F_SEEDS", "0-11")
SEEDS = list(range(int(_s.split("-")[0]), int(_s.split("-")[1]) + 1)) if "-" in _s else [int(x) for x in _s.split(",")]
PROZESSE = int(os.environ.get("WAHRHEIT_F_PROZESSE", "12"))
OUT = HERE/"wahrheit_f"


class Erzwinger:
    """Erzwingt an einem Punkt eine Option (nur bewertete Fälle, über window), merkt sich die Teilgruppe beim 1. Besuch."""

    def __init__(self, punkt, seite):
        self.orakel = ok.Orakel()
        self.punkt, self.seite = punkt, seite
        regel, andere, self.erster = self.orakel.PUNKTE[punkt]
        self.option = regel if seite == "regel" else andere
        self.gruppen = {}

    def choose(self, point, case, options, time, engine):
        if point.name != self.punkt or len(options) < 2:
            return None
        besuche = sum(1 for e in case.prefix if e.get("concept:name") == point.activity
                      and e.get("lifecycle:transition", "complete") == point.lifecycle)
        if case.case_id not in self.gruppen:
            from simulation.case_engine_lifecycle_aware import to_seconds
            dauer = (time - to_seconds(case.prefix[0]["time:timestamp"])) / 86400 if case.prefix else 0.0
            zeile = {"case:RequestedAmount": case.attributes.get("case:RequestedAmount"), "case:LoanGoal": case.attributes.get("case:LoanGoal"),
                     "case:ApplicationType": case.attributes.get("case:ApplicationType"), "dauer_tage": dauer}
            self.gruppen[case.case_id] = self.orakel.gruppe(zeile, point.name)
        if self.erster and besuche > 1:
            return None
        return self.option if self.option in options else None


def job(task):
    import torch
    torch.set_num_threads(1)
    punkt, seite, seed = task
    ordner = OUT/f"{punkt}_{seite}_seed{seed}"
    if (ordner/"faelle.csv").exists():
        return str(ordner)
    import kuenstliche_effekte as ke
    ke.aktivieren(ke.laden(ok.EFFEKTE))
    start = time.time()
    policy = Erzwinger(punkt, seite)
    events, _ = ok.simulator().run(ok.WARM + ok.SCORED + ok.COOL, policy=policy, window=(ok.WARM, ok.WARM + ok.SCORED), seed=seed)
    _, faelle = ok.bewerten(events)
    faelle["gruppe"] = pd.Series(policy.gruppen)
    ordner.mkdir(parents=True, exist_ok=True)
    faelle.to_csv(ordner/"faelle.csv")
    print("COMPLETE", json.dumps({"punkt": punkt, "seite": seite, "seed": seed, "abweichungen": float(faelle.abweichungen.mean()),
                                  "erreicht": int(faelle.gruppe.notna().sum()), "sekunden": round(time.time() - start)}), flush=True)
    return str(ordner)


def auswerten():
    alt = pd.read_csv(ok.WAHRHEIT).set_index(["Punkt", "Gruppe"])["n Abweichungen andere−Regel"]
    zeilen, gesamt = [], []
    for punkt in ok.Orakel.PUNKTE:
        teile = []
        for seed in SEEDS:
            a, b = (OUT/f"{punkt}_{s}_seed{seed}"/"faelle.csv" for s in ("regel", "andere"))
            if not (a.exists() and b.exists()):
                continue
            r = pd.read_csv(a, index_col=0)
            o = pd.read_csv(b, index_col=0)
            d = r.join(o, lsuffix="_r", rsuffix="_o", how="inner")
            d["seed"] = seed
            teile.append(d)
        if not teile:
            continue
        d = pd.concat(teile)
        for groesse in ("abweichungen", "konform", "dlz_tage"):
            d[f"diff_{groesse}"] = d[f"{groesse}_o"].astype(float) - d[f"{groesse}_r"].astype(float)
        je_seed = d.groupby("seed")[["diff_abweichungen", "diff_konform", "diff_dlz_tage"]].mean()
        gesamt.append({"Punkt": punkt, "Seeds": len(je_seed), "Fälle": len(d), "erreicht": int(d.gruppe_r.notna().sum()),
                       "andere−Regel Abw. je bewertetem Fall": je_seed.diff_abweichungen.mean(),
                       "SE": je_seed.diff_abweichungen.std(ddof=1) / np.sqrt(len(je_seed)),
                       "konform (pp)": 100 * je_seed.diff_konform.mean(), "DLZ (d)": je_seed.diff_dlz_tage.mean()})
        for g, t in d[d.gruppe_r.notna()].groupby("gruppe_r"):
            s = t.groupby("seed")[["diff_abweichungen", "diff_konform", "diff_dlz_tage"]].mean()
            zeilen.append({"Punkt": punkt, "Gruppe": g, "Fälle": len(t), "je Seed": round(len(t) / len(s), 1),
                           "Gruppe gleich in beiden Läufen": float((t.gruppe_r == t.gruppe_o).mean()),
                           "andere−Regel Abw. (F)": t.diff_abweichungen.mean(), "SE über Seeds": s.diff_abweichungen.std(ddof=1) / np.sqrt(len(s)),
                           "konform (pp)": 100 * t.diff_konform.mean(), "DLZ (d)": t.diff_dlz_tage.mean(),
                           "alte Tabelle (alle Fälle erzwungen)": alt.get((punkt, g), np.nan)})
    tab, ges = pd.DataFrame(zeilen), pd.DataFrame(gesamt)
    n_bewertet = len(SEEDS) * ok.SCORED

    def vorsprung(spalte):  # erwarteter Vorsprung eines Orakels vor der Regel je bewertetem Fall
        return float((tab["Fälle"] * np.minimum(tab[spalte].fillna(0), 0)).sum() / -n_bewertet)

    tab.to_csv(HERE/"wahrheit_f.csv", index=False)
    ges.to_csv(HERE/"wahrheit_f_gesamt.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 20, "display.max_colwidth", 80):
        text = (f"# Wahrheit der Conformance-Effekte in Aufbau F (2026-11-23)\n\n"
                f"Option nur für die {ok.SCORED} bewerteten Fälle an einem Punkt erzwungen, sonst Bank; {len(SEEDS)} Seeds; gepaart je Fall.\n\n"
                f"## Je Teilgruppe (Fälle, die den Punkt erreichen)\n\n```\n{tab.round(3).to_string(index=False)}\n```\n\n"
                f"## Über alle bewerteten Fälle je Punkt\n\n```\n{ges.round(3).to_string(index=False)}\n```\n\n"
                f"Erwarteter Vorsprung eines Orakels vor der Regel (Abweichungen je bewertetem Fall): nach F-Wahrheit {vorsprung('andere−Regel Abw. (F)'):.3f}, "
                f"nach alter Tabelle {vorsprung('alte Tabelle (alle Fälle erzwungen)'):.3f}; online gemessen Orakel − Regel 0,009 (online_konformanz.md).\n")
    (HERE/"wahrheit_f.md").write_text(text)
    return text


def main():
    OUT.mkdir(exist_ok=True)
    tasks = [(punkt, seite, seed) for seed in SEEDS for punkt in ok.Orakel.PUNKTE for seite in ("regel", "andere")]
    with ProcessPoolExecutor(PROZESSE, mp_context=multiprocessing.get_context("spawn")) as pool:
        for _ in pool.map(job, tasks):
            pass
    print(auswerten())
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
