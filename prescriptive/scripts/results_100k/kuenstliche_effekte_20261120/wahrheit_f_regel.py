"""Wahrheit der Conformance-Effekte in Aufbau F mit der REGEL als Basis (2026-11-23, Nutzer: „ja“). Anlass (befunde 10:05): Die
Einzelwahrheiten aus wahrheit_f.py gelten bei Bank an den anderen Punkten; die Effekte der Punkte überlappen (Orakel online nur +0,009
Abweichungen je Fall statt +0,15). Hier: wie viel bringt es, von der Regel abzuweichen, wenn an allen anderen Punkten die Regel entscheidet?

Gerüst wie online_konformanz.py (Effekte v3, 700 / 200 / 1.300 Fälle, Seeds 0–11, Policy nur für die bewerteten Fälle).
Stufe 1: basis = RulePolicy an allen Punkten (wie Arm regel; Teilgruppe je Fall und Punkt beim 1. Besuch festgehalten); je Wahlpunkt ein
  Lauf „andere Option an diesem Punkt für alle bewerteten Fälle (nachlegen nur 1. Besuch), sonst Regel“. Effekt je Teilgruppe gepaart je
  Fall gegen basis (Gruppe aus basis), SE über die 12 Seeds; dazu der Effekt über alle 200 bewerteten Fälle.
Stufe 2: kombi = Regel, aber in allen Teilgruppen mit Effekt < 0 aus Stufe 1 die andere Option – der direkt gemessene Vorsprung der besten
  Umschaltung über die Regel (keine Summe von Einzeleffekten).
4 × 12 + 12 + 12 = 72 Läufe. Ausgabe wahrheit_f_regel/<lauf>_seed<s>/faelle.csv, wahrheit_f_regel.md / .csv hier.
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
    os.environ["WAHRHEIT_FR_SEEDS"] = sys.argv[1]
    os.environ["WAHRHEIT_FR_PROZESSE"] = sys.argv[2] if len(sys.argv) > 2 else "12"
    del sys.argv[1:]
warnings.filterwarnings("ignore")
sys.path.insert(0, str(HERE))
import online_konformanz as ok  # noqa: E402  (Pfade, Simulator, bewerten, Orakel)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_s = os.environ.get("WAHRHEIT_FR_SEEDS", "0-11")
SEEDS = list(range(int(_s.split("-")[0]), int(_s.split("-")[1]) + 1)) if "-" in _s else [int(x) for x in _s.split(",")]
PROZESSE = int(os.environ.get("WAHRHEIT_FR_PROZESSE", "12"))
OUT = HERE/"wahrheit_f_regel"
GROESSEN = ("abweichungen", "konform", "dlz_tage")


class RegelMitSchalter:
    """RulePolicy; an den Punkten in schalter ({Punkt: None = alle Fälle | Menge von Teilgruppen}) die andere Option (nachlegen nur beim
    1. Besuch). Merkt sich die Teilgruppe je (Fall, Punkt) beim 1. Besuch."""

    def __init__(self, schalter):
        from prescriptive.policies.rule_policy import RulePolicy
        self.regel = RulePolicy()
        self.orakel = ok.Orakel()
        self.schalter = schalter
        self.gruppen = {}

    def choose(self, point, case, options, time, engine):
        regelwahl = self.regel.choose(point, case, options, time, engine)
        if point.name not in self.orakel.PUNKTE or len(options) < 2:
            return regelwahl
        regel, andere, erster = self.orakel.PUNKTE[point.name]
        schluessel = (case.case_id, point.name)
        if schluessel not in self.gruppen:
            from simulation.case_engine_lifecycle_aware import to_seconds
            dauer = (time - to_seconds(case.prefix[0]["time:timestamp"])) / 86400 if case.prefix else 0.0
            zeile = {"case:RequestedAmount": case.attributes.get("case:RequestedAmount"), "case:LoanGoal": case.attributes.get("case:LoanGoal"),
                     "case:ApplicationType": case.attributes.get("case:ApplicationType"), "dauer_tage": dauer}
            self.gruppen[schluessel] = self.orakel.gruppe(zeile, point.name)
            erster_besuch = True
        else:
            erster_besuch = False
        if point.name in self.schalter and (erster_besuch or not erster):
            gruppen = self.schalter[point.name]
            if (gruppen is None or self.gruppen[schluessel] in gruppen) and andere in options:
                return andere
        return regelwahl


def job(task):
    import torch
    torch.set_num_threads(1)
    name, schalter, seed = task
    ordner = OUT/f"{name}_seed{seed}"
    if (ordner/"faelle.csv").exists():
        return str(ordner)
    import kuenstliche_effekte as ke
    ke.aktivieren(ke.laden(ok.EFFEKTE))
    start = time.time()
    policy = RegelMitSchalter(schalter)
    events, _ = ok.simulator().run(ok.WARM + ok.SCORED + ok.COOL, policy=policy, window=(ok.WARM, ok.WARM + ok.SCORED), seed=seed)
    _, faelle = ok.bewerten(events)
    for punkt in ok.Orakel.PUNKTE:
        faelle[f"gruppe {punkt}"] = pd.Series({c: g for (c, p), g in policy.gruppen.items() if p == punkt})
    ordner.mkdir(parents=True, exist_ok=True)
    faelle.to_csv(ordner/"faelle.csv")
    (ordner/"schalter.json").write_text(json.dumps({k: (sorted(v) if v is not None else None) for k, v in schalter.items()}, ensure_ascii=False, indent=1))
    print("COMPLETE", json.dumps({"lauf": name, "seed": seed, "abweichungen": float(faelle.abweichungen.mean()),
                                  "konform": float(faelle.konform.mean()), "sekunden": round(time.time() - start)}), flush=True)
    return str(ordner)


def paare(name):
    teile = []
    for seed in SEEDS:
        a, b = OUT/f"basis_seed{seed}"/"faelle.csv", OUT/f"{name}_seed{seed}"/"faelle.csv"
        if a.exists() and b.exists():
            d = pd.read_csv(a, index_col=0).join(pd.read_csv(b, index_col=0), lsuffix="_b", rsuffix="_s", how="inner")
            d["seed"] = seed
            teile.append(d)
    d = pd.concat(teile)
    for g in GROESSEN:
        d[f"diff_{g}"] = d[f"{g}_s"].astype(float) - d[f"{g}_b"].astype(float)
    return d


def gesamt(name):
    d = paare(name)
    s = d.groupby("seed")[[f"diff_{g}" for g in GROESSEN]].mean()
    return {"Lauf": name, "Seeds": len(s), "Abw. je bewertetem Fall": s.diff_abweichungen.mean(), "SE": s.diff_abweichungen.std(ddof=1) / np.sqrt(len(s)),
            "konform (pp)": 100 * s.diff_konform.mean(), "SE konform (pp)": 100 * s.diff_konform.std(ddof=1) / np.sqrt(len(s)),
            "DLZ (d)": s.diff_dlz_tage.mean(), "SE DLZ": s.diff_dlz_tage.std(ddof=1) / np.sqrt(len(s))}


def stufe1_tabelle():
    alt = pd.read_csv(HERE/"wahrheit_f.csv").set_index(["Punkt", "Gruppe"])["andere−Regel Abw. (F)"]
    zeilen = []
    for punkt in ok.Orakel.PUNKTE:
        d = paare(punkt)
        spalte = f"gruppe {punkt}_b"
        for g, t in d[d[spalte].notna()].groupby(spalte):
            s = t.groupby("seed")[[f"diff_{x}" for x in GROESSEN]].mean()
            zeilen.append({"Punkt": punkt, "Gruppe": g, "Fälle": len(t), "andere−Regel Abw. (Regel sonst)": t.diff_abweichungen.mean(),
                           "SE über Seeds": s.diff_abweichungen.std(ddof=1) / np.sqrt(len(s)), "konform (pp)": 100 * t.diff_konform.mean(),
                           "DLZ (d)": t.diff_dlz_tage.mean(), "Bank sonst (wahrheit_f)": alt.get((punkt, g), np.nan)})
    return pd.DataFrame(zeilen)


def main():
    OUT.mkdir(exist_ok=True)
    tasks = [("basis", {}, seed) for seed in SEEDS] + [(punkt, {punkt: None}, seed) for seed in SEEDS for punkt in ok.Orakel.PUNKTE]
    with ProcessPoolExecutor(PROZESSE, mp_context=multiprocessing.get_context("spawn")) as pool:
        for _ in pool.map(job, tasks):
            pass
    tab = stufe1_tabelle()
    negativ = tab[tab["andere−Regel Abw. (Regel sonst)"] < 0]
    schalter = {punkt: set(t.Gruppe) for punkt, t in negativ.groupby("Punkt")}
    print("Stufe 2, umgeschaltete Teilgruppen:", {k: sorted(v) for k, v in schalter.items()}, flush=True)
    with ProcessPoolExecutor(PROZESSE, mp_context=multiprocessing.get_context("spawn")) as pool:
        for _ in pool.map(job, [("kombi", schalter, seed) for seed in SEEDS]):
            pass
    n = len(SEEDS) * ok.SCORED
    summe = float((negativ["Fälle"] * negativ["andere−Regel Abw. (Regel sonst)"]).sum() / n)
    ges = pd.DataFrame([gesamt(name) for name in list(ok.Orakel.PUNKTE) + ["kombi"]])
    basis = pd.concat([pd.read_csv(OUT/f"basis_seed{s}"/"faelle.csv", index_col=0) for s in SEEDS])
    online = json.loads((ok.OUT/"regel_seed0"/"result.json").read_text()) if (ok.OUT/"regel_seed0"/"result.json").exists() else {}
    basis0 = pd.read_csv(OUT/"basis_seed0"/"faelle.csv", index_col=0)
    tab.to_csv(HERE/"wahrheit_f_regel.csv", index=False)
    ges.to_csv(HERE/"wahrheit_f_regel_gesamt.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 20, "display.max_colwidth", 80):
        text = (f"# Wahrheit der Conformance-Effekte in Aufbau F, Regel als Basis (2026-11-23)\n\n"
                f"basis = Regel an allen Punkten ({len(SEEDS)} Seeds; Abweichungen je Fall {basis.abweichungen.mean():.3f}, konform "
                f"{100 * basis.konform.mean():.1f} %; Kontrolle Seed 0: basis {basis0.abweichungen.mean():.3f}, Online-Arm regel "
                f"{online.get('abweichungen', float('nan')):.3f}).\n\n"
                f"## Stufe 1: andere Option an einem Punkt, Regel sonst – je Teilgruppe (Fälle, die den Punkt unter der Regel erreichen)\n\n"
                f"```\n{tab.round(3).to_string(index=False)}\n```\n\n"
                f"## Über alle bewerteten Fälle (Lauf minus basis; kombi = alle Teilgruppen mit Effekt < 0 umgeschaltet)\n\n"
                f"```\n{ges.round(3).to_string(index=False)}\n```\n\n"
                f"Summe der negativen Einzeleffekte je bewertetem Fall: {summe:.3f}; kombi direkt gemessen: {ges.iloc[-1]['Abw. je bewertetem Fall']:.3f} "
                f"± {ges.iloc[-1]['SE']:.3f}. Umgeschaltet: {', '.join(f'{k}: {sorted(v)}' for k, v in schalter.items())}.\n")
    (HERE/"wahrheit_f_regel.md").write_text(text)
    print(text)
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
