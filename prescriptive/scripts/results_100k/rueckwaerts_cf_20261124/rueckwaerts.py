"""Causal Forest mit Rückwärtsinduktion für die Konformitätsaufgabe (2026-11-24, Nutzer: „ja mach beides“).

Stufen in umgekehrter Reihenfolge der Wahlpunkte im Fall: angebot_nachlegen → annehmen_nach_vervollstaendigung → annehmen_bei_konzept →
handle_leads. Je Stufe:
  1. Log im Simulator mit Effekten v3 (Katalog d_oe_zustand_20261115 wie online_konformanz.py): LAEUFE × FAELLE Fälle, Policy
     policies/ev_policy/rueckwaerts_forest.StufenPolitik – Stufenpunkt zufällig (nachlegen nur 1. Besuch), fertige Punkte ihr Forest,
     übrige Bank.
  2. Alignments gegen BPIC17_ToBe_angepasst (conformance_probe.py), Outcome je Fall = −Abweichungen.
  3. Causal Forest (CATEModel learner "cf") nur für den Stufenpunkt, Proben = die zufälligen Entscheidungen (nachlegen 1. Besuch);
     Testfälle wie train_cf_konformanz.py (helpers.training.test_cases) bleiben draußen.
  4. Je Teilgruppe der Effektregeln (online_konformanz.Orakel.gruppe): Differenz der mittleren Abweichungen zwischen den Optionen im
     zufälligen Log (unverzerrt, mit den Forests danach) und mittleres τ des Forests.
Ausgabe: stufe<k>_<punkt>/ (log_raw.pkl, abweichungen.csv, gruppen.csv), models/cate_model_cf_konformanz_rw_<kurz>.pkl,
rueckwaerts.md, fortschritt.log. Eine Stufe mit vorhandenem Forest wird übersprungen (Wiederaufnahme).
Aufruf im Ordner process-simulation-engine: .venv/bin/python -u <dieses Skript> [Läufe] [Fälle je Lauf] [Prozesse]
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
ROOT = HERE.parents[3]
PROBE = bool(os.environ.get("RW_PROBE"))  # Probelauf: alles (auch die Forests) nach probe/
OUT = HERE/"probe" if PROBE else HERE
K = ROOT/"prescriptive/scripts/results_100k/kuenstliche_effekte_20261120"
SOURCE = ROOT/"prescriptive/policies/ev_policy_event_costs/results/d_oe_zustand_20261115"
EFFEKTE = "kuenstliche_effekte/effekte_v3.yaml"
LAEUFE = int(sys.argv[1]) if len(sys.argv) > 1 else 8
FAELLE = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
PROZESSE = int(sys.argv[3]) if len(sys.argv) > 3 else 8
STUFEN = [("angebot_nachlegen", "nachlegen"), ("annehmen_nach_vervollstaendigung", "nachv"), ("annehmen_bei_konzept", "konzept"),
          ("handle_leads", "leads")]
FIRST_ONLY = ("angebot_nachlegen",)
CASE, ACT, LC, TIME = "case:concept:name", "concept:name", "lifecycle:transition", "time:timestamp"
if multiprocessing.current_process().name != "MainProcess":  # Arbeitsprozesse einfädig; der Forest im Hauptprozess darf alle Kerne nutzen
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PSP_CATALOG"] = str(SOURCE/"catalog.yaml")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"prescriptive"))
sys.path.insert(0, str(K))
from prescriptive.helpers.datasets import add_paths  # noqa: E402
add_paths()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def forest_pfad(kurz):
    return (OUT if PROBE else ROOT/"prescriptive/models")/f"cate_model_cf_konformanz_rw_{kurz}.pkl"


def note(text):
    line = f"{time.strftime('%Y-%m-%d %H:%M')} {text}"
    with open(OUT/"fortschritt.log", "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def simulator():
    import interventions
    interventions.CANDIDATES_FILE = str(SOURCE/"catalog.yaml")
    from prescriptive.helpers.datasets import load_dataset
    from preprocessing import preprocess
    from simulation.dataset_simulator import Simulator
    raw, _, _ = load_dataset("bpic17")
    return Simulator("bpic17", preprocess("bpic17", raw, verbose=False))


def lauf(task):
    stufe, punkt, forests, r = task
    import torch
    torch.set_num_threads(1)
    import kuenstliche_effekte as ke
    from prescriptive.policies.ev_policy.rueckwaerts_forest import StufenPolitik
    ke.aktivieren(ke.laden(EFFEKTE))
    start = time.time()
    policy = StufenPolitik(punkt, forests, seed=7000 + 100 * stufe + r, first_only=FIRST_ONLY)
    events, choices = simulator().run(FAELLE, policy=policy, seed=6000 + 100 * stufe + r)
    events[CASE] = f"R{stufe}_{r}_" + events[CASE].astype(str)
    choices = choices[choices.point == punkt].copy()
    choices["case"] = f"R{stufe}_{r}_" + choices["case"].astype(str)
    print(f"Stufe {stufe} Lauf {r}: {events[CASE].nunique():,} Fälle, {policy.zufall:,} Zufallsentscheidungen, {time.time() - start:.0f} s",
          flush=True)
    return events, choices


def align_teil(teil):
    import conformance_probe as cp
    return [(c, int(cost // 10000)) for c, _, cost, _ in cp.align(teil)]


def gruppen(log, punkt_obj, faelle):
    """Je Entscheidung am Stufenpunkt (in der Reihenfolge von CATEModel.samples): Teilgruppe der Effektregeln."""
    from interventions import option_at
    from simulation.case_engine_lifecycle_aware import to_seconds
    argv, env = sys.argv, {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    sys.argv = sys.argv[:1]  # online_konformanz liest beim Import seine Kommandozeile und setzt Thread-Variablen
    import online_konformanz as ok
    sys.argv = argv
    for k, v in env.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    orakel = ok.Orakel()
    log = log.sort_values([CASE, TIME], kind="stable")
    out = []
    for case, t in log[log[CASE].isin(faelle)].groupby(CASE, sort=False):
        toks = list(zip(t[ACT].astype(str), t[LC].astype(str)))
        start = t[TIME].iloc[0]
        erste = t.iloc[0]
        for i, (a, l) in enumerate(toks[:-1]):
            if (a, l) == (punkt_obj.activity, punkt_obj.lifecycle) and option_at(toks[i + 1][0], toks[i + 1][1], punkt_obj.options) is not None:
                dauer = (t[TIME].iloc[i] - start).total_seconds() / 86400
                out.append((case, to_seconds(t[TIME].iloc[i]), orakel.gruppe({"case:RequestedAmount": erste["case:RequestedAmount"], "case:LoanGoal": erste["case:LoanGoal"],
                                                 "case:ApplicationType": erste["case:ApplicationType"], "dauer_tage": dauer}, punkt_obj.name)))
    return out


def stufe_rechnen(k, punkt, kurz, forests):
    import conformance_probe as cp
    import interventions
    from helpers.training import test_cases
    from preprocessing import preprocess
    from prescriptive.estimators.cate_model import CATEModel
    ordner = OUT/f"stufe{k}_{kurz}"
    ordner.mkdir(exist_ok=True)
    t0 = time.time()
    if (ordner/"log_raw.pkl").exists():
        raw = pd.read_pickle(ordner/"log_raw.pkl")
    else:
        note(f"Stufe {k} ({punkt}): Log beginnt, {LAEUFE} × {FAELLE} Fälle, Forests an {sorted(forests) or 'keinem Punkt'}")
        with ProcessPoolExecutor(PROZESSE, mp_context=multiprocessing.get_context("spawn")) as pool:
            teile = list(pool.map(lauf, [(k, punkt, {n: str(p) for n, p in forests.items()}, r) for r in range(LAEUFE)]))
        raw = pd.concat([e for e, _ in teile], ignore_index=True)
        pd.concat([c for _, c in teile], ignore_index=True).to_csv(ordner/"choices.csv", index=False)
        raw.to_pickle(ordner/"log_raw.pkl")
    note(f"Stufe {k}: Log {raw[CASE].nunique():,} Fälle ({time.time() - t0:.0f} s)")
    if (ordner/"abweichungen.csv").exists():
        abw = pd.read_csv(ordner/"abweichungen.csv", index_col=0).iloc[:, 0]
    else:
        proj = cp.projektion(raw)
        faelle = proj[CASE].drop_duplicates().tolist()
        teile = [proj[proj[CASE].isin(set(faelle[i::PROZESSE * 4]))] for i in range(PROZESSE * 4)]
        with ProcessPoolExecutor(PROZESSE, mp_context=multiprocessing.get_context("spawn")) as pool:
            abw = pd.Series(dict(x for part in pool.map(align_teil, teile) for x in part), name="abweichungen")
        abw.to_csv(ordner/"abweichungen.csv")
    note(f"Stufe {k}: Alignments, Abweichungen je Fall {abw.mean():.3f}, konform {100 * (abw == 0).mean():.1f} % ({time.time() - t0:.0f} s)")
    log = preprocess("bpic17", raw, verbose=False)
    log["outcome"] = -log[CASE].astype(str).map(abw).astype(float)
    log = log[log["outcome"].notna()]
    held = test_cases(log)
    points = interventions.choice_points("bpic17")
    p = next(q for q in points if q.name == punkt)
    model = CATEModel("bpic17", points, seed=42, learner="cf")
    alle = model.samples(log)[punkt]
    gr = gruppen(log, p, set(c for c, _, _ in alle))
    if len(gr) != len(alle) or any(str(a[0]) != str(g[0]) for a, g in zip(alle, gr)):
        raise RuntimeError(f"Teilgruppen passen nicht zu den Proben ({len(gr)} gegen {len(alle)})")
    # nur die zufälligen Entscheidungen: Anfrage mit ≥ 2 zulässigen Optionen (choices.csv, gleiche Zeit) und – bei nachlegen – erster
    # Besuch je Fall (so entscheidet StufenPolitik zufällig; sonst Bank)
    ch = pd.read_csv(ordner/"choices.csv")
    mehrere = {}
    for c, s, n in zip(ch.case, ch.time, ch.options):
        if n >= 2:
            mehrere.setdefault(str(c), []).append(float(s))
    erst, gesehen = [], set()
    for (case, _, _), (_, sek, _) in zip(alle, gr):
        passt = any(abs(float(sek) - s) < 1.0 for s in mehrere.get(str(case), ()))  # Anfragezeit = Zeit des Auslöse-Events
        erst.append((punkt not in FIRST_ONLY or case not in gesehen) and passt)
        gesehen.add(case)
    train = [s for s, e in zip(alle, erst) if e and str(s[0]) not in held]
    note(f"Stufe {k}: {len(alle):,} Proben am Punkt, {sum(erst):,} zufällige, {len(train):,} im Training; Anfragen mit ≥ 2 Optionen "
         f"{sum(map(len, mehrere.values())):,} von {len(ch):,}")
    if sum(erst) < 0.5 * sum(map(len, mehrere.values())):
        raise RuntimeError("Abgleich Proben ↔ Anfragen fehlgeschlagen (Zeiten passen nicht)")
    model.fit(log[~log[CASE].astype(str).isin(held)], samples={punkt: train})
    if not model.knows(p):
        raise RuntimeError(f"Forest für {punkt} nicht geschätzt: {model.summary}")
    model.test_cases = held
    model.trained_on = {"log": str(ordner/"log_raw.pkl"), "outcome": "minus Abweichungen je Fall (BPIC17_ToBe_angepasst)",
                        "stufe": k, "punkt": punkt, "forests_danach": {n: str(x) for n, x in forests.items()}, "learner": "cf",
                        "proben": len(train)}
    model.save(str(forest_pfad(kurz)))
    proben = [(s, g[2]) for s, g, e in zip(alle, gr, erst) if e]
    # Teilgruppen: zufälliges Log (alle Proben) gegen mittleres τ
    cases, features, options = zip(*[s for s, _ in proben])
    tau = model.effects(p, list(features))[:, 0]
    d = pd.DataFrame({"case": cases, "option": options, "gruppe": [g for _, g in proben], "tau": tau,
                      "abw": [float(abw.get(str(c), np.nan)) for c in cases]})
    d["test"] = d.case.astype(str).isin(held)
    ref, akt = p.options[0], p.options[1]
    zeilen = []
    for g, t in d.groupby("gruppe"):
        a, r_ = t[t.option == akt].abw, t[t.option == ref].abw
        se = np.sqrt(a.var(ddof=1) / max(len(a), 1) + r_.var(ddof=1) / max(len(r_), 1))
        zeilen.append({"Punkt": punkt, "Gruppe": g, "Proben": len(t), "Anteil Aktion": round((t.option == akt).mean(), 3),
                       "Abw. Aktion − Referenz (zufällig)": a.mean() - r_.mean(), "SE": se,
                       "Forest −τ (Abw. Aktion − Referenz)": -t.tau.mean(), "Forest wählt Aktion": float((t.tau > 0).mean())})
    tab = pd.DataFrame(zeilen)
    tab.to_csv(ordner/"gruppen.csv", index=False)
    note(f"Stufe {k}: Forest {forest_pfad(kurz).name} gespeichert ({len(train):,} Proben, {time.time() - t0:.0f} s)")
    return tab, model.summary.get(punkt, {})


def main():
    OUT.mkdir(exist_ok=True)
    forests, tabellen, info = {}, [], {}
    for k, (punkt, kurz) in enumerate(STUFEN, start=1):
        if forest_pfad(kurz).exists():
            note(f"Stufe {k}: {forest_pfad(kurz).name} gibt es schon, übersprungen")
            if (OUT/f"stufe{k}_{kurz}"/"gruppen.csv").exists():
                tabellen.append(pd.read_csv(OUT/f"stufe{k}_{kurz}"/"gruppen.csv"))
        else:
            tab, info[punkt] = stufe_rechnen(k, punkt, kurz, forests)
            tabellen.append(tab)
        forests[punkt] = forest_pfad(kurz)
    tab = pd.concat(tabellen, ignore_index=True)
    with pd.option_context("display.width", 250, "display.max_colwidth", 90):
        text = (f"# Causal Forest mit Rückwärtsinduktion (Konformität, Effekte v3), {time.strftime('%Y-%m-%d %H:%M')}\n\n"
                f"Je Stufe {LAEUFE} × {FAELLE} Fälle; Stufenpunkt zufällig, spätere Punkte ihr Forest, frühere Bank. Abweichungen gegen "
                f"BPIC17_ToBe_angepasst; Aktion = zweite Option des Punkts (Referenz = erste). Negativ = Aktion hat weniger Abweichungen.\n\n"
                f"```\n{tab.round(3).to_string(index=False)}\n```\n\nForest-Zusammenfassung:\n```\n{json.dumps(info, indent=1, default=str)}\n```\n")
    (OUT/"rueckwaerts.md").write_text(text)
    print(text, flush=True)
    note("Alle Stufen fertig")


if __name__ == "__main__":
    main()
