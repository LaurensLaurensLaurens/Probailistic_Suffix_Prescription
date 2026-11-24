"""Probe der künstlichen Effekte (2026-11-20): derselbe Simulator (Aufbau F, Katalog d_oe_zustand_20261115, zufällige Zuteilung),
Bank-Policy, gleiche Saat, einmal ohne und einmal mit Effekten (process-simulation-engine/kuenstliche_effekte, Standard effekte_v1.yaml).
Prüft: gleiche Event-Typen, Verschiebung der Durchlaufzeit (wie nah bleibt der Log am Original), wie oft je Punkt und Option ein Effekt
greift und angewandt wird. Ausgabe probe.md / probe_effekte.csv in diesem Ordner.
Aufruf im Ordner process-simulation-engine: .venv/bin/python -u <dieses Skript> [Fälle] [Saat] [Effektdatei]
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/"prescriptive/policies/ev_policy_event_costs/results/d_oe_zustand_20261115"
OUT = Path(__file__).resolve().parent
CASES = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
EFFEKTE = sys.argv[3] if len(sys.argv) > 3 else ""
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PSP_CATALOG"] = str(SOURCE/"catalog.yaml")
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths  # noqa: E402
add_paths()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def simulator():
    import interventions
    interventions.CANDIDATES_FILE = str(SOURCE/"catalog.yaml")
    from prescriptive.helpers.datasets import load_dataset
    from preprocessing import preprocess
    from simulation.dataset_simulator import Simulator
    raw, _, _ = load_dataset("bpic17")
    return Simulator("bpic17", preprocess("bpic17", raw, verbose=False))


def run(task):
    mit_effekten, cases, seed, datei = task
    log = None
    if mit_effekten:
        import kuenstliche_effekte as ke
        ke.aktivieren(ke.laden(datei or None))
    events, choices = simulator().run(cases, policy=None, seed=seed)
    if mit_effekten:
        import kuenstliche_effekte as ke
        log = pd.DataFrame(ke.LETZTE_ENGINE.effect_log)
    return events, log


def durchlauf(events):
    t = events.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"])
    return (t["max"] - t["min"]).dt.total_seconds() / 86400


def main():
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
        (ohne, _), (mit, log) = list(pool.map(run, [(False, CASES, SEED, EFFEKTE), (True, CASES, SEED, EFFEKTE)]))
    tok = lambda e: set(e["concept:name"] + "|" + e["lifecycle:transition"])
    d_ohne, d_mit = durchlauf(ohne), durchlauf(mit)
    gemeinsam = d_ohne.index.intersection(d_mit.index)
    lines = [f"# Probe künstliche Effekte ({CASES} Fälle, Saat {SEED}, Bank-Policy)", "",
             f"Effektdatei: {EFFEKTE or 'effekte_v1.yaml'}", "",
             f"- Event-Typen ohne / mit: {len(tok(ohne))} / {len(tok(mit))}, nur ohne: {sorted(tok(ohne) - tok(mit))}, nur mit: {sorted(tok(mit) - tok(ohne))}",
             f"- Events je Fall ohne / mit: {len(ohne) / ohne['case:concept:name'].nunique():.2f} / {len(mit) / mit['case:concept:name'].nunique():.2f}",
             f"- Durchlaufzeit (Tage) ohne / mit: Mittel {d_ohne.mean():.2f} / {d_mit.mean():.2f}, Median {d_ohne.median():.2f} / {d_mit.median():.2f}, "
             f"P90 {d_ohne.quantile(.9):.2f} / {d_mit.quantile(.9):.2f}",
             f"- je Fall gepaart (gleiche Fall-ID): mit − ohne Mittel {(d_mit[gemeinsam] - d_ohne[gemeinsam]).mean():+.2f} d, "
             f"Fälle mit Effekt > 0: {int((log.groupby('case').tage.sum() > 0).sum()) if log is not None and len(log) else 0}", ""]
    wichtig = ["A_Incomplete|complete", "W_Call incomplete files|start", "A_Cancelled|complete", "O_Accepted|complete", "A_Pending|complete",
               "A_Denied|complete", "O_Create Offer|complete", "W_Validate application|start"]
    je_fall = lambda e: (e["concept:name"] + "|" + e["lifecycle:transition"]).value_counts() / e["case:concept:name"].nunique()
    h_ohne, h_mit = je_fall(ohne), je_fall(mit)
    lines += ["Event-Typen je Fall ohne / mit: " + ", ".join(f"{w.split('|')[0]} {h_ohne.get(w, 0):.3f} / {h_mit.get(w, 0):.3f}" for w in wichtig), ""]
    sys.path.insert(0, str(OUT))
    import conformance_probe as cp
    konf = {}
    for name, e in (("ohne", ohne), ("mit", mit)):
        erg = cp.align(cp.projektion(e))
        konf[name] = (np.mean([c < 10000 for _, _, c, _ in erg]), np.mean([c // 10000 for _, _, c, _ in erg]))
    lines += [f"Conformance (BPIC17_ToBe_angepasst) ohne / mit: konform {konf['ohne'][0]:.1%} / {konf['mit'][0]:.1%}, "
              f"Abweichungen je Fall {konf['ohne'][1]:.3f} / {konf['mit'][1]:.3f}", ""]
    if log is not None and len(log):
        log.to_csv(OUT/"probe_effekte.csv", index=False)
        log["mit"] = (log.tage != 0) | (log["ablauf"].fillna("") != "") | (log.get("abweichung", pd.Series("", index=log.index)).fillna("") != "")
        tab = log.groupby(["point", "option"]).agg(ausgefuehrt=("case", "size"), mit_effekt=("mit", "sum"),
                                                   tage_soll=("tage", "sum"), tage_angewandt=("angewandt_tage", "sum"),
                                                   abweichungen=("abweichung_angewandt", "sum") if "abweichung_angewandt" in log else ("tage", lambda s: 0)).reset_index()
        tab["Anteil mit Effekt"] = (tab.mit_effekt / tab.ausgefuehrt).round(3)
        tab["angewandt / soll"] = (tab.tage_angewandt / tab.tage_soll.where(tab.tage_soll != 0)).round(3)
        tab["Tage je Fall (angewandt)"] = (tab.tage_angewandt / log.case.nunique()).round(3)
        lines += ["```", tab.round(3).to_string(index=False), "```", ""]
    text = "\n".join(lines)
    (OUT/"probe.md").write_text(text + "\n")
    print(text)
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
