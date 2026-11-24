"""Conformance von echtem und simuliertem BPIC17-Log gegen das To-Be-Modell (2026-11-20, Nutzer: das PSP soll auch
Conformance-Abweichungen verhindern können; Vorgehen wie Mustroph, Conformance Checking Mustroph/src/conformance_checking).

Modell: process_models/BPIC17_ToBe_angepasst.bpmn (Standard, modell_anpassen.py; KONFORMANZ_MODELL=BPIC17_ToBe.bpmn für das Original) → Petri-Netz (pm4py). Projektion der Lifecycle-Events auf die Aktivitätsebene des Modells:
A_* und O_* mit lifecycle complete, Arbeitsschritte W_* mit start (tatsächliche Bearbeitung). Alignments (pm4py, Standardkosten:
Log- und Modellzug 1 je sichtbarem Schritt). Ausgabe je Log: Fitness-Verteilung, Anteil konformer Fälle, häufigste Abweichungen
(Logzug „(a, >>)“: Event im Log, das das Modell dort nicht erlaubt; Modellzug „(>>, b)“: vom Modell verlangtes Event fehlt).
Aufruf im Ordner process-simulation-engine: .venv/bin/python -u <dieses Skript> [Fälle je Log] [simulierter Log .pkl ...]
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
MODELL = ROOT/"process_models"/os.environ.get("KONFORMANZ_MODELL", "BPIC17_ToBe_angepasst.bpmn")  # Umgebung: anderes Modell
N = 500  # Fälle je Log (Kommandozeile erst in main gelesen: das Modul wird auch importiert)
SIM = [ROOT/"prescriptive/scripts/results_100k/rct_tlearner_20261118/rct_log_raw.pkl"]
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def projektion(events: pd.DataFrame) -> pd.DataFrame:
    """Lifecycle-Log → Aktivitätsebene des To-Be-Modells (A_/O_ complete, W_ start)."""
    lc = events["lifecycle:transition"].astype(str).str.lower()
    act = events["concept:name"].astype(str)
    keep = (act.str.startswith(("A_", "O_")) & (lc == "complete")) | (act.str.startswith("W_") & (lc == "start"))
    out = events.loc[keep, ["case:concept:name", "concept:name", "time:timestamp"]].copy()
    return out.sort_values(["case:concept:name", "time:timestamp"], kind="stable")


def align(teil: pd.DataFrame):
    import pm4py
    bpmn = pm4py.read_bpmn(str(MODELL))
    net, im, fm = pm4py.convert_to_petri_net(bpmn)
    res = pm4py.conformance_diagnostics_alignments(teil, net, im, fm, multi_processing=False,
                                                   activity_key="concept:name", case_id_key="case:concept:name",
                                                   timestamp_key="time:timestamp")
    cases = list(dict.fromkeys(teil["case:concept:name"]))
    return [(c, r["fitness"], r["cost"], [m for m in r["alignment"]]) for c, r in zip(cases, res)]


def pruefen(name: str, events: pd.DataFrame, seed: int = 0):
    proj = projektion(events)
    cases = proj["case:concept:name"].drop_duplicates().sample(n=min(N, proj["case:concept:name"].nunique()), random_state=seed)
    proj = proj[proj["case:concept:name"].isin(set(cases))]
    teile = [proj[proj["case:concept:name"].isin(set(cases.iloc[i::8]))] for i in range(8)]
    with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn")) as pool:
        ergebnisse = [x for part in pool.map(align, teile) for x in part]
    fitness = np.array([f for _, f, _, _ in ergebnisse])
    zuege = Counter()
    for _, _, _, al in ergebnisse:
        for a, b in al:
            if a == ">>" and b is not None:
                zuege[f"Modellzug (fehlt): {b}"] += 1
            elif b == ">>" and a is not None:
                zuege[f"Logzug (unerlaubt): {a}"] += 1
    lines = [f"## {name} ({len(ergebnisse)} Fälle)", "",
             f"- Fitness Mittel {fitness.mean():.3f}, Median {np.median(fitness):.3f}; konform (keine Log- oder Modellzüge; stille Übergänge "
             f"kosten in pm4py 1, Abweichungen 10.000): {np.mean([c < 10000 for _, _, c, _ in ergebnisse]):.1%}; Abweichungen je Fall "
             f"{np.mean([c // 10000 for _, _, c, _ in ergebnisse]):.2f}",
             "- häufigste Abweichungen (je Fall):"]
    lines += [f"  - {k}: {v / len(ergebnisse):.3f}" for k, v in zuege.most_common(12)]
    return "\n".join(lines) + "\n", ergebnisse


def main():
    global N, SIM
    N = int(sys.argv[1]) if len(sys.argv) > 1 else N
    SIM = [Path(p) for p in sys.argv[2:]] or SIM
    from prescriptive.helpers.datasets import add_paths, load_dataset
    add_paths()
    raw, _, _ = load_dataset("bpic17")
    texte = []
    text, _ = pruefen("Echter Log (BPIC17)", raw)
    texte.append(text)
    print(text, flush=True)
    for pfad in SIM:
        text, _ = pruefen(f"Simulierter Log ({pfad.parent.name}/{pfad.name})", pd.read_pickle(pfad))
        texte.append(text)
        print(text, flush=True)
    (OUT/f"conformance_probe_{MODELL.stem}.md").write_text(f"# Conformance gegen {MODELL.name} (Projektion A_/O_ complete, W_ start)\n\n" + "\n".join(texte))
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
