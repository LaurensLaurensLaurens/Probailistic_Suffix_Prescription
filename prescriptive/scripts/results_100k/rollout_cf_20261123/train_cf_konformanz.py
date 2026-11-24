"""Causal Forest für Conformance auf dem Simulator-Log mit Effekten v3 (2026-11-23, Nutzer: „Setze eine Version auf, die mittels Causal
Forest die späteren Entscheidungen in den Suffixen korrigiert“). Der Forest entscheidet im Rollout an späteren Wahlpunkten.

Log: kuenstliche_effekte_20261120/log_effekte_v3 (32.000 Fälle, Bank-Entscheidungen). Outcome je Fall = −(Zahl der Abweichungen) gegen
BPIC17_ToBe_angepasst (Alignment wie conformance_probe.py; höher ist besser, wie die Durchlaufzeit-Modelle mit negativen Tagen).
Testfälle wie beim PSP (helpers.training.test_cases), Training auf den übrigen; CATEModel learner "cf" (EconML CausalForestDML, wie 12.11.).
Ausgabe models/cate_model_cf_konformanz_simv3.pkl, abweichungen_je_fall.csv und forest_effekte.md hier.
Aufruf im Ordner process-simulation-engine: .venv/bin/python -u <dieses Skript> [Prozesse für Alignments]
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import time
import warnings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
K = ROOT/"prescriptive/scripts/results_100k/kuenstliche_effekte_20261120"
LOG = K/"log_effekte_v3/log_raw.pkl"
ZIEL = ROOT/"prescriptive/models/cate_model_cf_konformanz_simv3.pkl"
PROZESSE = int(sys.argv[1]) if len(sys.argv) > 1 else 12
os.environ["PSP_CATALOG"] = str(ROOT/"prescriptive/data/catalog_neue_punkte_zuteilung_random.yaml")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"prescriptive"))
sys.path.insert(0, str(K))
from prescriptive.helpers.datasets import add_paths  # noqa: E402
add_paths()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CASE = "case:concept:name"


def align_teil(teil):
    import conformance_probe as cp
    return [(c, int(cost // 10000)) for c, _, cost, _ in cp.align(teil)]


def main():
    if ZIEL.exists():
        sys.exit(f"{ZIEL} gibt es schon")
    import conformance_probe as cp
    import interventions
    from helpers.training import test_cases
    from preprocessing import preprocess
    from prescriptive.estimators.cate_model import CATEModel
    start = time.time()
    raw = pd.read_pickle(LOG)
    log = preprocess("bpic17", raw, verbose=False)
    proj = cp.projektion(raw)
    faelle = proj[CASE].drop_duplicates().tolist()
    teile = [proj[proj[CASE].isin(set(faelle[i::PROZESSE * 4]))] for i in range(PROZESSE * 4)]
    with ProcessPoolExecutor(PROZESSE, mp_context=multiprocessing.get_context("spawn")) as pool:
        abw = dict(x for part in pool.map(align_teil, teile) for x in part)
    abw = pd.Series(abw, name="abweichungen")
    abw.to_csv(HERE/"abweichungen_je_fall.csv")
    print(f"Alignments: {len(abw):,} Fälle, Abweichungen je Fall {abw.mean():.3f}, konform {100 * (abw == 0).mean():.1f} % "
          f"({time.time() - start:.0f} s)", flush=True)
    log["outcome"] = -log[CASE].astype(str).map(abw).astype(float)
    log = log[log["outcome"].notna()]
    held = test_cases(log)
    train = log[~log[CASE].astype(str).isin(held)]
    points = interventions.choice_points("bpic17")
    model = CATEModel("bpic17", points, seed=42, learner="cf").fit(train)
    model.test_cases = held
    model.trained_on = {"log": str(LOG), "outcome": "minus Abweichungen je Fall (BPIC17_ToBe_angepasst)", "cases": int(train[CASE].nunique()),
                        "points": [p.name for p in points], "learner": "cf"}
    model.save(str(ZIEL))
    zeilen = []
    test = log[log[CASE].astype(str).isin(held)]
    for point, rows in model.samples(test).items():
        p = next(q for q in points if q.name == point)
        if not model.knows(p) or not rows:
            continue
        _, features, options = zip(*rows)
        tau = model.effects(p, list(features))[:, 0]
        zeilen.append({"Wahlpunkt": point, "Referenz": p.options[0], "Aktion": p.options[1], "Entscheidungen": len(tau),
                       "τ Mittel (Abw. weniger mit Aktion)": tau.mean(), "τ SD": tau.std(), "Anteil Aktion besser": float((tau > 0).mean())})
    tab = pd.DataFrame(zeilen)
    text = (f"# Causal Forest Conformance (simv3), {time.strftime('%Y-%m-%d %H:%M')}\n\nτ = Outcome(Aktion) − Outcome(Referenz), Outcome = −Abweichungen; "
            f"τ > 0: Aktion hat weniger Abweichungen. Testfälle.\n\n```\n{tab.round(3).to_string(index=False)}\n```\n")
    (HERE/"forest_effekte.md").write_text(text)
    print(text, f"gespeichert: {ZIEL} ({time.time() - start:.0f} s)", sep="\n", flush=True)


if __name__ == "__main__":
    main()
