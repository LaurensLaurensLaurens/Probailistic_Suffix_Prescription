"""Hat der Causal Forest ein fallindividuelles Signal? (2026-11-16)

Gegenstück zur PSP-Prüfung (zwei unabhängige Schätzungen desselben Falls): Die Trainingsfälle des echten Logs werden
zufällig in zwei disjunkte Hälften geteilt, auf jeder wird derselbe Schätzer neu trainiert (estimators/cate_model.py,
unverändert), und beide schätzen den Effekt τ für dieselben Testentscheidungen. Korrelieren die beiden τ über die Fälle,
ist die Streuung zwischen den Fällen wiederholbar; die Kovarianz schätzt ihren wiederholbaren Anteil.

Teil 2: Stimmt die Streuung mit der Wahrheit im Simulator überein? Das vorhandene Modell
models/cate_model_cf_durchlaufzeit_real_alle.pkl bewertet die randomisierten Erstentscheidungen der Simulator-RCT-Daten
vom 14.11.; je Wahlpunkt werden die Fälle am Median von τ geteilt und die randomisierte Wirkung in beiden Hälften verglichen.

Es wird nichts in models/ geschrieben. Aufruf im Ordner process-simulation-engine:  .venv/bin/python -u <dieses Skript>
"""
import os
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[4]
P = ROOT/"prescriptive"
OUT = Path(__file__).resolve().parent/"forest_stabilitaet"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PSP_CATALOG"] = str(P/"data/catalog_neue_punkte_zuteilung_random.yaml")
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(P))
sys.path.insert(0, str(P/"scripts"))
from helpers.datasets import CASE, add_paths
add_paths()
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RCT = P/"scripts/results_100k/d_rct_tf1_training_20261114/rct_data/rct_bpic17_angebot_nachlegen_delta50/log.pkl"
ORDER = ["annehmen_bei_konzept", "annehmen_nach_vervollstaendigung", "angebot_nachlegen", "handle_leads"]


def main():
    import interventions
    from helpers.datasets import load_dataset
    from helpers.training import test_cases
    from preprocessing import preprocess
    from prescriptive.estimators.cate_model import LEARNERS, CATEModel
    from psp_diagnose import AIPW
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.time()
    raw, _, _ = load_dataset("bpic17")
    log = preprocess("bpic17", raw)
    del raw
    held_out = test_cases(log)
    is_test = log[CASE].astype(str).isin(held_out)
    train, test = log[~is_test], log[is_test]
    cases = np.array(sorted(train[CASE].astype(str).unique()), dtype=object)
    np.random.default_rng(0).shuffle(cases)
    half = {"A": set(cases[:len(cases)//2]), "B": set(cases[len(cases)//2:])}
    parts = {name: train[train[CASE].astype(str).isin(ids)] for name, ids in half.items()}
    points = interventions.choice_points("bpic17")
    first = CATEModel("bpic17", points, seed=42, learner="cf")
    samples = {name: first.samples(part) for name, part in parts.items()}
    test_samples = first.samples(test)
    print(f"Hälften: {len(half['A']):,} und {len(half['B']):,} Trainingsfälle, {len(held_out):,} Testfälle ({time.time()-started:.0f} s)", flush=True)
    rows, effects = [], {}
    for learner in ("cf", "x", "s"):
        models = {name: CATEModel("bpic17", points, seed=42, learner=learner).fit(parts[name], samples=samples[name]) for name in ("A", "B")}
        for point in points:
            if point.name not in ORDER or not all(model.knows(point) for model in models.values()):
                continue
            case_ids, features, _ = zip(*test_samples[point.name])
            a, b = (models[name].effects(point, list(features))[:, 0] for name in ("A", "B"))
            effects[(learner, point.name)] = pd.DataFrame({"case": case_ids, "tau_A": a, "tau_B": b})
            sign = np.sign(AIPW[point.name])
            cov = np.cov(a, b)[0, 1]
            rows.append({"Schätzer": LEARNERS[learner], "Wahlpunkt": point.name, "Testentscheidungen": len(a),
                         "Mittel A": a.mean(), "Mittel B": b.mean(), "SD über Fälle A": a.std(), "SD über Fälle B": b.std(),
                         "Korrelation A/B": np.corrcoef(a, b)[0, 1], "Rangkorrelation A/B": spearmanr(a, b)[0],
                         "wiederholbare SD (aus Kovarianz)": np.sqrt(max(cov, 0)), "mittlere |A - B|": np.abs(a-b).mean(),
                         "Vorzeichen wie AIPW A": (np.sign(a) == sign).mean(), "Vorzeichen wie AIPW B": (np.sign(b) == sign).mean(),
                         "beide gleiche Wahl": (np.sign(a) == np.sign(b)).mean()})
        print(f"{LEARNERS[learner]} fertig ({time.time()-started:.0f} s)", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT/"split_half.csv", index=False)
    pd.to_pickle(effects, OUT/"effekte_split_half.pkl")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.round(3).to_string(index=False), flush=True)

    # Teil 2: τ des vorhandenen Walds gegen die randomisierte Wirkung im Simulator
    try:
        forest = CATEModel.load(P/"models/cate_model_cf_durchlaufzeit_real_alle.pkl")
        rct = pd.read_pickle(RCT)
        rct = rct[rct["rct"].astype(bool)].copy()
        rct["concept:name"] = rct["concept:name"].astype(str).str.strip()
        times = rct.groupby(CASE)["time:timestamp"]
        total_hours = ((times.max()-times.min()).dt.total_seconds()/3600)
        rct_samples = forest.samples(rct)
        result = []
        for point in points:
            if point.name not in ORDER or not forest.knows(point):
                continue
            seen, picked = set(), []
            for case, features, option in rct_samples[point.name]:
                if case not in seen:  # erste Entscheidung des Falls an diesem Punkt
                    seen.add(case)
                    picked.append((case, features, option))
            ids, features, options = zip(*picked)
            tau = forest.effects(point, list(features))[:, 0]
            rest = (total_hours.reindex(list(ids)).to_numpy()-np.array([f["elapsed_hours"] for f in features]))/24
            action = np.array(options) == point.options[1]
            frame = pd.DataFrame({"tau": tau, "rest": rest, "action": action})
            frame["gruppe"] = np.where(frame.tau >= frame.tau.median(), "obere Hälfte von τ", "untere Hälfte von τ")
            for name, part in list(frame.groupby("gruppe")) + [("alle", frame)]:
                a, b = part.rest[part.action], part.rest[~part.action]
                result.append({"Wahlpunkt": point.name, "Gruppe": name, "n Aktion": len(a), "n Referenz": len(b), "τ Wald (Mittel)": part.tau.mean(),
                               "Wirkung im Simulator (Tage)": b.mean()-a.mean(), "SE": np.sqrt(a.var()/len(a)+b.var()/len(b))})
        truth = pd.DataFrame(result)
        truth.to_csv(OUT/"wald_gegen_simulator.csv", index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print("\nτ des Walds (echter Log) gegen die randomisierte Wirkung im Simulator, Erstentscheidungen:")
            print(truth.round(3).to_string(index=False), flush=True)
    except Exception as error:  # Teil 2 ist Zusatz; Teil 1 bleibt gültig
        print("Teil 2 nicht ausgewertet:", repr(error), flush=True)
    print(f"FERTIG in {time.time()-started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
