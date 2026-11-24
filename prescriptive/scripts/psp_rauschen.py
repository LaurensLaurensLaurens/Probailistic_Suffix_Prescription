"""
Rauschen der PSP-Bewertung an den Control-Flow-Punkten (2026-11-10): Wie deutlich unterscheidet das PSP die beiden Optionen
einer Entscheidung, gemessen am Standardfehler, den die gezogenen Suffixe selbst liefern?

Dieselben Entscheidungen wie scripts/psp_diagnose.py (erste Entscheidungen von Testfällen des echten Logs, je Punkt halb
mit der Aktion, halb mit der Referenz gewählt; ORDER und AIPW von dort), nur die Variante "Abfrage" (Fallattribute aus dem
Prefix, wie online). Je Entscheidung samples Suffixe je Option mit denselben Zufallszahlen (Suffix k jeder Option hat
dieselben Dropout-Masken und dasselbe Rauschen), Wert je Suffix wie die Policy (model.outcome, negative Durchlaufzeit).
Je Entscheidung:
- Diff = Mittel über k von (Wert Aktion_k - Wert Referenz_k), positiv = Aktion kürzer
- SE gepaart = Standardabweichung der Differenzen / sqrt(samples) (nutzt die gemeinsamen Zufallszahlen),
  SE ungepaart = sqrt(Var Aktion / samples + Var Referenz / samples) (als ob unabhängig gezogen)
- z = Diff / SE gepaart; nötige Ziehungen für |z| = 2 bei gleichem Diff: (2 · SD der Differenzen / |Diff|)^2
Je Punkt die Zusammenfassung (rauschen.csv) und je Entscheidung die Werte (entscheidungen.csv). Die Policy psp wählt nach
dem Mittel (Diff > 0: Aktion); "Regel 2 SE" wählt nur bei |z| > 2, sonst entscheidet die Bank.

Aufruf (im Ordner process-simulation-engine; Katalog wie beim Training über PSP_CATALOG):
    PSP_CATALOG=../prescriptive/data/catalog_neue_punkte.yaml .venv/bin/python -u ../prescriptive/scripts/psp_rauschen.py \
        --model ../prescriptive/models/psp_model_durchlaufzeit_<tag>.pkl --out ../prescriptive/scripts/results_100k/psp_rauschen_<tag>
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("MPLBACKEND", "Agg")
PRESCRIPTIVE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PRESCRIPTIVE)
sys.path.insert(0, os.path.join(PRESCRIPTIVE, "scripts"))

import numpy as np
import pandas as pd

from helpers.datasets import CASE, TIMESTAMP, add_paths
from psp_diagnose import AIPW, ORDER

_ARGS = {}


def _init(settings):
    _ARGS.update(settings)


def _run(job):
    import torch
    name, decisions = job
    torch.set_num_threads(_ARGS["threads"])
    torch.manual_seed(0)
    from prescriptive.estimators.psp_model import PSPModel
    model = PSPModel.load(_ARGS["model"])
    model.set_threads(_ARGS["threads"])
    point = model.points[name]
    options = list(ORDER[name])
    gaps = model.params.get("time_from") == "gaps"
    n = _ARGS["samples"]
    rows = []
    for case_id, events, now, chosen, total_days in decisions:
        drawn, elapsed = model.option_suffixes(point, events, options, now, n)
        values = np.array([[model.outcome(elapsed, s, "negative_cycle_days", gaps=gaps) for s in suffixes] for suffixes in drawn])
        d = values[0] - values[1]
        sd = d.std(ddof=1)
        rows.append({"Wahlpunkt": name, "Fall": case_id, "gewählt": chosen, "Durchlaufzeit echt": total_days,
                     "Diff": d.mean(), "SE gepaart": sd / np.sqrt(n),
                     "SE ungepaart": np.sqrt(values[0].var(ddof=1) / n + values[1].var(ddof=1) / n),
                     "Korrelation der Optionen": np.corrcoef(values[0], values[1])[0, 1],
                     "SD Durchlaufzeit gezogen": 0.5 * (values[0].std(ddof=1) + values[1].std(ddof=1)),
                     "Durchlaufzeit gezogen": -values[0 if chosen == options[0] else 1].mean(),
                     "nötige Ziehungen für |z|=2": (2 * sd / abs(d.mean())) ** 2 if d.mean() != 0 else np.inf})
    return pd.DataFrame(rows)


def summarize(table):
    out = {}
    for name, t in table.groupby("Wahlpunkt", sort=False):
        sign = np.sign(AIPW.get(name, np.nan))
        z = t["Diff"] / t["SE gepaart"]
        clear = z.abs() > 2
        out[name] = {
            "Entscheidungen": len(t), "Effekt laut AIPW (Tage)": AIPW.get(name, np.nan),
            "Diff Mittel (Tage)": t["Diff"].mean(), "Diff Betrag Median": t["Diff"].abs().median(),
            "SE gepaart Median": t["SE gepaart"].median(), "SE ungepaart Median": t["SE ungepaart"].median(),
            "Korrelation der Optionen Median": t["Korrelation der Optionen"].median(),
            "|z| Median": z.abs().median(), "Anteil |z| > 2": clear.mean(),
            "Vorzeichen wie AIPW (alle)": float((np.sign(t["Diff"]) == sign).mean()),
            "Vorzeichen wie AIPW (|z| > 2)": float((np.sign(t.loc[clear, "Diff"]) == sign).mean()) if clear.any() else np.nan,
            "Anteil Aktion nach Mittel": float((t["Diff"] > 0).mean()),
            "Regel 2 SE: Aktion": float((z > 2).mean()), "Regel 2 SE: Referenz": float((z < -2).mean()),
            "Regel 2 SE: Bank": float((~clear).mean()),
            "nötige Ziehungen für |z|=2 (Median)": t["nötige Ziehungen für |z|=2"].median(),
            "SD der gezogenen Durchlaufzeit (Tage)": t["SD Durchlaufzeit gezogen"].mean(),
            "Durchlaufzeit gezogen (Mittel)": t["Durchlaufzeit gezogen"].mean(), "Durchlaufzeit echt (Mittel)": t["Durchlaufzeit echt"].mean()}
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--decisions", type=int, default=200, help="Entscheidungen je Punkt (halb Aktion, halb Referenz)")
    parser.add_argument("--samples", type=int, default=100, help="gezogene Suffixe je Option (wie online)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    add_paths()
    from helpers.datasets import load_dataset
    from preprocessing import preprocess
    from prescriptive.estimators.psp_model import PSPModel
    started = time.time()
    os.makedirs(args.out, exist_ok=True)
    raw, _, _ = load_dataset("bpic17")
    log = preprocess("bpic17", raw)
    del raw
    model = PSPModel.load(args.model)
    test = log[log[CASE].astype(str).isin(set(map(str, model.test_cases)))].sort_values([CASE, TIMESTAMP], kind="stable")
    total = test.groupby(CASE)[TIMESTAMP].agg(lambda s: (s.max() - s.min()).total_seconds() / 86400)
    samples = model.samples(test)
    rng = np.random.default_rng(0)
    jobs = []
    for name in ORDER:
        if name not in model.points:
            continue
        seen, decisions = set(), []
        for case_id, (part, cut, now), option in samples.get(name, []):
            if case_id not in seen:
                seen.add(case_id)
                decisions.append((case_id, part.iloc[:cut].to_dict("records"), now, option, float(total[case_id])))
        action = [d for d in decisions if d[3] == ORDER[name][0]]
        reference = [d for d in decisions if d[3] != ORDER[name][0]]
        half = args.decisions // 2
        pick = [action[i] for i in rng.choice(len(action), min(half, len(action)), replace=False)] + \
               [reference[i] for i in rng.choice(len(reference), min(half, len(reference)), replace=False)]
        jobs.append((name, pick))
        print(f"{name}: {len(decisions):,} Entscheidungen in den Testfällen, {len(pick)} ausgewertet", flush=True)
    del log, test
    _ARGS.update(model=os.path.abspath(args.model), samples=args.samples, threads=args.threads)
    with ProcessPoolExecutor(min(args.workers, len(jobs)), mp_context=mp.get_context("spawn"), initializer=_init,
                             initargs=(dict(_ARGS),)) as pool:
        table = pd.concat(pool.map(_run, jobs), ignore_index=True)
    table.to_csv(os.path.join(args.out, "entscheidungen.csv"), index=False)
    summary = summarize(table)
    summary.to_csv(os.path.join(args.out, "rauschen.csv"))
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(summary.round(3).to_string())
    print(f"######## Fertig in {time.time() - started:.0f} s; Ergebnisse in {args.out}", flush=True)


if __name__ == "__main__":
    main()
