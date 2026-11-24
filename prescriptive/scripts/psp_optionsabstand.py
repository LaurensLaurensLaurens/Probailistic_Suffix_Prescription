"""
Test (2026-11-11): Liegt die schwache Options-Differenz von Variante (d) am Abstand des Options-Events?

Bei Control-Flow-Punkten ist der Zeitpunkt der Wahl die Zeit des Entscheidungs-Events (PSPModel.samples, online ebenso),
das Options-Event steht 1 ms danach (_option_event). Bei (d) ist die Abstandsklasse auch Eingang des Encoders: das
Options-Event hat dann immer die Klasse "0s", die es im Training bei A_Accepted, O_Create Offer, W_Complete application|resume
und W_Handle leads|withdraw nie gibt. Bei stetigem Abstand (c) sind 1 ms statt einiger Minuten praktisch derselbe Wert.

Dieselben Entscheidungen wie scripts/psp_rauschen.py (gleiche Auswahl bei gleichem --decisions); je Modell zwei Varianten,
beide je Option eigens kodiert und mit denselben Zufallszahlen für alle Optionen gezogen:
- "1 ms": Options-Event 1 ms nach dem Entscheidungs-Event (wie bisher)
- "typischer Abstand": Options-Event um den Median des echten Abstands dieser Option an diesem Punkt (Trainingsfälle) später;
  die Zeit bis dahin zählt mit zur Durchlaufzeit (sie ist Teil der Option); "ohne eigene Wartezeit" zieht sie wieder ab
  (dann ändert sich nur die Eingabe ins Netz, nicht die Rechnung)
Je Punkt wie psp_rauschen: Diff Mittel, Vorzeichen wie AIPW, SE gepaart, Korrelation der Optionen.

Aufruf (im Ordner process-simulation-engine):
    PSP_CATALOG=../prescriptive/data/catalog_neue_punkte.yaml CUDA_VISIBLE_DEVICES="" .venv/bin/python -u \
        ../prescriptive/scripts/psp_optionsabstand.py --psp "(c)=psp_model_durchlaufzeit_real_alle_s60_normal.pkl" \
        "(d)=psp_model_durchlaufzeit_real_alle_s60_klassen.pkl" --out ../prescriptive/scripts/results_100k/psp_optionsabstand
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

MODELS = os.path.join(PRESCRIPTIVE, "models")
_ARGS = {}


def _init(settings):
    _ARGS.update(settings)


def _run(job):
    import torch
    label, path, name, decisions, medians = job
    torch.set_num_threads(_ARGS["threads"])
    torch.manual_seed(0)
    from prescriptive.estimators.psp_model import PSPModel
    model = PSPModel.load(path)
    model.set_threads(_ARGS["threads"])
    point = model.points[name]
    options = list(ORDER[name])
    gaps = model.params.get("time_from") == "gaps"
    n = _ARGS["samples"]
    rows = []
    for variant, shift in (("1 ms", {o: 0.0 for o in options}), ("typischer Abstand", medians)):
        for case_id, events, now, chosen, _ in decisions:
            encoded = []
            for option in options:
                event = model._option_event(point, events, option, now)
                event[TIMESTAMP] = event[TIMESTAMP] + pd.Timedelta(seconds=float(shift[option]))
                prefix, static, mask, length, elapsed = model._encode(events, event)
                encoded.append(((prefix, static, mask, length), elapsed))
            seed = int(torch.randint(2 ** 31 - 1, (1,)))
            values = []
            for batch, elapsed in encoded:
                torch.manual_seed(seed)
                drawn = model.draw_cases([batch], n)[0]
                values.append([model.outcome(elapsed, s, "negative_cycle_days", gaps=gaps) for s in drawn])
            values = np.array(values)
            d = values[0] - values[1]
            own = (float(shift[options[0]]) - float(shift[options[1]])) / 86400  # eigene Wartezeit der Aktion minus der Referenz in Tagen
            rows.append({"Modell": label, "Variante": variant, "Wahlpunkt": name, "Fall": case_id, "Diff": d.mean(),
                         "Diff ohne eigene Wartezeit": d.mean() + own,
                         "SE gepaart": d.std(ddof=1) / np.sqrt(n), "Korrelation": np.corrcoef(values[0], values[1])[0, 1]})
    print(f"{label} {name}: fertig", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--psp", nargs="+", required=True, metavar="BESCHRIFTUNG=MODELL")
    parser.add_argument("--out", required=True)
    parser.add_argument("--decisions", type=int, default=200)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    add_paths()
    from helpers.datasets import load_dataset
    from preprocessing import preprocess
    from prescriptive.estimators.psp_model import PSPModel
    started = time.time()
    os.makedirs(args.out, exist_ok=True)
    models = [(label, path if os.path.isabs(path) else os.path.join(MODELS, path)) for label, path in (e.split("=", 1) for e in args.psp)]
    model = PSPModel.load(models[0][1])
    raw, _, _ = load_dataset("bpic17")
    log = preprocess("bpic17", raw)
    del raw
    is_test = log[CASE].astype(str).isin(set(map(str, model.test_cases)))
    # Median des echten Abstands Entscheidungs-Event -> Options-Event je Punkt und Option, aus den Trainingsfällen
    medians = {}
    for name, entries in model.samples(log[~is_test]).items():
        if name not in ORDER:
            continue
        gaps = {}
        for _, (part, cut, now), option in entries:
            when = pd.Timestamp(part[TIMESTAMP].iloc[cut])
            when = when.tz_convert("UTC").tz_localize(None) if when.tzinfo is not None else when
            gaps.setdefault(option, []).append((when - pd.Timestamp(now, unit="s")).total_seconds())
        medians[name] = {option: float(np.median(gaps[option])) for option in ORDER[name] if option in gaps}
        print(f"{name}: Median-Abstand je Option (s) {medians[name]}", flush=True)
    test = log[is_test].sort_values([CASE, TIMESTAMP], kind="stable")
    total = test.groupby(CASE)[TIMESTAMP].agg(lambda s: (s.max() - s.min()).total_seconds() / 86400)
    samples = model.samples(test)
    rng = np.random.default_rng(0)  # Auswahl wie psp_rauschen
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
        for label, path in models:
            jobs.append((label, path, name, pick, medians[name]))
    del log, test
    _ARGS.update(samples=args.samples, threads=args.threads)
    with ProcessPoolExecutor(min(args.workers, len(jobs)), mp_context=mp.get_context("spawn"), initializer=_init,
                             initargs=(dict(_ARGS),)) as pool:
        table = pd.concat(pool.map(_run, jobs), ignore_index=True)
    table.to_csv(os.path.join(args.out, "entscheidungen.csv"), index=False)
    out = {}
    for (label, variant, name), t in table.groupby(["Modell", "Variante", "Wahlpunkt"], sort=False):
        sign = np.sign(AIPW[name])
        out[(name, label, variant)] = {"AIPW": AIPW[name], "Diff Mittel": t["Diff"].mean(), "SE des Mittels": t["Diff"].std(ddof=1) / np.sqrt(len(t)),
                                       "Vorzeichen wie AIPW": float((np.sign(t["Diff"]) == sign).mean()),
                                       "Diff ohne eigene Wartezeit": t["Diff ohne eigene Wartezeit"].mean(),
                                       "Vorzeichen wie AIPW ohne eigene Wartezeit": float((np.sign(t["Diff ohne eigene Wartezeit"]) == sign).mean()),
                                       "SE gepaart Median": t["SE gepaart"].median(), "Korrelation Median": t["Korrelation"].median()}
    summary = pd.DataFrame(out).T
    summary.to_csv(os.path.join(args.out, "optionsabstand.csv"))
    with pd.option_context("display.width", 250):
        print(summary.round(3).to_string())
    print(f"######## Fertig in {time.time() - started:.0f} s; Ergebnisse in {args.out}", flush=True)


if __name__ == "__main__":
    main()
