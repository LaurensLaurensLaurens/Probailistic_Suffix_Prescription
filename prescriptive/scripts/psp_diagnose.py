"""
Diagnose eines PSP an den Control-Flow-Punkten auf den Testfällen des echten Logs: sagt es die Durchlaufzeit eines Falls
vorher, und hängen seine Suffixe an der Option?

Je Punkt (ORDER: Aktion und Referenz wie im Screening) decisions erste Entscheidungen von Testfällen (model.test_cases), zur
Hälfte mit der Aktion, zur Hälfte mit der Referenz. Je Entscheidung samples Suffixe je Option (gleiche Zufallszahlen) in drei
Varianten der statischen Merkmale: "Abfrage" (aus dem Prefix, wie online), "Training" (aus dem ganzen Fall, wie der Loader
im Training: auch Angebotswerte aus der Zukunft) und "Training, Accepted umgedreht". Kennzahlen je Punkt und Variante:
- Unterschied zwischen den Optionen (Aktion − Referenz, positiv = Aktion kürzer): Mittel, Betrag, Vorzeichen wie AIPW
  (AIPW: Effekt laut Screening auf dem echten Log, screening_alle 2026-11-06)
- faktisch (die gewählte Option): Fehler und Korrelation gegen die echte Durchlaufzeit, Unterschied der Gruppen
- Länge der gezogenen Suffixe, Unterschied nach 5 Schritten (so weit trainiert der Decoder im Original)
Hat das Netz einen Restzeit-Kopf (TRAIN remaining_head), dieselben Kennzahlen (ohne Länge) auch für die Bewertung über den
Kopf (Spalten "Kopf ..."; wie die Policy mit value_from head: Erwartungswert des Kopfs, gemittelt über samples Dropout-Masken).

Aufruf (im Ordner process-simulation-engine, Wahlpunkte des Modells; Katalog über PSP_CATALOG, z. B. data/catalog_neue_punkte.yaml):
    .venv/bin/python -u ../prescriptive/scripts/psp_diagnose.py --model ../prescriptive/models/psp_model_durchlaufzeit_<tag>.pkl \
        --out ../prescriptive/scripts/results_100k/psp_diagnose_<tag>
Ergebnis: <out>/diagnose.csv (je Punkt), <out>/entscheidungen.csv (je Entscheidung).
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

import numpy as np
import pandas as pd

from helpers.datasets import CASE, TIMESTAMP, add_paths

AIPW = {"annehmen_bei_konzept": 5.2, "annehmen_nach_vervollstaendigung": 3.0, "angebot_nachlegen": -1.4, "handle_leads": -2.2}
ORDER = {"annehmen_bei_konzept": ("A_Accepted", "W_Complete application|start"),
         "annehmen_nach_vervollstaendigung": ("A_Accepted", "W_Complete application|resume"),
         "angebot_nachlegen": ("O_Create Offer", "O_Sent (mail and online)"),
         "handle_leads": ("W_Handle leads|start", "W_Handle leads|withdraw")}
LEAKY = ["Accepted", "Selected", "CreditScore", "FirstWithdrawalAmount", "NumberOfTerms", "MonthlyCost"]
_ARGS = {}


def _init(settings):
    _ARGS.update(settings)  # in jedem Arbeitsprozess (spawn: das Modul ist dort neu geladen)


def _flip(value):
    if value is None or pd.isna(value):
        return None
    return (not bool(value)) if isinstance(value, (bool, np.bool_)) else ("False" if str(value) == "True" else "True")


def _run(job):
    import torch
    name, decisions = job
    torch.set_num_threads(2)
    torch.manual_seed(0)
    from prescriptive.estimators.psp_model import PSPModel
    model = PSPModel.load(_ARGS["model"])
    model.set_threads(2)
    point = model.points[name]
    options = list(ORDER[name])
    head = getattr(model.net, "remaining_head", None) is not None
    rows = []
    for case_id, events, now, chosen, statics, total_days in decisions:
        row = {"Wahlpunkt": name, "Fall": case_id, "gewählt": chosen, "Durchlaufzeit echt": total_days}
        variants = {"Abfrage": events, "Training": [{**e, **statics} for e in events]}
        flipped = _flip(statics.get("Accepted"))
        if flipped is not None:
            variants["Training, Accepted umgedreht"] = [{**e, **statics, "Accepted": flipped} for e in events]
        for label, evs in variants.items():
            drawn, elapsed = model.option_suffixes(point, evs, options, now, _ARGS["samples"])
            gaps = model.params.get("time_from") == "gaps"  # wie die Policy: Durchlaufzeit aus der Summe der Abstände, wenn so eingestellt
            values = np.array([[model.outcome(elapsed, s, "negative_cycle_days", gaps=gaps) for s in suffixes] for suffixes in drawn])
            at5 = [np.mean([(s[min(4, len(s) - 1)][1] if s else elapsed) / 86400 for s in suffixes]) for suffixes in drawn]
            row.update({f"{label}: Wert Aktion": values[0].mean(), f"{label}: Wert Referenz": values[1].mean(),
                        f"{label}: Diff nach 5 Schritten": -(at5[0] - at5[1]),
                        f"{label}: Länge Aktion": np.mean([len(s) for s in drawn[0]]), f"{label}: Länge Referenz": np.mean([len(s) for s in drawn[1]])})
            if head:
                remaining, at = model.option_remaining_many(point, [evs], options, now, _ARGS["samples"], expected=True)
                row.update({f"Kopf {label}: Wert Aktion": -(at[0] / 86400 + remaining[0][0].mean()),
                            f"Kopf {label}: Wert Referenz": -(at[0] / 86400 + remaining[1][0].mean())})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(table):
    rows = {}
    for name, t in table.groupby("Wahlpunkt", sort=False):
        act = (t["gewählt"] == ORDER[name][0]).to_numpy()
        out = {"Entscheidungen": len(t), "Effekt laut AIPW (echt)": AIPW.get(name, np.nan),
               "echt: Unterschied der Gruppen": -(t.loc[act, "Durchlaufzeit echt"].mean() - t.loc[~act, "Durchlaufzeit echt"].mean())}
        for source in ("", "Kopf "):  # Bewertung über die gezogenen Suffixe, und über den Restzeit-Kopf, wenn das Netz einen hat
            for variant in ("Abfrage", "Training", "Training, Accepted umgedreht"):
                label = source + variant
                if f"{label}: Wert Aktion" not in t or t[f"{label}: Wert Aktion"].isna().all():
                    continue
                diff = t[f"{label}: Wert Aktion"] - t[f"{label}: Wert Referenz"]
                out.update({f"{label}: Option-Diff Mittel": diff.mean(), f"{label}: Option-Diff Betrag": diff.abs().mean(),
                            f"{label}: Vorzeichen wie AIPW": float((np.sign(diff) == np.sign(AIPW.get(name, np.nan))).mean())})
                if variant == "Training, Accepted umgedreht":
                    out[f"{source}Accepted umdrehen: Änderung (Betrag)"] = (t[f"{label}: Wert Aktion"] - t[f"{source}Training: Wert Aktion"]).abs().mean()
                    continue
                factual = np.where(act, t[f"{label}: Wert Aktion"], t[f"{label}: Wert Referenz"])
                out.update({f"{label}: Fehler faktisch (MAE, Tage)": float(np.mean(np.abs(-factual - t["Durchlaufzeit echt"]))),
                            f"{label}: Korrelation faktisch": float(np.corrcoef(-factual, t["Durchlaufzeit echt"])[0, 1]),
                            f"{label}: Unterschied der Gruppen faktisch": float(factual[act].mean() - factual[~act].mean())})
                if not source:
                    out.update({f"{label}: Diff nach 5 Schritten": t[f"{label}: Diff nach 5 Schritten"].mean(),
                                f"{label}: Suffixlänge": float(np.where(act, t[f"{label}: Länge Aktion"], t[f"{label}: Länge Referenz"]).mean())})
        rows[name] = out
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--decisions", type=int, default=100, help="Entscheidungen je Punkt (halb Aktion, halb Referenz)")
    parser.add_argument("--samples", type=int, default=30, help="gezogene Suffixe je Option")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    add_paths()
    from helpers.datasets import load_dataset
    from preprocessing import preprocess
    from prescriptive.estimators.psp_model import PSPModel
    started = time.time()
    os.makedirs(args.out, exist_ok=True)
    raw, _, _ = load_dataset("bpic17")
    log = preprocess("bpic17", raw)
    model = PSPModel.load(args.model)
    test = log[log[CASE].astype(str).isin(set(map(str, model.test_cases)))].sort_values([CASE, TIMESTAMP], kind="stable")
    statics = test.groupby(CASE)[LEAKY].agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan).to_dict("index")
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
                decisions.append((case_id, part.iloc[:cut].to_dict("records"), now, option, statics[case_id], float(total[case_id])))
        action = [d for d in decisions if d[3] == ORDER[name][0]]
        reference = [d for d in decisions if d[3] != ORDER[name][0]]
        half = args.decisions // 2
        pick = [action[i] for i in rng.choice(len(action), min(half, len(action)), replace=False)] + \
               [reference[i] for i in rng.choice(len(reference), min(half, len(reference)), replace=False)]
        jobs.append((name, pick))
        print(f"{name}: {len(decisions):,} Entscheidungen in den Testfällen, {len(pick)} ausgewertet", flush=True)
    _ARGS.update(model=os.path.abspath(args.model), samples=args.samples)
    with ProcessPoolExecutor(min(args.workers, len(jobs)), mp_context=mp.get_context("spawn"), initializer=_init,
                             initargs=(dict(_ARGS),)) as pool:
        table = pd.concat(pool.map(_run, jobs), ignore_index=True)
    table.to_csv(os.path.join(args.out, "entscheidungen.csv"), index=False)
    summary = summarize(table)
    summary.to_csv(os.path.join(args.out, "diagnose.csv"))
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(summary.round(2).to_string())
    print(f"######## Fertig in {time.time() - started:.0f} s; Ergebnisse in {args.out}", flush=True)


if __name__ == "__main__":
    main()
