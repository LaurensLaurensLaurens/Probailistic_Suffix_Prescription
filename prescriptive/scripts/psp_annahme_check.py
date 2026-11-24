"""
Sieht das PSP an den Wahlpunkten einen Unterschied in der Angebotsannahme (2026-11-12)? Offline-Prüfung vor einem
Online-Test der Policy psp_annahme (policies/accepted_policy/psp_accepted_policy.py), wie scripts/psp_rauschen.py für die
Durchlaufzeit.

Entscheidungen: je Punkt die erste Entscheidung von Testfällen des echten Logs, halb mit der Aktion, halb mit der
Referenz gewählt (ORDER), ohne Fälle mit O_Accepted schon im Prefix. Je Entscheidung samples Suffixe je Option mit
denselben Zufallszahlen; Wert je Suffix: enthält es O_Accepted (psp_accepted_policy.reached)? Je Entscheidung:
- Diff = Anteil Aktion − Anteil Referenz (Prozentpunkte), SE gepaart aus den Differenzen je Suffix
Je Punkt: mittlere Diff gegen den AIPW-Effekt auf dem echten Log (screening/results_accepted, Prozentpunkte), Anteil mit
dem Vorzeichen des AIPW, wie oft die Policy die Aktion wählen würde, Anteil gezogener Suffixe mit O_Accepted und mit
Fallende (A_Pending, A_Denied, A_Cancelled) gegen den echten Anteil dieser Entscheidungen.

Aufruf (im Ordner process-simulation-engine; Katalog wie beim Training über PSP_CATALOG):
    PSP_CATALOG=../prescriptive/data/catalog_accepted_punkte.yaml CUDA_VISIBLE_DEVICES="" .venv/bin/python -u \
        ../prescriptive/scripts/psp_annahme_check.py --model <pfad> --out ../prescriptive/scripts/results_100k/psp_annahme_<tag>
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

from helpers.datasets import ACTIVITY, CASE, TIMESTAMP, add_paths

TARGET = "O_Accepted"
ENDS = ("A_Pending", "A_Denied", "A_Cancelled")
# Aktion, Referenz je Punkt und AIPW-Effekt der Aktion auf O_Accepted (Prozentpunkte, screening/results_accepted, echtes Log)
ORDER = {"annehmen_bei_konzept": ("A_Accepted", "W_Complete application|start"),
         "annehmen_nach_vervollstaendigung": ("A_Accepted", "W_Complete application|resume"),
         "handle_leads": ("W_Handle leads|start", "W_Handle leads|withdraw"),
         "neues_angebot_beim_nachfassen": ("O_Create Offer", "W_Call after offers|resume")}
AIPW = {"annehmen_bei_konzept": 10.74, "annehmen_nach_vervollstaendigung": 5.35, "handle_leads": -11.83, "neues_angebot_beim_nachfassen": 12.57}
_ARGS = {}


def _init(settings):
    _ARGS.update(settings)


def _has(suffix, names):
    return any(str(label).split("|")[0] in names for label, _, _ in suffix)


def _run(job):
    import torch
    name, decisions = job
    torch.set_num_threads(_ARGS["threads"])
    torch.manual_seed(0)
    from prescriptive.estimators.psp_model import PSPModel
    model = PSPModel.load(_ARGS["model"])
    point = model.points[name]
    options = list(ORDER[name])
    n = _ARGS["samples"]
    rows = []
    for case_id, events, now, chosen, accepted in decisions:
        drawn, _ = model.option_suffixes(point, events, options, now, n)
        hit = np.array([[_has(s, (TARGET,)) for s in suffixes] for suffixes in drawn], dtype=float)
        end = np.array([[_has(s, ENDS) for s in suffixes] for suffixes in drawn], dtype=float)
        d = hit[0] - hit[1]
        rows.append({"Wahlpunkt": name, "Fall": case_id, "gewählt": chosen, "O_Accepted echt": accepted,
                     "Diff (pp)": 100 * d.mean(), "SE gepaart (pp)": 100 * d.std(ddof=1) / np.sqrt(n),
                     "Anteil Annahme Aktion": hit[0].mean(), "Anteil Annahme Referenz": hit[1].mean(),
                     "Anteil Annahme gewählte Option": hit[0 if chosen == options[0] else 1].mean(), "Anteil mit Fallende": end.mean()})
    return pd.DataFrame(rows)


def summarize(table):
    out = {}
    for name, t in table.groupby("Wahlpunkt", sort=False):
        sign = np.sign(AIPW[name])
        z = t["Diff (pp)"] / t["SE gepaart (pp)"].replace(0, np.nan)
        out[name] = {"Entscheidungen": len(t), "Effekt laut AIPW (pp)": AIPW[name], "Diff Mittel (pp)": t["Diff (pp)"].mean(),
                     "SE gepaart Median (pp)": t["SE gepaart (pp)"].median(), "Anteil |z| > 2": float((z.abs() > 2).mean()),
                     "Vorzeichen wie AIPW (Diff ≠ 0)": float((np.sign(t.loc[t["Diff (pp)"] != 0, "Diff (pp)"]) == sign).mean()),
                     "Anteil Diff = 0": float((t["Diff (pp)"] == 0).mean()),
                     "Anteil Aktion nach Mittel": float((t["Diff (pp)"] > 0).mean()),
                     "Annahme gezogen (gewählte Option)": t["Anteil Annahme gewählte Option"].mean(), "Annahme echt": t["O_Accepted echt"].mean(),
                     "Suffixe mit Fallende": t["Anteil mit Fallende"].mean()}
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
    accepted = set(test.loc[test[ACTIVITY] == TARGET, CASE])
    samples = model.samples(test)
    rng = np.random.default_rng(0)
    jobs = []
    for name in ORDER:
        if name not in model.points:
            print(f"{name}: nicht im Modell", flush=True)
            continue
        seen, decisions = set(), []
        for case_id, (part, cut, now), option in samples.get(name, []):
            events = part.iloc[:cut].to_dict("records")
            if case_id in seen:
                continue
            seen.add(case_id)
            if any(event.get(ACTIVITY) == TARGET for event in events):
                continue  # Ziel schon erreicht: hängt nicht mehr von der Wahl ab (wie im Screening)
            decisions.append((case_id, events, now, option, float(case_id in accepted)))
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
    summary.to_csv(os.path.join(args.out, "annahme.csv"))
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(summary.round(3).to_string())
    print(f"######## Fertig in {time.time() - started:.0f} s; Ergebnisse in {args.out}", flush=True)


if __name__ == "__main__":
    main()
