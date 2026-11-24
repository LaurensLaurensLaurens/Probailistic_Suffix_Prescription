"""
Woher kommt der Unterschied zwischen den Optionen im PSP (2026-11-11)? Zerlegung je Wahlpunkt in Event-Pfad und Zeiten.

Dieselben Entscheidungen wie scripts/psp_rauschen.py (erste Entscheidungen von Testfällen des echten Logs, je Punkt halb
mit der Aktion, halb mit der Referenz gewählt, gleiche Auswahl bei gleichem --decisions). Je Entscheidung und Modell
samples Suffixe je Option mit denselben Zufallszahlen (option_suffixes). Je gezogenem Suffix:
- Restzeit ab dem Options-Event in Tagen (wie die Bewertung der Policy: model.outcome)
- Ende: das erste von A_Pending / A_Denied / A_Cancelled im Suffix, sonst "keins"
- Anzahl Events, Summe der Abstände ab 1 Tag ("lange Wartezeiten") und darunter
Je Punkt (alle Ziehungen aller Entscheidungen zusammen) mit p_o(e) = Anteil Ende e und T_o(e) = mittlere Restzeit bei
Ende e unter Option o (Aktion a, Referenz r):
    T_a - T_r = Σ_e (p_a(e) - p_r(e)) · (T_a(e) + T_r(e)) / 2     Pfad: die Option ändert, wie der Fall endet
              + Σ_e (p_a(e) + p_r(e)) / 2 · (T_a(e) - T_r(e))     Zeit: die Option ändert die Dauer bei gleichem Ende
Vorzeichen wie AIPW: Vorteil der Aktion = T_r - T_a (positiv = Aktion kürzer). Dazu dieselbe Zerlegung für die echten
Suffixe der Entscheidungen nach der gewählten Option (beobachtet, also verzerrt durch die Auswahl, kein kausaler Effekt).

Aufruf (im Ordner process-simulation-engine):
    PSP_CATALOG=../prescriptive/data/catalog_neue_punkte.yaml CUDA_VISIBLE_DEVICES="" .venv/bin/python -u \
        ../prescriptive/scripts/psp_zerlegung.py --psp "(c)=psp_model_durchlaufzeit_real_alle_s60_normal.pkl" \
        "(d)=psp_model_durchlaufzeit_real_alle_s60_klassen.pkl" --out ../prescriptive/scripts/results_100k/psp_zerlegung
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

DAY = 86400.0
ENDS = ("A_Pending", "A_Denied", "A_Cancelled")
MODELS = os.path.join(PRESCRIPTIVE, "models")
_ARGS = {}


def end_state(activities):
    for activity in activities:
        if activity in ENDS:
            return activity
    return "keins"


def _init(settings):
    _ARGS.update(settings)


def _run(job):
    import torch
    label, path, name, decisions = job
    torch.set_num_threads(_ARGS["threads"])
    torch.manual_seed(0)
    from prescriptive.estimators.psp_model import PSPModel
    model = PSPModel.load(path)
    model.set_threads(_ARGS["threads"])
    point = model.points[name]
    options = list(ORDER[name])
    gaps = model.params.get("time_from") == "gaps"
    rows = []
    for case_id, events, now, chosen, _ in decisions:
        drawn, elapsed = model.option_suffixes(point, events, options, now, _ARGS["samples"])
        for o, suffixes in enumerate(drawn):
            for s in suffixes:
                gap_days = np.array([g for _, _, g in s], dtype=float) / DAY
                rows.append({"Modell": label, "Wahlpunkt": name, "Fall": case_id, "Option": "Aktion" if o == 0 else "Referenz",
                             "Restzeit": -model.outcome(elapsed, s, "negative_cycle_days", gaps=gaps) - elapsed / DAY,
                             "Ende": end_state([e[0].split("|")[0] for e in s]), "Events": len(s),
                             "lange Wartezeiten": gap_days[gap_days >= 1].sum(), "kurze Abstände": gap_days[gap_days < 1].sum()})
    print(f"{label} {name}: fertig", flush=True)
    return pd.DataFrame(rows)


def decompose(t):
    """Zerlegung von T_r - T_a (Vorteil der Aktion) in Pfad und Zeit, dazu Anteile und Restzeit je Ende."""
    out = {}
    stats = {}
    for option, g in t.groupby("Option"):
        counts = g["Ende"].value_counts()
        stats[option] = (counts / len(g), g.groupby("Ende")["Restzeit"].mean(), g["Restzeit"].mean())
    (pa, ta, mean_a), (pr, tr, mean_r) = stats["Aktion"], stats["Referenz"]
    path = time_part = 0.0
    for e in ("A_Pending", "A_Denied", "A_Cancelled", "keins"):
        p_a, p_r = pa.get(e, 0.0), pr.get(e, 0.0)
        t_a, t_r = ta.get(e, np.nan), tr.get(e, np.nan)
        t_a, t_r = (t_r if np.isnan(t_a) else t_a), (t_a if np.isnan(t_r) else t_r)
        if np.isnan(t_a):
            continue
        path += (p_r - p_a) * (t_a + t_r) / 2
        time_part += (p_a + p_r) / 2 * (t_r - t_a)
        out[f"Anteil {e}: Aktion / Referenz"] = f"{p_a:.0%} / {p_r:.0%}"
        out[f"Restzeit bei {e}: Aktion / Referenz"] = f"{t_a:.1f} / {t_r:.1f}"
    out.update({"Vorteil Aktion gesamt (Tage)": mean_r - mean_a, "davon Pfad": path, "davon Zeit": time_part,
                "Restzeit Aktion / Referenz": f"{mean_a:.1f} / {mean_r:.1f}"})
    for column in ("Events", "lange Wartezeiten", "kurze Abstände"):
        if column in t:
            means = t.groupby("Option")[column].mean()
            out[f"{column}: Aktion / Referenz"] = f"{means['Aktion']:.1f} / {means['Referenz']:.1f}"
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--psp", nargs="+", required=True, metavar="BESCHRIFTUNG=MODELL")
    parser.add_argument("--out", required=True)
    parser.add_argument("--decisions", type=int, default=200, help="Entscheidungen je Punkt (halb Aktion, halb Referenz), wie psp_rauschen")
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
    test = log[log[CASE].astype(str).isin(set(map(str, model.test_cases)))].sort_values([CASE, TIMESTAMP], kind="stable")
    total = test.groupby(CASE)[TIMESTAMP].agg(lambda s: (s.max() - s.min()).total_seconds() / 86400)
    samples = model.samples(test)
    rng = np.random.default_rng(0)  # Auswahl wie psp_rauschen
    jobs, real = [], []
    for name in ORDER:
        if name not in model.points:
            continue
        seen, decisions = set(), []
        for case_id, (part, cut, now), option in samples.get(name, []):
            if case_id not in seen:
                seen.add(case_id)
                decisions.append((case_id, part.iloc[:cut].to_dict("records"), now, option, float(total[case_id]), part, cut))
        action = [d for d in decisions if d[3] == ORDER[name][0]]
        reference = [d for d in decisions if d[3] != ORDER[name][0]]
        half = args.decisions // 2
        pick = [action[i] for i in rng.choice(len(action), min(half, len(action)), replace=False)] + \
               [reference[i] for i in rng.choice(len(reference), min(half, len(reference)), replace=False)]
        for case_id, _, now, option, _, part, cut in pick:
            suffix = part.iloc[cut:]
            gap_days = suffix[TIMESTAMP].diff().dt.total_seconds().fillna(0).to_numpy() / DAY
            real.append({"Modell": "echt (beobachtet)", "Wahlpunkt": name, "Fall": case_id,
                         "Option": "Aktion" if option == ORDER[name][0] else "Referenz",
                         "Restzeit": (part[TIMESTAMP].max() - pd.Timestamp(now, unit="s")).total_seconds() / DAY,
                         "Ende": end_state(list(suffix["concept:name"]))})
        for label, path in models:
            jobs.append((label, path, name, [d[:5] for d in pick]))
    del log, test
    _ARGS.update(samples=args.samples, threads=args.threads)
    with ProcessPoolExecutor(min(args.workers, len(jobs)), mp_context=mp.get_context("spawn"), initializer=_init,
                             initargs=(dict(_ARGS),)) as pool:
        table = pd.concat([pd.DataFrame(real), *pool.map(_run, jobs)], ignore_index=True)
    table.to_csv(os.path.join(args.out, "ziehungen.csv.gz"), index=False)
    summary = {}
    for (label, name), t in table.groupby(["Modell", "Wahlpunkt"], sort=False):
        summary[(name, label)] = {"AIPW (Tage)": AIPW[name], **decompose(t)}
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(args.out, "zerlegung.csv"))
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 20):
        for name in ORDER:
            if name in summary.columns.get_level_values(0):
                print(f"\n==== {name}")
                print(summary[name].to_string())
    print(f"######## Fertig in {time.time() - started:.0f} s; Ergebnisse in {args.out}", flush=True)


if __name__ == "__main__":
    main()
