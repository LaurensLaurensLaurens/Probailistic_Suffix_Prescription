"""Die Ziehungen hinter "D, Zeit, ohne Rückfall" ansehen (2026-11-16).

Für eine Stichprobe echter Zweierwahlen aus den drei Läufen
policies/ev_policy_event_costs/results/d_oe_zustand_20261115/d_time_uncertainty_ablation/without_guard_seed{0,1,2}
wird der Prefix des Falls bis zum Entscheidungs-Event aus den gespeicherten Events wiederhergestellt und mit demselben
Modell (models/D_stateful.pkl, unverändert geladen) je Option 100 Suffixe gezogen, mit gemeinsamen Zufallszahlen wie
online. Gespeichert wird je Ziehung, woraus sich die Restzeit zusammensetzt (erster Abstand der Option, Rest, Endzustand,
Zahl der Events und Angebote, längster Abstand), dazu der tatsächliche weitere Verlauf des Falls im Simulatorlauf.

Aufruf im Ordner process-simulation-engine:  .venv/bin/python -u <dieses Skript> --workers 5
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/"prescriptive/policies/ev_policy_event_costs/results/d_oe_zustand_20261115"
RUN = "d_time_uncertainty_ablation/without_guard_seed{}"
MODEL = SOURCE/"models/D_stateful.pkl"
OUT = Path(__file__).resolve().parent/"d_ohne_ziehungen"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PSP_CATALOG"] = str(SOURCE/"catalog.yaml")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths
add_paths()
import numpy as np
import pandas as pd

DAY = 86400.0
CASE, ACT, LIFE, TIME = "case:concept:name", "concept:name", "lifecycle:transition", "time:timestamp"
RULE = {"annehmen_bei_konzept": "A_Accepted", "annehmen_nach_vervollstaendigung": "A_Accepted",
        "angebot_nachlegen": "O_Sent (mail and online)", "handle_leads": "W_Handle leads|withdraw"}
ENDS = ("A_Pending", "A_Denied", "A_Cancelled")
PER_POINT, DRAWS = 60, 100


def end_state(names):
    for name in names:
        for end in ENDS:
            if name.startswith(end):
                return end
    return "ohne"


def pick():
    """Je Wahlpunkt PER_POINT Zweierwahlen (angebot_nachlegen: je zur Hälfte erster und späterer Besuch), über die Seeds gemischt."""
    rng = np.random.default_rng(0)
    rows = []
    for seed in (0, 1, 2):
        seen = {}
        for index, r in enumerate(json.load(open(SOURCE/RUN.format(seed)/"decision_estimates.json"))):
            key = (r["case"], r["point"])
            seen[key] = seen.get(key, 0)+1
            if len(r["options"]) == 2 and r["estimates"]:
                rows.append({"seed": seed, "index": index, "case": r["case"], "point": r["point"], "time": r["time"], "options": r["options"],
                             "selected": r["selected"], "visit": seen[key], "online_values": r["estimates"][-1]["mean_values"],
                             "online_n": r["estimates"][-1]["samples_per_option"]})
    table = pd.DataFrame(rows)
    chosen = []
    for point, part in table.groupby("point", sort=False):
        groups = [part[part.visit == 1], part[part.visit > 1]] if point == "angebot_nachlegen" else [part]
        for group in groups:
            chosen.append(group.iloc[rng.choice(len(group), min(PER_POINT, len(group)), replace=False)])
    return pd.concat(chosen).reset_index(drop=True)


def work(task):
    number, states = task
    import torch
    from prescriptive.estimators.psp_model import PSPModel
    from prescriptive.estimators.conditional_gaps import SuffixSamplingError
    torch.set_num_threads(1)
    torch.manual_seed(1000+number)
    model = PSPModel.load(MODEL)
    model.set_threads(1)
    events = {seed: pd.read_pickle(SOURCE/RUN.format(seed)/"events.pkl") for seed in sorted({s["seed"] for s in states})}
    draws, summary, examples = [], [], []
    for s in states:
        part = events[s["seed"]]
        part = part.loc[part[CASE].astype(str).eq(s["case"])].sort_values(TIME, kind="stable").reset_index(drop=True)
        stamps = part[TIME].map(lambda t: pd.Timestamp(t).timestamp()).to_numpy()
        cut = int(np.searchsorted(stamps, s["time"]+1e-6, side="right"))
        prefix, future = part.iloc[:cut], part.iloc[cut:]
        point = model.points[s["point"]]
        last = prefix.iloc[-1]
        ok = str(last[ACT]) == point.activity and str(last[LIFE]) == point.lifecycle and abs(stamps[cut-1]-s["time"]) < 1e-3
        real_tokens = (future[ACT].astype(str)+"|"+future[LIFE].astype(str)).tolist()
        info = {**{k: s[k] for k in ("seed", "case", "point", "visit", "selected", "online_n")}, "prefix_events": cut, "prefix_ok": bool(ok),
                "prefix_days": (stamps[cut-1]-stamps[0])/DAY, "real_rest_days": (stamps[-1]-stamps[cut-1])/DAY, "real_next": real_tokens[0] if real_tokens else None,
                "real_end": end_state(future[ACT].astype(str)), "real_events": len(future), "real_offers": int((future[ACT].astype(str) == "O_Create Offer").sum()),
                "online_value_0": s["online_values"][0], "online_value_1": s["online_values"][1], "option_0": s["options"][0], "option_1": s["options"][1]}
        if not ok:
            summary.append(info)
            continue
        try:
            drawn, elapsed = model.option_suffixes(point, prefix.to_dict("records"), list(s["options"]), s["time"], DRAWS)
        except SuffixSamplingError as error:
            summary.append({**info, "sampling_error": str(error)[:120]})
            continue
        info["elapsed_days_model"] = elapsed/DAY
        summary.append(info)
        for option, suffixes in zip(s["options"], drawn):
            for k, suffix in enumerate(suffixes):
                names = [name for name, _, _ in suffix]
                gaps = np.array([gap for _, _, gap in suffix], dtype=float)/DAY
                end = end_state(names)
                reached = next((float(gaps[:i+1].sum()) for i, name in enumerate(names) if name.startswith(ENDS)), np.nan)
                draws.append({"seed": s["seed"], "case": s["case"], "point": s["point"], "visit": s["visit"], "time": s["time"], "option": option,
                              "rule_option": option == RULE[s["point"]], "k": k, "first_gap": float(gaps[0]) if len(gaps) else 0.0,
                              "rest": float(gaps[1:].sum()), "total": float(gaps.sum()), "events": len(names), "end": end,
                              "days_to_end_marker": reached, "offers": sum(name.startswith("O_Create Offer") for name in names[1:]),
                              "max_gap": float(gaps[1:].max()) if len(gaps) > 1 else 0.0, "gaps_over_5d": int((gaps[1:] > 5).sum()),
                              "sum_gaps_over_5d": float(gaps[1:][gaps[1:] > 5].sum()) if len(gaps) > 1 else 0.0,
                              "second": names[1] if len(names) > 1 else "EOS", "last": names[-1] if names else "EOS"})
                if k < 3 and len(examples) < 400:
                    examples.append({"case": s["case"], "point": s["point"], "option": option, "k": k,
                                     "suffix": [(name, round(float(gap)/3600, 2)) for name, _, gap in suffix]})
    return pd.DataFrame(draws), pd.DataFrame(summary), examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.time()
    states = pick().to_dict("records")
    order = np.random.default_rng(1).permutation(len(states))
    tasks = [(w, [states[i] for i in order[w::args.workers]]) for w in range(args.workers)]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(work, tasks))
    pd.concat([r[0] for r in results], ignore_index=True).to_pickle(OUT/"ziehungen.pkl")
    pd.concat([r[1] for r in results], ignore_index=True).to_csv(OUT/"zustaende.csv", index=False)
    json.dump(sum((r[2] for r in results), []), open(OUT/"beispiele.json", "w"), ensure_ascii=False, indent=1)
    print(f"FERTIG in {time.time()-started:.0f} s: {len(states)} Zustände", flush=True)


if __name__ == "__main__":
    main()
