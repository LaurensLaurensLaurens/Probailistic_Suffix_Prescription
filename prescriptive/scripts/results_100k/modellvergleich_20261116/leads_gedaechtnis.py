"""Erinnert sich OE später im Fall an die Lead-Bearbeitung? (2026-11-16)

Echte Testfälle (für alle Modelle dieselben 6.301), Prefix bis zum ersten O_Sent (mail and online). Zwei Gruppen:
Fälle mit W_Handle leads|start davor und Fälle mit W_Handle leads|withdraw ohne start. Je Prefix 50 gezogene Suffixe
(PSPModel.draw_cases wie in der Suffixgüte), Restzeit als Summe der Abstände. Verglichen wird die Differenz der Gruppen
(Lead minus ohne Lead) im echten Log und in den Modellen; beobachtet, kein kausaler Effekt, aber für alle Modelle
dieselbe Zielgröße. Aufruf im Ordner process-simulation-engine.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d_ohne_ziehungen as base
from leads_oe import MODELS, TAG
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent/("leads_gedaechtnis"+TAG)
PER_GROUP, DRAWS = 250, 50
SENT = "O_Sent (mail and online)"


def prefixes():
    from prescriptive.estimators.psp_model import PSPModel
    test = set(map(str, PSPModel.load(next(iter(MODELS.values()))).test_cases))
    raw = pd.read_pickle(base.ROOT/"prescriptive/data/cache/bpic17_raw.pkl")
    raw = raw[raw[base.CASE].astype(str).isin(test)].copy()
    raw[base.ACT] = raw[base.ACT].astype(str).str.strip()
    raw = raw.sort_values([base.CASE, base.TIME], kind="stable")
    rows = []
    for case, part in raw.groupby(base.CASE, sort=False):
        part = part.reset_index(drop=True)
        tok = (part[base.ACT]+"|"+part[base.LIFE].astype(str)).tolist()
        if SENT+"|complete" not in tok:
            continue
        cut = tok.index(SENT+"|complete")+1
        before = tok[:cut]
        if "W_Handle leads|start" in before:
            group = "Lead bearbeitet"
        elif "W_Handle leads|withdraw" in before:
            group = "Lead zurückgezogen"
        else:
            continue
        stamps = part[base.TIME]
        rows.append({"case": str(case), "gruppe": group, "cut": cut, "events_vorher": cut,
                     "real_rest": (stamps.iloc[-1]-stamps.iloc[cut-1]).total_seconds()/base.DAY,
                     "alter": (stamps.iloc[cut-1]-stamps.iloc[0]).total_seconds()/base.DAY, "prefix": part.iloc[:cut].to_dict("records")})
    frame = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    return pd.concat([g.iloc[rng.choice(len(g), min(PER_GROUP, len(g)), replace=False)] for _, g in frame.groupby("gruppe")]), frame


def work(task):
    number, items = task
    import torch
    from prescriptive.estimators.psp_model import PSPModel
    torch.set_num_threads(1)
    out = []
    for label, path in MODELS.items():
        model = PSPModel.load(path)
        model.set_threads(1)
        for index, item in enumerate(items):
            torch.manual_seed(11000+97*number+index)
            suffixes = model.draw_cases([model._encode(item["prefix"])[:4]], DRAWS)[0]
            rest = [sum(gap for _, _, gap in s)/base.DAY for s in suffixes]
            ends = [base.end_state([name for name, _, _ in s]) for s in suffixes]
            out.append({"modell": label, "case": item["case"], "gruppe": item["gruppe"], "rest": float(np.mean(rest)),
                        "pending": np.mean([e == "A_Pending" for e in ends]), "cancelled": np.mean([e == "A_Cancelled" for e in ends]),
                        "events": float(np.mean([len(s) for s in suffixes]))})
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    picked, everything = prefixes()
    items = picked.to_dict("records")
    with ProcessPoolExecutor(6, mp_context=multiprocessing.get_context("spawn")) as pool:
        drawn = pd.concat(list(pool.map(work, [(w, items[w::6]) for w in range(6)])), ignore_index=True)
    drawn.to_csv(OUT/"ziehungen.csv", index=False)
    picked.drop(columns="prefix").to_csv(OUT/"prefixe.csv", index=False)
    pd.set_option("display.width", 250)
    print("Alle Testfälle mit O_Sent:", everything.groupby("gruppe").agg(n=("case", "size"), rest=("real_rest", "mean"), alter=("alter", "mean")).round(2).to_string(), sep="\n")
    real = picked.groupby("gruppe").agg(n=("case", "size"), rest=("real_rest", "mean"), alter=("alter", "mean"), events=("events_vorher", "mean"))
    print("\nAusgewählte Prefixe, echter Log:\n" + real.round(2).to_string())
    table = drawn.groupby(["modell", "gruppe"])[["rest", "pending", "cancelled", "events"]].mean().unstack("gruppe")
    print("\nGezogen (Mittel je Gruppe):\n" + table.round(2).to_string())
    rows = {"echter Log": picked.groupby("gruppe").real_rest.mean()}
    for label, part in drawn.groupby("modell"):
        rows[label] = part.groupby("gruppe").rest.mean()
    diff = pd.DataFrame(rows).T
    diff["Lead minus ohne"] = diff["Lead bearbeitet"]-diff["Lead zurückgezogen"]
    real_diff = picked[picked.gruppe == "Lead bearbeitet"].real_rest
    other = picked[picked.gruppe == "Lead zurückgezogen"].real_rest
    print("\nRestzeit ab erstem O_Sent (Tage):\n" + diff.round(2).to_string())
    print(f"SE der echten Differenz: {np.sqrt(real_diff.var()/len(real_diff)+other.var()/len(other)):.2f}")
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
