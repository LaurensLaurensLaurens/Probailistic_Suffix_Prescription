"""Wirkt ein zusätzliches Angebot im Verlauf auf die vorhergesagte Nachfassphase? (2026-11-16)

Dieselben 250 echten Test-Prefixe mit genau einem Angebot wie angebote_gedaechtnis.py (bis zum ersten
W_Call after offers|start). Variante B fügt direkt vor diesem letzten Event eine Kopie der drei Angebots-Events des Falls
ein (O_Create Offer, O_Created, O_Sent mit denselben Angebotswerten, Zeitstempel kurz davor). Sonst ist alles gleich.
Je Prefix und Variante 50 Ziehungen mit denselben Zufallszahlen. Differenz B minus A = vom Modell vorhergesagte
Wirkung des zweiten Angebots auf die Restzeit ab hier. Aufruf im Ordner process-simulation-engine.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d_ohne_ziehungen as base
from leads_oe import MODELS, TAG
from angebote_gedaechtnis import prefixes
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent/("angebote_einfuegen"+TAG)
DRAWS = 50
OFFER = ["O_Create Offer", "O_Created", "O_Sent (mail and online)"]


def with_extra_offer(prefix):
    records = [dict(e) for e in prefix]
    copies = []
    for name in OFFER:
        source = next((e for e in records if str(e[base.ACT]).strip() == name), None)
        if source is None:
            return None
        copies.append(dict(source))
    before, last = pd.Timestamp(records[-2][base.TIME]), pd.Timestamp(records[-1][base.TIME])
    span = max((last-before).total_seconds(), 0.004)
    if (last-before).total_seconds() < 0.004:
        records[-1][base.TIME] = before+pd.Timedelta(seconds=span)
    for k, copy in enumerate(copies, 1):
        copy[base.TIME] = before+pd.Timedelta(seconds=span*k/4)
    return records[:-1]+copies+[records[-1]]


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
            for variant, prefix in (("A", item["prefix"]), ("B", item["extra"])):
                torch.manual_seed(23000+97*number+index)
                suffixes = model.draw_cases([model._encode(prefix)[:4]], DRAWS)[0]
                tokens = [[n for n, _, _ in s] for s in suffixes]
                out.append({"modell": label, "case": item["case"], "variante": variant,
                            "rest": float(np.mean([sum(g for _, _, g in s)/base.DAY for s in suffixes])),
                            "events": float(np.mean([len(t) for t in tokens])),
                            "angebote": float(np.mean([sum(x.startswith("O_Create Offer") for x in t) for t in tokens])),
                            "storniert": float(np.mean([sum(x.startswith("O_Cancelled") for x in t) for t in tokens])),
                            "pending": float(np.mean([any(x.startswith("A_Pending") for x in t) for t in tokens]))})
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    picked, _ = prefixes()
    picked = picked[picked.gruppe == "1 Angebot"].copy()
    picked["extra"] = picked.prefix.map(with_extra_offer)
    picked = picked[picked.extra.notna()]
    items = picked.to_dict("records")
    with ProcessPoolExecutor(6, mp_context=multiprocessing.get_context("spawn")) as pool:
        drawn = pd.concat(list(pool.map(work, [(w, items[w::6]) for w in range(6)])), ignore_index=True)
    drawn.to_csv(OUT/"ziehungen.csv", index=False)
    wide = drawn.pivot_table(index=["modell", "case"], columns="variante", values=["rest", "events", "angebote", "storniert", "pending"])
    diff = (wide.xs("B", axis=1, level=1)-wide.xs("A", axis=1, level=1)).groupby(level=0)
    table = diff.mean()
    table["SE rest"] = diff.rest.std()/np.sqrt(diff.rest.count())
    table["Prefixe"] = diff.rest.count()
    pd.set_option("display.width", 250)
    print("Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):\n" + table.round(3).to_string())
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
