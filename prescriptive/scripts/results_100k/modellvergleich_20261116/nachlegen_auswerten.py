"""Auswertung von nachlegen_erklaerung.py: Abschnitte und Kennzahlen je Quelle und Option (Create Offer minus Sent)."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

O = Path(__file__).resolve().parent
ENDS = ("A_Pending", "A_Cancelled", "A_Denied")


def first(tokens, gaps, prefix):
    total = 0.0
    for token, gap in zip(tokens, gaps):
        total += gap
        if token.startswith(prefix):
            return total
    return np.nan


def metrics(row):
    t, g = row.tokens, row.gaps
    end = next((e for tok in t for e in ENDS if tok.startswith(e)), "ohne")
    call = first(t, g, "W_Call after offers|start")
    validate = first(t, g, "W_Validate application|start")
    total = float(sum(g))
    long = Counter()
    for token, gap in zip(t, g):
        if gap >= 1.0:
            long[token] += gap
    return {"Restzeit": total, "Events": len(t), "Angebote": sum(x.startswith("O_Create Offer") for x in t),
            "versendet": sum(x.startswith("O_Sent") for x in t), "O_Cancelled": sum(x.startswith("O_Cancelled") for x in t),
            "Nachfass-Schleifen": sum(x in ("W_Call after offers|suspend", "W_Call after offers|resume") for x in t),
            "Ende": end, "bis Nachfassen": call, "Nachfassen bis Validierung": validate-call, "Validierung bis Ende": total-validate,
            "erreicht Validierung": not np.isnan(validate), "lange": dict(long)}


def main():
    folder = O/(sys.argv[1] if len(sys.argv) > 1 else "nachlegen_erklaerung")
    frame = pd.read_pickle(folder/"verlaeufe.pkl")
    frame["opt"] = np.where(frame.option.str.startswith("O_Create Offer"), "extra", "senden")
    m = pd.concat([frame[["quelle", "opt", "case", "seed"]].reset_index(drop=True), pd.DataFrame([metrics(r) for r in frame.itertuples()])], axis=1)
    for e in ENDS+("ohne",):
        m[e] = m.Ende == e
    cols = ["Restzeit", "Events", "Angebote", "versendet", "O_Cancelled", "Nachfass-Schleifen", "bis Nachfassen",
            "Nachfassen bis Validierung", "Validierung bis Ende", "erreicht Validierung", "A_Pending", "A_Cancelled", "A_Denied", "ohne"]
    table = m.groupby(["quelle", "opt"])[cols].mean()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("Mittel je Verlauf (ab Entscheidung, Tage):\n" + table.round(2).T.to_string())
    diff = table.xs("extra", level=1)-table.xs("senden", level=1)
    sim = m[m.quelle.str.startswith("Simulator")]
    a, b = sim[sim.opt == "extra"].Restzeit, sim[sim.opt == "senden"].Restzeit
    print(f"\nSimulator (Bank danach): Fälle extra {len(a)}, senden {len(b)}, SE der Restzeitdifferenz {np.sqrt(a.var()/len(a)+b.var()/len(b)):.2f}")
    print("\nZusätzliches Angebot minus senden:\n" + diff.round(2).T.to_string())
    print("\nTage in langen Abständen (>= 1 d) nach dem folgenden Event, je Verlauf, extra minus senden (größte Unterschiede):")
    for source, part in m.groupby("quelle", sort=False):
        sums = {}
        for opt, group in part.groupby("opt"):
            c = Counter()
            for entry in group.lange:
                c.update(entry)
            sums[opt] = {k: v/len(group) for k, v in c.items()}
        keys = set(sums["extra"]) | set(sums["senden"])
        delta = sorted(((sums["extra"].get(k, 0)-sums["senden"].get(k, 0), k) for k in keys), key=lambda x: -abs(x[0]))[:5]
        print(f"  {source:<24}", "; ".join(f"{k} {v:+.2f}" for v, k in delta))
    table.to_csv(folder/"kennzahlen.csv")


if __name__ == "__main__":
    main()
