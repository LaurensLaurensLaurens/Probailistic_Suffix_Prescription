"""Wie viel der gezogenen Restzeit liegt auf ungültigen Schritten? (2026-11-17, Frage: warum macht „Zeit je Event“ die Zeiten instabil)

Für die gespeicherten Nachlegen-Verläufe je Quelle: Restzeit gesamt, davon auf Schritten, deren Event ein im echten Log nie
beobachteter Übergang ist (zwischen Events oder Lifecycle innerhalb der Aktivität, Masken wie uebergaenge_pruefen.py), und die
mittleren Abstände je Event-Typ gegen den echten Log (Prüffälle). Aufruf im Ordner process-simulation-engine.
"""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import uebergaenge_pruefen as u  # noqa: E402

u.pruefen(verbose=False)
between, within = u.between, u.within
R = OUT.parent
rows, per_token = [], defaultdict(dict)


def split(tokens, gaps):
    valid_time, invalid_time, last, previous = 0.0, 0.0, {}, None
    for tok, gap in zip(tokens, gaps):
        a, _, l = tok.rpartition("|")
        bad = (previous is not None and (previous, tok) not in between) or (a in last and (a, last[a], l) not in within)
        if bad:
            invalid_time += gap
        else:
            valid_time += gap
        last[a], previous = l, tok
    return valid_time, invalid_time


for path in sorted((R/"modellvergleich_20261116").glob("nachlegen_erklaerung*/verlaeufe.pkl")):
    frame = pd.read_pickle(path)
    for source, part in frame.groupby("quelle"):
        if any(r["Quelle"] == source for r in rows):
            continue
        parts = np.array([split(list(t), list(g)) for t, g in zip(part.tokens, part.gaps)])
        total = parts.sum(axis=1)
        rows.append({"Quelle": source, "Restzeit (Tage)": total.mean(), "davon auf ungültigen Schritten (Tage)": parts[:, 1].mean(),
                     "Anteil ungültig an der Zeit": parts[:, 1].sum()/total.sum(), "Restzeit nur gültige Schritte": parts[:, 0].mean(),
                     "SD Restzeit": total.std()})
        sums, counts = defaultdict(float), defaultdict(int)
        for t, g in zip(part.tokens, part.gaps):
            for tok, gap in zip(t, g):
                sums[tok] += gap
                counts[tok] += 1
        for tok in sums:
            per_token[source][tok] = (sums[tok]/counts[tok], counts[tok]/len(part))
table = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(table.round(2).to_string(index=False))
table.to_csv(OUT/"zeit_ungueltig.csv", index=False)
# Abstände je Event (Tage) und Häufigkeit je Verlauf für die Events mit den größten Zeitbeiträgen
focus = ["(c)", "ZZ ep3", "ZZ ep7", "ZZ ep9", "Simulator (Bank danach)"]
contrib = defaultdict(float)
for source in focus:
    for tok, (mean_gap, freq) in per_token.get(source, {}).items():
        contrib[tok] = max(contrib[tok], mean_gap*freq)
top = sorted(contrib, key=lambda k: -contrib[k])[:10]
print("\nAbstand je Event (Tage) × Häufigkeit je Verlauf = Zeitbeitrag (Tage):")
for tok in top:
    print(f"  {tok:<42}", "  ".join(f"{s}: {per_token[s].get(tok, (0, 0))[0]:.2f}×{per_token[s].get(tok, (0, 0))[1]:.2f}" for s in focus if s in per_token))
