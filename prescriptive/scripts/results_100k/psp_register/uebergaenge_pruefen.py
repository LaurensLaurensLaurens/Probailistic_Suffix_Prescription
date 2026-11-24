"""Wie oft verletzen gezogene Suffixe (Modelle und Simulator) die im echten Log beobachteten Übergänge? (2026-11-16)

Zwei Übergangsmodelle aus dem echten BPIC17-Log (80 % der Fälle, zufällig nach Fall; die übrigen 20 % prüfen, wie oft ein
echter, aber ungesehener Übergang vorkommt = Fehlalarm-Rate):
- zwischen Events: Token (Aktivität|Lifecycle) → nächstes Token im Fall (wie die Maske allowed[a, b] des Simulators)
- innerhalb einer Aktivität: letzter Lifecycle derselben Aktivität → nächster Lifecycle (z. B. suspend → resume), nur
  innerhalb des Suffixes geprüft (das erste Vorkommen einer Aktivität im Suffix hat unbekannten Vorzustand und wird übersprungen)
Dazu die Regel der Optionsprüfung (interventions/work_state.py: kein start bei aktiver/pausierter Arbeit, resume nur nach suspend).
Geprüft werden die gespeicherten Verläufe aus nachlegen_erklaerung*.py (1. Besuch Nachlegen, je Quelle und Option).
Aufruf im Ordner process-simulation-engine: .venv/bin/python <dieses Skript>
"""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
R = OUT.parent
P = R.parents[1]
sys.path.insert(0, str(P.parent/"process-simulation-engine"))
from interventions.work_state import WorkState  # noqa: E402

def pruefen(verbose=True):
    global between, within
    raw = pd.read_pickle(P/"data/cache/bpic17_raw.pkl").sort_values(["case:concept:name", "time:timestamp"], kind="stable")
    raw["concept:name"] = raw["concept:name"].astype(str).str.strip()
    raw["token"] = raw["concept:name"]+"|"+raw["lifecycle:transition"].astype(str)
    cases = raw["case:concept:name"].unique()
    rng = np.random.default_rng(0)
    learn = set(rng.choice(cases, int(0.8*len(cases)), replace=False))
    between, within = set(), set()
    holdout = []
    for case, tokens in raw.groupby("case:concept:name", sort=False)["token"]:
        tokens = list(tokens)
        if case in learn:
            between.update(zip(tokens[:-1], tokens[1:]))
            last = {}
            for tok in tokens:
                a, _, l = tok.rpartition("|")
                within.add((a, last.get(a, "keins"), l))
                last[a] = l
        else:
            holdout.append(tokens)


    def check(tokens):
        """Verstöße in einem Suffix: zwischen Events, innerhalb einer Aktivität, Regel der Optionsprüfung."""
        b = sum((x, y) not in between for x, y in zip(tokens[:-1], tokens[1:]))
        w, rule, last, state = 0, 0, {}, WorkState()
        for tok in tokens:
            a, _, l = tok.rpartition("|")
            if a in last:  # nur Paare innerhalb des Suffixes
                w += (a, last[a], l) not in within
                rule += not state.permits(tok)
            last[a] = l
            state.push(tok)
        return b, w, rule


    def summary(label, suffixes):
        rows = [check(list(t)) for t in suffixes]
        n_events = sum(len(t) for t in suffixes)
        arr = np.array(rows)
        return {"Quelle": label, "Suffixe": len(suffixes), "Events je Suffix": n_events/len(suffixes),
                "mit Verstoß zwischen Events": (arr[:, 0] > 0).mean(), "mit Verstoß innerhalb Aktivität": (arr[:, 1] > 0).mean(),
                "mit Verstoß Optionsprüfung": (arr[:, 2] > 0).mean(), "Verstöße zwischen je 100 Events": 100*arr[:, 0].sum()/n_events,
                "Verstöße innerhalb je 100 Events": 100*arr[:, 1].sum()/n_events}


    results, top = [], {}
    # echte Suffixe der 20 % Prüffälle ab dem ersten O_Created (wie der Wahlpunkt Nachlegen)
    real = [t[t.index("O_Created|complete")+1:] for t in holdout if "O_Created|complete" in t]
    results.append(summary("echter Log (20 % Prüffälle, Fehlalarm-Rate)", real))
    folders = sorted((R/"modellvergleich_20261116").glob("nachlegen_erklaerung*/verlaeufe.pkl"))
    for path in folders:
        frame = pd.read_pickle(path)
        for source, part in frame.groupby("quelle"):
            label = source if source.startswith("Simulator") else source
            if any(r["Quelle"] == label for r in results):
                continue
            results.append(summary(label, list(part.tokens)))
            counter = Counter()
            for t in part.tokens:
                t = list(t)
                last = {}
                for tok in t:
                    a, _, l = tok.rpartition("|")
                    if a in last and (a, last[a], l) not in within:
                        counter[f"{a}: {last[a]} → {l}"] += 1
                    last[a] = l
            top[label] = counter.most_common(4)
    table = pd.DataFrame(results)
    pd.set_option("display.width", 250)
    verbose and print(table.round(3).to_string(index=False))
    verbose and print("\nHäufigste Verstöße innerhalb einer Aktivität (Anzahl):")
    for label, items in top.items():
        verbose and print(f"  {label:<26}", "; ".join(f"{k} ({v})" for k, v in items))
    table.to_csv(OUT/"uebergaenge.csv", index=False)
    verbose and print(f"\nGelernt: {len(between)} Übergänge zwischen Events, {len(within)} Lifecycle-Übergänge innerhalb von Aktivitäten")
    return table


if __name__ == "__main__":
    pruefen()
