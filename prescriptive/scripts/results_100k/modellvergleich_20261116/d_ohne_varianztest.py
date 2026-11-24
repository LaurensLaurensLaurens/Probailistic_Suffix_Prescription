"""Offline-Test zweier Rauschminderungen für "D, Zeit" (2026-11-16), ohne Projektcode oder Modelle zu ändern.

Dieselben Entscheidungszustände wie d_ohne_ziehungen.py (je Gruppe 40), je Option 100 Suffixe, zwei Ziehungsarten:
- "bisher":    models/D_stateful.pkl wie online. ConditionalGaps.draw zieht torch.rand(Anzahl noch laufender Suffixe);
               sobald unter einer Option ein Suffix früher endet, verschiebt sich der Zufallszahlenstrom der anderen.
- "gekoppelt": nur im Speicher dieses Prozesses ersetzt: je Suffix-Zeile und Schritt eine feste Zufallszahl aus einem
               eigenen Generator (derselbe Startwert für beide Optionen), die Trainingsabstände je Pool sortiert, die
               Zufallszahl also als Quantil. Gewichte, Pools, Zustandsregeln und Verteilungen bleiben dieselben.
Und zwei Bewertungen derselben gezogenen Suffixe:
- "gezogen":   Restzeit = Summe der gezogenen Abstände (wie online)
- "erwartet":  Restzeit = Summe der Mittelwerte der Abstandspools, aus denen je Schritt gezogen wurde (bedingter
               Erwartungswert je Schritt; gleicher Erwartungswert der Summe, weniger Streuung)

Aufruf im Ordner process-simulation-engine:  .venv/bin/python -u <dieses Skript> --workers 6
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d_ohne_ziehungen as base  # setzt Umgebung und Pfade
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent/"d_ohne_varianztest"
PER_GROUP, DRAWS = 40, 100


class CoupledGaps:
    """Wie ConditionalGaps (dieselben Pools und Rückfallregel), aber mit festen Zufallszahlen je Zeile und Schritt und sortierten Pools."""

    def __init__(self, source):
        self.source, self.min_observations = source, source.min_observations
        self.pools = {key: np.sort(values) for key, values in source.pools.items()}
        self.rows, self.kept, self.seed, self.step = {}, [], 0, 0

    def distribution(self, state, token):
        own, active = state.context(token)
        specific = self.pools.get((state.previous, token, own, active))
        if specific is not None and len(specific) >= self.min_observations:
            return specific
        return self.pools.get((None, token, own, active))

    def draw(self, states, labels):
        import torch
        ids = [id(state) for state in states]
        if any(i not in self.rows for i in ids):  # neuer Aufruf von draw_cases: Zeilen neu nummerieren
            self.kept = list(states)  # hält die Objekte am Leben, ihre ids bleiben eindeutig
            self.rows = {i: k for k, i in enumerate(ids)}
            saved = torch.get_rng_state()
            self.seed = int(torch.randint(2**31-1, (1,)))  # hängt nur vom bisherigen (für beide Optionen gleichen) Strom ab
            torch.set_rng_state(saved)
            self.step = 0
        uniforms = np.random.default_rng([self.seed, self.step]).random(len(self.rows))
        self.step += 1
        gaps = np.zeros(len(labels), dtype=float)
        for i, (state, token) in enumerate(zip(states, labels)):
            values = self.distribution(state, token)
            if values is None or not len(values):
                raise ValueError(f"No training gap support for {token}")
            u = uniforms[self.rows[ids[i]]]
            gaps[i] = values[min(int(u*len(values)), len(values)-1)]
        return gaps


def pick():
    table = base.pick()
    table["gruppe"] = np.where(table.point == "angebot_nachlegen", np.where(table.visit == 1, "nachlegen 1. Besuch", "nachlegen ab 2. Besuch"), table.point)
    return pd.concat([part.head(PER_GROUP) for _, part in table.groupby("gruppe", sort=False)]).reset_index(drop=True)


def work(task):
    number, states = task
    import torch
    from prescriptive.estimators.psp_model import PSPModel
    from prescriptive.estimators.conditional_gaps import SuffixSamplingError
    torch.set_num_threads(1)
    model = PSPModel.load(base.MODEL)
    model.set_threads(1)
    original = model.conditional_gaps
    coupled = CoupledGaps(original)
    means = {key: float(values.mean()) for key, values in original.pools.items()}

    def expected(state, token):
        own, active = state.context(token)
        specific = original.pools.get((state.previous, token, own, active))
        key = (state.previous, token, own, active) if specific is not None and len(specific) >= original.min_observations else (None, token, own, active)
        return means.get(key)

    events = {seed: pd.read_pickle(base.SOURCE/base.RUN.format(seed)/"events.pkl") for seed in sorted({s["seed"] for s in states})}
    rows = []
    for index, s in enumerate(states):
        part = events[s["seed"]]
        part = part.loc[part[base.CASE].astype(str).eq(s["case"])].sort_values(base.TIME, kind="stable").reset_index(drop=True)
        stamps = part[base.TIME].map(lambda t: pd.Timestamp(t).timestamp()).to_numpy()
        prefix = part.iloc[:int(np.searchsorted(stamps, s["time"]+1e-6, side="right"))].to_dict("records")
        point = model.points[s["point"]]
        start = model._work_states([prefix])[0]
        for variant, gaps in (("bisher", original), ("gekoppelt", coupled)):
            model.conditional_gaps = gaps
            torch.manual_seed(5000+97*number+index)
            try:
                drawn, _ = model.option_suffixes(point, prefix, list(s["options"]), s["time"], DRAWS)
            except SuffixSamplingError:
                continue
            for option, suffixes in zip(s["options"], drawn):
                for k, suffix in enumerate(suffixes):
                    state, total = start.copy(), 0.0
                    for name, _, gap in suffix:
                        mean = expected(state, name)
                        total += gap if mean is None else mean
                        state.push(name)
                    rows.append({"seed": s["seed"], "case": s["case"], "point": s["point"], "visit": s["visit"], "gruppe": s["gruppe"],
                                 "variante": variant, "rule_option": option == base.RULE[s["point"]], "k": k,
                                 "gezogen": sum(gap for _, _, gap in suffix)/base.DAY, "erwartet": total/base.DAY, "events": len(suffix),
                                 "end": base.end_state([name for name, _, _ in suffix])})
    model.conditional_gaps = original
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.time()
    states = pick().to_dict("records")
    order = np.random.default_rng(1).permutation(len(states))
    tasks = [(w, [states[i] for i in order[w::args.workers]]) for w in range(args.workers)]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        table = pd.concat(list(pool.map(work, tasks)), ignore_index=True)
    table.to_pickle(OUT/"ziehungen.pkl")
    key = ["seed", "case", "point", "visit"]
    result = []
    for (variant, group), part in table.groupby(["variante", "gruppe"], sort=False):
        for value in ("gezogen", "erwartet"):
            wide = part.pivot_table(index=key+["k"], columns="rule_option", values=value)
            per = wide.groupby(level=[0, 1, 2, 3])
            diff = per.apply(lambda t: (t[False]-t[True]).mean())
            sd = per.apply(lambda t: (t[False]-t[True]).std(ddof=1))
            result.append({"Ziehung": variant, "Bewertung": value, "Gruppe": group, "Zustände": len(diff),
                           "Gewinn Regel-Option (Tage)": diff.mean(), "SE des Mittels": diff.std(ddof=1)/np.sqrt(len(diff)),
                           "Restzeit Mittel": part[value].mean(), "SD Restzeit je Ziehung": part[value].std(),
                           "Korrelation der Optionen": per.apply(lambda t: np.corrcoef(t[False], t[True])[0, 1]).median(),
                           "MC-SE bei 100 Ziehungen": sd.median()/np.sqrt(DRAWS), "Vorzeichen wie Regel": (diff > 0).mean(),
                           "Anteil |z| > 2": ((diff/(sd/np.sqrt(DRAWS))).abs() > 2).mean()})
    summary = pd.DataFrame(result)
    summary.to_csv(OUT/"zusammenfassung.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(summary.round(3).to_string(index=False))
    print(f"FERTIG in {time.time()-started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
