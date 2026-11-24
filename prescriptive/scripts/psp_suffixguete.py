"""
Güte der gezogenen Suffixe auf Testfällen des echten Logs (2026-11-11): Events UND Zeiten je PSP-Modell, dazu eine
Semi-Markov-Kette als einfache Vergleichsgröße. Anders als zwischenergebnisse.py suffix (Prefixe aus dem ganzen Log, bei
auf dem echten Log trainierten Modellen also meist Trainingsfälle) nur Prefixe aus den Testfällen der Modelle (alle Modelle
müssen dieselben haben).

- Prefixe: --prefixes zufällige (Testfall, Länge 1 bis Fallende - 1) des echten Logs nach dem Preprocessing der Pipeline
- je Prefix --draws gezogene Suffixe (PSPModel.draw_cases wie online: Dropout-Masken und Rauschen je Ziehung)
- Events wie zwischenergebnisse.scores: Treffer je Schritt (häufigstes gezogenes Token an Position k gleich dem echten),
  DLS auf den Token, Suffixlänge-MAE (|mittlere gezogene Länge - echte Länge|)
- Zeiten:
  Abstand Schritt 1 MAE (Stunden): |Mittel der gezogenen Abstände zum ersten Suffix-Event - echter Abstand|
  Restzeit = Zeit vom letzten Prefix-Event bis Fallende, gezogen wie die Bewertung der Policy (model.outcome, bei
  time_from "gaps" die Summe der gezogenen Abstände): MAE und mittlerer Fehler (gezogen - echt) des Mittels über die
  Ziehungen, CRPS der gezogenen Verteilung (Stichprobenform E|X - y| - E|X - X'| / 2) und Anteil der echten Restzeit im
  80-%-Intervall der Ziehungen (10. bis 90. Perzentil; gut kalibriert: 80 %)
- Semi-Markov-Kette 1. Ordnung auf denselben Token, Ende als Zustand, Abstand je Übergang zufällig aus den echten Abständen
  dieses Übergangs; gelernt aus allen Fällen des echten Logs außer den Testfällen (die Trainingsfälle der Modelle)

Aufruf (im Ordner process-simulation-engine; Katalog wie beim Training über PSP_CATALOG):
    PSP_CATALOG=../prescriptive/data/catalog_neue_punkte.yaml CUDA_VISIBLE_DEVICES="" .venv/bin/python -u \
        ../prescriptive/scripts/psp_suffixguete.py --psp "s40=psp_model_durchlaufzeit_real_np_s40_ohne_zukunft_abstaende.pkl" \
        "(c)=psp_model_durchlaufzeit_real_alle_s60_normal.pkl" --out ../prescriptive/scripts/results_100k/suffixguete
"""
import argparse
import datetime
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

DAY, HOUR = 86400.0, 3600.0
SEED = 0
BATCH = 10  # Prefixe je Aufruf von draw_cases
EVENT_STEPS = (1, 2, 3, 5, 10, 20)
END = "Ende"
MAX_MARKOV_STEPS = 500
MODELS = os.path.join(PRESCRIPTIVE, "models")


def event_scores(true, drawn):
    """Treffer je Schritt, DLS und Längenfehler eines Prefix (true, drawn: Token-Listen), wie zwischenergebnisse.scores."""
    from collections import Counter
    from prescriptive.estimators.kpis import dls
    out = {}
    for k in EVENT_STEPS:
        at = Counter(s[k - 1] if len(s) >= k else END for s in drawn).most_common(1)[0][0]
        out[f"Schritt {k}"] = float(at == true[k - 1]) if len(true) >= k else np.nan
    out["DLS"] = float(np.mean([dls(s, true) for s in drawn]))
    out["Suffixlänge-MAE"] = abs(np.mean([len(s) for s in drawn]) - len(true))
    return out


def time_scores(true_first_gap, true_remaining, first_gaps, remaining):
    """Zeitmaße eines Prefix: Sekunden (echt) und Arrays der Ziehungen (Abstand zum ersten Event, Restzeit) in Sekunden."""
    y, x = true_remaining / DAY, np.asarray(remaining, dtype=float) / DAY
    low, high = np.quantile(x, [0.1, 0.9])
    first = np.asarray(first_gaps, dtype=float)
    return {"Abstand Schritt 1 MAE (h)": abs(first.mean() - true_first_gap) / HOUR if len(first) and true_first_gap is not None else np.nan,
            "Restzeit MAE (Tage)": abs(x.mean() - y), "Restzeit Fehler (Tage)": x.mean() - y,
            "Restzeit CRPS (Tage)": np.abs(x - y).mean() - 0.5 * np.abs(x[:, None] - x[None, :]).mean(),
            "Abdeckung 80 %": float(low <= y <= high),
            "Restzeit gezogen (Tage)": x.mean(), "Restzeit echt (Tage)": y}


def truth(part, k, tokens):
    """Echtes Suffix ab Prefixlänge k: Token, Abstand zum ersten Suffix-Event und Restzeit in Sekunden."""
    times = part[TIMESTAMP]
    first = (times.iloc[k] - times.iloc[k - 1]).total_seconds() if len(part) > k else None
    return tokens[k:], first, (times.iloc[-1] - times.iloc[k - 1]).total_seconds()


class SemiMarkov:
    """Markov-Kette 1. Ordnung auf den Token mit Ende als Zustand, Abstände je Übergang aus den echten Abständen."""

    def __init__(self, cases, seed=SEED):
        table = {}
        for tokens, seconds in cases:
            for i, a in enumerate(tokens):
                b, gap = (tokens[i + 1], seconds[i + 1] - seconds[i]) if i + 1 < len(tokens) else (END, 0.0)
                table.setdefault(a, {}).setdefault(b, []).append(gap)
        self.next = {a: (list(d), np.array([len(g) for g in d.values()], dtype=float) / sum(len(g) for g in d.values()),
                         [np.array(g) for g in d.values()]) for a, d in table.items()}
        self.rng = np.random.default_rng(seed)

    def draw(self, last, n):
        out = []
        for _ in range(n):
            suffix, token = [], last
            while len(suffix) < MAX_MARKOV_STEPS and token in self.next:
                tokens, p, gaps = self.next[token]
                j = self.rng.choice(len(tokens), p=p)
                if tokens[j] == END:
                    break
                token = tokens[j]
                suffix.append((token, float(gaps[j][self.rng.integers(len(gaps[j]))])))
            out.append(suffix)
        return out


_ARGS = {}


def _init(settings):
    _ARGS.update(settings)


def _run(job):
    """Ein PSP-Modell auf allen Prefixen: je Prefix die Event- und Zeitmaße."""
    import torch
    label, path, prefixes = job
    torch.set_num_threads(_ARGS["threads"])
    from prescriptive.estimators.psp_model import PSPModel
    model = PSPModel.load(path)
    model.set_threads(_ARGS["threads"])
    gaps = model.params.get("time_from") == "gaps"
    torch.manual_seed(SEED)
    started, rows = time.time(), []
    for start in range(0, len(prefixes), BATCH):
        batch = prefixes[start:start + BATCH]
        encoded, info = [], []
        for case_id, records, true_tokens, true_first, true_remaining in batch:
            prefix, static, mask, length, elapsed = model._encode(records)
            encoded.append((prefix, static, mask, length))
            info.append((case_id, len(records), true_tokens, true_first, true_remaining, elapsed))
        drawn = model.draw_cases(encoded, _ARGS["draws"])
        for (case_id, k, true_tokens, true_first, true_remaining, elapsed), suffixes in zip(info, drawn):
            remaining = [-model.outcome(elapsed, s, "negative_cycle_days", gaps=gaps) * DAY - elapsed for s in suffixes]
            first = [s[0][2] for s in suffixes if s]
            rows.append({"Modell": label, "Fall": case_id, "Prefixlänge": k,
                         **event_scores(true_tokens, [[e[0] for e in s] for s in suffixes]),
                         **time_scores(true_first, true_remaining, first, remaining)})
    print(f"{label}: {time.time() - started:.0f} s", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--psp", nargs="+", required=True, metavar="BESCHRIFTUNG=MODELL", help="PSP-Modelle (Datei in prescriptive/models oder Pfad)")
    parser.add_argument("--out", required=True, help="Ordner für prefixe.csv, suffixguete.csv und die Bilder")
    parser.add_argument("--prefixes", type=int, default=500)
    parser.add_argument("--draws", type=int, default=100, help="gezogene Suffixe je Prefix (wie online)")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    add_paths()
    from helpers.datasets import load_dataset
    from preprocessing import preprocess
    from prescriptive.estimators.psp_model import PSPModel
    from zwischenergebnisse import number, render
    started = time.time()
    os.makedirs(args.out, exist_ok=True)
    models = [(label, path if os.path.isabs(path) else os.path.join(MODELS, path)) for label, path in (e.split("=", 1) for e in args.psp)]
    first = PSPModel.load(models[0][1])
    test = set(map(str, first.test_cases))
    for label, path in models[1:]:
        if set(map(str, PSPModel.load(path).test_cases)) != test:
            raise SystemExit(f"{label}: andere Testfälle als {models[0][0]}")
    raw, _, _ = load_dataset("bpic17")
    log = preprocess("bpic17", raw).sort_values([CASE, TIMESTAMP], kind="stable")
    del raw
    log = log.assign(_token=first.labels(log).to_numpy())
    is_test = log[CASE].astype(str).isin(test)
    train = [(list(p["_token"]), list((p[TIMESTAMP] - p[TIMESTAMP].iloc[0]).dt.total_seconds())) for _, p in log[~is_test].groupby(CASE, sort=False)]
    markov = SemiMarkov(train)
    parts = [p.reset_index(drop=True) for _, p in log[is_test].groupby(CASE, sort=True) if len(p) > 1]
    rng = np.random.default_rng(SEED)
    picks = [(int(c), int(rng.integers(1, len(parts[c])))) for c in rng.choice(len(parts), args.prefixes, replace=False)]
    print(f"{len(picks)} Prefixe aus {len(parts):,} Testfällen, Semi-Markov-Kette aus {len(train):,} Trainingsfällen, {args.draws} Ziehungen je Prefix", flush=True)
    prefixes = []
    for c, k in picks:
        part = parts[c]
        tokens = list(part["_token"])
        true_tokens, true_first, true_remaining = truth(part, k, tokens)
        prefixes.append((part[CASE].iloc[0], part.drop(columns="_token").iloc[:k].to_dict("records"), true_tokens, true_first, true_remaining))
    markov_rows = []
    for (c, k), (case_id, records, true_tokens, true_first, true_remaining) in zip(picks, prefixes):
        drawn = markov.draw(parts[c]["_token"].iloc[k - 1], args.draws)
        markov_rows.append({"Modell": "Semi-Markov-Kette", "Fall": case_id, "Prefixlänge": k,
                            **event_scores(true_tokens, [[t for t, _ in s] for s in drawn]),
                            **time_scores(true_first, true_remaining, [s[0][1] for s in drawn if s], [sum(g for _, g in s) for s in drawn])})
    del log, parts
    _ARGS.update(draws=args.draws, threads=args.threads)
    with ProcessPoolExecutor(min(args.workers, len(models)), mp_context=mp.get_context("spawn"), initializer=_init,
                             initargs=(dict(_ARGS),)) as pool:
        tables = list(pool.map(_run, [(label, path, prefixes) for label, path in models]))
    table = pd.concat([pd.DataFrame(markov_rows), *tables], ignore_index=True)
    table.to_csv(os.path.join(args.out, "prefixe.csv"), index=False)
    order = ["Semi-Markov-Kette", *[label for label, _ in models]]
    summary = table.drop(columns=["Fall", "Prefixlänge"]).groupby("Modell", sort=False).mean().loc[order]  # mean ohne NaN (nicht gewertete Schritte)
    summary.to_csv(os.path.join(args.out, "suffixguete.csv"))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.round(3).to_string(), flush=True)
    counts = pd.DataFrame(markov_rows)[[f"Schritt {k}" for k in EVENT_STEPS]].notna().sum()
    stand = f"Stand {datetime.date.today():%d.%m.%Y}."
    header = ["Modell", *[f"Schritt {k}" for k in EVENT_STEPS], "DLS", "Suffixlänge-MAE"]
    rows = [[label, *[f"{summary.loc[label, f'Schritt {k}'] * 100:.0f} %" for k in EVENT_STEPS], number(summary.loc[label, "DLS"]),
             number(summary.loc[label, "Suffixlänge-MAE"], 1)] for label in order]
    render("Suffix-Vorhersage BPIC17 (Testfälle): Events", header, rows, os.path.join(args.out, "suffixguete_events.png"),
           f"{len(picks)} Prefixe aus Testfällen, je {args.draws} gezogene Suffixe. Treffer: häufigstes gezogenes Token an dieser Position gleich dem echten, "
           f"gewertet nur Prefixe mit echtem Event dort: {' / '.join(str(n) for n in counts)} an Schritt {' / '.join(str(k) for k in EVENT_STEPS)}. "
           f"DLS auf Aktivität|Lifecycle. {stand}", highlight=0)
    header = ["Modell", "Abstand Schritt 1\nMAE (h)", "Restzeit\nMAE (Tage)", "Restzeit\nFehler (Tage)", "Restzeit\nCRPS (Tage)",
              "Abdeckung\n80-%-Intervall", "Restzeit gezogen\n/ echt (Tage)"]
    rows = [[label, number(summary.loc[label, "Abstand Schritt 1 MAE (h)"], 1), number(summary.loc[label, "Restzeit MAE (Tage)"], 1),
             number(summary.loc[label, "Restzeit Fehler (Tage)"], 1, sign=True), number(summary.loc[label, "Restzeit CRPS (Tage)"], 1),
             f"{summary.loc[label, 'Abdeckung 80 %'] * 100:.0f} %",
             f"{number(summary.loc[label, 'Restzeit gezogen (Tage)'], 1)} / {number(summary.loc[label, 'Restzeit echt (Tage)'], 1)}"] for label in order]
    render("Suffix-Vorhersage BPIC17 (Testfälle): Zeiten", header, rows, os.path.join(args.out, "suffixguete_zeiten.png"),
           f"{len(picks)} Prefixe aus Testfällen, je {args.draws} gezogene Suffixe. Restzeit: letztes Prefix-Event bis Fallende, wie in der Bewertung der Policy "
           f"(Summe der gezogenen Abstände). MAE und Fehler des Mittels über die Ziehungen; CRPS der gezogenen Verteilung (kleiner ist besser); "
           f"Abdeckung: echte Restzeit zwischen 10. und 90. Perzentil der Ziehungen (Soll 80 %). Semi-Markov-Kette: Abstände je Übergang aus den Trainingsfällen. {stand}",
           highlight=0)
    print(f"######## Fertig in {time.time() - started:.0f} s; Ergebnisse in {args.out}", flush=True)


if __name__ == "__main__":
    main()
