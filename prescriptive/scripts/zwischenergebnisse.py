"""
Die Tabellen in EXTRAZWISCHENERGEBNISSE/<Datensatz>/ als Bild, im Stil von EXTRAZWISCHENERGEBNISSE/2017:

suffix  Suffix-Vorhersage: Trefferquote je Schritt, DLS, Suffixlänge-MAE für eine Markov-Kette und PSP-Modelle, auf
        zufälligen Prefixen des echten Logs (keines der Modelle hat es gesehen, alle lernen aus den RCT-Daten).
        - Prefixe: PREFIXES zufällige (Fall, Länge 1 bis Fallende - 1) des echten Logs nach dem Preprocessing der Pipeline
        - je Prefix DRAWS gezogene Suffixe; Token sind Aktivität|Lifecycle wie beim PSP (PSPModel.labels)
        - Treffer an Schritt k: das häufigste gezogene Token an Position k gleich dem echten, gewertet nur Prefixe, deren
          echtes Suffix mindestens k Events hat; ein gezogenes Suffix, das vor k endet, zählt dort als "Ende" (kein Treffer)
        - DLS: Damerau-Levenshtein-Ähnlichkeit (estimators/kpis.dls) von gezogenem und echtem Suffix, Mittel über die Ziehungen
        - Suffixlänge-MAE: |mittlere gezogene Länge - echte Länge| in Events
        - Markov-Kette 1. Ordnung auf denselben Token mit Ende als Zustand, gelernt aus den Trainingsfällen des ersten
          PSP-Modells (seine RCT-Daten ohne die Testfälle); unbekanntes letztes Token: Ende
online  Online-Auswertung im gelernten Simulator: Durchlaufzeit, Uplift gegen die Bank mit 95-%-KI, P90 und CVaR 90 %
        gegen die Bank, aus den Tabellen der Pipeline (<out>/prescription/<Ansatz>/<Datensatz>_online.csv); --extra holt
        weitere Zeilen aus anderen Läufen derselben Seeds (z. B. ein anderes PSP-Modell, psp_erm je γ)
truth   die echte Werkstatt (scripts/repair_truth.py, <Ordner>/laeufe.csv): Durchlaufzeit, Abstand zur Hausregel mit 95-%-KI,
        P90 und CVaR 90 % gegen die Hausregel, Anteile der Wahlen je Fähigkeit; gepaart je Seed mit der Bank desselben Ordners

Aufruf aus process-simulation-engine/, z. B.:
    .venv/bin/python -u ../prescriptive/scripts/zwischenergebnisse.py suffix --dataset bpic19 \
        --psp "PSP (5.000 Fälle, 25 Epochen, 1e-4)=psp_model_bpic19.pkl" --out ../EXTRAZWISCHENERGEBNISSE/2019/suffix_vorhersage_bpic19.png
    .venv/bin/python -u ../prescriptive/scripts/zwischenergebnisse.py online --dataset bpic19 --results results_bpic19_durchlaufzeit \
        --out ../EXTRAZWISCHENERGEBNISSE/2019/online_auswertung_bpic19.png
"""
import argparse
import datetime
import os
import sys
import textwrap
import time
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "process-simulation-engine"), os.path.join(ROOT, "prescriptive"), ROOT]
MODELS = os.path.join(ROOT, "prescriptive", "models")
SCRIPTS = os.path.join(ROOT, "prescriptive", "scripts")
CASE, TIMESTAMP = "case:concept:name", "time:timestamp"
PREFIXES, DRAWS, SEED = 200, 50, 0
STEPS = (1, 2, 3, 5, 10, 20)
END = "Ende"
MAX_MARKOV_STEPS = 500
NOTE_WIDTH = 185  # Zeichen je Zeile der Fußnote


# ------------------------------------------------------------------ Bild

def number(value, digits=2, sign=False):
    """Deutsche Schreibweise: Komma, echtes Minus, mit sign auch +."""
    text = f"{value:{'+' if sign else ''}.{digits}f}".replace(".", ",").replace("-", "−")
    return "0,0" if text in ("+0,0", "−0,0") else text


def render(title, header, rows, path, note, highlight=None):
    """Tabelle als PNG wie in EXTRAZWISCHENERGEBNISSE/2017: Kopf grau, Zeile highlight gelb, Fußnote grau."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Breite je Spalte in Zeichen: die längste Zeile der Überschrift (fett, etwa 15 % breiter) oder der Zellen
    widths = [max(1.15 * max(map(len, str(header[j]).split("\n"))), *(len(str(row[j])) for row in rows)) + 4 for j in range(len(header))]
    widths[0] += 6
    total = sum(widths)
    note = textwrap.fill(note, NOTE_WIDTH)
    lines = note.count("\n") + 1
    fig = plt.figure(figsize=(20, 1.3 + 0.28 * (lines - 1) + 0.65 * (len(rows) + 1)), dpi=130)
    fig.text(0.012, 1 - 0.3 / fig.get_figheight(), title, fontsize=21, fontweight="bold", color="#1f2329", va="top")
    top, bottom = 1 - 1.0 / fig.get_figheight(), (0.75 + 0.28 * (lines - 1)) / fig.get_figheight()
    ax = fig.add_axes([0.012, bottom, 0.976, top - bottom])
    ax.axis("off")
    table = ax.table(cellText=[list(map(str, row)) for row in rows], colLabels=header, colWidths=[w / total for w in widths],
                     cellLoc="center", loc="upper left", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d5d8de")
        cell.set_fontsize(16)
        cell.get_text().set_color("#1f2329")
        if c == 0:
            cell.get_text().set_horizontalalignment("left")
            cell.PAD = 0.02
        if r == 0:
            cell.set_facecolor("#eef0f4")
            cell.get_text().set_fontweight("bold")
        elif r - 1 in ([] if highlight is None else [highlight] if isinstance(highlight, int) else highlight):
            cell.set_facecolor("#fcf3d9")
    fig.text(0.012, (0.32 + 0.14 * (lines - 1)) / fig.get_figheight(), note, fontsize=14, color="#5f6670", va="center", linespacing=1.5)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"gespeichert: {path}", flush=True)


# ------------------------------------------------------------------ Suffix-Vorhersage

def training_cases(model, folder=None):
    """
    Die Trainingsfälle des PSP-Modells: seine RCT-Daten (folder: Ordner unter prescriptive/data, Standard der beim Training),
    gezogen wie helpers/training.train_estimators, ohne die Testfälle.
    """
    from helpers.parameters import TRAIN_SEED
    from helpers.training import sample_cases
    data = pd.read_pickle(os.path.join(ROOT, "prescriptive", "data", folder or model.trained_on["rct"], "log.pkl"))
    if model.trained_on["cases"] < data[CASE].nunique():
        data = sample_cases(data, model.trained_on["cases"], TRAIN_SEED)
    return data[~data[CASE].isin(model.test_cases)]


class Markov:
    """Markov-Kette 1. Ordnung auf den Token, Ende als Zustand."""

    def __init__(self, sequences, seed=SEED):
        counts = {}
        for tokens in sequences:
            for a, b in zip(tokens, tokens[1:] + [END]):
                counts.setdefault(a, Counter())[b] += 1
        self.next = {a: (list(c), np.array(list(c.values()), dtype=float) / sum(c.values())) for a, c in counts.items()}
        self.rng = np.random.default_rng(seed)

    def draw(self, last, n):
        suffixes = []
        for _ in range(n):
            suffix, token = [], last
            while len(suffix) < MAX_MARKOV_STEPS and token in self.next:
                tokens, p = self.next[token]
                token = tokens[self.rng.choice(len(tokens), p=p)]
                if token == END:
                    break
                suffix.append(token)
            suffixes.append(suffix)
        return suffixes


def scores(true, drawn):
    """Treffer je Schritt (0/1), DLS (Mittel über die Ziehungen) und Längenfehler eines Prefix."""
    from prescriptive.estimators.kpis import dls
    hits = {}
    for k in STEPS:
        at = Counter(suffix[k - 1] if len(suffix) >= k else END for suffix in drawn).most_common(1)[0][0]
        hits[k] = float(at == true[k - 1]) if len(true) >= k else np.nan
    return {**{f"Schritt {k}": hits[k] for k in STEPS}, "DLS": float(np.mean([dls(suffix, true) for suffix in drawn])),
            "Suffixlänge-MAE": abs(np.mean([len(suffix) for suffix in drawn]) - len(true))}


def suffix_table(args):
    import torch

    from prescriptive.helpers.datasets import add_paths
    add_paths()
    from helpers.datasets import load_dataset
    from preprocessing import preprocess
    from prescriptive.estimators.psp_model import PSPModel
    models = [(label, PSPModel.load(path if os.path.isabs(path) else os.path.join(MODELS, path))) for label, path in
              (entry.split("=", 1) for entry in args.psp)]
    for _, model in models:
        model.set_threads(args.threads)
    raw, _, _ = load_dataset(args.dataset)
    log = preprocess(args.dataset, raw).sort_values([CASE, TIMESTAMP], kind="stable")
    parts = [part.reset_index(drop=True) for _, part in log.groupby(CASE, sort=True) if len(part) > 1]
    rng = np.random.default_rng(SEED)
    picks = [(int(c), int(rng.integers(1, len(parts[c])))) for c in rng.choice(len(parts), args.prefixes, replace=False)]
    first = models[0][1]
    train = training_cases(first, args.rct).sort_values([CASE, TIMESTAMP], kind="stable")
    train = train.assign(_token=first.labels(train).to_numpy())
    markov = Markov([list(tokens) for _, tokens in train.groupby(CASE, sort=False)["_token"]])
    print(f"{args.dataset}: {len(picks)} Prefixe aus {len(parts):,} Fällen des echten Logs, Markov-Kette aus {train[CASE].nunique():,} Trainingsfällen", flush=True)
    results = {"Markov-Kette": []}
    for c, k in picks:
        tokens = list(first.labels(parts[c]))
        results["Markov-Kette"].append(scores(tokens[k:], markov.draw(tokens[k - 1], args.draws)))
    for label, model in models:
        torch.manual_seed(SEED)
        started, rows = time.time(), []
        for c, k in picks:
            tokens = list(model.labels(parts[c]))
            prefix, static, mask, length, _ = model._encode(parts[c].iloc[:k].to_dict("records"))
            drawn = model.draw([prefix], static, mask, length, args.draws)[0]
            rows.append(scores(tokens[k:], [[event[0] for event in suffix] for suffix in drawn]))
        results[label] = rows
        print(f"{label}: {time.time() - started:.0f} s", flush=True)
    table = pd.DataFrame({label: pd.DataFrame(rows).mean() for label, rows in results.items()}).T  # mean: ohne die nicht gewerteten (NaN)
    table.index.name = "Modell"
    table.to_csv(os.path.splitext(args.out)[0] + ".csv")
    counts = pd.DataFrame(results["Markov-Kette"])[[f"Schritt {k}" for k in STEPS]].notna().sum()
    print(table.round(3).to_string(), flush=True)
    header = ["Modell", *[f"Schritt {k}" for k in STEPS], "DLS", "Suffixlänge-MAE"]
    rows = [[label, *[f"{table.loc[label, f'Schritt {k}'] * 100:.0f} %" for k in STEPS], number(table.loc[label, "DLS"]),
             number(table.loc[label, "Suffixlänge-MAE"], 1)] for label in table.index]
    note = (f"{len(picks)} echte Prefixe, je {args.draws} gezogene Suffixe. Treffer: häufigstes gezogenes Token an dieser Position gleich dem echten, "
            f"gewertet nur Prefixe mit echtem Event dort: {' / '.join(str(n) for n in counts)} an Schritt {' / '.join(str(k) for k in STEPS)}. DLS auf Aktivitäten. "
            f"Stand {datetime.date.today():%d.%m.%Y}.")
    render(f"Suffix-Vorhersage {args.name or args.dataset}: Trefferquote je Schritt", header, rows, args.out, note, highlight=0)


# ------------------------------------------------------------------ Online-Auswertung

def online_table(args):
    folder = args.results if os.path.isabs(args.results) else os.path.join(SCRIPTS, args.results)
    tables = []
    for approach in args.approaches:
        path = os.path.join(folder, "prescription", approach, f"{args.dataset}_online.csv")
        if os.path.exists(path):
            tables.append(pd.read_csv(path, index_col=0))
    table = pd.concat(tables)
    table = table[~table.index.duplicated()]
    order = [name for name in ("bank", "random", "psp", "psp_erm", "psp_lookahead", "xgb", "lstm") if name in table.index]
    labels = {"bank": "Bank", "random": "Zufall", "psp": "PSP", "psp_erm": "PSP-ERM", "psp_lookahead": "PSP-Vorausschau", "xgb": "XGB", "lstm": "LSTM"}
    labels.update(dict(entry.split("=", 1) for entry in args.label))
    # Weitere Zeilen (Ordner|Ansatz|Policy|Beschriftung) vor XGB, unter eigenem Namen
    at = order.index("xgb") if "xgb" in order else len(order)
    policies = {name: name for name in order}
    for k, entry in enumerate(args.extra):
        other, approach, policy, label = entry.split("|", 3)
        other = other if os.path.isabs(other) else os.path.join(SCRIPTS, other)
        name = f"_extra{k}"
        table.loc[name] = pd.read_csv(os.path.join(other, "prescription", approach, f"{args.dataset}_online.csv"), index_col=0).loc[policy]
        order.insert(at + k, name)
        labels[name], policies[name] = label, policy
    header = ["Policy", "Durchlaufzeit (Tage)", "Uplift gegen Bank (95-%-KI)", "P90 gegen Bank (Tage)", "CVaR 90 % gegen Bank (Tage)"]
    rows = []
    for name in order:
        row = table.loc[name]
        if name == "bank":
            rows.append([labels[name], number(row["Durchlaufzeit (Tage)"]), "–", "–", "–"])
            continue
        rows.append([labels[name], number(row["Durchlaufzeit (Tage)"]),
                     f"{number(row['Uplift ggü. Bank (%)'], 1, True)} % ({number(row['Uplift 95-%-KI unten'], 1)} bis {number(row['Uplift 95-%-KI oben'], 1)})",
                     number(row["Durchlaufzeit P90 (Tage) ggü. Bank"], 1, True), number(row["CVaR 90 % (Tage) ggü. Bank"], 1, True)])
    seeds, cases = int(table["Wiederholungen"].max()), int(round(table["Fälle"].mean()))
    count = f"{cases:,}".replace(",", ".")
    note = (f"{seeds} Seeds, je {count} gewertete Fälle, Outcome Durchlaufzeit, Vergleich gepaart je Seed. "
            f"Uplift positiv = kürzer. {args.note + ' ' if args.note else ''}Stand {datetime.date.today():%d.%m.%Y}.")
    highlight = [k for k, name in enumerate(order) if policies[name].startswith("psp")]
    render(f"Online-Auswertung im Simulator: {args.name or args.dataset}", header, rows, args.out, note, highlight=highlight)


# ------------------------------------------------------------------ Echte Werkstatt

def truth_table(args):
    header = ["Policy", "Durchlaufzeit\n(Tage)", "gegen Hausregel\n(Tage, 95-%-KI)", "P90 gegen\nHausregel (Tage)", "CVaR 90 % gegen\nHausregel (Tage)",
              "Wahl stark / neutral /\nschwach / Expertin"]
    rows, highlight, seeds, cases = [], [], 0, 0
    for k, entry in enumerate(args.rows):
        folder, policy, label = entry.split("|", 2)
        folder = folder if os.path.isabs(folder) else os.path.join(SCRIPTS, folder)
        runs = pd.read_csv(os.path.join(folder, "laeufe.csv"))
        mine, bank = (runs[runs["Policy"] == name].set_index("seed").sort_index() for name in (policy, "bank"))
        seeds, cases = max(seeds, len(mine)), int(round(mine["Fälle"].mean()))

        def gap(column, digits=2, interval=False):
            if column not in mine:
                return "\u2013"
            diff = mine[column] - bank[column]
            text = number(diff.mean(), digits, True)
            return f"{text} (± {number(2.262 * diff.std(ddof=1) / np.sqrt(len(diff)), digits)})" if interval else text  # t mit 9 Freiheitsgraden

        if policy == "bank":
            rows.append([label, number(mine["Durchlaufzeit (Tage)"].mean()), "\u2013", "\u2013", "\u2013", "\u2013"])
            continue
        shares = " / ".join(f"{mine[f'gewählt: {g}'].mean() * 100:.0f}" for g in ("stark", "neutral", "schwach", "expertin")) + " %"
        rows.append([label, number(mine["Durchlaufzeit (Tage)"].mean()), gap("Durchlaufzeit (Tage)", interval=True),
                     gap("Durchlaufzeit P90 (Tage)", 1), gap("Durchlaufzeit CVaR 90 % (Tage)", 1), shares])
        if policy.startswith("psp"):
            highlight.append(k)
    count = f"{cases:,}".replace(",", ".")
    note = (f"Die echte Werkstatt (bekannte Wahrheit), {seeds} Seeds, je {count} gewertete Fälle, gepaart je Seed; negativ = kürzer als die Hausregel. "
            f"{args.note + ' ' if args.note else ''}Stand {datetime.date.today():%d.%m.%Y}.")
    render(f"Echte Werkstatt: {args.name or args.dataset}", header, rows, args.out, note, highlight=highlight)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=("suffix", "online", "truth"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--name", default=None, help="Name im Titel (Standard: der Datensatz)")
    parser.add_argument("--out", required=True, help="PNG (suffix: dazu die Zahlen als CSV daneben)")
    parser.add_argument("--psp", nargs="+", default=[], metavar="BESCHRIFTUNG=MODELL", help="suffix: PSP-Modelle (Datei in prescriptive/models oder Pfad)")
    parser.add_argument("--prefixes", type=int, default=PREFIXES)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rct", default=None, help="suffix: RCT-Ordner des ersten Modells unter prescriptive/data, falls umbenannt")
    parser.add_argument("--results", default=None, help="online: Ergebnisordner der Pipeline (unter prescriptive/scripts oder Pfad)")
    parser.add_argument("--approaches", nargs="+", default=["XGBoost", "PSP"], help="online: Ordner unter prescription/")
    parser.add_argument("--label", nargs="*", default=[], metavar="POLICY=BESCHRIFTUNG", help="online: Beschriftung je Policy, z. B. xgb='XGB, 5.000 Fälle'")
    parser.add_argument("--note", default="", help="online, truth: Zusatz für die Fußnote")
    parser.add_argument("--extra", nargs="*", default=[], metavar="ORDNER|ANSATZ|POLICY|BESCHRIFTUNG", help="online: weitere Zeilen aus anderen Läufen")
    parser.add_argument("--rows", nargs="*", default=[], metavar="ORDNER|POLICY|BESCHRIFTUNG", help="truth: Zeilen in dieser Reihenfolge (Ordner mit laeufe.csv)")
    args = parser.parse_args()
    if args.what == "suffix":
        if not args.psp:
            parser.error("suffix braucht --psp")
        suffix_table(args)
    elif args.what == "online":
        if not args.results:
            parser.error("online braucht --results")
        online_table(args)
    else:
        if not args.rows:
            parser.error("truth braucht --rows")
        truth_table(args)


if __name__ == "__main__":
    main()
