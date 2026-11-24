"""Stornierung in gezogenen Suffixen (2026-11-17): trifft das Netz die 30-Tage-Frist nach dem letzten O_Sent? Hauptzweck der Uhren
(ZZU-TF1). Gleiche Prefixe für alle Modelle: Testfälle (äußere Testfälle von (c)), Schnitt direkt nach dem ersten O_Sent, danach noch
mindestens ein Event. Je Prefix n Ziehungen bis EOS (completion_steps). Gemessen je Quelle (echter Log = die echten Suffixe derselben
Prefixe): Anteil Verläufe mit A_Cancelled, Abstand vor A_Cancelled (Tage seit dem vorigen Event), Tage vom letzten O_Sent bis
A_Cancelled (Mittel, Median, P10/P90, Anteil 29–32 Tage), Restzeit bis A_Cancelled ab Schnitt, Restzeit insgesamt.
Aufruf im Ordner process-simulation-engine: .venv/bin/python <dieses Skript> [Prefixe] [Ziehungen]
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PSP_CATALOG", str(ROOT/"prescriptive/data/catalog_neue_punkte_zuteilung_random.yaml"))
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths  # noqa: E402
add_paths()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from prescriptive.estimators.psp_model import PSPModel  # noqa: E402

OUT = Path(__file__).resolve().parent
N_PREFIXES = int(sys.argv[1]) if len(sys.argv) > 1 else 300
N_DRAWS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
R = ROOT/"prescriptive/scripts/results_100k"
MODELS = {"(c)": ROOT/"prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal.pkl",
          "(c)-TF1": ROOT/"prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal_tf1.pkl",
          "ZZ-TF1 ep12": R/"zaehler_zeit_tf1_20261117/modelle/psp_zztf1_ep12.pkl",
          "ZZU-TF1 ep3": R/"zaehler_uhren_tf1_20261117/modelle/psp_zzu_ep3.pkl"}
REFERENZ = MODELS["(c)"]
EXTRA = dict(a.split("=", 1) for a in sys.argv[3:])  # label=pfad: nur diese Modelle rechnen, frühere Zeilen der Tabelle behalten
if EXTRA:
    MODELS = {label: Path(path) for label, path in EXTRA.items()}
SENT = ("O_Sent (mail and online)", "O_Sent (online only)")
DAY = 86400.0


def measure(suffixes, cut_time, last_sent):
    """suffixes: Liste von (Aktivität, Sekunden seit Fallbeginn, Abstand) je Verlauf; cut_time/last_sent in Sekunden seit Fallbeginn."""
    rows = []
    for suffix in suffixes:
        sent, row = last_sent, {"storniert": False}
        for name, t, gap in suffix:
            activity = name.split("|")[0]
            if activity in SENT:
                sent = t
            if activity == "A_Cancelled":
                row = {"storniert": True, "abstand_vorher": gap / DAY, "seit_o_sent": (t - sent) / DAY, "rest_bis_storno": (t - cut_time) / DAY}
                break
        row["rest"] = (suffix[-1][1] - cut_time) / DAY if suffix else 0.0
        rows.append(row)
    return rows


def main():
    raw = pd.read_pickle(ROOT/"prescriptive/data/cache/bpic17_raw.pkl")
    raw["time:timestamp"] = raw["time:timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    raw["concept:name"] = raw["concept:name"].astype(str).str.strip()
    reference = PSPModel.load(str(REFERENZ))  # Testfälle von (c): gleiche Prefixe für alle Modelle
    grouped = {c: p for c, p in raw.groupby("case:concept:name", sort=False)}
    rng = np.random.default_rng(0)
    prefixes, real = [], []
    for case in rng.permutation(sorted(map(str, reference.test_cases))):
        part = grouped.get(case)
        if part is None:
            continue
        events = part.sort_values("time:timestamp", kind="stable").to_dict("records")
        hits = [i for i, e in enumerate(events) if e["concept:name"] in SENT]
        if not hits or hits[0] + 1 >= len(events):
            continue
        cut = hits[0] + 1
        start = events[0]["time:timestamp"]
        seconds = [(e["time:timestamp"] - start).total_seconds() for e in events]
        suffix = [(f"{e['concept:name']}|{e.get('lifecycle:transition', 'complete')}", seconds[cut + k],
                   seconds[cut + k] - seconds[cut + k - 1]) for k, e in enumerate(events[cut:])]
        prefixes.append(events[:cut])
        real.extend({**row, "quelle": "echter Log"} for row in measure([suffix], seconds[cut - 1], seconds[hits[0]]))
        if len(prefixes) >= N_PREFIXES:
            break
    rows = list(real)
    for label, path in MODELS.items():
        model = PSPModel.load(str(path))
        torch.manual_seed(0)
        encoded = [model._encode(p) for p in prefixes]
        drawn = model.draw_cases([e[:4] for e in encoded], N_DRAWS, completion_steps=200)
        for e, p, suffixes in zip(encoded, prefixes, drawn):
            start = p[0]["time:timestamp"]
            cut_time = (p[-1]["time:timestamp"] - start).total_seconds()
            sent = max((ev["time:timestamp"] - start).total_seconds() for ev in p if ev["concept:name"] in SENT)
            rows.extend({**row, "quelle": label} for row in measure(suffixes, cut_time, sent))
        print(label, "fertig", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/("storno_abstand_roh.csv" if not EXTRA else f"storno_abstand_roh_{len(EXTRA)}_{list(EXTRA)[0].replace(' ', '_')}.csv"), index=False)
    c = frame[frame.storniert]
    summary = pd.DataFrame({
        "Verläufe": frame.groupby("quelle").size(),
        "Anteil storniert": frame.groupby("quelle").storniert.mean(),
        "Abstand vor A_Cancelled (Mittel)": c.groupby("quelle").abstand_vorher.mean(),
        "seit letztem O_Sent: Mittel": c.groupby("quelle").seit_o_sent.mean(),
        "Median": c.groupby("quelle").seit_o_sent.median(),
        "P10": c.groupby("quelle").seit_o_sent.quantile(0.1),
        "P90": c.groupby("quelle").seit_o_sent.quantile(0.9),
        "Anteil 29–32 d": c.groupby("quelle").seit_o_sent.apply(lambda s: s.between(29, 32).mean()),
        "Restzeit bis Storno": c.groupby("quelle").rest_bis_storno.mean(),
        "Restzeit gesamt": frame.groupby("quelle").rest.mean()})
    if EXTRA and (OUT/"storno_abstand.csv").exists():  # frühere Modelle behalten (gleiche Prefixe und Ziehungen: Saat 0)
        previous = pd.read_csv(OUT/"storno_abstand.csv", index_col=0)
        summary = pd.concat([previous[~previous.index.isin(summary.index)], summary])
    order = ["echter Log"] + [i for i in summary.index if i != "echter Log"]
    summary = summary.reindex(order)
    summary.to_csv(OUT/"storno_abstand.csv")
    pd.set_option("display.width", 250)
    print(f"{len(prefixes)} Prefixe (Schnitt nach dem ersten O_Sent), {N_DRAWS} Ziehungen je Prefix; Tage")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
