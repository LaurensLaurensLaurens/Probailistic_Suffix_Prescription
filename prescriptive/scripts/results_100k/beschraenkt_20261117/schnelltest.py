"""Schnelltest beschränktes Ziehen (2026-11-17): je Modell ohne/mit Beschränkung an echten Testfällen (Nachlegen, 1. Besuch)
Optionssuffixe ziehen und Gültigkeit, Endzustände, Restzeit, Optionsunterschied und Kopplung vergleichen; dazu der Anteil der
Netz-Wahrscheinlichkeit auf gesperrten Tokens. Aufruf im Ordner process-simulation-engine: .venv/bin/python <dieses Skript> [Fälle]"""
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
sys.path.insert(0, str(ROOT/"prescriptive/scripts/results_100k/psp_register"))
import uebergaenge_pruefen as u  # noqa: E402

u.pruefen(verbose=False)


def check(tokens):
    """Ungültige Übergänge zwischen Events und Lifecycle-Folgen innerhalb einer Aktivität (Masken aus uebergaenge_pruefen)."""
    bad, last = sum((x, y) not in u.between for x, y in zip(tokens[:-1], tokens[1:])), {}
    for tok in tokens:
        a, _, l = tok.rpartition("|")
        if a in last:
            bad += (a, last[a], l) not in u.within
        last[a] = l
    return bad


OUT = Path(__file__).resolve().parent
CASES = int(sys.argv[1]) if len(sys.argv) > 1 else 60
ENDS = ("A_Pending", "A_Cancelled", "A_Denied")
models = {"(c)": ROOT/"prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal.pkl",
          "(c) Gumbel": OUT/"modelle/c_gumbel.pkl",
          "(c) Masken": OUT/"modelle/c_masken.pkl",
          "(c) Masken+Gumbel": OUT/"modelle/c_masken_gumbel.pkl",
          "ZZ ep9": ROOT/"prescriptive/scripts/results_100k/zaehler_zeit_20261116/modelle/psp_zz_ep9.pkl",
          "ZZ ep9 Masken": OUT/"modelle/zz9_masken.pkl",
          "ZZ ep9 Masken+Gumbel": OUT/"modelle/zz9_masken_gumbel.pkl"}
EXTRA = {"OE": ROOT/"prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_token_oe.pkl",
         "OE-TF1 Ep. 26": ROOT/"prescriptive/scripts/results_100k/modellvergleich_20261116/oe_tf1_schnappschuss/psp_oe_tf1_letzte.pkl",
         "D": ROOT/"prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_klassen_option_encoder.pkl"}
if len(sys.argv) > 2:  # nur die genannten Modelle (aus models oder EXTRA), eigene Ergebnisdatei
    models = {name: {**models, **EXTRA}[name] for name in sys.argv[2:]}
raw = pd.read_pickle(ROOT/"prescriptive/data/cache/bpic17_raw.pkl")
raw["time:timestamp"] = raw["time:timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
raw["concept:name"] = raw["concept:name"].astype(str).str.strip()
reference = PSPModel.load(str(ROOT/"prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal.pkl"))
test = sorted(map(str, reference.test_cases))
prefixes = []
for case in test:
    part = raw[raw["case:concept:name"].astype(str) == case].sort_values("time:timestamp", kind="stable")
    events = part.to_dict("records")
    cut = next((i+1 for i, e in enumerate(events) if e["concept:name"] == "O_Created" and e["lifecycle:transition"] == "complete"), None)
    if cut:
        prefixes.append(events[:cut])
    if len(prefixes) >= CASES:
        break
rows = []
for label, path in models.items():
    model = PSPModel.load(str(path))
    point = model.points["angebot_nachlegen"]
    torch.manual_seed(0)
    for events in prefixes:
        now = pd.Timestamp(events[-1]["time:timestamp"]).timestamp()
        drawn, _ = model.option_suffixes(point, events, list(point.options), now, 30)
        for option, suffixes in zip(point.options, drawn):
            for k, s in enumerate(suffixes):
                tokens = [t for t, _, _ in s]
                rows.append({"modell": label, "fall": events[0]["case:concept:name"], "option": option, "k": k,
                             "rest": sum(g for _, _, g in s)/86400, "events": len(s), "ungueltig": check(tokens),
                             "ende": next((e for t in tokens for e in ENDS if t.startswith(e)), "ohne")})
    stats = getattr(model, "ablauf_statistik", None)
    if stats:
        print(f"{label}: Wahrscheinlichkeit auf gesperrten Tokens im Mittel {stats['masse_gesperrt']/max(stats['schritte'], 1):.1%}, "
              f"Zeilen ohne erlaubtes Token {stats['ohne_erlaubtes_token']} von {stats['zeilen']}")
frame = pd.DataFrame(rows)
frame.to_csv(OUT/("schnelltest.csv" if len(sys.argv) <= 2 else "schnelltest_" + "_".join(n.replace(" ", "").replace(".", "") for n in sys.argv[2:]) + ".csv"), index=False)
summary = frame.groupby("modell").agg(rest=("rest", "mean"), sd_rest=("rest", "std"), events=("events", "mean"),
                                      ungueltig_je_100=("ungueltig", lambda x: np.nan),
                                      ohne_ende=("ende", lambda x: (x == "ohne").mean()), pending=("ende", lambda x: (x == "A_Pending").mean()),
                                      cancelled=("ende", lambda x: (x == "A_Cancelled").mean()))
summary["ungueltig_je_100"] = frame.groupby("modell")[["ungueltig", "events"]].sum().pipe(lambda g: 100*g.ungueltig/g.events)
wide = frame.pivot_table(index=["modell", "fall", "k"], columns="option", values="rest")
extra = next(c for c in wide.columns if c.startswith("O_Create Offer"))
send = next(c for c in wide.columns if c != extra)
a, b = extra, send
summary["diff_extra_minus_senden"] = (wide[extra]-wide[send]).groupby(level=0).mean()
summary["se_diff_je_fall"] = (wide[extra]-wide[send]).groupby(level=[0, 1]).mean().groupby(level=0).agg(lambda x: x.std()/len(x)**0.5)
summary["korrelation_optionen"] = wide.groupby(level=0).apply(lambda g: g[a].corr(g[b]))
pd.set_option("display.width", 200)
print(summary.round(3).to_string())
