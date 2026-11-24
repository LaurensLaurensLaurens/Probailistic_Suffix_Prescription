"""Wo weichen die gezogenen Verläufe nach einem Zusatzangebot von den echten ab? (2026-11-19, Nutzer: „ja mach das setze das um“)

Anlass: angebote_sonde.py – mit den echten Events trifft der β-NLL-Kopf die Wartezeiten nach zwei Angeboten; trotzdem zieht die
β-NLL-Variante beim ersten nachlegen keinen Effekt. Hier: Verläufe ab der Entscheidung (O_Created → O_Sent oder O_Create Offer) bis zum
ersten W_Validate application|start (Kundenantwort; ohne Validierung bis Fallende), je Quelle und Option:
  - Zahl jedes Tokens je Verlauf,
  - Zeit je Token: Summe der Abstände VOR diesem Token (wohin die Zeit fällt),
  - Kennzahlen: Zeit bis 2. / letztes O_Sent, Nachfass-Schleifen (suspend/resume), Anteil mit Validierung, Restzeit.
Unterschied extra − senden je Quelle; Zerlegung des Zeitunterschieds nach Token zeigt, welche Schritte im Gezogenen fehlen.
Quellen: Simulator mit Bank-Fortsetzung (randomisiert), T-Learner, Steigung, Steigung + β-NLL 1 (je 60 Zustände × 100 Ziehungen,
nachlegen_erklaerung*), echter Log (1. O_Created je Fall, beobachtet). Ausgabe nachlegen_verlaeufe.md / _token.csv / _kennzahlen.csv.
Aufruf: process-simulation-engine/.venv/bin/python nachlegen_verlaeufe.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
VAL = "W_Validate application|start"
SOURCES = [("nachlegen_erklaerung_t_kopf_steigung_bnll1", None), ("nachlegen_erklaerung_t_kopf_steigung", "T-Learner Kopf je Event (Steigung)"),
           ("nachlegen_erklaerung_t_zztf1", "T-Learner (ZZ-TF1)")]
SHORT = {"Simulator (Bank danach)": "Simulator", "T-Learner Kopf je Event (Steigung, β-NLL 1)": "β-NLL 1",
         "T-Learner Kopf je Event (Steigung)": "Steigung", "T-Learner (ZZ-TF1)": "T-Learner", "echter Log (beobachtet)": "echt"}


def traces():
    frames, seen = [], set()
    for folder, only in SOURCES:
        v = pd.read_pickle(OUT/folder/"verlaeufe.pkl")
        v = v[~v.quelle.isin(seen)] if only is None else v[v.quelle == only]
        seen |= set(v.quelle)
        frames.append(pd.DataFrame({"quelle": v.quelle.map(SHORT).values,
                                    "opt": np.where(v.option.str.startswith("O_Create"), "extra", "senden"),
                                    "tokens": [list(t) for t in v.tokens], "gaps": [list(map(float, g)) for g in v.gaps]}))
    d = pd.read_pickle(ROOT/"prescriptive/data/cache/bpic17_raw.pkl")
    d["concept:name"] = d["concept:name"].str.strip()
    d = d.sort_values(["case:concept:name", "time:timestamp"], kind="stable")
    d["tok"] = d["concept:name"] + "|" + d["lifecycle:transition"]
    rows = []
    for _, g in d.groupby("case:concept:name", sort=False):
        names = g["concept:name"].tolist()
        if "O_Created" not in names:
            continue
        i = names.index("O_Created")
        if i + 1 >= len(names) or names[i + 1] not in ("O_Sent (mail and online)", "O_Create Offer"):
            continue
        days = g["time:timestamp"].astype("int64").to_numpy() / 86400e9
        rows.append({"quelle": "echt", "opt": "extra" if names[i + 1] == "O_Create Offer" else "senden",
                     "tokens": g.tok.tolist()[i + 1:], "gaps": list(np.diff(days[i:]))})
    frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def cut(tokens, gaps):
    stop = next((j for j, t in enumerate(tokens) if t == VAL), None)
    end = len(tokens) if stop is None else stop + 1
    return tokens[:end], gaps[:end], stop is not None


def measures(row):
    t, g, validated = cut(row.tokens, row.gaps)
    cum = np.cumsum(g)
    sends = [j for j, x in enumerate(t) if x.startswith("O_Sent")]
    return {"bis Validierung bzw. Ende (d)": cum[-1] if len(cum) else 0.0, "erreicht Validierung": validated,
            "Events bis dahin": len(t), "O_Sent": len(sends), "O_Created": sum(x.startswith("O_Created") for x in t),
            "Nachfass-Schleifen": sum(x in ("W_Call after offers|suspend", "W_Call after offers|resume") for x in t),
            "bis 2. O_Sent (d)": cum[sends[1]] if len(sends) > 1 else np.nan,
            "bis letztes O_Sent (d)": cum[sends[-1]] if sends else np.nan,
            "letztes O_Sent → Validierung (d)": cum[-1] - cum[sends[-1]] if validated and sends else np.nan,
            "Restzeit gesamt (d)": float(np.sum(row.gaps))}


def per_token(frame):
    """Je Quelle und Option: mittlere Zahl und mittlere Zeit (Abstände vor dem Token) je Token und Verlauf bis zur Validierung."""
    out = []
    for (source, opt), part in frame.groupby(["quelle", "opt"]):
        counts, times = {}, {}
        for row in part.itertuples():
            t, g, _ = cut(row.tokens, row.gaps)
            for x, y in zip(t, g):
                counts[x] = counts.get(x, 0) + 1
                times[x] = times.get(x, 0.0) + y
        n = len(part)
        for x in counts:
            out.append({"quelle": source, "opt": opt, "token": x, "zahl": counts[x] / n, "zeit": times[x] / n})
    return pd.DataFrame(out)


def main():
    frame = traces()
    m = pd.concat([frame[["quelle", "opt"]], pd.DataFrame([measures(r) for r in frame.itertuples()])], axis=1)
    order = ["echt", "Simulator", "T-Learner", "Steigung", "β-NLL 1"]
    level = m.groupby(["quelle", "opt"]).mean(numeric_only=True)
    level["n"] = m.groupby(["quelle", "opt"]).size()
    diff = (level.xs("extra", level="opt") - level.xs("senden", level="opt")).drop(columns="n").reindex(order)
    level.to_csv(OUT/"nachlegen_verlaeufe_kennzahlen.csv")
    tok = per_token(frame)
    tok.to_csv(OUT/"nachlegen_verlaeufe_token.csv", index=False)
    wide = tok.pivot_table(index="token", columns=["quelle", "opt"], values=["zahl", "zeit"], fill_value=0.0)
    rows = {}
    for kind in ("zahl", "zeit"):
        for source in order:
            if (kind, source, "extra") in wide.columns:
                rows[(kind, source)] = wide[(kind, source, "extra")] - wide[(kind, source, "senden")]
    d = pd.DataFrame(rows)
    top = d[("zeit", "echt")].abs().add(d[("zeit", "β-NLL 1")].abs()).sort_values(ascending=False).index[:14]
    time_diff = d["zeit"].loc[top].reindex(columns=order)
    count_diff = d["zahl"].loc[top].reindex(columns=order)
    lines = ["# Verläufe nach dem 1. nachlegen: extra − senden je Quelle (2026-11-19)", "",
             "Bis zur ersten W_Validate application|start (sonst Fallende). Echt = beobachtet, Simulator = randomisiert mit Bank-Fortsetzung.", "",
             "## Kennzahlen (extra − senden)", "", "```", diff.T.round(2).to_string(), "```", "",
             "## Kennzahlen je Option", "", "```", level.round(2).to_string(), "```", "",
             "## Zeit vor Token je Verlauf, extra − senden (Tage; die 14 Token mit dem größten Unterschied echt + β-NLL)", "", "```",
             time_diff.round(2).to_string(), "```", "", "## Zahl je Verlauf, extra − senden (gleiche Token)", "", "```",
             count_diff.round(2).to_string(), "```", ""]
    (OUT/"nachlegen_verlaeufe.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
