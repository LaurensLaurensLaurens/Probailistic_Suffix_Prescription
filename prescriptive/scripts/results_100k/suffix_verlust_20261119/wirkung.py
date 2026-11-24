"""Wirkung des Suffix-Verlusts an den Wahlpunkten (2026-11-19, nach dem Training beider T-Learner-Arme mit sequence_loss).

Dieselbe Diagnose wie modellvergleich_20261116/crn_schluessel.py (gleiche 300 Simulatorzustände, gleiche Saaten je Zustand, je Option
100 gezogene Suffixe, Bedingungen schritt / event), nur für die neue Hülle; die Ziehungen des bisherigen T-Learners (ZZ-TF1) kommen aus
crn_schluessel/ziehungen.pkl (gleiche Zustände und Saaten, daher je Zustand gepaart vergleichbar).
Ausgaben in wirkung/: ziehungen.pkl (nur neues Modell), wirkung.csv / wirkung.md mit
  - Zusammenfassung je Modell, Bedingung, Gruppe (Spalten wie crn_schluessel, dazu Restzeit je Option und erwarteter Anteil falscher
    Entscheidungen bei 300 Ziehungen: Mittel über Zustände von Φ(−Vorteil / Rauschen bei 300))
  - je Gruppe gepaart über Zustände: Vorteil der Regel-Option neu minus alt (Bedingung event)
Aufruf im Ordner process-simulation-engine: .venv/bin/python <dieses Skript> <Hülle.pkl>
Umgebung (2026-11-19 abends): WIRKUNG_LABEL (Name des neuen Modells), WIRKUNG_OUT (Ausgabeordner hier, Standard wirkung),
WIRKUNG_EXTRA (weitere ziehungen.pkl, nur für die Zusammenfassung).
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
VERGLEICH = HERE.parent/"modellvergleich_20261116"
sys.path.insert(0, str(VERGLEICH))
import crn_schluessel as crn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

OUT = HERE/os.environ.get("WIRKUNG_OUT", "wirkung")
NEW = os.environ.get("WIRKUNG_LABEL", "T-Learner Suffix-Verlust")
EXTRA = [Path(x) for x in os.environ.get("WIRKUNG_EXTRA", "").split(":") if x]
OLD = "T-Learner (ZZ-TF1)"
# beim Import gesetzt, damit die Arbeitsprozesse (spawn importieren dieses Skript neu) dasselbe Modell laden
_args = [a for a in sys.argv[1:] if a.endswith(".pkl")]
if _args:
    crn.MODELS = {NEW: Path(_args[0]).resolve()}


def summarize(drawn):
    table = crn.summarize(drawn)
    extra = []
    for (label, condition, g), part in drawn.groupby(["modell", "bedingung", "gruppe"], sort=False):
        wide = part.pivot_table(index=["seed", "case", "visit", "k"], columns="rule", values="rest")
        wrong = []
        for _, t in wide.groupby(level=[0, 1, 2]):
            d = t[False].to_numpy() - t[True].to_numpy()
            se300 = d.std(ddof=1) / np.sqrt(300)
            wrong.append(norm.cdf(-d.mean() / se300) if se300 > 0 else float(d.mean() <= 0))
        extra.append({"Modell": label, "Bedingung": condition, "Gruppe": g,
                      "Restzeit Regel-Option (d)": part.loc[part.rule, "rest"].mean(),
                      "Restzeit andere Option (d)": part.loc[~part.rule, "rest"].mean(),
                      "erwartet falsch bei 300": float(np.mean(wrong))})
    return table.merge(pd.DataFrame(extra), on=["Modell", "Bedingung", "Gruppe"])


def paired(drawn):
    rows = []
    event = drawn[drawn.bedingung == "event"]
    per_state = (event.groupby(["modell", "gruppe", "seed", "case", "visit", "rule"]).rest.mean().unstack("rule"))
    advantage = (per_state[False] - per_state[True]).unstack("modell").dropna()
    for g, part in advantage.groupby(level="gruppe", sort=False):
        d = part[NEW] - part[OLD]
        rows.append({"Gruppe": g, "Zustände": len(d), f"Vorteil Regel-Option {OLD} (d)": part[OLD].mean(),
                     f"Vorteil Regel-Option {NEW} (d)": part[NEW].mean(), "neu − alt (d)": d.mean(),
                     "SE": d.std(ddof=1) / np.sqrt(len(d)), "Korrelation je Zustand": np.corrcoef(part[OLD], part[NEW])[0, 1]})
    return pd.DataFrame(rows)


def main():
    assert _args, "Aufruf: wirkung.py <Hülle.pkl>"
    OUT.mkdir(parents=True, exist_ok=True)
    states = crn.base.pick().to_dict("records")
    order = np.random.default_rng(2).permutation(len(states))  # wie crn_schluessel: gleiche Aufteilung, gleiche Saaten je Zustand
    with ProcessPoolExecutor(6, mp_context=multiprocessing.get_context("spawn")) as pool:
        drawn = pd.concat(list(pool.map(crn.work, [(w, [states[i] for i in order[w::6]]) for w in range(6)])), ignore_index=True)
    drawn.to_pickle(OUT/"ziehungen.pkl")
    old = pd.read_pickle(crn.OUT/"ziehungen.pkl")
    both = pd.concat([old[old.modell == OLD], drawn] + [pd.read_pickle(x) for x in EXTRA], ignore_index=True)
    table, pairs = summarize(both), paired(both)
    table.to_csv(OUT/"wirkung.csv", index=False)
    pairs.to_csv(OUT/"wirkung_gepaart.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 30):
        text = table.round(3).to_string(index=False)
        text_pairs = pairs.round(3).to_string(index=False)
    (OUT/"wirkung.md").write_text(f"# Suffix-Verlust: Wirkung an den Wahlpunkten (Diagnose, 2026-11-19)\n\nModell neu: {_args[0]}\n\n```\n{text}\n```\n\n"
                                  f"## Gepaart je Zustand (Bedingung event)\n\n```\n{text_pairs}\n```\n")
    print(text)
    print(text_pairs)
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
