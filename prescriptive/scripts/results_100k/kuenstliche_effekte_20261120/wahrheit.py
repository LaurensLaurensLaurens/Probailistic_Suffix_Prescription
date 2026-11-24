"""Wahre Effekte je Teilgruppe im Simulator mit künstlichen Effekten (2026-11-20).

Je Wahlpunkt zwei Läufe mit gleicher Saat: die Policy erzwingt an diesem Punkt die Regel-Option bzw. die andere Option (nachlegen nur
beim 1. Besuch, sonst bei jedem Besuch), an den anderen Punkten die Bank. Vorteil der Regel-Option je Fall = Durchlaufzeit mit der
anderen Option − mit der Regel-Option (positiv = Regel-Option besser), getrennt nach Teilgruppe (Bedingung der Effektregel trifft zu
oder nicht). Mehrere Saaten gegen das Paarungsrauschen (zufällige Zuteilung). Ohne Effekte (Effektdatei "-") zum Vergleich mit der
bisherigen Simulator-Wahrheit.
Ausgabe wahrheit_<name>.md / .csv. Aufruf im Ordner process-simulation-engine:
  .venv/bin/python -u <dieses Skript> [Fälle je Lauf] [Saaten, z. B. 11-13] [Effektdatei oder -]
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/"prescriptive/policies/ev_policy_event_costs/results/d_oe_zustand_20261115"
OUT = Path(__file__).resolve().parent
CASES = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
_seeds = sys.argv[2] if len(sys.argv) > 2 else "11-13"
SEEDS = list(range(int(_seeds.split("-")[0]), int(_seeds.split("-")[1]) + 1)) if "-" in _seeds else [int(s) for s in _seeds.split(",")]
EFFEKTE = sys.argv[3] if len(sys.argv) > 3 else ""
KONFORMANZ = os.environ.get("KONFORMANZ", "1") == "1"  # Conformance je Fall mitmessen (pm4py-Alignments, angepasstes To-Be-Modell)
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PSP_CATALOG"] = str(SOURCE/"catalog.yaml")
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT))
from prescriptive.helpers.datasets import add_paths  # noqa: E402
add_paths()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Wahlpunkt -> (Regel-Option, andere Option) wie im Katalog (test.action), nur 1. Besuch?
PUNKTE = {"annehmen_bei_konzept": ("A_Accepted", "W_Complete application|start", False),
          "annehmen_nach_vervollstaendigung": ("A_Accepted", "W_Complete application|resume", False),
          "angebot_nachlegen": ("O_Sent (mail and online)", "O_Create Offer", True),
          "handle_leads": ("W_Handle leads|withdraw", "W_Handle leads|start", False)}


WICHTIG = ["W_Call incomplete files|schedule", "A_Incomplete|complete", "A_Cancelled|complete", "O_Accepted|complete", "A_Denied|complete", "O_Create Offer|complete"]


class Erzwingen:
    def __init__(self, punkt, option, erster):
        self.punkt, self.option, self.erster = punkt, option, erster

    def choose(self, point, case, options, time, engine):
        if point.name != self.punkt or self.option not in options:
            return None
        if self.erster:
            visits = sum(1 for e in case.prefix if e.get("concept:name") == point.activity and e.get("lifecycle:transition", "complete") == point.lifecycle)
            if visits > 1:
                return None
        return self.option


def simulator():
    import interventions
    interventions.CANDIDATES_FILE = str(SOURCE/"catalog.yaml")
    from prescriptive.helpers.datasets import load_dataset
    from preprocessing import preprocess
    from simulation.dataset_simulator import Simulator
    raw, _, _ = load_dataset("bpic17")
    return Simulator("bpic17", preprocess("bpic17", raw, verbose=False))


def run(task):
    punkt, welche, seed = task
    import kuenstliche_effekte as ke
    effekte = None
    if EFFEKTE != "-":
        effekte = ke.aktivieren(ke.laden(EFFEKTE or None))
    regel, andere, erster = PUNKTE[punkt]
    events, choices = simulator().run(CASES, policy=Erzwingen(punkt, regel if welche == "regel" else andere, erster), seed=seed)
    t = events.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"])
    dlz = ((t["max"] - t["min"]).dt.total_seconds() / 86400).rename("dlz")
    attrs = events.groupby("case:concept:name")[["case:RequestedAmount", "case:LoanGoal", "case:ApplicationType"]].first()
    tok = events["concept:name"] + "|" + events["lifecycle:transition"]
    zaehl = pd.crosstab(events["case:concept:name"], tok).reindex(columns=WICHTIG, fill_value=0)
    zaehl.columns = [f"n {w.split('|')[0]}" for w in WICHTIG]
    attrs = attrs.join(zaehl)
    if KONFORMANZ:  # Conformance-Abweichungen je Fall gegen BPIC17_ToBe_angepasst (wie conformance_probe.py)
        import conformance_probe as cp
        erg = cp.align(cp.projektion(events))
        attrs = attrs.join(pd.Series({c: cost // 10000 for c, _, cost, _ in erg}, name="n Abweichungen"))
    gefragt = choices[(choices.point == punkt) & choices.choice.notna()]
    erste = gefragt.groupby("case").time.min().rename("zeit_wahl")
    beginn = t["min"].map(lambda x: x.timestamp() if hasattr(x, "timestamp") else float(x))
    frame = pd.concat([dlz, attrs], axis=1).join(erste, how="inner")
    frame["dauer_tage"] = (frame.zeit_wahl - beginn.reindex(frame.index).to_numpy()) / 86400
    frame["punkt"], frame["welche"], frame["seed"] = punkt, welche, seed
    return frame.reset_index().rename(columns={"index": "case", "case:concept:name": "case"})


def gruppe(row, effekte, punkt):
    """Teilgruppe des Falls an diesem Punkt: die Bedingungen der Effektregeln, die auf ihn zutreffen (bei beiden Optionen), als Text;
    „keine“, wenn keine Regel zutrifft."""
    import kuenstliche_effekte as ke
    attrs = {"case:RequestedAmount": row["case:RequestedAmount"], "case:LoanGoal": row["case:LoanGoal"],
             "case:ApplicationType": row["case:ApplicationType"]}
    regeln = [r for regeln in ((effekte.get("punkte") or {}).get(punkt) or {}).values() for r in regeln]
    texte = sorted({", ".join(f"{k}={v}" for k, v in r.items() if k in ke.BEDINGUNGEN) for r in regeln
                    if ke.trifft_zu(r, attrs, row["dauer_tage"], 1)})
    return " | ".join(texte) if texte else "keine"


def main():
    tasks = [(p, w, s) for p in PUNKTE for w in ("regel", "andere") for s in SEEDS]
    with ProcessPoolExecutor(min(12, len(tasks)), mp_context=multiprocessing.get_context("spawn")) as pool:
        frames = list(pool.map(run, tasks))
    alle = pd.concat(frames, ignore_index=True)
    import kuenstliche_effekte as ke
    effekte = ke.laden(EFFEKTE or None) if EFFEKTE != "-" else ke.laden()
    name = "ohne" if EFFEKTE == "-" else Path(effekte["pfad"]).stem
    rows = []
    for punkt in PUNKTE:
        teil = alle[alle.punkt == punkt]
        r = teil[teil.welche == "regel"].set_index(["seed", "case"])
        a = teil[teil.welche == "andere"].set_index(["seed", "case"])
        spalten = [f"n {w.split('|')[0]}" for w in WICHTIG] + (["n Abweichungen"] if KONFORMANZ else [])
        both = r.join(a[["dlz"] + spalten], rsuffix="_andere", how="inner")
        both["vorteil"] = both.dlz_andere - both.dlz
        both["gruppe"] = both.apply(lambda row: gruppe(row, effekte, punkt), axis=1)
        for g, part in [("alle", both)] + [(v, both[both.gruppe == v]) for v in sorted(both.gruppe.unique())]:
            je_seed = part.groupby(level="seed").vorteil.mean()
            rows.append({"Punkt": punkt, "Gruppe": g, "Fälle": len(part), "Anteil": len(part) / max(1, len(both)),
                         "Vorteil Regel-Option (d)": part.vorteil.mean(),
                         "SE über Saaten": je_seed.std(ddof=1) / np.sqrt(len(je_seed)) if len(je_seed) > 1 else float("nan"),
                         "je Saat": " / ".join(f"{v:+.2f}" for v in je_seed),
                         **{f"{s} andere−Regel": (part[s + "_andere"] - part[s]).mean() for s in spalten}})
    out = pd.DataFrame(rows)
    out.to_csv(OUT/f"wahrheit_{name}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        text = out.round(3).to_string(index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        text = out.drop(columns=["je Saat"]).round(3).to_string(index=False)
    (OUT/f"wahrheit_{name}.md").write_text(f"# Wahre Effekte je Teilgruppe ({name}, {CASES} Fälle je Lauf, Saaten {SEEDS})\n\n"
                                          f"Vorteil Regel-Option = Durchlaufzeit andere − Regel-Option (positiv: Regel-Option besser)\n\n```\n{text}\n```\n")
    print(text)
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
