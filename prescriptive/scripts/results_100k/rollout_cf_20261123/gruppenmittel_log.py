"""Gruppenmittel der Abweichungen je Option im Trainings-Log (2026-11-23, Nutzer: „sind die Befunde auch alle bezogen auf mein aktuelles
Modell?“). Wie die Prüfung 23:00 (dort auf log_effekte_v3), hier auf dem Log, auf dem k3x4ol trainiert wurde (log_effekte_v3_x4):
je erstem Besuch von konzept / nach V. / leads die Option der Bank (nächstes Event nach dem Auslöser) und die Abweichungen des Falls
(Alignment wie conformance_probe.py gegen BPIC17_ToBe_angepasst), je Teilgruppe; dazu Anteil der Optionen (Confounding-Prüfung).
Ausgabe gruppenmittel_<log>.md hier. Aufruf im Ordner process-simulation-engine: .venv/bin/python -u <dieses Skript> [log-Ordner] [Prozesse]
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import time
import warnings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
K = ROOT/"prescriptive/scripts/results_100k/kuenstliche_effekte_20261120"
ORDNER = sys.argv[1] if len(sys.argv) > 1 else "log_effekte_v3_x4"
PROZESSE = int(sys.argv[2]) if len(sys.argv) > 2 else 12
os.environ["PSP_CATALOG"] = str(ROOT/"prescriptive/data/catalog_neue_punkte_zuteilung_random.yaml")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(K))
from prescriptive.helpers.datasets import add_paths  # noqa: E402
add_paths()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CASE = "case:concept:name"
PUNKTE = {"annehmen_bei_konzept": ("A_Accepted", "W_Complete application|start"),
          "annehmen_nach_vervollstaendigung": ("A_Accepted", "W_Complete application|resume"),
          "handle_leads": ("W_Handle leads|withdraw", "W_Handle leads|start")}


def align_teil(teil):
    import conformance_probe as cp
    return [(c, int(cost // 10000)) for c, _, cost, _ in cp.align(teil)]


def main():
    import conformance_probe as cp
    import interventions
    from interventions import option_token
    start = time.time()
    raw = pd.read_pickle(K/ORDNER/"log_raw.pkl")
    proj = cp.projektion(raw)
    faelle = proj[CASE].drop_duplicates().tolist()
    teile = [proj[proj[CASE].isin(set(faelle[i::PROZESSE * 4]))] for i in range(PROZESSE * 4)]
    with ProcessPoolExecutor(PROZESSE, mp_context=multiprocessing.get_context("spawn")) as pool:
        abw = dict(x for part in pool.map(align_teil, teile) for x in part)
    print(f"Alignments: {len(abw):,} Fälle ({time.time() - start:.0f} s)", flush=True)
    pts = {p.name: p for p in interventions.choice_points("bpic17")}
    raw = raw.sort_values([CASE, "time:timestamp"], kind="stable")
    raw["tok"] = raw["concept:name"] + "|" + raw["lifecycle:transition"].fillna("complete")
    z = []
    for c, t in raw.groupby(CASE, sort=False):
        toks = t.tok.tolist()
        amt, goal = t["case:RequestedAmount"].iloc[0], t["case:LoanGoal"].iloc[0]
        for name in PUNKTE:
            p = pts[name]
            trig = f"{p.activity}|{p.lifecycle}"
            if trig in toks[:-1]:
                nxt = toks[toks.index(trig) + 1]
                opts = {option_token(o): o for o in p.options}
                if nxt in opts:
                    z.append((name, amt, goal, opts[nxt], abw.get(str(c), np.nan)))
    d = pd.DataFrame(z, columns=["punkt", "betrag", "ziel", "option", "abw"])
    d["gruppe"] = np.where(d.punkt == "annehmen_nach_vervollstaendigung",
                           np.where(d.ziel.isin(["Car", "Existing loan takeover"]), "Car/Übernahme", "sonst"),
                           np.where(d.betrag < 11000, "< 11k", "≥ 11k"))
    zeilen = []
    for (punkt, gruppe), t in d.groupby(["punkt", "gruppe"]):
        regel, andere = PUNKTE[punkt]
        m = t.groupby("option").abw.agg(["size", "mean", "sem"])
        zeilen.append({"Punkt": punkt, "Gruppe": gruppe, "n Regel-Option": int(m.loc[regel, "size"]), "n andere": int(m.loc[andere, "size"]),
                       "Anteil andere (Bank)": round(float((t.option == andere).mean()), 3),
                       "Abw. Regel-Option": round(float(m.loc[regel, "mean"]), 3), "Abw. andere": round(float(m.loc[andere, "mean"]), 3),
                       "andere − Regel": round(float(m.loc[andere, "mean"] - m.loc[regel, "mean"]), 3),
                       "SE": round(float(np.hypot(m.loc[regel, "sem"], m.loc[andere, "sem"])), 3)})
    tab = pd.DataFrame(zeilen)
    text = (f"# Gruppenmittel der Abweichungen je Option im Log {ORDNER} ({time.strftime('%Y-%m-%d %H:%M')})\n\n"
            f"Erste Besuche, Option = Wahl der Bank, Abweichungen des ganzen Falls (BPIC17_ToBe_angepasst). {len(abw):,} Fälle.\n\n"
            f"```\n{tab.to_string(index=False)}\n```\n")
    (HERE/f"gruppenmittel_{ORDNER}.md").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
