"""Entscheidet das PSP online so wie der Causal Forest? (2026-11-23, Frage des Nutzers: „Vielleicht entscheidet dort das spätere PSP wie
Causal Forest“ – Voraussetzung dafür, dass der Forest-Rollout zur tatsächlich ausgeführten Politik passt.)

Für jede Online-Entscheidung eines Arms (erste Besuche je Fall und Punkt, bewertete Fälle) wird an genau demselben Zustand (Events des
Falls bis zum Entscheidungszeitpunkt) die Wahl des Conformance-Forests models/cate_model_cf_konformanz_simv3.pkl berechnet und mit der
Wahl des Arms verglichen, je Wahlpunkt und Teilgruppe (wie online_konformanz.Orakel.gruppe) und nach Abstand der PSP-Schätzungen
(|Differenz der erwarteten Abweichungen| aus entscheidungen.csv).
Ausgabe uebereinstimmung_forest_<arme>.md / .csv hier.
Aufruf im Ordner process-simulation-engine: .venv/bin/python <dieses Skript> [Arme, Komma; Standard konformanz_ol300]
"""
from pathlib import Path
import sys
import warnings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
K = ROOT/"prescriptive/scripts/results_100k/kuenstliche_effekte_20261120"
FOREST = ROOT/"prescriptive/models/cate_model_cf_konformanz_simv3.pkl"
ARME = (sys.argv[1] if len(sys.argv) > 1 else "konformanz_ol300").split(",")
del sys.argv[1:]
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(K))
import online_konformanz as ok  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from simulation.case_engine_lifecycle_aware import to_seconds  # noqa: E402

CASE, TIME = "case:concept:name", "time:timestamp"


def main():
    import interventions
    from prescriptive.policies.ev_policy.rollout import ForestRollout
    points = {p.name: p for p in interventions.choice_points("bpic17")}
    ro = ForestRollout(str(FOREST), points)
    orakel = ok.Orakel()
    zeilen = []
    for arm in ARME:
        for ordner in sorted(ok.OUT.glob(f"{arm}_seed*")):
            if not (ordner/"entscheidungen.csv").exists():
                continue
            e = pd.read_csv(ordner/"entscheidungen.csv").drop_duplicates(["case", "point"])
            ev = pd.read_pickle(ordner/"events.pkl").sort_values([CASE, TIME], kind="stable")
            ev["_s"] = [to_seconds(t) for t in ev[TIME]]
            faelle = {c: t for c, t in ev.groupby(CASE, sort=False)}
            je_punkt = {}
            for _, z in e.iterrows():
                if z.case not in faelle or z.point not in ro.punkte or not isinstance(z.wahl, str):
                    continue
                t = faelle[z.case]
                prefix_ev = t[t["_s"] <= z.time + 1e-6].drop(columns="_s").to_dict("records")
                if not prefix_ev:
                    continue
                prefix = ro._prefix(prefix_ev)
                regel, andere, _ = orakel.PUNKTE[z.point]
                schaetz = {o: z.get(f"abweichungen {o}") for o in (regel, andere)}
                dauer = (z.time - t["_s"].iloc[0]) / 86400
                g = orakel.gruppe({"case:RequestedAmount": t["case:RequestedAmount"].iloc[0], "case:LoanGoal": t["case:LoanGoal"].iloc[0],
                                   "case:ApplicationType": t["case:ApplicationType"].iloc[0], "dauer_tage": dauer}, z.point)
                je_punkt.setdefault(z.point, []).append((prefix.row(z.time), z.wahl, g, abs(schaetz[regel] - schaetz[andere])))
            for name, liste in je_punkt.items():
                rows, wahl, gruppe, abstand = zip(*liste)
                werte = np.column_stack([np.zeros(len(rows)), ro.forest.effects(ro.punkte[name], list(rows))])
                optionen = list(ro.forest.options[name])
                regel, andere, _ = orakel.PUNKTE[name]
                for w, wa, g, a in zip(werte, wahl, gruppe, abstand):
                    fw = optionen[int(np.argmax(w))]
                    zeilen.append({"arm": arm, "seed": ordner.name, "punkt": name, "gruppe": g, "psp_andere": wa == andere,
                                   "forest_andere": fw == andere, "gleich": wa == fw, "abstand": a})
    d = pd.DataFrame(zeilen)
    d["abstand_klasse"] = pd.cut(d.abstand, [-1e-9, 0.02, 0.05, 0.1, 0.2, 10], labels=["<0,02", "0,02–0,05", "0,05–0,1", "0,1–0,2", "≥0,2"])
    je_gruppe = d.groupby(["arm", "punkt", "gruppe"]).agg(n=("gleich", "size"), psp_andere=("psp_andere", "mean"),
                                                         forest_andere=("forest_andere", "mean"), gleich=("gleich", "mean")).round(3)
    je_punkt = d.groupby(["arm", "punkt"]).agg(n=("gleich", "size"), psp_andere=("psp_andere", "mean"),
                                               forest_andere=("forest_andere", "mean"), gleich=("gleich", "mean")).round(3)
    je_abstand = d.groupby(["arm", "abstand_klasse"], observed=True).agg(n=("gleich", "size"), gleich=("gleich", "mean")).round(3)
    gesamt = d.groupby("arm").agg(n=("gleich", "size"), gleich=("gleich", "mean")).round(3)
    d.to_csv(HERE/f"uebereinstimmung_forest_{'_'.join(ARME)}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        text = (f"# Übereinstimmung PSP-Wahl mit Forest-Wahl am selben Online-Zustand (erste Besuche, bewertete Fälle)\n\n"
                f"Forest: {FOREST.name}. psp_andere / forest_andere = Anteil „andere Option“ (nicht Regel); gleich = gleiche Wahl.\n\n"
                f"## Gesamt\n```\n{gesamt.to_string()}\n```\n\n## Je Punkt\n```\n{je_punkt.to_string()}\n```\n\n"
                f"## Je Teilgruppe\n```\n{je_gruppe.to_string()}\n```\n\n"
                f"## Nach Abstand der PSP-Schätzungen (|Δ erwartete Abweichungen|)\n```\n{je_abstand.to_string()}\n```\n")
    (HERE/f"uebereinstimmung_forest_{'_'.join(ARME)}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
