"""Wie entscheidet der reine Forest-Arm online? (2026-11-24, nach cf_konformanz in online_konformanz.py, Nutzer: „teste parallel mal, wie
gut der Causal Forest bei der Konformitätsaufgabe performt“.)

Je Arm (choices.csv der Online-Läufe, bewertete Fälle, erster Besuch je Fall und Punkt) der Anteil „andere Option“ (nicht Regel-Option) je
Wahlpunkt und Teilgruppe der Effektregeln (online_konformanz.Orakel.gruppe, Dauer seit Fallbeginn zum Entscheidungszeitpunkt), dazu die
Ziel-Wahl der besten Kombination mit der Regel als Basis (wahrheit_f_regel.md: umgeschaltet nachlegen ≥ 1,71 d, konzept ≥ 11k,
nach Vervollständigung alle) und des Orakels (Wahrheit mit der Bank sonst, wahrheit_effekte_v3.csv).
Ausgabe forest_wahlen.md / .csv hier.
Aufruf im Ordner process-simulation-engine: .venv/bin/python <dieses Skript> [Arme, Komma]
"""
from pathlib import Path
import sys
import warnings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
K = ROOT/"prescriptive/scripts/results_100k/kuenstliche_effekte_20261120"
ARME = (sys.argv[1] if len(sys.argv) > 1 else "cf_konformanz,orakel,regel,konformanz_ol300").split(",")
del sys.argv[1:]
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(K))
import online_konformanz as ok  # noqa: E402
import pandas as pd  # noqa: E402
from simulation.case_engine_lifecycle_aware import to_seconds  # noqa: E402

CASE, TIME = "case:concept:name", "time:timestamp"
KOMBI = {("angebot_nachlegen", "dauer_ab_tage=0.84, nur_erster_besuch=True | dauer_ab_tage=1.71, nur_erster_besuch=True"),
         ("annehmen_bei_konzept", "betrag_ab=11000"), ("annehmen_nach_vervollstaendigung", "keine"),
         ("annehmen_nach_vervollstaendigung", "ziel_in=['Car', 'Existing loan takeover']")}


def main():
    orakel = ok.Orakel()
    zeilen = []
    for arm in ARME:
        for ordner in sorted(ok.OUT.glob(f"{arm}_seed*")):
            c = pd.read_csv(ordner/"choices.csv").drop_duplicates(["case", "point"])  # erster Besuch je Fall und Punkt
            ev = pd.read_pickle(ordner/"events.pkl").sort_values([CASE, TIME], kind="stable")
            erste = ev.groupby(CASE, sort=False).first()
            for z in c.itertuples():
                if z.point not in orakel.PUNKTE or z.case not in erste.index:
                    continue
                f = erste.loc[z.case]
                dauer = (z.time - to_seconds(f[TIME])) / 86400
                g = orakel.gruppe({"case:RequestedAmount": f["case:RequestedAmount"], "case:LoanGoal": f["case:LoanGoal"],
                                   "case:ApplicationType": f["case:ApplicationType"], "dauer_tage": dauer}, z.point)
                regel, andere, _ = orakel.PUNKTE[z.point]
                d = orakel.delta.get((z.point, g))
                zeilen.append({"arm": arm, "seed": ordner.name, "punkt": z.point, "gruppe": g, "andere": z.choice == andere,
                               "kombi_andere": (z.point, g) in KOMBI, "orakel_andere": bool(d is not None and d < 0)})
    d = pd.DataFrame(zeilen)
    d.to_csv(HERE/"forest_wahlen.csv", index=False)
    breit = d.pivot_table(index=["punkt", "gruppe"], columns="arm", values="andere", aggfunc="mean").round(3)
    ziel = d.groupby(["punkt", "gruppe"]).agg(n_je_arm=("andere", lambda s: round(len(s) / len(ARME))),
                                              kombi=("kombi_andere", "first"), orakel_bank=("orakel_andere", "first"))
    tab = ziel.join(breit[[a for a in ARME if a in breit.columns]])
    d["wie_kombi"] = d.andere == d.kombi_andere
    treffer = d.groupby("arm").wie_kombi.mean().round(3)
    with pd.option_context("display.width", 250, "display.max_colwidth", 90):
        text = (f"# Online-Wahlen je Teilgruppe: Anteil „andere Option“ (nicht Regel), erste Besuche, bewertete Fälle, 12 Seeds\n\n"
                f"kombi / orakel_bank = Ziel-Wahl (True = andere Option) der besten Kombination mit Regel sonst (wahrheit_f_regel) bzw. des "
                f"Orakels (Wahrheit mit Bank sonst).\n\n```\n{tab.to_string()}\n```\n\nAnteil Entscheidungen wie die Kombination:\n"
                f"```\n{treffer.to_string()}\n```\n")
    (HERE/"forest_wahlen.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
