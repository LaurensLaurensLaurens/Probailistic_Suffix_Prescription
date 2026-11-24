"""Wo weichen die Online-Policies (Aufbau F, 12 Seeds) von der festen Regel ab, und was kosten die Abweichungen? (2026-11-19)

Je Arm und Wahlpunkt: echte Wahlen (2 Optionen) je Fall, Anteil Abweichungen von der Regel-Option. Bei angebot_nachlegen zählt nur der
1. Besuch (first_only: spätere Besuche entscheidet die Bank). Zwei Kostenschätzungen:
  erwartet  = Abweichungen je Fall × wahrer mittlerer Nachteil der Nicht-Regel-Option im Simulator (aus randomisierten Läufen, s. WAHR);
              so viel kosten die Abweichungen, wenn die Policy die Fälle dafür ZUFÄLLIG auswählt (kein Signal).
  beobachtet = gepaarte Differenz je Fall (Arm − Regel, gleicher Seed, gleiche Fall-ID), getrennt nach Fällen mit und ohne Abweichung.
              Fälle ohne Abweichung messen die Entkopplung der gepaarten Läufe (Ressourcen, Zufallsstrom), Fälle mit Abweichung die Kosten.
Liegt der beobachtete Nachteil der abweichenden Fälle beim wahren Mittel, wählt die Policy die Abweichungen ohne Signal aus.
dlz_unter_regel: Durchlaufzeit derselben Fälle im Regel-Lauf (sind die abweichenden Fälle die langen?). se über Fälle, se_seeds über Seed-Mittel.
Ausgabe abweichungen_regel.md / .csv. Aufruf: process-simulation-engine/.venv/bin/python abweichungen_regel.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import f_zwoelf_auswerten as fz  # noqa: E402

ARMS = ["c_fest100", "c_fest300", "zztf1_fest100", "tl_fest300", "tl_fest500", "tlks_fest300", "tlko_fest300", "tlrct_fest300", "tlcrn_fest300", "tlsv_fest300", "tlsvk_fest300", "tlsvp_fest300", "tlsvh_fest300", "olsv_fest300", "plsv_fest300", "olsvcf_fest300"]
POINTS = ["annehmen_bei_konzept", "annehmen_nach_vervollstaendigung", "angebot_nachlegen", "handle_leads"]
KURZ = {"annehmen_bei_konzept": "konzept", "annehmen_nach_vervollstaendigung": "nach V.", "angebot_nachlegen": "nachlegen (1.)",
        "handle_leads": "leads"}
# Wahrer Nachteil der Nicht-Regel-Option in Tagen (Simulator, randomisiert; befunde/psp-model-comparison-1116):
# konzept / nach V. / leads aus dem Simulator-RCT, nachlegen 1. Besuch mit Bank-Fortsetzung (nur 1. Besuch randomisiert, 7.647 Fälle).
WAHR = {"annehmen_bei_konzept": 2.79, "annehmen_nach_vervollstaendigung": 2.40, "angebot_nachlegen": 1.55, "handle_leads": 1.32}


def choices(path):
    c = pd.read_csv(path/"choices.csv")
    c["besuch"] = c.groupby(["case", "point"]).cumcount() + 1
    c = c[(c.options == 2) & c.choice.notna()]
    c = c[(c.point != "angebot_nachlegen") | (c.besuch == 1)]
    c["abweichung"] = [ch != fz.REGEL.get(p) for p, ch in zip(c.point, c.choice)]
    return c


def markdown(frame, index=True):
    """Markdown-Tabelle ohne tabulate (im venv nicht installiert)."""
    frame = frame.reset_index() if index else frame
    cells = [[str(c) for c in frame.columns]] + [["–" if isinstance(v, float) and np.isnan(v) else str(v) for v in row] for row in frame.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cells[0]) + " |", "|" + "---|" * len(cells[0])] + ["| " + " | ".join(r) + " |" for r in cells[1:]])


def se(x):
    x = pd.Series(x).dropna()
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def main():
    point_rows, case_rows, arm_rows = [], [], []
    for arm in ARMS:
        seeds = [s for s in range(12) if fz.folder(arm, s) is not None and fz.folder("regel", s) is not None]
        if not seeds:
            continue
        per_case = []
        for seed in seeds:
            path = fz.folder(arm, seed)
            c = choices(path)
            costs = pd.read_csv(path/"case_costs.csv", index_col=0).cycle_days
            regel = pd.read_csv(fz.folder("regel", seed)/"case_costs.csv", index_col=0).cycle_days
            for point in POINTS:
                p = c[c.point == point]
                point_rows.append({"arm": arm, "seed": seed, "punkt": point, "wahlen_je_fall": len(p) / len(costs),
                                   "abweichungen_je_fall": p.abweichung.sum() / len(costs), "anteil_abweichung": p.abweichung.mean() if len(p) else np.nan})
            dev = c[c.abweichung].groupby("case").point.agg(lambda s: "+".join(sorted(set(s))))
            both = costs.index.intersection(regel.index)
            frame = pd.DataFrame({"seed": seed, "differenz": costs[both] - regel[both], "regel_dlz": regel[both]})
            frame["abweichung_an"] = dev.reindex(both).fillna("")
            per_case.append(frame)
        cases = pd.concat(per_case)
        points = pd.DataFrame([r for r in point_rows if r["arm"] == arm])
        expected = sum(points[points.punkt == p].abweichungen_je_fall.mean() * WAHR[p] for p in POINTS)
        seed_diff = cases.groupby("seed").differenz.mean()
        arm_rows.append({"arm": arm, "seeds": len(seeds), "nachteil_gegen_regel": seed_diff.mean(), "se": se(seed_diff),
                         "erwartet_bei_zufaelliger_auswahl": expected,
                         "ohne_abweichung": cases[cases.abweichung_an == ""].differenz.mean(),
                         "ohne_abweichung_se": se(cases[cases.abweichung_an == ""].differenz)})
        for group, part in cases.groupby(cases.abweichung_an.where(cases.abweichung_an.str.count(r"\+") == 0, "mehrere")):
            case_rows.append({"arm": arm, "abweichung_an": group or "keine", "faelle": len(part), "anteil_faelle": len(part) / len(cases),
                              "differenz_je_fall": part.differenz.mean(), "se": se(part.differenz),
                              "se_seeds": se(part.groupby("seed").differenz.mean()) if part.seed.nunique() > 1 else np.nan,
                              "dlz_unter_regel": part.regel_dlz.mean(),
                              "wahr_bei_zufall": WAHR.get(group, np.nan), "beitrag_je_fall_insgesamt": part.differenz.sum() / len(cases)})
    points = pd.DataFrame(point_rows).groupby(["arm", "punkt"], sort=False)[["wahlen_je_fall", "abweichungen_je_fall", "anteil_abweichung"]].mean().reset_index()
    arms, groups = pd.DataFrame(arm_rows), pd.DataFrame(case_rows)
    points.to_csv(OUT/"abweichungen_regel_punkte.csv", index=False)
    groups.to_csv(OUT/"abweichungen_regel_faelle.csv", index=False)
    arms.to_csv(OUT/"abweichungen_regel.csv", index=False)

    share = points.pivot(index="arm", columns="punkt", values="anteil_abweichung").reindex(index=arms.arm, columns=POINTS).rename(columns=KURZ)
    lines = ["# Abweichungen von der festen Regel in Aufbau F (2026-11-19)", "",
             "Regel: konzept → A_Accepted, nach V. → A_Accepted, nachlegen → O_Sent (kein Zusatzangebot), leads → withdraw. "
             "Differenz = Durchlaufzeit Arm − Regel in Tagen (positiv = Arm langsamer), gepaart je Seed und Fall-ID. "
             "Wahrer Nachteil der Nicht-Regel-Option (Simulator, randomisiert): " + ", ".join(f"{KURZ[p]} {v:+.2f}" for p, v in WAHR.items()) + " d.", "",
             "## Anteil Abweichungen je Punkt (an echten Wahlen; nachlegen nur 1. Besuch)", "", markdown(share.map(lambda v: f"{v:.0%}")), "",
             "## Nachteil gegen Regel: beobachtet vs. erwartet bei zufälliger Auswahl der abweichenden Fälle", "",
             markdown(arms.round(2), index=False), "",
             "## Je Fallgruppe (nach Punkt der Abweichung)", "", markdown(groups.round(2), index=False), ""]
    (OUT/"abweichungen_regel.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
