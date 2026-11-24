"""Auswertung der nachgezogenen Suffixe aus d_ohne_ziehungen.py: woraus besteht die gezogene Restzeit je Option?"""
from pathlib import Path

import numpy as np
import pandas as pd

O = Path(__file__).resolve().parent
d = pd.read_pickle(O/"d_ohne_ziehungen/ziehungen.pkl")
z = pd.read_csv(O/"d_ohne_ziehungen/zustaende.csv")
RULE = {"annehmen_bei_konzept": "A_Accepted", "annehmen_nach_vervollstaendigung": "A_Accepted",
        "angebot_nachlegen": "O_Sent (mail and online)", "handle_leads": "W_Handle leads|withdraw"}
ORDER = ["annehmen_bei_konzept", "annehmen_nach_vervollstaendigung", "nachlegen 1. Besuch", "nachlegen ab 2. Besuch", "handle_leads"]
ENDS = ["A_Pending", "A_Cancelled", "A_Denied", "ohne"]
KEY = ["seed", "case", "point", "visit"]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def group(frame):
    return np.where(frame.point == "angebot_nachlegen", np.where(frame.visit == 1, "nachlegen 1. Besuch", "nachlegen ab 2. Besuch"), frame.point)


d["gruppe"], z["gruppe"] = group(d), group(z)
errors = z["sampling_error"].notna().sum() if "sampling_error" in z else 0
print("Zustände:", len(z), "| Prefix korrekt wiederhergestellt:", int(z.prefix_ok.sum()), "| Ziehungsfehler:", int(errors))

# Kontrolle gegen die online gespeicherten Schätzungen
z["rule_index"] = [0 if RULE[p] == o else 1 for p, o in zip(z.point, z.option_0)]
z["online_diff"] = np.where(z.rule_index == 0, z.online_value_0-z.online_value_1, z.online_value_1-z.online_value_0)
m = d.groupby(KEY+["rule_option"]).total.mean().unstack()
m["redraw_diff"] = m[False]-m[True]  # > 0: Regel-Option kürzer
check = z.merge(m.reset_index(), on=KEY)
print("\nKontrolle: mittlere Differenz (Regel-Option kürzer, Tage), online gespeichert und nachgezogen:")
print(check.groupby("gruppe")[["online_diff", "redraw_diff"]].agg(["size", "mean"]).round(3).to_string())


def block(g):
    out = {}
    for flag, name in ((True, "Regel"), (False, "andere")):
        p = g[g.rule_option == flag]
        out[f"Restzeit {name}"] = p.total.mean()
        out[f"1. Abstand {name}"] = p.first_gap.mean()
        out[f"Rest nach Option {name}"] = p.rest.mean()
        out[f"Events {name}"] = p.events.mean()
        out[f"Angebote danach {name}"] = p.offers.mean()
        for e in ENDS:
            out[f"Ende {e} {name}"] = (p.end == e).mean()
        for e in ("A_Pending", "A_Cancelled"):
            out[f"Restzeit | {e} {name}"] = p.total[p.end == e].mean()
        out[f"längster Abstand {name}"] = p.max_gap.mean()
        out[f"Zeit in Abständen > 5 d {name}"] = p.sum_gaps_over_5d.mean()
    a, b = g[g.rule_option], g[~g.rule_option]
    out["Gewinn Regel-Option (Tage)"] = b.total.mean()-a.total.mean()
    out["  davon 1. Abstand"] = b.first_gap.mean()-a.first_gap.mean()
    out["  davon Rest"] = b.rest.mean()-a.rest.mean()
    pa = np.array([(a.end == e).mean() for e in ENDS])
    pb = np.array([(b.end == e).mean() for e in ENDS])
    ta = np.array([a.total[a.end == e].mean() if (a.end == e).any() else np.nan for e in ENDS])
    tb = np.array([b.total[b.end == e].mean() if (b.end == e).any() else np.nan for e in ENDS])
    out["  Pfadanteil (andere Endzustände)"] = float(np.nansum((pb-pa)*np.nanmean([ta, tb], axis=0)))
    out["  Zeitanteil (gleicher Endzustand)"] = float(np.nansum((pa+pb)/2*(np.nan_to_num(tb)-np.nan_to_num(ta))))
    wide = g.pivot_table(index=KEY+["k"], columns="rule_option", values="total")
    per = wide.groupby(level=[0, 1, 2, 3])
    out["SD Restzeit je Ziehung"] = g.total.std()
    out["SD gepaarte Differenz je Ziehung"] = per.apply(lambda t: (t[False]-t[True]).std(ddof=1)).median()
    out["Korrelation der Optionen (Median)"] = per.apply(lambda t: np.corrcoef(t[False], t[True])[0, 1]).median()
    out["MC-SE bei 100 Ziehungen (Median)"] = out["SD gepaarte Differenz je Ziehung"]/10
    out["Anteil Zustände mit Regel-Vorzeichen (100)"] = (per.apply(lambda t: (t[False]-t[True]).mean()) > 0).mean()
    within = g.groupby("end").total.var().mul(g.groupby("end").size()).sum()
    out["Varianzanteil des Endzustands an der Restzeit"] = 1-within/(g.total.var()*len(g))
    return pd.Series(out)


table = pd.DataFrame({name: block(part) for name, part in d.groupby("gruppe", sort=False)})
print("\n=== Gezogene Suffixe von D (stateful), je 60 Zustände x 100 Ziehungen je Option ===")
print(table[ORDER].round(2).to_string())
real = z[z.prefix_ok].groupby("gruppe").agg(n=("real_rest_days", "size"), real_rest=("real_rest_days", "mean"), real_events=("real_events", "mean"),
                                            real_offers=("real_offers", "mean"), pending=("real_end", lambda s: (s == "A_Pending").mean()),
                                            cancelled=("real_end", lambda s: (s == "A_Cancelled").mean()))
print("\nTatsächlicher weiterer Verlauf derselben Zustände im Simulatorlauf (unter D's Policy):")
print(real.loc[ORDER].round(2).to_string())
table.to_csv(O/"d_ohne_ziehungen/zerlegung.csv")
print("\nZweites Suffix-Event (nach der Option), häufigste je Punkt und Option:")
for name in ("annehmen_bei_konzept", "handle_leads", "nachlegen ab 2. Besuch"):
    for flag in (True, False):
        p = d[(d.gruppe == name) & (d.rule_option == flag)]
        print(name, "Regel" if flag else "andere", "|", "; ".join(f"{k} {v:.0%}" for k, v in p.second.value_counts(normalize=True).head(5).items()))
