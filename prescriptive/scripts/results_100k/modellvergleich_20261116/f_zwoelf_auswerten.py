"""Auswertung Aufbau F auf 12 Seeds (2026-11-17): Seeds 0-2 aus den bisherigen Läufen, weitere aus f_zwoelf_seeds/ (f_zwoelf_seeds.py).
Je Arm: Durchlaufzeit, Gewinn gegen Bank, gepaarte Differenzen (Arm − Regel, c_fest300 − c_fest100, zztf1_fest100 − c_fest100) mit SE über die
Seeds, Angebote je Fall, Anteil Regel-Option an den Wahlen. Ausgabe f_zwoelf_auswertung.md/.csv.
Aufruf: process-simulation-engine/.venv/bin/python f_zwoelf_auswerten.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
NEW = OUT/"f_zwoelf_seeds"
OLD = {"bank": lambda s: ROOT/f"prescriptive/policies/ev_policy_event_costs/results/d_oe_zustand_20261115/bank_reference_seed{s}",
       "regel": lambda s: OUT/f"cf_im_korrigierten_simulator/regel_seed{s}",
       "c_fest100": lambda s: OUT/f"c_first_only_im_korrigierten_simulator/fest100_seed{s}",
       "c_fest300": lambda s: OUT/f"c_first_only_300_im_korrigierten_simulator/fest300_seed{s}"}
ARMS = ["bank", "regel", "c_fest100", "c_fest300", "zztf1_fest100", "tl_fest300", "tl_fest500", "zzrct_fest100", "tlrct_fest300", "tlbnll_fest300", "tlks_fest300", "tlko_fest300", "tlkb_fest300", "tlcrn_fest300", "tlsv_fest300", "tlsvk_fest300", "tlsvp_fest300", "tlsvh_fest300", "olsv_fest300", "plsv_fest300", "olsvcf_fest300"]
REGEL = {"annehmen_bei_konzept": "A_Accepted", "annehmen_nach_vervollstaendigung": "A_Accepted",
         "angebot_nachlegen": "O_Sent (mail and online)", "handle_leads": "W_Handle leads|withdraw"}


def folder(arm, seed):
    new = NEW/f"{arm}_seed{seed}"
    if (new/"case_costs.csv").exists():
        return new
    old = OLD.get(arm, lambda s: None)(seed)
    return old if old is not None and (old/"case_costs.csv").exists() else None


def measures(path):
    cases = pd.read_csv(path/"case_costs.csv", index_col=0)
    out = {"dlz": float(cases.cycle_days.mean())}
    events = path/"events.pkl"
    if events.exists():
        e = pd.read_pickle(events)
        e = e[e["case:concept:name"].isin(cases.index)]
        out["angebote"] = float((e["concept:name"] == "O_Created").sum() / len(cases))
    choices = path/"choices.csv"
    if choices.exists():
        c = pd.read_csv(choices)
        c = c[(c.options == 2) & c.choice.notna()]
        if len(c):
            out["regel_option"] = float(np.mean([ch == REGEL.get(p) for p, ch in zip(c.point, c.choice)]))
    return out


def paired(frame, a, b):
    both = frame[[a, b]].dropna()
    d = both[a] - both[b]
    return len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan


def main():
    rows = []
    for arm in ARMS:
        for seed in range(12):
            path = folder(arm, seed)
            if path is not None:
                rows.append({"arm": arm, "seed": seed, **measures(path), "quelle": str(path.relative_to(ROOT))})
    long = pd.DataFrame(rows)
    long.to_csv(OUT/"f_zwoelf_auswertung_je_seed.csv", index=False)
    wide = long.pivot(index="seed", columns="arm", values="dlz")
    lines = ["# Aufbau F auf bis zu 12 Seeds × 200 Fälle (gepaart je Seed)", ""]
    summary = []
    for arm in [a for a in ARMS if a in wide]:
        n, gain, se = paired(wide.assign(_b=wide.get("bank")), "_b", arm) if arm != "bank" else (wide[arm].notna().sum(), 0.0, np.nan)
        part = long[long.arm == arm]
        summary.append({"arm": arm, "seeds": int(wide[arm].notna().sum()), "dlz_tage": wide[arm].mean(), "gewinn_gegen_bank": gain,
                        "gewinn_se": se, "angebote_je_fall": part.get("angebote", pd.Series(dtype=float)).mean(),
                        "anteil_regel_option": part.get("regel_option", pd.Series(dtype=float)).mean()})
    summary = pd.DataFrame(summary)
    lines += ["```", summary.round(3).to_string(index=False), "```", "", "## Gepaarte Differenzen (Tage, positiv = erste schneller)", "", "```"]
    pairs = []
    for a, b in (("regel", "c_fest100"), ("regel", "c_fest300"), ("regel", "zztf1_fest100"), ("c_fest100", "c_fest300"),
                 ("c_fest100", "zztf1_fest100"), ("regel", "tl_fest300"), ("c_fest300", "tl_fest300"), ("zztf1_fest100", "tl_fest300"),
                 ("regel", "tl_fest500"), ("c_fest300", "tl_fest500"), ("tl_fest300", "tl_fest500"),
                 ("regel", "zzrct_fest100"), ("regel", "tlrct_fest300"), ("zztf1_fest100", "zzrct_fest100"), ("tl_fest300", "tlrct_fest300"),
                 ("regel", "tlbnll_fest300"), ("c_fest300", "tlbnll_fest300"), ("tl_fest300", "tlbnll_fest300"),
                 ("regel", "tlks_fest300"), ("c_fest300", "tlks_fest300"), ("tl_fest300", "tlks_fest300"),
                 ("regel", "tlko_fest300"), ("c_fest300", "tlko_fest300"), ("tl_fest300", "tlko_fest300"), ("tlko_fest300", "tlks_fest300"),
                 ("regel", "tlkb_fest300"), ("c_fest300", "tlkb_fest300"), ("tl_fest300", "tlkb_fest300"), ("tlks_fest300", "tlkb_fest300"),
                 ("regel", "tlcrn_fest300"), ("c_fest300", "tlcrn_fest300"), ("tl_fest300", "tlcrn_fest300"),
                 ("regel", "tlsv_fest300"), ("tlcrn_fest300", "tlsv_fest300"), ("c_fest300", "tlsv_fest300"),
                 ("regel", "tlsvk_fest300"), ("tlsv_fest300", "tlsvk_fest300"), ("tlcrn_fest300", "tlsvk_fest300"),
                 ("regel", "tlsvp_fest300"), ("tlsvk_fest300", "tlsvp_fest300"), ("tlsv_fest300", "tlsvp_fest300"),
                 ("regel", "tlsvh_fest300"), ("tlsvk_fest300", "tlsvh_fest300"), ("tlsv_fest300", "tlsvh_fest300"),
                 ("regel", "olsv_fest300"), ("tlsvk_fest300", "olsv_fest300"), ("tlsv_fest300", "olsv_fest300"),
                 ("regel", "plsv_fest300"), ("tlsvk_fest300", "plsv_fest300"), ("olsv_fest300", "plsv_fest300"),
                 ("regel", "olsvcf_fest300"), ("olsv_fest300", "olsvcf_fest300"), ("tlsvk_fest300", "olsvcf_fest300")):
        if a in wide and b in wide:
            n, d, se = paired(wide, b, a)  # Durchlaufzeit b − a: positiv = a schneller
            pairs.append({"vergleich": f"{a} schneller als {b}", "seeds": n, "tage": d, "se": se, "z": d / se if se else np.nan})
    pairs = pd.DataFrame(pairs)
    lines += [pairs.round(3).to_string(index=False), "```", ""]
    summary.to_csv(OUT/"f_zwoelf_auswertung.csv", index=False)
    pairs.to_csv(OUT/"f_zwoelf_paare.csv", index=False)
    (OUT/"f_zwoelf_auswertung.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
