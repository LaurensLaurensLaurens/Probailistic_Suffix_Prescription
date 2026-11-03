"""
Warum wählt das PSP beim Outcome kosten in der echten Werkstatt anders als XGBoost? Das PSP entscheidet (wie die Policy psp),
an jeder Wahl werden festgehalten: die freien Optionen, je Option die Schätzung von PSP und XGBoost, die wahren erwarteten Kosten
(wie die Spezialisten-Regel in repair_truth.py), und aus den gezogenen Suffixen des PSP die Dauer der ersten Reparatur und die
Zahl der Nacharbeiten, gegen die wahren Erwartungen (repair_engine.PROCESS, repair_workshop.skill).

Aufruf aus process-simulation-engine/:
    .venv/bin/python -u ../prescriptive/scripts/repair_cost_diagnose.py [--outcome kosten] [--seed 0] [--out datei.csv]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "process-simulation-engine"), os.path.join(ROOT, "prescriptive"), ROOT]
CASES, WARMUP, COOLDOWN = 3000, 700, 1300  # wie repair_truth.py
LEVEL = {0.6: "stark", 1.0: "neutral", 1.6: "schwach"}


class Recorder:
    """Wählt wie die Policy psp (Mittel der gezogenen Outcomes) und hält je Wahl alles fest."""

    def __init__(self, psp, xgb, spec, seed):
        self.psp, self.xgb, self.spec, self.rows = psp, xgb, spec, []
        self.kind = psp._outcome_kind()
        self.rng = np.random.default_rng(seed)

    def choose(self, point, case, options, now, engine):
        from prescriptive.costs.repair import rates
        from simulation.repair_workshop import EXPERTS, skill
        events = case.prefix
        drawn, elapsed = self.psp.option_suffixes(point, events, options, now, self.psp.samples_per_option)
        costs = self.psp._option_costs(point, events, options, now)
        psp = np.array([np.mean([self.psp.outcome(elapsed, s, self.kind, c) for s in suffixes]) for suffixes, c in zip(drawn, costs)])
        xgb = np.asarray(self.xgb.values(point, events, options, now), dtype=float)
        device, process = case.draw["device"], engine.process
        days = np.array([process["normal_days"] * skill(r, device)[0] * (1 + skill(r, device)[1] * process["rework_share"]) for r in options])
        start = pd.Timestamp(now, unit="s")
        true = -(rates.labor_cost(list(options), [start] * len(options), [start + pd.Timedelta(days=d) for d in days], self.spec["tarif"])
                 + self.spec["cost_per_day"] * days)
        best = np.flatnonzero(psp == psp.max())
        pick = int(best[0] if len(best) == 1 else self.rng.choice(best))
        decision = len(self.rows) and self.rows[-1]["Wahl"] + 1 or 1
        for k, option in enumerate(options):
            # Dauer der ersten Reparatur aus den gezogenen Abständen bis W_repair|complete (wie die Kosten, costs/repair/outcome.py)
            first = [elapsed + next((np.cumsum([g for _, _, g in s])[j] for j, (label, _, _) in enumerate(s) if label == "W_repair|complete"), np.nan)
                     for s in drawn[k]]
            self.rows.append({"Wahl": decision, "Option": option, "Optionen": len(options),
                              "Fähigkeit": "expertin" if option in EXPERTS else LEVEL[skill(option, device)[0]],
                              "psp": psp[k], "xgb": xgb[k], "wahr": true[k], "gewählt psp": k == pick,
                              "Reparatur gezogen (Tage)": (np.nanmean(first) - elapsed) / 86400,
                              "Reparatur wahr (Tage)": process["normal_days"] * skill(option, device)[0],
                              "Nacharbeit gezogen": np.mean([sum(label == "W_repair|start" for label, _, _ in s) for s in drawn[k]]),
                              "Nacharbeit wahr": skill(option, device)[1],
                              "Satz jetzt (€/h)": rates.rate(option, start, self.spec["tarif"])})
        return options[pick]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome", default="kosten")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--psp-model", default=os.path.join(ROOT, "prescriptive", "models", "psp_model_repair.pkl"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    from prescriptive import profiles
    profiles.set_outcome("repair", args.outcome)
    from helpers.training import model_class, model_path
    from preprocessing import chosen_outcome_spec
    from prescriptive.estimators.psp_model import PSPModel
    from simulation.repair_workshop import RepairWorkshop
    psp = PSPModel.load(args.psp_model)
    psp.set_threads(8)
    xgb = model_class("xgb").load(model_path("xgb", "repair"))
    recorder = Recorder(psp, xgb, chosen_outcome_spec("repair"), args.seed)
    started = time.time()
    engine = RepairWorkshop(seed=args.seed)
    engine.set_policy(recorder, [engine.point], window=(WARMUP, CASES - COOLDOWN))
    engine.run(CASES)
    frame = pd.DataFrame(recorder.rows)
    print(f"{frame['Wahl'].nunique()} Wahlen in {time.time() - started:.0f} s, Optionen je Wahl im Mittel {frame.groupby('Wahl')['Optionen'].first().mean():.1f}")
    if args.out:
        frame.to_csv(args.out, index=False)
    analyse(frame)


def analyse(frame):
    groups = frame.groupby("Wahl")
    for model in ("psp", "xgb"):
        chosen = groups.apply(lambda f: f.loc[f[model].idxmax()])
        best = groups["wahr"].max()
        print(f"{model}: beste Wahl {np.mean(np.isclose(chosen['wahr'].to_numpy(), best.to_numpy())):.1%}, Verlust je Wahl {(best - chosen['wahr'].to_numpy()).mean():,.0f} €, "
              f"gewählt nach Fähigkeit {chosen['Fähigkeit'].value_counts(normalize=True).round(3).to_dict()}")
    # Unterschiede zwischen den Optionen einer Wahl: Schätzung gegen Wahrheit (Steigung 1 = richtig skaliert)
    centered = frame.assign(**{m: frame[m] - groups[m].transform("mean") for m in ("psp", "xgb", "wahr")})
    centered = centered[groups["Optionen"].transform("first") > 1]
    for model in ("psp", "xgb"):
        slope = np.polyfit(centered["wahr"], centered[model], 1)[0]
        print(f"{model}: Unterschiede zwischen Optionen, Steigung gegen die Wahrheit {slope:.2f}")
    print("\nJe Fähigkeit (Mittel über alle Optionen aller Wahlen):")
    print(frame.groupby("Fähigkeit")[["Reparatur gezogen (Tage)", "Reparatur wahr (Tage)", "Nacharbeit gezogen", "Nacharbeit wahr", "Satz jetzt (€/h)"]]
          .mean().reindex(["expertin", "stark", "neutral", "schwach"]).round(3).to_string())
    # Die typische Fehlentscheidung: stark frei, PSP nimmt neutral
    rows = []
    for _, f in groups:
        pick = f.loc[f["psp"].idxmax()]
        if pick["Fähigkeit"] == "neutral" and (f["Fähigkeit"] == "stark").any():
            strong = f[f["Fähigkeit"] == "stark"].loc[lambda g: g["psp"].idxmax()]
            rows.append({"PSP: stark minus neutral (€)": strong["psp"] - pick["psp"], "wahr: stark minus neutral (€)": strong["wahr"] - pick["wahr"],
                         "Satz neutral (€/h)": pick["Satz jetzt (€/h)"], "Satz stark (€/h)": strong["Satz jetzt (€/h)"]})
    if rows:
        rows = pd.DataFrame(rows)
        print(f"\nPSP wählt neutral, obwohl ein starker frei ist: {len(rows)} Wahlen; Mittel:")
        print(rows.mean().round(0).to_string())


if __name__ == "__main__":
    main()
