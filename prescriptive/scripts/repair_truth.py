"""
Die Policies der Pipeline auf dem Datensatz repair (im gelernten Simulator trainiert) in der echten Werkstatt
(process-simulation-engine/simulation/repair_workshop.py): Stimmt die Rangfolge aus dem gelernten Simulator mit der Wahrheit überein?
Als Bezug die Spezialisten-Regel, die die wahren Fähigkeiten kennt (je Gerätetyp die kürzeste erwartete Reparatur mit Nacharbeit).

Wie die Online-Auswertung der Pipeline: je Seed CASES Ankünfte, gewertet die Fälle WARMUP bis CASES - COOLDOWN - 1 (alle, auch abgebrochene),
Durchlaufzeit in Tagen. Alle Policies sehen dieselben Fälle (jeder Fall zieht aus eigenen Zufallszahlen), die Unterschiede sind also gepaart.
Mit --outcome kosten bzw. kosten_b (Katalog repair, outcome_optionen; prescriptive/costs/repair) dazu Lohnkosten und Kosten je Fall in €,
die Modelle mit diesem Outcome (models/<name>_model_repair_<Option>.pkl), und die Spezialisten-Regel wählt die geringsten erwarteten
Kosten (Näherung: die erwartete Reparatur mit Nacharbeit am Stück ab jetzt zum Satz des Mitarbeiters plus Satz je Tag).
Policies: bank (die Hausregel der Werkstatt), random, xgb, lstm und psp (models/<name>_model_repair.pkl), psp_erm (das PSP mit dem
entropischen Risikomaß, γ: --erm-gamma), psp_lookahead (das PSP mit Blick auf die anderen Fälle, darf warten), spezialist.

Aufruf aus process-simulation-engine/:
    .venv/bin/python -u ../prescriptive/scripts/repair_truth.py --policies bank random xgb psp spezialist --seeds 10
Ergebnisse in prescriptive/scripts/results_repair_pipeline/wahrheit/ (--out: anderer Ordner).
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "process-simulation-engine"), os.path.join(ROOT, "prescriptive"), ROOT]

OUT = os.path.join(ROOT, "prescriptive", "scripts", "results_repair_pipeline", "wahrheit")
CASES, WARMUP, COOLDOWN = 3000, 700, 1300
POLICIES = ("bank", "random", "xgb", "lstm", "psp", "psp_erm", "psp_lookahead", "spezialist")


class RandomChoice:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def choose(self, point, case, options, time, engine):
        return options[int(self.rng.integers(len(options)))]


class Specialist:
    """
    Kennt die wahren Fähigkeiten: die kürzeste erwartete Reparatur einschließlich Nacharbeit; mit dem Outcome kosten (spec)
    die geringsten erwarteten Kosten dieser Reparatur ab jetzt zum Satz des Mitarbeiters plus Satz je Tag.
    """

    def __init__(self, spec=None):
        self.spec = spec if spec and spec["type"] == "negative_repair_cost" else None

    def choose(self, point, case, options, time, engine):
        from simulation.repair_workshop import skill
        device, share = case.draw["device"], engine.process["rework_share"]
        days = np.array([engine.process["normal_days"] * skill(r, device)[0] * (1 + skill(r, device)[1] * share) for r in options])
        if self.spec is None:
            return options[int(np.argmin(days))]
        from prescriptive.costs.repair import rates
        start = pd.Timestamp(time, unit="s")
        cost = rates.labor_cost(list(options), [start] * len(options), [start + pd.Timedelta(days=d) for d in days], self.spec["tarif"])
        return options[int(np.argmin(cost + self.spec["cost_per_day"] * days))]


def outcome_spec():
    """Das gewählte Outcome der Werkstatt (--outcome, über profiles.set_outcome auch in den Arbeitsprozessen)."""
    from preprocessing import chosen_outcome_spec
    return chosen_outcome_spec("repair")


def make(name, seed, gamma=None, model=None, xgb_model=None):
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    if name == "bank":
        return None
    if name == "random":
        return RandomChoice(seed)
    if name == "spezialist":
        return Specialist(outcome_spec())
    from helpers.training import model_path
    if name == "xgb":
        from prescriptive.policies.xgb_policy import XGBPolicy
        return XGBPolicy(xgb_model or model_path("xgb", "repair"), seed=seed)
    if name == "lstm":
        from prescriptive.policies.lstm_policy import LSTMPolicy
        return LSTMPolicy(model_path("lstm", "repair"), seed=seed)
    if name == "psp_lookahead":
        from prescriptive.policies.psp_lookahead_policy import PSPLookaheadPolicy
        return PSPLookaheadPolicy(model or model_path("psp", "repair"), seed=seed)
    if name == "psp_erm":
        from prescriptive.policies.psp_erm_policy import PSPERMPolicy
        return PSPERMPolicy(model or model_path("psp", "repair"), seed=seed, **({"gamma": gamma} if gamma else {}))
    from prescriptive.policies.psp_policy import PSPPolicy
    return PSPPolicy(model or model_path("psp", "repair"), seed=seed)


def simulate(job):
    from simulation.repair_workshop import EXPERTS, RepairWorkshop, skill
    name, seed, gamma, model, xgb_model = job
    started = time.time()
    policy = make(name, seed, gamma, model, xgb_model)
    engine = RepairWorkshop(seed=seed)
    engine.set_policy(policy, [engine.point] if policy is not None else [], window=(WARMUP, CASES - COOLDOWN))
    events = pd.DataFrame(engine.run(CASES))
    table = engine.case_table()
    part = table[table["number"].between(WARMUP, CASES - COOLDOWN - 1)]
    costs = {}
    spec = outcome_spec()
    if spec["type"] == "negative_repair_cost":
        from prescriptive.costs.repair.outcome import case_costs
        money = case_costs(events[events["case:concept:name"].isin(part["case"])], spec)
        costs = {"Kosten (€)": money["Kosten (€)"].mean(), "Lohnkosten (€)": money["Lohnkosten (€)"].mean(),
                 "Kosten P90 (€)": money["Kosten (€)"].quantile(0.9)}
    devices = dict(zip(table["case"], table["device"]))
    from interventions import WAIT
    chosen = [c for c in engine.choices if c["choice"] is not None and c["choice"] != WAIT]
    quality = [("expertin" if c["choice"] in EXPERTS else
                {0.6: "stark", 1.0: "neutral", 1.6: "schwach"}[skill(c["choice"], devices[c["case"]])[0]]) for c in chosen]
    shares = pd.Series(quality, dtype=object).value_counts(normalize=True) if quality else pd.Series(dtype=float)
    return {"Policy": name, "seed": seed, "Fälle": len(part), "Durchlaufzeit (Tage)": part["days"].mean(), "Durchlaufzeit P90 (Tage)": part["days"].quantile(0.9),
            "Durchlaufzeit CVaR 90 % (Tage)": part["days"][part["days"] >= part["days"].quantile(0.9)].mean(), **costs,
            "Warten auf Reparatur (Tage)": part["waited"].mean(), "Nacharbeit je Fall": part["reworks"].mean(), "Wahlen": len(engine.choices),
            "Warten (Antworten)": sum(c["choice"] == WAIT for c in engine.choices),
            **{f"gewählt: {k}": shares.get(k, 0.0) for k in ("stark", "neutral", "schwach", "expertin")}, "Sekunden": time.time() - started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--erm-gamma", type=float, default=None, help="psp_erm: Risikoaversion γ in 1/Tag (Standard policies/psp_erm_policy.py)")
    parser.add_argument("--psp-model", default=None, help="psp, psp_erm: Pfad des PSP-Modells (Standard models/psp_model_repair.pkl)")
    parser.add_argument("--out", default=OUT, help="Ordner der Ergebnisse")
    parser.add_argument("--xgb-model", default=None, help="xgb: Pfad des XGBoost-Modells (Standard models/xgb_model_repair<outcome>.pkl)")
    parser.add_argument("--outcome", default=None, help="eine der outcome_optionen von repair (kosten, kosten_b): Kosten in €, Modelle mit _<Option>")
    args = parser.parse_args()
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    from prescriptive import profiles
    profiles.set_outcome("repair", args.outcome)  # in der Umgebung, gilt so auch in den Arbeitsprozessen
    os.makedirs(args.out, exist_ok=True)
    jobs = [(name, seed, args.erm_gamma, args.psp_model, args.xgb_model) for name in args.policies for seed in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = pd.DataFrame(list(pool.map(simulate, jobs)))
    runs.to_csv(os.path.join(args.out, "laeufe.csv"), index=False)
    table = runs.groupby("Policy").mean(numeric_only=True).drop(columns="seed").reindex([p for p in POLICIES if p in args.policies])
    if "bank" in args.policies:
        wide = runs.pivot(index="seed", columns="Policy", values="Durchlaufzeit (Tage)")
        gap = wide.sub(wide["bank"], axis=0)
        table["ggü. bank (Tage)"] = gap.mean()
        table["KI ±"] = 2.262 * gap.std(ddof=1) / np.sqrt(len(gap))  # 95 %, t mit 9 Freiheitsgraden (10 Seeds)
        if "Kosten (€)" in runs:
            wide = runs.pivot(index="seed", columns="Policy", values="Kosten (€)")
            gap = wide.sub(wide["bank"], axis=0)
            table["ggü. bank (€)"] = gap.mean()
            table["KI ± (€)"] = 2.262 * gap.std(ddof=1) / np.sqrt(len(gap))
    table.to_csv(os.path.join(args.out, "ergebnis.csv"))
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(table.round(3).to_string(), flush=True)
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
