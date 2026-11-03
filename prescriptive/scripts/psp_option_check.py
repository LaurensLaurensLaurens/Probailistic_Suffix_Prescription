"""
Unterscheidet das PSP die Optionen eines Wahlpunkts? Auf Testfällen der RCT-Daten (model.test_cases) bewertet das PSP je
Probe alle Optionen (option_suffixes wie die Policy psp, gleiche Zufallszahlen je Option), die Vergleichsmodelle (--models,
Standard XGBoost; auch lstm) ebenso; die Proben zieht immer das PSP-Modell (dieselben für alle Modelle).
Ausgabe je Probe der Wert jeder Option minus das Mittel über die Optionen derselben Probe (Tage bzw. €, positiv = besser), die
Spanne zwischen bester und schlechtester Option und welche Option gewählt würde.
Ohne Wahrheit (nicht repair): Rangkorrelation und gleiche Wahl des PSP mit den Vergleichsmodellen.
Bei repair zusätzlich die Wahrheit aus der Werkstatt (repair_workshop.skill: erwartete Reparatur eines normalen Falls
mit Nacharbeit; mit --outcome kosten deren Kosten zum Satz des Mitarbeiters ab jetzt plus Satz je Tag, eine Näherung),
gemittelt je Fähigkeit (Expertin, stark, neutral, schwach), die Rangkorrelation mit ihr und das
Zusammenspiel von Mechaniker und Gerät: Wert je Mechanikergruppe (1-3, 4-6, 7-9) und Gerät, zerlegt in den Effekt der
Gruppe allein und die Wechselwirkung (in der Wahrheit nur Wechselwirkung).

Aufruf aus process-simulation-engine/:
    .venv/bin/python -u ../prescriptive/scripts/psp_option_check.py --dataset repair [--model models/psp_model_repair_25ep.pkl]
    .venv/bin/python -u ../prescriptive/scripts/psp_option_check.py --dataset repair --models xgb lstm   (ohne Ziehen des PSP)
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "process-simulation-engine"), os.path.join(ROOT, "prescriptive"), ROOT]
RCT = {"repair": "rct_repair_nacharbeit_delta50"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="repair")
    parser.add_argument("--model", default=None, help="PSP-Modell (Standard: models/psp_model_<dataset>.pkl)")
    parser.add_argument("--models", nargs="+", default=["psp", "xgb"], choices=("psp", "xgb", "lstm"), help="bewertete Modelle")
    parser.add_argument("--rct", default=None, help="Ordner der RCT-Daten unter prescriptive/data")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--draws", type=int, default=30, help="gezogene Suffixe je Option (Policy psp: 30)")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--out", default=None, help="CSV je Probe und Option")
    parser.add_argument("--paths", nargs="*", default=[], metavar="MODELL=PFAD", help="Vergleichsmodell aus dieser Datei (Standard models/<name>_model_<dataset>.pkl)")
    parser.add_argument("--outcome", default=None, help="eine der outcome_optionen des Datensatzes (z. B. repair: kosten), Standard das Outcome des Katalogs")
    args = parser.parse_args()
    paths = dict(entry.split("=", 1) for entry in args.paths)

    import torch

    from prescriptive.helpers.datasets import add_paths
    add_paths()
    from prescriptive import profiles
    profiles.set_outcome(args.dataset, args.outcome)
    from helpers.training import model_class, model_path
    from preprocessing import chosen_outcome_spec
    from prescriptive.estimators.psp_model import PSPModel
    psp = PSPModel.load(args.model or model_path("psp", args.dataset))
    psp.set_threads(args.threads)
    others = {name: model_class(name).load(paths.get(name) or model_path(name, args.dataset)) for name in args.models if name != "psp"}
    names = list(args.models)
    log = pd.read_pickle(os.path.join(ROOT, "prescriptive", "data", args.rct or RCT[args.dataset], "log.pkl"))
    log = log[log["case:concept:name"].isin(psp.test_cases)]
    truth = None
    if args.dataset == "repair":
        from simulation.repair_engine import PROCESS
        from simulation.repair_workshop import skill
        from prescriptive.costs.repair import rates
        level = {0.6: "stark", 1.0: "neutral", 1.6: "schwach"}
        spec = chosen_outcome_spec(args.dataset)

        def truth(option, device, now):
            factor, rework = skill(option, device)
            days = PROCESS["normal_days"] * factor * (1 + rework * PROCESS["rework_share"])
            if spec["type"] != "negative_repair_cost":
                return -days
            # Kosten (Näherung): die erwartete Reparatur mit Nacharbeit am Stück ab jetzt zum Satz des Mitarbeiters, dazu der Satz je Tag
            start = pd.Timestamp(now, unit="s")
            return -(rates.labor_cost(option, start, start + pd.Timedelta(days=days), spec["tarif"]) + spec["cost_per_day"] * days)

        def group(option, device):
            return "expertin" if option.startswith("expert") else level[skill(option, device)[0]]

    unit = "€" if chosen_outcome_spec(args.dataset)["type"] == "negative_repair_cost" else "Tage"
    torch.manual_seed(0)
    rows, started = [], time.time()
    for name, samples in psp.samples(log).items():
        point, options = psp.points[name], psp.options[name]
        if not samples or not psp.knows(point) or not all(model.knows(point) for model in others.values()):
            continue
        chosen = sorted(np.random.default_rng(0).choice(len(samples), min(args.samples, len(samples)), replace=False))
        for case, (part, cut, now), _ in (samples[k] for k in chosen):
            events = part.iloc[:cut].to_dict("records")
            values = {}
            if "psp" in names:
                values["psp"] = psp.outcome_samples(point, events, options, now, args.draws).mean(axis=1)
            for model_name, model in others.items():
                values[model_name] = np.asarray(model.values(point, events, options, now), dtype=float)
            device = events[0].get("case:device")
            t = np.array([truth(o, device, now) for o in options]) if truth else np.full(len(options), np.nan)
            for k, option in enumerate(options):
                row = {"Wahlpunkt": name, "Fall": case, "Option": option, "Gerät": device, "Gruppe": group(option, device) if truth else option,
                       "wahr": t[k] - t.mean()}
                for model_name, v in values.items():
                    row[model_name] = v[k] - v.mean()
                    row[f"{model_name} gewählt"] = k == int(np.argmax(v))
                rows.append(row)
    print(f"{len({(r['Wahlpunkt'], r['Fall']) for r in rows})} Proben in {time.time() - started:.0f} s ({', '.join(names)}; PSP: "
          f"{args.model or model_path('psp', args.dataset)})", flush=True)
    frame = pd.DataFrame(rows)
    if args.out:
        frame.to_csv(args.out, index=False)
    columns = names + (["wahr"] if truth else [])
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        for name, part in frame.groupby("Wahlpunkt"):
            n = part.groupby("Fall").ngroups
            table = part.groupby("Gruppe")[columns].mean()
            for model_name in names:
                table[f"{model_name}_gewählt"] = part.groupby("Gruppe")[f"{model_name} gewählt"].sum() / n
            if truth:
                table = table.reindex(["expertin", "stark", "neutral", "schwach"])
            print(f"\n{name}: {n} Proben, {table.shape[0]} Optionen; Mittel je Gruppe, Wert minus Mittel über die Optionen derselben Probe ({unit}, positiv = besser)")
            if truth or table.shape[0] <= 20:
                print(table.round(3).to_string())
            per_case = part.groupby("Fall")
            spread = per_case[columns].agg(lambda v: v.max() - v.min()).median()
            print(f"Spanne zwischen bester und schlechtester Option je Probe ({unit}, Median):", spread.round(3).to_dict())
            if not truth:
                # Ohne Wahrheit: Übereinstimmung des PSP mit den Vergleichsmodellen
                if "psp" in names and len(names) > 1:
                    others_ = [m for m in names if m != "psp"]
                    rank = per_case.apply(lambda f: pd.Series({m: f["psp"].corr(f[m], method="spearman") for m in others_})).median()
                    same = {m: round(per_case.apply(lambda f: f.loc[f["psp gewählt"], "Option"].iloc[0] == f.loc[f[f"{m} gewählt"], "Option"].iloc[0]).mean(), 3)
                            for m in others_}
                    print("Rangkorrelation PSP mit den Vergleichsmodellen je Probe (Spearman, Median):", rank.round(3).to_dict(), "gleiche Wahl:", same)
                continue
            rank = per_case.apply(lambda f: pd.Series({m: f[m].corr(f["wahr"], method="spearman") for m in names})).median()
            print("Rangkorrelation mit der Wahrheit je Probe (Spearman, Median):", rank.round(3).to_dict())
            # Wie oft die Wahl die nach der Wahrheit beste Option ist, und was sie im Mittel gegen diese verliert
            hit = {m: round(per_case.apply(lambda f: np.isclose(f.loc[f[f"{m} gewählt"], "wahr"].iloc[0], f["wahr"].max())).mean(), 3) for m in names}  # Gleichstand zählt
            loss = {m: round(per_case.apply(lambda f: f["wahr"].max() - f.loc[f[f"{m} gewählt"], "wahr"].iloc[0]).mean(), 3) for m in names}
            print(f"Wahl gleich der besten nach der Wahrheit (Anteil): {hit}; Verlust gegen die beste ({unit}, Mittel je Probe): {loss}")
            # Zusammenspiel Mechaniker x Gerät: je Gerät zentriert, zerlegt in Effekt der Gruppe allein und Wechselwirkung
            mech = part[part["Option"].str.startswith("mechanic")].copy()
            mech["Mechaniker"] = mech["Option"].map(lambda o: "{0}-{1}".format(*(3 * ((int(o.rsplit("_", 1)[1]) - 1) // 3) + k for k in (1, 3))))
            for model_name in columns:
                grid = mech.pivot_table(index="Mechaniker", columns="Gerät", values=model_name, aggfunc="mean")
                centered = grid - grid.mean(axis=0)
                main_effect = centered.mean(axis=1)
                interaction = centered.sub(main_effect, axis=0)
                print(f"{model_name}: Wert je Mechanikergruppe und Gerät; Streuung Gruppe allein {main_effect.std():.2f}, Wechselwirkung "
                      f"{interaction.values.std():.2f}")
                print(grid.round(2).to_string())
            best = {m: mech.loc[mech.groupby("Fall")[m].idxmax(), "Gruppe"].value_counts(normalize=True).round(3).to_dict() for m in names}
            print("Bester Mechaniker je Probe (nur Mechaniker):", best)
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
