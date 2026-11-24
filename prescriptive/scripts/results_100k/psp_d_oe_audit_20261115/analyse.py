"""Independently audit saved D/OE draws; leave original inputs untouched."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
BASE = OUT.parent / "d_sampling_diagnosis_20261115"
OE = OUT.parent / "oe_vs_d_sampling_20261115"


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    provenance = json.loads((BASE / "provenance.json").read_text())
    provenance.update(json.loads((OE / "manifest.json").read_text())["original_inputs_sha256"])
    checks = []
    for filename, expected in provenance.items():
        path = Path(filename)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        checks.append({"path": filename, "matches": actual == expected, "actual_sha256": actual})
    write("input_validation.json", checks)
    assert all(row["matches"] for row in checks), "Historical inputs have changed; inspect before comparing"

    frames = {"D": pd.read_csv(BASE / "psp_draws.csv"), "OE": pd.read_csv(OE / "draws.csv")}
    frames["OE"] = frames["OE"].loc[frames["OE"].random_sample].copy()
    keys = ["id", "option", "repeat", "draw"]
    assert len(frames["D"]) == len(frames["OE"]) == 36000
    assert frames["D"].set_index(keys).index.sort_values().equals(frames["OE"].set_index(keys).index.sort_values())
    simulator = pd.read_csv(BASE / "simulator_branches.csv")
    reference = simulator.groupby(["id", "point", "option"]).agg(
        sim_cost=("total_eur", "mean"), sim_days=("remaining_days", "mean"),
        sim_cost_se=("total_eur", lambda x: x.std(ddof=1)/np.sqrt(len(x))),
        sim_days_se=("remaining_days", lambda x: x.std(ddof=1)/np.sqrt(len(x))),
    ).reset_index()
    predictions, choices, summaries = [], [], []
    for name, frame in frames.items():
        means = frame.groupby(["id", "point", "option"]).agg(
            pred_cost=("total_eur", "mean"), pred_days=("days", "mean"),
            pred_hours=("active_hours", "mean"), pred_events=("events", "mean"),
        ).reset_index().merge(reference, validate="one_to_one")
        means.insert(0, "model", name)
        predictions.append(means)
        summary = {"model": name, "prefixes": int(frame.id.nunique()), "suffixes": len(frame),
            "mean_cost": float(frame.total_eur.mean()), "median_cost": float(frame.total_eur.median()),
            "mean_active_hours": float(frame.active_hours.mean()), "mean_days": float(frame.days.mean()),
            "missing_business_end_fraction": float(frame.missing_business_end.mean()),
            "open_session_end_fraction": float(frame.open_at_end.gt(0).mean()),
            "top10pct_cost_share": float(frame.total_eur.nlargest(int(len(frame)*.1)).sum()/frame.total_eur.sum()),
            "cost_mae_vs_3_simulator_branches": float((means.pred_cost-means.sim_cost).abs().mean()),
            "days_mae_vs_3_simulator_branches": float((means.pred_days-means.sim_days).abs().mean()),
        }
        for objective, column, sim_column in (("cost", "total_eur", "total_eur"), ("days", "days", "remaining_days")):
            flips, agreement, zvalues, paired_ses = [], [], [], []
            for rid, group in frame.groupby("id", sort=True):
                options = sorted(group.option.unique())
                a, b = options
                paired = group.pivot(index=["repeat", "draw"], columns="option", values=column)
                sim_paired = simulator[simulator.id.eq(rid)].pivot(index="repeat", columns="option", values=sim_column)
                differences = paired[a]-paired[b]
                mean_difference = float(differences.mean())
                sim_difference = float((sim_paired[a]-sim_paired[b]).mean())
                sim_se = float((sim_paired[a]-sim_paired[b]).std(ddof=1)/np.sqrt(len(sim_paired)))
                reps = differences.groupby(level="repeat").agg(["mean", "std", "count"])
                ses = reps["std"]/np.sqrt(reps["count"])
                flip = bool((reps["mean"] < 0).nunique() > 1)
                flips.append(flip)
                agreement.append(bool((mean_difference < 0) == (sim_difference < 0)))
                zvalues.extend((reps["mean"].abs()/ses).tolist())
                paired_ses.extend(ses.tolist())
                choices.append({"model": name, "objective": objective, "id": int(rid), "point": group.point.iloc[0],
                    "option_a": a, "option_b": b, "predicted_a_minus_b": mean_difference,
                    "simulator_a_minus_b": sim_difference, "simulator_paired_se_n3": sim_se,
                    "simulator_abs_z_gt2": abs(sim_difference) > 2*sim_se,
                    "repeat_choice_flip": flip,
                    "agrees_with_small_simulator_reference": agreement[-1],
                    "predicted_repeat_se_median": float(ses.median()),
                })
            summary[objective+"_repeat_flip_fraction"] = float(np.mean(flips))
            summary[objective+"_median_paired_se_n500"] = float(np.median(paired_ses))
            summary[objective+"_abs_z_gt2_fraction"] = float(np.mean(np.array(zvalues)>2))
            summary[objective+"_agreement_with_n3_simulator_states_of12"] = int(sum(agreement))
        summaries.append(summary)
    pd.concat(predictions, ignore_index=True).to_csv(OUT/"option_predictions.csv", index=False)
    pd.DataFrame(choices).to_csv(OUT/"option_differences.csv", index=False)
    write("recomputed_summary.json", summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
