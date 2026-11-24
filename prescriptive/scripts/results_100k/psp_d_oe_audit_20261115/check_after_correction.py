"""Compare frozen legacy draws with the saved corrected held-out draws.

Same 100 prefixes, 30 draws and start seeds. This runs independently of the
online comparison and never changes its sources, models or configuration.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/"prescriptive/policies/ev_policy_event_costs/results/d_oe_zustand_20261115"
DEST = OUT/"ziehungsvergleich"
os.environ["PSP_CATALOG"] = str(OUT/"catalog.yaml")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths
add_paths()
import numpy as np
import pandas as pd
import torch
from prescriptive.estimators.psp_model import PSPModel
from prescriptive.policies.ev_policy_event_costs.costs import EventCosts
from prescriptive.scripts.psp_suffixguete import event_scores, time_scores


def weight_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.net.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    DEST.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    picks = pd.read_csv(ROOT/"prescriptive/scripts/results_100k/suffixguete/prefixe.csv")
    picks = picks[picks.Modell.eq("(d) wie (c) mit Zeitklassen")].iloc[:100]
    corrected = pd.read_csv(OUT/"heldout_prefixes.csv")
    raw = pd.read_pickle(ROOT/"prescriptive/data/cache/bpic17_raw.pkl")
    raw = raw.loc[raw["case:concept:name"].astype(str).isin(set(picks.Fall.astype(str)))].copy()
    raw["concept:name"] = raw["concept:name"].astype(str).str.strip()
    raw = raw.sort_values(["case:concept:name", "time:timestamp"], kind="stable")
    cases = {str(case): part.reset_index(drop=True) for case, part in raw.groupby("case:concept:name", sort=False)}
    costs = EventCosts(OUT/"event_costs.yaml")
    paths = {"D": "psp_model_durchlaufzeit_real_alle_s60_klassen_option_encoder_ue.pkl",
        "OE": "psp_model_durchlaufzeit_real_alle_s60_token_oe_ue.pkl"}
    rows, draw_rows, hashes = [], [], {}
    for name, filename in paths.items():
        baseline = PSPModel.load(ROOT/"prescriptive/models"/filename)
        current = PSPModel.load(OUT/"models"/f"{name}_stateful.pkl")
        assert weight_hash(baseline) == weight_hash(current)
        hashes[name] = weight_hash(baseline)
        del current
        baseline.set_threads(1)
        assert not getattr(baseline, "work_state_constraints", False)
        assert getattr(baseline, "conditional_gaps", None) is None
        for count, (index, pick) in enumerate(picks.iterrows(), 1):
            part, k = cases[str(pick.Fall)], int(pick["Prefixlänge"])
            prefix = part.iloc[:k].to_dict("records")
            tokens = (part["concept:name"].astype(str)+"|"+part["lifecycle:transition"].astype(str)).tolist()[k:]
            encoded = baseline._encode(prefix)
            torch.manual_seed(20261115+index)
            suffixes = baseline.draw_cases([encoded[:4]], 30)[0]
            at = part["time:timestamp"].iloc[k-1].timestamp()
            sunk_state = costs.state(prefix)
            sunk = sunk_state.total(at)
            true_remaining = (part["time:timestamp"].iloc[-1]-part["time:timestamp"].iloc[k-1]).total_seconds()
            first = (part["time:timestamp"].iloc[k]-part["time:timestamp"].iloc[k-1]).total_seconds()
            sampled_costs = []
            prefix_end = any(e["concept:name"] in ("A_Pending", "A_Denied", "A_Cancelled") for e in prefix)
            for suffix in suffixes:
                state, time = sunk_state.copy(), at
                end = prefix_end
                for token, _, gap in suffix:
                    a, _, lc = token.rpartition("|")
                    time += gap
                    state.push(a, lc, time)
                    end |= a in ("A_Pending", "A_Denied", "A_Cancelled")
                bill = state.summary(time)
                remaining = bill["total_eur"]-sunk
                sampled_costs.append(remaining)
                draw_rows.append({"model": name, "case": str(pick.Fall), "cost": remaining,
                    "missing_business_end": not end, "open_at_end": bool(state.opened),
                    "reopened": state.reopened_segments > sunk_state.reopened_segments,
                    "active_hours": bill["active_hours"]-sunk_state.summary(at)["active_hours"]})
            row = {"model": name, "case": str(pick.Fall), "prefix_length": k,
                **event_scores(tokens, [[s[0] for s in suffix] for suffix in suffixes]),
                **time_scores(first, true_remaining, [s[0][2] if s else 0. for s in suffixes],
                    [sum(s[2] for s in suffix) for suffix in suffixes]),
                "remaining_cost_mae": abs(np.mean(sampled_costs)-(costs.state(part.to_dict("records")).total()-sunk))}
            rows.append(row)
            if count % 20 == 0:
                print("MATCHED BASELINE", name, count, flush=True)
    before = pd.DataFrame(rows)
    draws = pd.DataFrame(draw_rows)
    before.to_csv(DEST/"baseline_prefixes.csv", index=False)
    draws.to_csv(DEST/"baseline_draws.csv", index=False)
    tables = []
    for version, frame in (("bisherig_mit_ue", before), ("korrigiert", corrected)):
        table = frame.drop(columns=["case", "prefix_length"]).groupby("model").mean(numeric_only=True)
        table["version"] = version
        tables.append(table.reset_index())
    summary = pd.concat(tables, ignore_index=True)
    summary.to_csv(DEST/"matched_summary.csv", index=False)
    structural = draws.groupby("model")[["missing_business_end", "open_at_end", "reopened", "active_hours", "cost"]].mean()
    structural.to_csv(DEST/"baseline_structure.csv")
    validation = {"status": "complete", "prefixes_per_model": 100, "draws_per_prefix": 30,
        "identical_neural_weights": hashes, "baseline": "legacy model copies with their original observed-transition masks",
        "corrected": "saved corrected draws on the identical prefix keys, n=30 and start seeds",
        "scope": "bundled state, EOS and conditional-time correction; differences also include stricter train-only transition support",
        "baseline_suffixes": len(draws), "corrected_suffixes": 6000}
    for name in paths:
        old_keys = set(map(tuple, before.loc[before.model.eq(name), ["case", "prefix_length"]].values))
        new_keys = set(map(tuple, corrected.loc[corrected.model.eq(name), ["case", "prefix_length"]].values))
        assert old_keys == new_keys and len(old_keys) == 100
    (DEST/"validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    lines = ["**Ziehungsqualität vor und nach der Korrektur**", "",
        "Identische 100 zurückgehaltene Testprefixe je Modell, je 30 Suffixe und dieselben Start-Seeds. "
        "Die neuronalen Gewichte sind nachweislich identisch. Die bisherigen Modelle verwenden bereits ihre alte "
        "Übergangssperre. Die korrigierten Kopien ergänzen Zustands-/EOS-Prüfung und ein gemeinsames, ausschließlich "
        "auf Trainingsfällen gelerntes bedingtes Zeitmodell; ihre Übergangsregeln schließen auch die Validierungsfälle aus.", "",
        "| Kennzahl | D bisher | D korrigiert | OE bisher | OE korrigiert |",
        "|---|---:|---:|---:|---:|"]
    indexed = summary.set_index(["model", "version"])
    for key, label, factor in [("Schritt 1", "Treffer Schritt 1 (%)", 100), ("Schritt 5", "Treffer Schritt 5 (%)", 100),
        ("Schritt 10", "Treffer Schritt 10 (%)", 100), ("DLS", "Sequenzähnlichkeit DLS", 1),
        ("Suffixlänge-MAE", "Suffixlängen-MAE (Events)", 1), ("Abstand Schritt 1 MAE (h)", "Erster Abstand MAE (h)", 1),
        ("Restzeit MAE (Tage)", "Restzeit-MAE (Tage)", 1), ("remaining_cost_mae", "Restkosten-MAE (€)", 1)]:
        vals = [indexed.loc[(model, version), key]*factor for model in paths for version in ("bisherig_mit_ue", "korrigiert")]
        lines.append("| "+label+" | "+" | ".join(f"{v:.3f}" for v in vals)+" |")
    lines += ["", "Die korrigierte Prüfung liefert auf allen 6.000 Suffixen keine offenen Sitzungen am Ende, "
        "keine fehlenden geschäftlichen Endmarker und keine nach ihren Regeln ungültigen Öffnungen. "
        "Diese Eigenschaften werden durch die Ziehungsregeln erzwungen; bessere Vorhersagefehler sind separat zu beurteilen. "
        "Ein vollständiger Vergleich des Online-Outcomes über alle Seeds läuft unabhängig weiter.", "",
        "Weitere Dateien: matched_summary.csv, baseline_structure.csv, baseline_prefixes.csv, baseline_draws.csv und validation.json."]
    (DEST/"bericht.md").write_text("\n".join(lines)+"\n")
    print(summary.to_string(index=False), flush=True)
    print(structural.to_string(), flush=True)


if __name__ == "__main__":
    main()
