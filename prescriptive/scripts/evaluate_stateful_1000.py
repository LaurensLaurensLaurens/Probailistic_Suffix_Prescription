"""Exact 1000-prefix/100-draw comparison against the original D/OE tables.

Independent of the running online pipeline; model copies and its sources stay
unchanged. Partial progress and failures are saved rather than silently dropping
prefixes from the reference population.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import warnings
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths
add_paths()
import numpy as np
import pandas as pd

REFERENCE = ROOT/"prescriptive/scripts/results_100k"
LABELS = {"D": "(d) wie (c) mit Zeitklassen", "OE": "real_alle_s60_token_oe"}
REF_FILES = {"D": REFERENCE/"suffixguete/prefixe.csv",
    "OE": REFERENCE/"voll_real_alle_s60_token_oe_suffixguete/prefixe.csv"}


def now():
    return datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds")


def write(path, value):
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False)+"\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(name):
    table = pd.read_csv(REF_FILES[name])
    table = table.loc[table.Modell.eq(LABELS[name])].reset_index(drop=True)
    assert len(table) == 1000 and table.Fall.nunique() == 1000
    return table


def evaluate(job):
    name, source, out = job
    os.environ["PSP_CATALOG"] = str(source/"catalog.yaml")
    import torch
    from interventions.work_state import WorkState
    from prescriptive.estimators.psp_model import PSPModel
    from prescriptive.estimators.conditional_gaps import SuffixSamplingError
    from prescriptive.scripts.psp_suffixguete import event_scores, time_scores
    torch.set_num_threads(1)
    started = time.time()
    target = out/name
    target.mkdir(exist_ok=False)
    model = PSPModel.load(source/"models"/f"{name}_stateful.pkl")
    model.set_threads(1)
    picks = reference(name)
    raw = pd.read_pickle(ROOT/"prescriptive/data/cache/bpic17_raw.pkl")
    raw = raw.loc[raw["case:concept:name"].astype(str).isin(set(picks.Fall.astype(str)))].copy()
    raw["concept:name"] = raw["concept:name"].astype(str).str.strip()
    raw = raw.sort_values(["case:concept:name", "time:timestamp"], kind="stable")
    cases = {str(case): part.reset_index(drop=True) for case, part in raw.groupby("case:concept:name", sort=False)}
    del raw
    rows, failures = [], []
    returned = 0
    torch.manual_seed(0)

    def save(status):
        write(target/"status.json", {"status": status, "model": name, "updated_at": now(),
            "completed_prefixes": len(rows), "failed_prefixes": len(failures), "requested_prefixes": 1000,
            "draws_per_prefix": 100, "valid_suffixes": returned, "invalid_returned_suffixes": 0,
            "elapsed_seconds": round(time.time()-started, 1)})
        pd.DataFrame(rows).to_csv(target/"prefixes.csv", index=False)
        write(target/"failures.json", failures)

    def record_output(info, suffixes):
        nonlocal returned
        pick, prefix, true_tokens, true_first, true_remaining, work = info
        for suffix in suffixes:
            check = work.copy()
            for token, _, gap in suffix:
                assert np.isfinite(gap) and gap >= 0
                assert check.permits(token), (name, pick.Fall, token)
                check.push(token)
            assert check.complete, (name, pick.Fall)
        returned += len(suffixes)
        # The original evaluator excludes empty suffixes for first-gap MAE;
        # use exactly the same definition here.
        first_gaps = [suffix[0][2] for suffix in suffixes if suffix]
        remaining = [sum(event[2] for event in suffix) for suffix in suffixes]
        rows.append({"Modell": name, "Fall": str(pick.Fall), "Prefixlänge": len(prefix),
            **event_scores(true_tokens, [[event[0] for event in suffix] for suffix in suffixes]),
            **time_scores(true_first, true_remaining, first_gaps, remaining)})

    save("running")
    for start in range(0, 1000, 10):
        encoded, infos = [], []
        for _, pick in picks.iloc[start:start+10].iterrows():
            case_id, k = str(pick.Fall), int(pick["Prefixlänge"])
            assert case_id in set(map(str, model.test_cases))
            part = cases[case_id]
            prefix = part.iloc[:k].to_dict("records")
            work = WorkState.from_events(prefix)
            tokens = (part["concept:name"].astype(str)+"|"+part["lifecycle:transition"].astype(str)).tolist()[k:]
            first_gap = (part["time:timestamp"].iloc[k]-part["time:timestamp"].iloc[k-1]).total_seconds()
            remaining = (part["time:timestamp"].iloc[-1]-part["time:timestamp"].iloc[k-1]).total_seconds()
            reference_truth = float(pick["Restzeit echt (Tage)"])*86400
            assert np.isclose(remaining, reference_truth, atol=1e-5, rtol=0), (case_id, remaining, reference_truth)
            encoded.append(model._encode(prefix)[:4])
            infos.append((pick, prefix, tokens, first_gap, remaining, work))
        try:
            drawn = model.draw_cases(encoded, 100, work_states=[info[-1] for info in infos],
                previous_tokens=[info[-1].previous for info in infos])
            for info, suffixes in zip(infos, drawn):
                record_output(info, suffixes)
        except SuffixSamplingError:
            # Identify unsupported individual prefixes; don't discard a whole
            # batch because a different prefix exhausted rejection attempts.
            for entry, info in zip(encoded, infos):
                try:
                    drawn = model.draw_cases([entry], 100, work_states=[info[-1]], previous_tokens=[info[-1].previous])[0]
                    record_output(info, drawn)
                except SuffixSamplingError as error:
                    failures.append({"Fall": str(info[0].Fall), "Prefixlänge": len(info[1]), "error": str(error)})
        if (start+10) % 50 == 0:
            save("running")
            print("VALIDATION", name, start+10, "of 1000; failures", len(failures), flush=True)
    save("complete" if not failures else "incomplete")
    return name


def report(source, out):
    records, paired = [], []
    all_complete = True
    for name in LABELS:
        old = reference(name)
        new = pd.read_csv(out/name/"prefixes.csv")
        status = json.loads((out/name/"status.json").read_text())
        complete = status["status"] == "complete" and len(new) == 1000
        all_complete &= complete
        merged = old.merge(new, on=["Fall", "Prefixlänge"], suffixes=("_alt", "_neu"), validate="one_to_one")
        assert len(merged) == len(new)
        for version, frame in (("ursprünglich", old), ("korrigiert", new)):
            row = frame.drop(columns=["Modell", "Fall", "Prefixlänge"]).mean(numeric_only=True).to_dict()
            records.append({"Modell": name, "Version": version, "Prefixe": len(frame), **row})
        delta = merged["Restzeit MAE (Tage)_neu"]-merged["Restzeit MAE (Tage)_alt"]
        rng = np.random.default_rng(42)
        boot = np.asarray([rng.choice(delta.to_numpy(), len(delta), replace=True).mean() for _ in range(5000)])
        paired.append({"model": name, "complete_1000_prefix_comparison": complete,
            "paired_prefixes": len(merged), "mae_new_minus_old_days": float(delta.mean()),
            "paired_case_bootstrap95_low": float(np.quantile(boot, .025)),
            "paired_case_bootstrap95_high": float(np.quantile(boot, .975))})
        merged.to_csv(out/f"{name}_paired_prefixes.csv", index=False)
    summary = pd.DataFrame(records)
    summary.to_csv(out/"comparison.csv", index=False)
    write(out/"paired_mae_difference.json", paired)
    labels = [("Schritt 1", "Treffer Schritt 1 (%)", 100), ("Schritt 2", "Treffer Schritt 2 (%)", 100),
        ("Schritt 3", "Treffer Schritt 3 (%)", 100), ("Schritt 5", "Treffer Schritt 5 (%)", 100),
        ("Schritt 10", "Treffer Schritt 10 (%)", 100), ("Schritt 20", "Treffer Schritt 20 (%)", 100),
        ("DLS", "Sequenzähnlichkeit DLS", 1), ("Suffixlänge-MAE", "Suffixlängen-MAE (Events)", 1),
        ("Abstand Schritt 1 MAE (h)", "Erster Abstand MAE (h)", 1),
        ("Restzeit MAE (Tage)", "Restzeit-MAE (Tage)", 1), ("Restzeit Fehler (Tage)", "Restzeit-Bias (Tage)", 1),
        ("Restzeit CRPS (Tage)", "Restzeit CRPS (Tage)", 1), ("Abdeckung 80 %", "80-%-Abdeckung (%)", 100)]
    indexed = summary.set_index(["Modell", "Version"])
    lines = ["**D/OE: dieselben 1.000 Testprefixe, jeweils 100 Ziehungen**", "",
        "Die ursprünglichen Referenzen sind D 7,81438 und OE 7,98412 Tage Restzeit-MAE. "
        "Die gespeicherten ursprünglichen Einzelwerte werden mit den korrigierten Modellkopien auf identischen "
        "Fall-/Prefixschlüsseln verglichen. Die echte Restzeit wurde für jeden Prefix gegen die Referenz geprüft. "
        "Die ursprünglichen Modelle hatten noch keine Übergangssperre; die neue Variante umfasst Zustands-/EOS-Regeln, "
        "Trainings-Übergangsregeln und das zusätzliche bedingte empirische Zeitmodell. "
        "Neuronale Gewichte und die laufende Online-Auswertung werden nicht verändert.", "",
        "| Kennzahl | D ursprünglich | D korrigiert | OE ursprünglich | OE korrigiert |",
        "|---|---:|---:|---:|---:|"]
    for key, label, factor in labels:
        values = [indexed.loc[(name, version), key]*factor for name in LABELS for version in ("ursprünglich", "korrigiert")]
        lines.append("| "+label+" | "+" | ".join(f"{value:.4f}" for value in values)+" |")
    lines += ["", "Gepaarte Änderung der Restzeit-MAE, neu minus alt (negative Werte bedeuten Verbesserung). "
        "95-%-Intervalle: 5.000 Bootstrap-Wiederholungen über die gemeinsamen Testfälle; kein Intervall eines Interventionseffekts.", ""]
    for row in paired:
        lines.append(f"- {row['model']}: {row['mae_new_minus_old_days']:+.4f} Tage, "
            f"95-%-Intervall [{row['paired_case_bootstrap95_low']:+.4f}; {row['paired_case_bootstrap95_high']:+.4f}].")
    if not all_complete:
        lines += ["", "**Unvollständig:** Mindestens ein Modell konnte nicht für alle 1.000 Prefixe vollständige unterstützte "
            "Suffixe liefern. Die neue Gesamt-MAE ist dann nicht direkt mit der ursprünglichen 1.000-Prefix-MAE vergleichbar. "
            "Die gepaarte Differenz bezieht sich ausschließlich auf die tatsächlich gemeinsam ausgewerteten Prefixe. "
            "Fehlgeschlagene Prefixe stehen in den failures.json-Dateien."]
    (out/"bericht.md").write_text("\n".join(lines)+"\n")
    write(out/"status.json", {"status": "complete" if all_complete else "incomplete", "updated_at": now(),
        "requested_prefixes_per_model": 1000, "draws_per_prefix": 100,
        "report": str(out/"bericht.md")})
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    original = {name: reference(name) for name in LABELS}
    assert original["D"][["Fall", "Prefixlänge"]].equals(original["OE"][["Fall", "Prefixlänge"]])
    files = [Path(__file__), *REF_FILES.values(), source/"catalog.yaml", source/"event_costs.yaml",
        source/"models/D_stateful.pkl", source/"models/OE_stateful.pkl", ROOT/"prescriptive/estimators/psp_model.py",
        ROOT/"prescriptive/estimators/conditional_gaps.py", ROOT/"process-simulation-engine/interventions/work_state.py"]
    hashes = {str(path): sha(path) for path in files}
    write(out/"manifest.json", {"prepared_at": now(), "prefixes_per_model": 1000, "draws_per_prefix": 100,
        "batch_size": 10, "torch_initial_seed": 0, "workers": 2, "source_sha256": hashes,
        "baseline_mae_days": {name: float(frame["Restzeit MAE (Tage)"].mean()) for name, frame in original.items()}})
    write(out/"status.json", {"status": "running", "updated_at": now(), "completed_models": []})
    try:
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(evaluate, (name, source, out)) for name in LABELS]
            complete = []
            for future in as_completed(futures):
                complete.append(future.result())
                write(out/"status.json", {"status": "running", "updated_at": now(), "completed_models": complete})
        for path, expected in hashes.items():
            assert sha(path) == expected, f"Source changed during evaluation: {path}"
        report(source, out)
    except BaseException as error:
        write(out/"status.json", {"status": "failed", "updated_at": now(), "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
