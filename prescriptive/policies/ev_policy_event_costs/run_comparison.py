"""Train all three event-cost CATE methods and compare them with PSP and bank.

Persistent logs/status allow the CPU pipeline to continue independently of chat.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from .costs import DEFAULT_CONFIG, EventCosts

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
MODULE = "prescriptive.policies.ev_policy_event_costs"
DEFAULT_PSP = ROOT / "prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_klassen_option_encoder.pkl"
DEFAULT_CATALOG = ROOT / "prescriptive/data/catalog_neue_punkte.yaml"
LABELS = {"bank": "Bank", "psp": "PSP (d), 30 Ziehungen", "s_learner": "S-Learner",
          "x_learner": "X-Learner", "causal_forest": "Causal Forest"}


def now():
    return datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds")


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(out, psp, catalog, config):
    if not psp.is_file() or not catalog.is_file():
        raise FileNotFoundError("PSP model and catalog must exist")
    costs = EventCosts(config)
    out.mkdir(parents=True, exist_ok=False)
    (out / "event_costs.yaml").write_text(yaml.safe_dump(costs.to_dict(), sort_keys=False, allow_unicode=True))
    (out / "catalog.yaml").write_bytes(catalog.read_bytes())
    sources = list(PACKAGE.glob("*.py")) + [ROOT / "prescriptive/estimators/cate_model.py",
               ROOT / "prescriptive/estimators/psp_model.py",
               ROOT / "process-simulation-engine/simulation/case_engine_lifecycle_aware.py",
               ROOT / "process-simulation-engine/resources/case_allocation.py"]
    train = [sys.executable, "-u", "-W", "ignore::FutureWarning", "-m", MODULE + ".train",
             "--tag", "real_alle", "--learner", "s", "x", "cf", "--cases", "0", "--seed", "42",
             "--catalog", str(out / "catalog.yaml"), "--cost-config", str(out / "event_costs.yaml"),
             "--out", str(out / "models")]
    evaluate = [sys.executable, "-u", "-W", "ignore::FutureWarning", "-m", MODULE + ".evaluate",
                "--policies", *LABELS, "--psp-model", str(psp), "--samples", "30", "--scored-cases", "200",
                "--warmup", "700", "--cooldown", "1300", "--seeds", "1", "--seed", "0", "--workers", "1",
                "--catalog", str(out / "catalog.yaml"), "--cost-config", str(out / "event_costs.yaml"),
                "--out", str(out / "online")]
    for name, learner in (("s_learner", "s"), ("x_learner", "x"), ("causal_forest", "cf")):
        evaluate += ["--cate-model", f"{name}={out / 'models' / ('cate_model_' + learner + '_event_costs.pkl')}"]
    write(out / "manifest.json", {"prepared_at": now(), "training_seed": 42, "training_cases": "all outside test split",
          "seed": 0, "scored_cases": 200, "samples_per_option": 30, "warmup": 700, "cooldown": 1300,
          "policies": list(LABELS), "objective": "negative_event_costs", "cost_fingerprint": costs.fingerprint,
          "psp_model": str(psp), "psp_sha256": sha(psp), "source_catalog": str(catalog),
          "catalog_sha256": sha(out / "catalog.yaml"), "train_command": train, "evaluate_command": evaluate,
          "source_sha256": {str(path): sha(path) for path in sources}})
    write(out / "status.json", {"status": "prepared", "updated_at": now()})


def check_sources(out, manifest):
    for name, digest in manifest["source_sha256"].items():
        if sha(Path(name)) != digest:
            raise RuntimeError(f"Source changed after preparation: {name}")
    if sha(Path(manifest["psp_model"])) != manifest["psp_sha256"]:
        raise RuntimeError("PSP model changed after preparation")
    if EventCosts(out / "event_costs.yaml").fingerprint != manifest["cost_fingerprint"]:
        raise RuntimeError("Tariffs changed after preparation")
    if sha(out / "catalog.yaml") != manifest["catalog_sha256"]:
        raise RuntimeError("Catalog changed after preparation")


def report(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = read(out / "online/run.json")
    training = read(out / "models/training.json")
    assert run["status"] == training["status"] == "complete"
    assert run["samples"] == 30 and run["scored_cases"] == 200 and run["seed"] == 0 and run["seeds"] == 1
    assert training["cost_fingerprint"] == run["cost_fingerprint"]
    table = pd.read_csv(out / "online/comparison.csv").set_index("Policy").loc[list(LABELS)]
    bills = {name: pd.read_csv(out / f"online/{name}_seed0/case_costs.csv").set_index("case:concept:name")
             for name in LABELS}
    for name, bill in bills.items():
        assert len(bill) == 200 and set(bill.index) == set(bills["bank"].index)
        assert np.isclose(bill["total_eur"].mean(), table.loc[name, "Eventkosten je Fall (€)"])
    paired = pd.DataFrame({name: bill["total_eur"] for name, bill in bills.items()})
    paired.to_csv(out / "paired_case_costs.csv")
    bank = float(table.loc["bank", "Eventkosten je Fall (€)"])
    table["Ersparnis ggü. Bank (€)"] = bank - table["Eventkosten je Fall (€)"]
    table.to_csv(out / "comparison.csv")
    lines = ["# EV-Vergleich nach Eventkosten", "",
             "200 ausgewertete Fälle je Verfahren, Seed 0; 700 Warmup und 1.300 Cooldown. "
             "PSP (d): 30 Ziehungen je Option, Optionsabstand je Suffix gezogen. "
             "Die Tarife sind die gemeinsam festgehaltenen Beispielwerte aus event_costs.yaml.", "",
             "| Verfahren | Kosten/Fall (€) | Ersparnis ggü. Bank (€) | Ersparnis (%) | Durchlaufzeit (Tage) |",
             "|---|---:|---:|---:|---:|"]
    for name, row in table.iterrows():
        lines.append(f"| {LABELS[name]} | {row['Eventkosten je Fall (€)']:.2f} | "
                     f"{row['Ersparnis ggü. Bank (€)']:+.2f} | {row['Uplift ggü. Bank (%)']:+.2f} | "
                     f"{row['Durchlaufzeit (Tage)']:.2f} |")
    lines += ["", "Niedrigere Kosten sind besser; positive Ersparnisse bedeuten Verbesserung gegenüber der Bank.", "",
              f"CATE-Training: {training['train_cases']:,} echte Fälle; {training['test_cases']:,} Fälle im äußeren "
              "Testsplit ausgeschlossen. S-Learner, X-Learner und Causal Forest verwenden dasselbe Kostenziel, "
              "dieselben Trainingsfälle und dieselben Wahlpunkte. PSP verwendet die vorhandenen trainierten Suffixgewichte.", "",
              "Alle Verfahren werden auf denselben 200 Fall-IDs verglichen. Die tatsächlichen Eventfolgen ändern sich "
              "durch die Entscheidungen. Die Kosten stammen aus den Simulatorereignissen, nicht aus den Vorhersagen.", "",
              "Ein Seed mit 200 Fällen ist ein Mini-Vergleich. Konfidenzintervalle über mehrere Seeds liegen hier nicht vor.", "",
              "![Kostenbestandteile je Verfahren](event_cost_comparison.png)", "",
              "Die einzelnen Kosten und Entscheidungen stehen unter online/<policy>_seed0/. "
              "paired_case_costs.csv stellt die Kosten der gleichen Fälle nebeneinander."]
    (out / "auswertung.md").write_text("\n".join(lines) + "\n")
    columns = ["Basiskosten je Fall (€)", "Lifecycle-Kosten je Fall (€)", "Zeitkosten je Fall (€)"]
    fig, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    left = np.zeros(len(table))
    for column, label, color in zip(columns, ("Basis", "Lifecycle", "Aktive Zeit"), ("#577590", "#43aa8b", "#f9c74f")):
        values = table[column].to_numpy()
        axis.barh(range(len(table)), values, left=left, label=label, color=color)
        left += values
    axis.set_yticks(range(len(table)), [LABELS[name] for name in table.index])
    axis.invert_yaxis()
    axis.axvline(bank, color="#444444", linestyle="--", linewidth=1)
    for index, value in enumerate(left):
        axis.text(value + max(left) * .015, index, f"{value:.2f} €", va="center", fontsize=9)
    axis.set_xlim(0, max(left) * 1.18)
    axis.set_xlabel("Mittlere Eventkosten je Fall (€)")
    axis.set_title("200 Fälle · Seed 0 · PSP mit 30 Ziehungen je Option")
    axis.legend(loc="lower right")
    fig.savefig(out / "event_cost_comparison.png", dpi=160)
    plt.close(fig)
    print(table[["Eventkosten je Fall (€)", "Ersparnis ggü. Bank (€)", "Uplift ggü. Bank (%)"]].to_string(), flush=True)


def run(out):
    manifest = read(out / "manifest.json")
    status = {"pid": os.getpid(), "started_at": now()}
    def update(phase, **extra):
        status.update(status=phase, updated_at=now(), **extra)
        write(out / "status.json", status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg", MPLCONFIGDIR=str(out / "matplotlib"),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        check_sources(out, manifest)
        update("training", learners=["s", "x", "cf"], training_cases="all outside test split")
        with (out / "training.log").open("x") as log:
            subprocess.run(manifest["train_command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        check_sources(out, manifest)
        update("online", scored_cases=200, psp_samples_per_option=30)
        with (out / "online.log").open("x") as log:
            subprocess.run(manifest["evaluate_command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        check_sources(out, manifest)
        update("reporting")
        os.environ["MPLCONFIGDIR"] = env["MPLCONFIGDIR"]
        report(out)
        update("complete", report=str(out / "auswertung.md"))
    except BaseException as error:
        update("failed", error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--psp-model", type=Path, default=DEFAULT_PSP)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--cost-config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.prepare:
        prepare(out, args.psp_model.resolve(), args.catalog.resolve(), args.cost_config.resolve())
        print(f"Prepared: {out}")
    elif args.launch:
        if read(out / "status.json")["status"] != "prepared":
            raise RuntimeError("Only a prepared run can be launched")
        with (out / "pipeline.log").open("x") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", MODULE + ".run_comparison", "--run", "--out", str(out)],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        (out / "pipeline.pid").write_text(str(process.pid) + "\n")
        print(json.dumps({"pid": process.pid, "out": str(out)}))
    else:
        run(out)


if __name__ == "__main__":
    main()
