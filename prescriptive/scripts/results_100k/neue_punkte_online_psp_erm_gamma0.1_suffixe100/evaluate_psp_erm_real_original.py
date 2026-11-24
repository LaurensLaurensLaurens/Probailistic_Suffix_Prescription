"""Dauerhafte PSP-ERM-Auswertung auf BPIC17, mit Checkpoints und automatischem CATE-ERM-Vergleich.

Ein CPU-Prozess je Seed, bis zu --workers parallel. Neue Prozesse starten nur bei ausreichendem freiem RAM.
Die Bank-Durchlaufzeiten werden aus dem identischen, geprüften CATE-ERM-Lauf übernommen.
Worker schreiben alle 30 Sekunden Fortschritt und nach Abschluss einen atomaren Checkpoint mit allen Durchlaufzeiten.
Erneuter Aufruf setzt ausschließlich fehlende Seeds fort. Nach allen Seeds entstehen CSVs, Diagramm und Berichte.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
PRESCRIPTIVE = ROOT / "prescriptive"
DEFAULT_OUT = PRESCRIPTIVE / "scripts/results_100k/neue_punkte_online_psp_erm_gamma0.1_suffixe100"
REFERENCE = PRESCRIPTIVE / "scripts/results_100k/neue_punkte_online_cate_erm_gamma0.1"
DEFAULT_MODEL = PRESCRIPTIVE / "models/psp_model_durchlaufzeit_real_alle_s60_normal.pkl"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    def convert(item):
        if hasattr(item, "tolist"):
            return item.tolist()
        raise TypeError(type(item).__name__)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, default=convert, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def environment(out):
    sys.path.insert(0, str(ROOT))
    os.environ["PSP_CATALOG"] = str(out / "catalog.yaml")
    os.environ["PRESCRIPTIVE_OUTCOME_BPIC17"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(out / "matplotlib_cache")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        os.environ[key] = "1"
    from prescriptive.helpers.datasets import add_paths
    add_paths()


def initialise(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for directory in ("checkpoints", "progress", "logs"):
        (out / directory).mkdir(exist_ok=True)
    reference = json.loads((REFERENCE / "laufparameter.json").read_text())
    assert reference["gamma"] == args.gamma
    assert reference["simulation_seeds"] == list(range(10))
    for relative in ("process-simulation-engine/simulation/dataset_simulator.py",
                     "process-simulation-engine/simulation/case_engine_lifecycle_aware.py",
                     "process-simulation-engine/models/next_event/bpic17/config.json",
                     "process-simulation-engine/models/next_event/bpic17/model.weights.h5"):
        assert digest(ROOT / relative) == reference["source_sha256"][relative], f"Simulator geändert: {relative}"
    catalog = REFERENCE / "catalog.yaml"
    assert digest(catalog) == reference["catalog_sha256"]
    configuration = {"dataset": "bpic17", "gamma": args.gamma, "samples": args.samples,
                     "model": str(args.model.resolve()), "model_sha256": digest(args.model),
                     "seeds": list(range(10)), "n_cases": 3000, "window": [700, 1700],
                     "evaluated_cases": 1000, "first_only": [], "value_source": "suffix",
                     "catalog_sha256": digest(catalog), "reference": str(REFERENCE),
                     "simulator_sha256": {key: value for key, value in reference["source_sha256"].items()
                                          if key.startswith("process-simulation-engine/")},
                     "source_sha256": digest(__file__)}
    manifest = out / "laufparameter.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == configuration, "Laufparameter oder Code verändert; anderen Ergebnisordner verwenden"
    else:
        shutil.copy2(catalog, out / "catalog.yaml")
        atomic(manifest, configuration)
    assert digest(out / "catalog.yaml") == configuration["catalog_sha256"]
    for seed in configuration["seeds"]:
        source = REFERENCE / f"checkpoints/bank_seed{seed}.json"
        bank = json.loads(source.read_text())
        assert bank["seed"] == seed and bank["Policy"] == "bank" and len(bank["_days"]) == 1000
        target = out / f"checkpoints/bank_seed{seed}.json"
        if target.exists():
            assert digest(target) == digest(source)
        else:
            shutil.copy2(source, target)
    return configuration


def worker(out, seed):
    environment(out)
    started = time.monotonic()
    progress = out / f"progress/seed{seed}.json"
    config = json.loads((out / "laufparameter.json").read_text())
    atomic(progress, {"seed": seed, "state": "initialising", "updated": now(), "questions": 0})
    assert digest(config["model"]) == config["model_sha256"]
    from helpers.datasets import load_dataset
    from preprocessing import preprocess
    from prescriptive.evaluation import online_evaluation as online
    from simulation.dataset_simulator import is_trained, model_folder
    assert is_trained(model_folder("bpic17"))
    raw, _, _ = load_dataset("bpic17")
    log = preprocess("bpic17", raw)
    del raw
    params = {"psp_erm": {"model": config["model"], "gamma": config["gamma"], "samples": config["samples"]}}
    online._init_worker("bpic17", log, None, None, params)
    factory = online._policy
    seen = set()
    last_write = 0.0

    def traced_factory(name, policy_seed):
        nonlocal last_write
        policy = factory(name, policy_seed)
        assert policy.model.value_source() == "suffix"
        assert policy.model._outcome_kind() == "negative_cycle_days"
        assert set(policy.model.points) == set(json.loads((REFERENCE / "laufparameter.json").read_text())["choice_points"])
        choose = policy.choose
        atomic(progress, {"seed": seed, "state": "running", "updated": now(), "questions": 0,
                          "elapsed_seconds": time.monotonic() - started})

        def traced_choose(point, case, options, moment, engine):
            nonlocal last_write
            result = choose(point, case, options, moment, engine)
            seen.add(case.case_id)
            current = time.monotonic()
            if current - last_write >= 30:
                atomic(progress, {"seed": seed, "state": "running", "updated": now(),
                                  "questions": policy.stats["Fragen (ERM)"], "cases_with_decisions": len(seen),
                                  "elapsed_seconds": current - started, "stats": policy.stats})
                last_write = current
            return result
        policy.choose = traced_choose
        return policy

    online._policy = traced_factory
    row = online._run(("psp_erm", seed, config["n_cases"], tuple(config["window"]), None))
    assert row["Fälle"] == len(row["_days"]) == 1000
    row["Suffixe je Option"] = config["samples"]
    atomic(out / f"checkpoints/psp_erm_seed{seed}.json", row)
    atomic(progress, {"seed": seed, "state": "complete", "updated": now(),
                      "elapsed_seconds": time.monotonic() - started, "questions": row["Fragen (ERM)"]})


def completed(out):
    result = []
    for seed in range(10):
        path = out / f"checkpoints/psp_erm_seed{seed}.json"
        if path.exists():
            row = json.loads(path.read_text())
            assert row["Policy"] == "psp_erm" and row["seed"] == seed
            assert row["Fälle"] == len(row["_days"]) == 1000
            result.append(seed)
    return result


def report(out):
    environment(out)
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from prescriptive.evaluation.online_evaluation import compare
    from helpers.report import Report, set_outcome, uplift_chart
    config = json.loads((out / "laufparameter.json").read_text())
    assert completed(out) == list(range(10))
    rows = [json.loads((out / f"checkpoints/{name}_seed{seed}.json").read_text())
            for seed in range(10) for name in ("bank", "psp_erm")]
    for row in rows:
        row["_days"] = np.array(row["_days"], dtype=float)
    table, runs = compare(rows)
    table.to_csv(out / "ergebnis.csv")
    runs.to_csv(out / "laeufe.csv", index=False)
    set_outcome("negative_cycle_days")
    document = Report(str(out), "PSP-ERM: BPIC17, γ=0,1, 100 Suffixe je Option")
    document.heading("10 Seeds, je 3.000 Ankünfte, Warmup 700, Cooldown 1.300; je 1.000 gewertete Fälle")
    sub = "prescription/PSP_ERM"
    document.figure("bpic17", "online_diagramm", uplift_chart(table, runs), "PSP-ERM gegen Bank", sub=sub)
    document.table("bpic17", "online", table.round(3), "Mittel über zehn Seeds", sub=sub)
    document.table("bpic17", "online_laeufe", runs.round(3), "Einzelläufe", sub=sub)
    table.to_csv(out / sub / "bpic17_online.csv")
    runs.to_csv(out / sub / "bpic17_online_laeufe.csv")
    cate_rows = [json.loads(path.read_text()) for path in sorted((REFERENCE / "checkpoints").glob("*.json"))]
    combined_rows = cate_rows + [row for row in rows if row["Policy"] == "psp_erm"]
    for row in combined_rows:
        row["_days"] = np.array(row["_days"], dtype=float)
    all_table, all_runs = compare(combined_rows)
    all_table.to_csv(out / "vergleich_alle_erm.csv")
    all_runs.to_csv(out / "vergleich_alle_erm_laeufe.csv", index=False)
    metric = f"ERM γ={config['gamma']:g} (Tage)"
    metrics = ["Durchlaufzeit (Tage)", "Durchlaufzeit P90 (Tage)", "Durchlaufzeit P95 (Tage)", "CVaR 90 % (Tage)", metric]
    order = ["bank", "s_learner_erm", "x_learner_erm", "causal_forest_erm", "psp_erm"]
    summary = all_table.loc[order, metrics + [f"{metric} ggü. Bank", f"{metric} ggü. Bank 95-%-KI unten", f"{metric} ggü. Bank 95-%-KI oben"]]
    summary.to_csv(out / "zusammenfassung.csv")
    names = ["Bank", "S-Learner ERM", "X-Learner ERM", "Causal Forest ERM", "PSP-ERM"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    colors = ["#2a78d6", "#db7b27", "#35976a", "#8d58ba"]
    from scipy import stats
    bank = all_runs[all_runs.Policy == "bank"].set_index("seed")
    differences = []
    for ax, column, title in zip(axes, ["Durchlaufzeit (Tage)", metric], ["Mittlere Durchlaufzeit", "Entropisches Risiko, γ=0,1"]):
        for i, policy in enumerate(order[1:]):
            values = all_runs[all_runs.Policy == policy].set_index("seed")[column] - bank[column]
            mean = values.mean()
            half = stats.t.ppf(.975, 9) * values.std(ddof=1) / np.sqrt(10)
            ax.errorbar(mean, i, xerr=half, fmt="o", color=colors[i], capsize=4)
            differences.append({"Policy": policy, "Kennzahl": column, "Differenz zur Bank": mean,
                                "95-%-KI unten": mean - half, "95-%-KI oben": mean + half})
        ax.axvline(0, color="#555", linestyle="--", linewidth=1)
        ax.set_yticks(range(4), names[1:])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Differenz zur Bank (Tage), 95-%-KI\n← kleiner ist besser")
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "vergleich_alle_erm.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(differences).to_csv(out / "differenzen_zur_bank.csv", index=False)
    psp = table.loc["psp_erm"]
    low, high = psp[f"{metric} ggü. Bank 95-%-KI unten"], psp[f"{metric} ggü. Bank 95-%-KI oben"]
    verdict = "gesicherter ERM-Vorteil" if high < 0 else "gesicherter ERM-Nachteil" if low > 0 else "kein statistisch gesicherter ERM-Unterschied"
    question_count = int(runs.loc[runs.Policy == "psp_erm", "Fragen (ERM)"].sum())
    different = int(runs.loc[runs.Policy == "psp_erm", "ERM anders als das Mittel"].sum())
    # Standard-Markdown ohne optionale tabulate-Abhängigkeit.
    compact = summary[metrics].copy()
    compact.insert(0, "Policy", names)
    lines = ["| " + " | ".join(compact.columns) + " |", "| " + " | ".join(["---"] * len(compact.columns)) + " |"]
    lines += ["| " + " | ".join(str(v) if isinstance(v, str) else f"{v:.3f}" for v in values) + " |"
              for values in compact.itertuples(index=False, name=None)]
    markdown = ["# PSP-ERM-Auswertung", "", "BPIC17, γ=0,1 / Tag; zehn Seeds (0–9), je 3.000 Ankünfte, "
                "Warmup 700, Cooldown 1.300, je 1.000 gewertete Fälle. Vier Control-Flow-Punkte wie bei CATE-ERM, "
                "Zuteilung random, keine first_only-Einschränkung. Das vorhandene PSP real_alle_s60_normal "
                "bewertet jede Option mit 100 Suffixziehungen; das Modell wurde nicht neu trainiert.", "",
                "## Ergebnisse", "", "Alle Werte in Tagen; kleiner ist besser. Mittel über die je Seed berechneten Kennzahlen.", "",
                *lines, "", f"PSP-ERM: Differenz des empirischen ERM zur Bank {psp[f'{metric} ggü. Bank']:+.3f} Tage, "
                f"95-%-KI [{low:+.3f}; {high:+.3f}]: {verdict}.", "",
                f"Bei {different:,} von {question_count:,} Entscheidungen ({100 * different / max(question_count, 1):.2f} %) "
                "wählte ERM anders als die Mittelwertbewertung derselben gezogenen Suffixe. "
                "Dies ist kein separater EV-Simulationslauf: Nach einer anderen Wahl würde sich der weitere Verlauf ändern.", "",
                "![Gepaarter Vergleich](vergleich_alle_erm.png)", "", "## Vergleichbarkeit und Grenzen", "",
                "Die Bank- und CATE-ERM-Rohdaten stammen aus dem vorherigen identischen Lauf. Katalog und "
                "Simulator-Prüfsummen wurden vor dem Start geprüft; die Bank wurde nicht unnötig neu simuliert. "
                "Die KIs beruhen auf zehn gepaarten Seed-Differenzen (t-Intervall), nicht auf unabhängigen Einzelentscheidungen. "
                "Das empirische ERM reagiert stark auf seltene lange Fälle; die Modellschätzungen und die "
                "Übertragbarkeit auf echte Prozesse sind durch diese KIs nicht abgesichert.", "",
                "CATE-ERM: Der S-Learner schätzte auf allen Testproben Nulleffekte und entscheidet bei Gleichstand zufällig. "
                "Bei γ=0,1 konzentrierte sich das exponentielle CATE-Trainingsgewicht je Wahlpunkt zu 73–100 % "
                "auf einen einzelnen Fall. Diese Einschränkung gilt bei der Interpretation der Vergleichsmethoden.", "",
                "## Dateien", "", "- bericht.html / ergebnis.csv / laeufe.csv: PSP-ERM und Bank.",
                "- vergleich_alle_erm.csv / vergleich_alle_erm_laeufe.csv: Vergleich mit allen drei CATE-ERM-Verfahren.",
                "- zusammenfassung.csv / differenzen_zur_bank.csv: kompakte Tabellen und Konfidenzintervalle.",
                "- checkpoints/: Durchlaufzeiten je Fall, einschließlich der übernommenen Bank-Läufe.",
                "- laufparameter.json / catalog.yaml / logs/: Parameter, Prüfsummen und Seed-Logs.", ""]
    (out / "auswertung.md").write_text("\n".join(markdown))
    print(summary.to_string(), flush=True)


def available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2
    return 0.0


def supervise(args):
    config = initialise(args)
    out = args.out
    active = {}
    attempts = {seed: 0 for seed in range(10)}
    pending = [seed for seed in range(10) if seed not in completed(out)]
    next_start = 0.0
    print(f"Start {now()}: {len(pending)} fehlende Seeds, bis zu {args.workers} Prozesse", flush=True)
    while pending or active:
        for seed, (process, stream, launched) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            del active[seed]
            if code == 0 and seed in completed(out):
                print(f"Seed {seed} vollständig gespeichert ({now()})", flush=True)
            elif attempts[seed] < 2:
                print(f"Seed {seed} beendet mit {code}; einmaliger Neustart", flush=True)
                pending.append(seed)
            else:
                raise RuntimeError(f"Seed {seed} zweimal fehlgeschlagen; siehe logs/seed{seed}.log")
        startup = sum(time.monotonic() - launched < 30 for _, _, launched in active.values())
        if pending and len(active) < args.workers and time.monotonic() >= next_start and available_gib() >= 6 + 2.5 * (1 + startup):
            seed = pending.pop(0)
            stream = (out / f"logs/seed{seed}.log").open("a")
            process = subprocess.Popen([sys.executable, "-u", __file__, "--out", str(out), "--worker-seed", str(seed)],
                                       stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, cwd=str(ROOT))
            active[seed] = (process, stream, time.monotonic())
            attempts[seed] += 1
            next_start = time.monotonic() + 5
            print(f"Seed {seed} gestartet; verfügbar {available_gib():.1f} GiB RAM", flush=True)
        atomic(out / "status.json", {"state": "running", "updated": now(), "completed_seeds": completed(out),
                                    "active_seeds": sorted(active), "pending_seeds": pending,
                                    "available_ram_gib": available_gib(), "gamma": config["gamma"], "samples": config["samples"]})
        time.sleep(5)
    report(out)
    atomic(out / "status.json", {"state": "complete", "updated": now(), "completed_seeds": completed(out),
                                "gamma": config["gamma"], "samples": config["samples"]})
    print(f"Fertig {now()}: {out / 'auswertung.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gamma", type=float, default=.1)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--worker-seed", type=int, choices=range(10), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    assert args.gamma == .1 and args.samples > 0 and 0 < args.workers <= 6
    if args.worker_seed is not None:
        worker(args.out, args.worker_seed)
    elif args.report_only:
        report(args.out)
    else:
        supervise(args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        # Seed-Logs und vorhandene Checkpoints bleiben für die Fortsetzung erhalten.
        if "--worker-seed" not in sys.argv:
            out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else DEFAULT_OUT
            if out.exists():
                atomic(out / "status.json", {"state": "failed", "updated": now(), "error": traceback.format_exc()})
        raise
