"""Repeat the existing PSP-D event-cost evaluation with 300 draws, then report."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
PACKAGE = OUT.parents[1]
ROOT = PACKAGE.parents[2]
REFERENCE = PACKAGE / "results/vergleich_200_faelle_30_ziehungen_seed0_20261113"


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


def check(manifest):
    for name, digest in manifest["sha256"].items():
        if sha(Path(name)) != digest:
            raise RuntimeError(f"Comparison input changed: {name}")


def prepare():
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Run already prepared; inspect its status before restarting")
    original = read(REFERENCE / "manifest.json")
    for name, digest in original["source_sha256"].items():
        if sha(Path(name)) != digest:
            raise RuntimeError(f"Source differs from the 30-draw reference: {name}")
    model = Path(original["psp_model"])
    if sha(model) != original["psp_sha256"]:
        raise RuntimeError("PSP weights differ from the 30-draw reference")
    for filename in ("event_costs.yaml", "catalog.yaml"):
        (OUT / filename).write_bytes((REFERENCE / filename).read_bytes())
    command = [sys.executable, "-u", "-W", "ignore::FutureWarning", "-m",
               "prescriptive.policies.ev_policy_event_costs.evaluate",
               "--policies", "bank", "psp", "--psp-model", str(model),
               "--samples", "300", "--scored-cases", "200", "--seed", "0", "--seeds", "1",
               "--warmup", "700", "--cooldown", "1300", "--workers", "1",
               "--catalog", str(OUT / "catalog.yaml"), "--cost-config", str(OUT / "event_costs.yaml"),
               "--out", str(OUT / "online")]
    sources = [Path(name) for name in original["source_sha256"]]
    sources += [model, Path(__file__).resolve(), OUT / "catalog.yaml", OUT / "event_costs.yaml",
                REFERENCE / "online/comparison.csv", REFERENCE / "online/run.json",
                REFERENCE / "online/bank_seed0/case_costs.csv",
                REFERENCE / "online/psp_seed0/case_costs.csv"]
    manifest = {"prepared_at": now(), "objective": "negative_event_costs", "variant": "d",
                "samples_per_option": 300, "scored_cases": 200, "seed": 0, "seeds": 1,
                "warmup": 700, "cooldown": 1300, "reference_30_draws": str(REFERENCE),
                "psp_model": str(model), "cost_fingerprint": original["cost_fingerprint"],
                "command": command, "sha256": {str(path): sha(path) for path in sources}}
    write(OUT / "manifest.json", manifest)
    return manifest


def report(manifest):
    import numpy as np
    import pandas as pd

    run = read(OUT / "online/run.json")
    prior_run = read(REFERENCE / "online/run.json")
    assert run["status"] == prior_run["status"] == "complete"
    assert run["samples"] == 300 and prior_run["samples"] == 30
    assert run["points"] == prior_run["points"]
    for key in ("scored_cases", "warmup", "cooldown", "seeds", "seed", "cost_fingerprint"):
        assert run[key] == prior_run[key], key
    assert run["cost_fingerprint"] == manifest["cost_fingerprint"]
    current = pd.read_csv(OUT / "online/comparison.csv").set_index("Policy")
    previous = pd.read_csv(REFERENCE / "online/comparison.csv").set_index("Policy")
    bills = {}
    for key, folder in (("bank", OUT / "online/bank_seed0"),
                        ("bank_previous", REFERENCE / "online/bank_seed0"),
                        ("psp_30", REFERENCE / "online/psp_seed0"),
                        ("psp_300", OUT / "online/psp_seed0")):
        bills[key] = pd.read_csv(folder / "case_costs.csv").set_index("case:concept:name").sort_index()
        assert len(bills[key]) == 200
        assert bills[key].index.equals(bills["bank"].index)
    assert np.allclose(bills["bank"].to_numpy(), bills["bank_previous"].to_numpy(), rtol=0, atol=1e-10)
    rows = pd.DataFrame([current.loc["bank"], previous.loc["psp"], current.loc["psp"]],
                        index=pd.Index(["Bank", "PSP-D, 30 Ziehungen", "PSP-D, 300 Ziehungen"], name="Verfahren"))
    bank = float(current.loc["bank", "Eventkosten je Fall (€)"])
    rows["Ersparnis ggü. Bank (€)"] = bank - rows["Eventkosten je Fall (€)"]
    rows["Ersparnis ggü. Bank (%)"] = rows["Ersparnis ggü. Bank (€)"] / bank * 100
    for name, row in zip(("bank", "psp_30", "psp_300"), rows.to_dict("records")):
        assert np.isclose(bills[name]["total_eur"].mean(), row["Eventkosten je Fall (€)"])
    rows.to_csv(OUT / "comparison_30_vs_300.csv")
    pd.DataFrame({name: bills[name]["total_eur"] for name in ("bank", "psp_30", "psp_300")}).to_csv(
        OUT / "paired_case_costs.csv")
    delta = rows.loc["PSP-D, 300 Ziehungen", "Eventkosten je Fall (€)"] - rows.loc[
        "PSP-D, 30 Ziehungen", "Eventkosten je Fall (€)"]
    lines = ["# PSP-D: Eventkosten mit 30 und 300 Ziehungen", "",
             "Jeweils 200 bewertete Fälle, Seed 0, 700 Warmup und 1.300 Cooldown. "
             "Identische Modellgewichte, Beispieltarife, Wahlpunkte und Ressourcenstrategie. "
             "Optionsabstand je Suffix gezogen. Einziger geänderter PSP-Parameter: Ziehungen je Option.", "",
             "| Verfahren | Kosten/Fall (€) | Ersparnis ggü. Bank (€) | Ersparnis (%) | Durchlaufzeit (Tage) |",
             "|---|---:|---:|---:|---:|"]
    for name, row in rows.iterrows():
        lines.append(f"| {name} | {row['Eventkosten je Fall (€)']:.2f} | "
                     f"{row['Ersparnis ggü. Bank (€)']:+.2f} | {row['Ersparnis ggü. Bank (%)']:+.2f} | "
                     f"{row['Durchlaufzeit (Tage)']:.2f} |")
    lines += ["", "Niedrigere Kosten sind besser. Positive Ersparnisse bedeuten geringere Kosten als die Bank.",
              f"300 minus 30 Ziehungen: {delta:+.2f} €/Fall.", "",
              "Die Bank wurde neu simuliert. Alle Kostenbestandteile und Arbeitszeiten der 200 Bankfälle "
              "stimmen mit der Referenz überein. Die Fall-IDs stimmen auch zwischen beiden PSP-Läufen überein.", "",
              "Ein Seed mit 200 Fällen; kein Konfidenzintervall über mehrere Seeds. "
              "Kosten stammen aus den tatsächlich simulierten Events.", "",
              "Kostenbestandteile und weitere Kennzahlen: comparison_30_vs_300.csv. "
              "Einzelkosten: paired_case_costs.csv; Events und Entscheidungen: online/."]
    (OUT / "auswertung.md").write_text("\n".join(lines) + "\n")
    write(OUT / "validation.json", {"status": "passed", "same_200_case_ids": True,
          "bank_case_cost_components_identical": True, "same_model_tariffs_points_and_window": True,
          "completed_at": now()})
    print(rows[["Eventkosten je Fall (€)", "Ersparnis ggü. Bank (%)", "Durchlaufzeit (Tage)"]].to_string(), flush=True)


def main():
    status = {"pid": os.getpid(), "started_at": now(), "samples_per_option": 300,
              "scored_cases": 200, "seed": 0}

    def update(phase, **extra):
        status.update(status=phase, updated_at=now(), **extra)
        write(OUT / "status.json", status)
        print(json.dumps(status, ensure_ascii=False), flush=True)

    try:
        manifest = prepare()
        check(manifest)
        update("online")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLCONFIGDIR=str(OUT / "matplotlib"))
        with (OUT / "online.log").open("x") as log:
            started = time.monotonic()
            process = subprocess.Popen(manifest["command"], cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            update("online", evaluator_pid=process.pid, heartbeat_at=now())
            while process.poll() is None:
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    extra = {"evaluator_pid": process.pid, "elapsed_seconds": round(time.monotonic() - started),
                             "heartbeat_at": now()}
                    stat = Path(f"/proc/{process.pid}/stat")
                    if stat.exists():
                        fields = stat.read_text().split()
                        extra["evaluator_cpu_seconds"] = round((int(fields[13]) + int(fields[14])) / os.sysconf("SC_CLK_TCK"), 1)
                    update("online", **extra)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, manifest["command"])
        check(manifest)
        update("reporting")
        report(manifest)
        update("complete", report=str(OUT / "auswertung.md"))
    except BaseException as error:
        update("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
