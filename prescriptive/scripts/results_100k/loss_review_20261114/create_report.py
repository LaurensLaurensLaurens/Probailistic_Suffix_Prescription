"""Snapshot and review existing PSP training losses without changing training."""
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
P = ROOT / "prescriptive"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "matplotlib"))
MODELS = [
    ("Früher Nachtlauf", P / "data/PSP_lr1e4.log", "Früher Nachtlauf"),
    ("Echtlog, alte Wahlpunkte", P / "data/PSP_real_w3_10k.log", "Ursprüngliches Echtlog-PSP"),
    ("Echtlog, neue Wahlpunkte", P / "data/PSP_real_w3_10k_np.log", "Echtlog, neue Wahlpunkte"),
    ("s40", P / "data/PSP_real_np_s40_ohne_zukunft.log", "s40"),
    ("(a)", P / "data/PSP_real_alle_s60_loggaps.log", "a"),
    ("(b)", P / "data/PSP_real_alle_s60_loggaps_rest.log", "b"),
    ("(c)", P / "data/PSP_real_alle_s60_normal.log", "c"),
    ("(d)", P / "data/PSP_real_alle_s60_klassen.log", "d"),
    ("OE", P / "data/PSP_real_alle_s60_token_oe.log", "OE"),
    ("(c), TF 1", P / "data/PSP_real_alle_s60_normal_tf1.log", "c TF1"),
    ("(e)", P / "scripts/results_100k/e_suffixe30_cases200_seed0_20261113/pipeline.log", "e"),
    ("Neues D, TF 1 (echter Log)", P / "scripts/results_100k/d_tf1_training_20261114/pipeline.log", None),
    ("Neues D, TF 1 (RCT)", P / "scripts/results_100k/d_rct_tf1_training_20261114/pipeline.log", None),
]
NUMBER = r"([-+0-9.eE]+)"
EPOCH = re.compile(r"Epoch \[(\d+)/(\d+)\], Learning Rate: " + NUMBER + r", Teacher forcing ratio: " + NUMBER)
LOSSES = {"train_attenuated": r"Training: Avg Attenuated Training Loss: ",
          "validation_standard": r"Validation: Avg Standard Validation Loss: ",
          "validation_attenuated": r"Validation: Avg Attenuated Validation Loss: ",
          "scheduler_loss": r"Validation Loss for Scheduler: "}


def parse(text):
    blocks, rows, current = [], [], None
    for line in text.splitlines():
        match = EPOCH.search(line)
        if match:
            if current is not None:
                rows.append(current)
            epoch = int(match[1])
            if rows and epoch <= rows[-1]["epoch"]:
                blocks.append(rows)
                rows = []
            current = {"epoch": epoch, "planned_epochs": int(match[2]),
                       "learning_rate": float(match[3]), "teacher_forcing": float(match[4])}
        if current is not None:
            for key, prefix in LOSSES.items():
                match = re.search(prefix + NUMBER, line)
                if match:
                    current[key] = float(match[1])
    if current:
        rows.append(current)
    if rows:
        blocks.append(rows)
    return [[row for row in block if "validation_standard" in row] for block in blocks]


def slope(rows, key):
    if len(rows) < 2:
        return None
    xs, ys = [row["epoch"] for row in rows], [row[key] for row in rows]
    xmean, ymean = sum(xs)/len(xs), sum(ys)/len(ys)
    return sum((x-xmean)*(y-ymean) for x,y in zip(xs,ys)) / sum((x-xmean)**2 for x in xs)


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    metadata = {row["Variante"]: row for row in json.loads((P / "scripts/results_100k/psp_gesamtuebersicht_20261114/trainingsdaten.json").read_text())}
    snapshot = OUT / "logs"
    snapshot.mkdir(exist_ok=True)
    all_rows, summaries, series, sources = [], [], {}, {}
    for index, (label, path, key) in enumerate(MODELS):
        text = path.read_text()
        (snapshot / f"{index+1:02d}.log").write_text(text)
        sources[str(path)] = hashlib.sha256(text.encode()).hexdigest()
        blocks = [block for block in parse(text) if block]
        assert len(blocks) == 1, (label, len(blocks))
        rows = blocks[0]
        assert [r["epoch"] for r in rows] == list(range(1, len(rows)+1))
        if key:
            expected = metadata[key]["summary"]["alle Prefixe"]
            assert len(rows) == expected["Epochen"]
            assert abs(rows[-1]["validation_standard"] - expected["Validierungsverlust letzte Epoche"]) < .00011
            assert abs(min(r["validation_standard"] for r in rows) - expected["bester Validierungsverlust"]) < .00011
        best = min(rows, key=lambda r: r["validation_standard"])
        same_tf = [row for row in rows if row["teacher_forcing"] == rows[-1]["teacher_forcing"]]
        best_same = min(same_tf, key=lambda r: r["validation_standard"])
        tail = same_tf[-5:]
        series[label] = rows
        summaries.append({"model": label, "completed_epochs": len(rows), "planned_epochs": rows[-1]["planned_epochs"],
            "best_epoch_raw": best["epoch"], "best_validation_raw": best["validation_standard"],
            "best_epoch_at_final_tf": best_same["epoch"], "best_validation_at_final_tf": best_same["validation_standard"],
            "final_tf": rows[-1]["teacher_forcing"], "first_epoch_at_final_tf": same_tf[0]["epoch"],
            "last_validation": rows[-1]["validation_standard"], "tail_first_epoch": tail[0]["epoch"],
            "tail_first_validation": tail[0]["validation_standard"],
            "tail_validation_change": rows[-1]["validation_standard"] - tail[0]["validation_standard"],
            "tail_validation_slope": slope(tail,"validation_standard"),
            "tail_training_start": tail[0]["train_attenuated"], "tail_training_end": rows[-1]["train_attenuated"],
            "tail_training_slope": slope(tail,"train_attenuated"), "source": str(path)})
        all_rows += [{"model": label, **row, "source": str(path)} for row in rows]
    save_csv(OUT / "epochs.csv", all_rows)
    save_csv(OUT / "summary.csv", summaries)
    stamp = datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds")
    (OUT / "manifest.json").write_text(json.dumps({"created_at":stamp,"source_sha256":sources},indent=2)+"\n")
    f = lambda value: f"{value:.4f}".replace(".",",")
    lines = ["# PSP-Verlustverläufe", "", f"Stand: {stamp}. Protokolle wurden als unveränderliche Kopien unter logs/ gesichert.", "",
        "Training verwendet den attenuierten, mit GradNorm gewichteten Trainingsverlust; "
        "die Tabelle zeigt den Standard-Validierungsverlust. Beide sind unterschiedliche Größen. "
        "Absolute Verluste verschiedener Modellvarianten sind wegen verschiedener Aufgaben nicht direkt zu vergleichen.", "",
        "## Minima und Trainingsende", "",
        "| Modell | Fertige Epochen | Rohminimum (Epoche) | Minimum bei TF wie am Ende (Epoche) | Letzter Val-Verlust | Letzte bis zu fünf Val-Werte bei gleicher TF |",
        "|---|---:|---:|---:|---:|---|"]
    for r in summaries:
        tail = [row for row in series[r["model"]] if row["teacher_forcing"] == r["final_tf"]][-5:]
        tail_values = " → ".join(f(row['validation_standard']) for row in tail)
        lines.append(f"| {r['model']} | {r['completed_epochs']}/{r['planned_epochs']} | {f(r['best_validation_raw'])} ({r['best_epoch_raw']}) | "
                     f"{f(r['best_validation_at_final_tf'])} ({r['best_epoch_at_final_tf']}) | {f(r['last_validation'])} | "
                     + tail_values + " |")
    lines += ["", "## Einordnung", "",
        "Die Validierung übernimmt im Trainer die aktuelle Teacher-Forcing-Quote des Trainings. "
        "Bei den geplanten 15-Epochen-Läufen ohne TF-Untergrenze fällt sie von 1 auf 0 und bleibt ab Epoche 9 bei 0. "
        "Ein Rohminimum vor dieser Phase wurde daher unter anderen Bedingungen gemessen. "
        "Die zusätzliche Spalte vergleicht nur Epochen mit derselben Quote wie am Ende.", "",
        "Die gezeigten Endwerte stammen aus den tatsächlich gespeicherten Modellen: "
        "Die vorhandene Pipeline übernimmt die letzte Epoche und lädt nicht automatisch den besten Validierungsstand. "
        "Der Trainer überschreibt last.pt beim Speichern. Die laufenden Trainings wurden durch diese Analyse nicht verändert.", "",
        "Aus den Protokollen lässt sich der beobachtete Trend bis zum Ende beurteilen. "
        "Ob zusätzliche, bisher nicht trainierte Epochen danach besser würden, ist daraus nicht feststellbar.", "",
        "[Standard-Validierungsverluste](validation_curves.png) · [Attenuierte Trainingsverluste](training_curves.png) · "
        "[Alle Epochen als CSV](epochs.csv) · [Zusammenfassung als CSV](summary.csv)"]
    (OUT / "auswertung.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncols, nrows = 4, math.ceil(len(series)/4)
    for key, title, filename in [("validation_standard", "Standard-Validierungsverlust", "validation_curves.png"),
                                 ("train_attenuated", "Attenuierter Trainingsverlust", "training_curves.png")]:
        fig, axes = plt.subplots(nrows,ncols,figsize=(16,3*nrows),layout="constrained")
        for axis,(label,rows) in zip(axes.flat,series.items()):
            axis.plot([r['epoch'] for r in rows],[r[key] for r in rows],marker='.',color='#1967a3')
            final_tf = rows[-1]['teacher_forcing']
            start = next(r['epoch'] for r in rows if r['teacher_forcing']==final_tf)
            if start>1: axis.axvline(start,color='gray',linestyle=':',linewidth=1)
            axis.set_title(f"{label} ({len(rows)}/{rows[-1]['planned_epochs']})",fontsize=10)
            axis.set_xlabel('Epoche',fontsize=8)
            axis.grid(alpha=.2)
            axis.tick_params(labelsize=8)
        for axis in list(axes.flat)[len(series):]: axis.set_visible(False)
        fig.suptitle(title+" – gestrichelt: Beginn der finalen TF-Quote",fontsize=14)
        fig.savefig(OUT/filename,dpi=160)
        plt.close(fig)
    for r in summaries:
        print(f"{r['model']}: {r['completed_epochs']}/{r['planned_epochs']}; "
              f"best={r['best_validation_raw']}@{r['best_epoch_raw']}; last={r['last_validation']}")
    print(OUT / "auswertung.md")


if __name__ == "__main__":
    main()
