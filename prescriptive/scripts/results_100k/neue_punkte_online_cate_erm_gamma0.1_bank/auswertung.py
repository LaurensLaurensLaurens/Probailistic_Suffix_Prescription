"""Ergebnistabellen dieses Laufs zusammenführen; nach Abschluss von online.log ausführen."""
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
GAMMA = 0.1
ERM = "ERM γ=0.1 (Tage)"
POLICIES = {"s_learner_erm": ("S_Learner_ERM", "S-Learner ERM", "S_Learner", "s_learner"),
            "x_learner_erm": ("X_Learner_ERM", "X-Learner ERM", "X_Learner", "x_learner"),
            "causal_forest_erm": ("Causal_Forest_ERM", "Causal Forest ERM", "Causal_Forest", "causal_forest")}
METRICS = ["Durchlaufzeit (Tage)", "Durchlaufzeit P90 (Tage)", "Durchlaufzeit P95 (Tage)", "CVaR 90 % (Tage)", ERM]


def markdown(table):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}" if np.isfinite(value) else "—"
        return str(value)
    return "\n".join(["| " + " | ".join(map(str, table.columns)) + " |",
                      "| " + " | ".join(["---"] * len(table.columns)) + " |"] +
                     ["| " + " | ".join(map(cell, row)) + " |" for row in table.itertuples(index=False, name=None)])


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return float(mean), float(mean - half), float(mean + half)


tables, batches, bank = [], [], None
for policy, (folder, label, _, _) in POLICIES.items():
    directory = OUT / "prescription" / folder
    table = pd.read_csv(directory / "bpic17_online.csv").set_index("Policy")
    runs = pd.read_csv(directory / "bpic17_online_laeufe.csv").drop(columns="Unnamed: 0", errors="ignore")
    assert set(runs.Policy) == {"bank", policy}
    for name, rows in runs.groupby("Policy"):
        assert sorted(rows.seed.tolist()) == list(range(10))
        assert rows["Fälle"].eq(1000).all()
    assert runs.loc[runs.Policy == policy, "ERM γ (1/Tag)"].eq(GAMMA).all()
    current_bank = runs[runs.Policy == "bank"].set_index("seed").sort_index()
    if bank is None:
        bank = current_bank
        tables.append(table.loc[["bank"]])
        batches.append(runs[runs.Policy == "bank"])
    else:
        pd.testing.assert_frame_equal(bank[METRICS], current_bank[METRICS])
    tables.append(table.loc[[policy]])
    batches.append(runs[runs.Policy == policy])

table = pd.concat(tables)
runs = pd.concat(batches, ignore_index=True)
table.to_csv(OUT / "ergebnis_gesamt.csv")
runs.to_csv(OUT / "laeufe_gesamt.csv", index=False)
labels = {"bank": "Bank", **{key: value[1] for key, value in POLICIES.items()}}
summary = table[METRICS + [f"{ERM} ggü. Bank", f"{ERM} ggü. Bank 95-%-KI unten", f"{ERM} ggü. Bank 95-%-KI oben"]].copy()
summary.insert(0, "Methode", [labels[name] for name in summary.index])
summary.to_csv(OUT / "zusammenfassung.csv", index=False)

paired = []
old_comparisons = []
old_bank_matches = []
for policy, (_, label, old_folder, old_policy) in POLICIES.items():
    current = runs[runs.Policy == policy].set_index("seed").sort_index()
    for metric in METRICS:
        mean, low, high = interval(current[metric] - bank[metric])
        paired.append({"Methode": label, "Kennzahl": metric, "Differenz zur Bank": mean,
                       "95-%-KI unten": low, "95-%-KI oben": high})
    old_path = ROOT / "prescriptive/scripts/results_100k/neue_punkte_online_cate/prescription" / old_folder / "bpic17_online_laeufe.csv"
    old = pd.read_csv(old_path)
    old_bank = old[old.Policy == "bank"].set_index("seed").sort_index()
    same_bank = np.allclose(bank[METRICS[:-1]], old_bank[METRICS[:-1]], atol=0.00101, rtol=0)
    old_bank_matches.append(bool(same_bank))
    previous = old[old.Policy == old_policy].set_index("seed").sort_index()
    assert previous.index.equals(current.index)
    for metric in METRICS[:-1]:
        mean, low, high = interval(current[metric] - previous[metric])
        old_comparisons.append({"Methode": label, "Kennzahl": metric, "EV Mittel": previous[metric].mean(),
                                "ERM Mittel": current[metric].mean(), "ERM minus EV": mean,
                                "95-%-KI unten": low if same_bank else np.nan,
                                "95-%-KI oben": high if same_bank else np.nan,
                                "Bank-Replikation identisch (Toleranz 0.001 Tage)": bool(same_bank)})
paired = pd.DataFrame(paired)
paired.to_csv(OUT / "differenzen_zur_bank.csv", index=False)
pd.DataFrame(old_comparisons).to_csv(OUT / "vergleich_ev_erm.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
colors = ["#2a78d6", "#db7b27", "#35976a"]
for ax, metric, title in zip(axes, [METRICS[0], ERM], ["Mittlere Durchlaufzeit", "Entropisches Risiko, γ = 0,1"]):
    part = paired[paired.Kennzahl == metric]
    for i, row in enumerate(part.itertuples(index=False, name=None)):
        label, _, mean, low, high = row
        ax.errorbar(mean, i, xerr=[[mean - low], [high - mean]], fmt="o", color=colors[i], capsize=4)
    ax.axvline(0, color="#555", linestyle="--", linewidth=1)
    ax.set_yticks(range(3), [value[1] for value in POLICIES.values()])
    ax.invert_yaxis()
    ax.set_xlabel("Differenz zur Bank (Tage), 95-%-KI\n← kleiner ist besser")
    ax.set_title(title)
    ax.grid(axis="x", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("CATE-ERM: 10 Seeds, je 1.000 gewertete Fälle", fontsize=13)
fig.savefig(OUT / "vergleich_zur_bank.png", dpi=200)
plt.close(fig)

source_files = ["prescriptive/estimators/cate_erm_model.py", "prescriptive/estimators/cate_model.py",
                "prescriptive/estimators/xgb_model.py", "prescriptive/policies/erm_policy/cate_policy.py",
                "prescriptive/evaluation/online_evaluation.py", "prescriptive/scripts/main.py",
                "prescriptive/scripts/train_cate_real.py", "process-simulation-engine/simulation/dataset_simulator.py",
                "process-simulation-engine/simulation/case_engine_lifecycle_aware.py",
                "process-simulation-engine/models/next_event/bpic17/config.json",
                "process-simulation-engine/models/next_event/bpic17/model.weights.h5"]
manifest = json.loads((OUT / "laufparameter.json").read_text())
manifest["source_sha256"] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_files}
manifest["historical_ev_bank_replication_matches"] = all(old_bank_matches)
manifest["resumption_script"] = str(OUT / "resume_simulation.py")
manifest["resumption_script_sha256"] = hashlib.sha256((OUT / "resume_simulation.py").read_bytes()).hexdigest()
manifest["checkpoint_settings_sha256"] = hashlib.sha256((OUT / "checkpoint_settings.json").read_bytes()).hexdigest()
manifest["raw_case_durations"] = "checkpoints/*_seed*.json (_days)"
(OUT / "laufparameter.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
shutil.copy2(ROOT / "prescriptive/data/CATE_ERM_gamma0.1_real_alle.log", OUT / "training.log")
shutil.copy2(ROOT / "prescriptive/scripts/results_100k/cate_erm_gamma0.1_real_alle/effekte_testfaelle.csv", OUT / "effekte_testfaelle.csv")
diagnostics = pd.read_csv(OUT / "training_diagnostik.csv")
findings = []
for policy, (_, label, _, _) in POLICIES.items():
    row = table.loc[policy]
    lower, upper = row[f"{ERM} ggü. Bank 95-%-KI unten"], row[f"{ERM} ggü. Bank 95-%-KI oben"]
    verdict = "niedriger als bei der Bank" if upper < 0 else "höher als bei der Bank" if lower > 0 else "im KI nicht klar von der Bank getrennt"
    findings.append(f"- {label}: ERM {row[ERM]:.3f} Tage; Differenz zur Bank {row[f'{ERM} ggü. Bank']:+.3f} Tage "
                    f"(95-%-KI [{lower:+.3f}; {upper:+.3f}]), {verdict}.")
report = ["# Auswertung der drei CATE-ERM-Policies", "",
          "BPIC17, γ = 0,1 / Tag. Alle Modelle auf denselben 25.208 Fällen trainiert, 6.301 echte Fälle zurückgehalten. "
          "Training-Seed 42. Online: Seeds 0–9, je 3.000 Ankünfte, Warmup 700, Cooldown 1.300; "
          "je 1.000 gewertete Fälle, insgesamt 10.000 pro Policy. Vier parallele Prozesse. "
          "Vier Control-Flow-Punkte, keine first_only-Einschränkung, Mitarbeiterzuteilung nach gelerntem Bankverhalten (learned). "
          "Bestehendes Simulator-Modell models/next_event/bpic17; kein neues Simulator-Training.", "",
          "## Ergebnisse", "", "Alle Größen in Tagen; kleiner ist besser. ERM ist das empirische entropische Risiko "
          "der simulierten Durchlaufzeiten bei γ = 0,1. Ausgewiesen ist das Mittel der zehn je Seed berechneten Kennzahlen; "
          "die Daten werden nicht vor der Risikoberechnung über Seeds zusammengeworfen.", "", markdown(summary), "",
          "## Gepaarter Vergleich zur Bank", "", *findings, "",
          "Die 95-%-Konfidenzintervalle beruhen auf den zehn gepaarten Seed-Differenzen (t-Intervall). "
          "Sie beschreiben die Streuung über Simulationswiederholungen, nicht die Unsicherheit der kausalen Schätzer "
          "oder die Übertragbarkeit auf den echten Betrieb. Die neuen CSVs enthalten volle Rechengenauigkeit; "
          "die Tabellen im HTML-Bericht und die historischen EV-CSVs sind auf drei Nachkommastellen gerundet.", "",
          "![Vergleich zur Bank](vergleich_zur_bank.png)", "",
          "## Auffälligkeit des Trainings", "",
          "Der S-Learner schätzt auf sämtlichen Entscheidungsproben der zurückgehaltenen Fälle an allen vier Wahlpunkten "
          "Nulleffekte; die Policy löst diese Gleichstände zufällig auf. Ihre Ergebnisse belegen damit "
          "keine gelernte risikosensitive Unterscheidung der Aktionen. Frühes Stoppen wählte je Punkt nur ein bis zwei Bäume. "
          "Auch mehrere Outcome-Hilfsmodelle des X-Learners und Causal Forest stoppten nach ein bis zwei Bäumen.", "",
          "![Aktionsanteile auf echten Testfällen](aktionsanteile_testfaelle.png)", "",
          "Bei diesem Gamma konzentriert sich das exponentielle Trainingsgewicht extrem auf einzelne lange Fälle. "
          "Die folgende Diagnose bezieht sich auf den inneren Trainingsteil je Wahlpunkt. Wiederholte "
          "Entscheidungsproben desselben Falls wurden vor der Gewichtsdiagnose zusammengefasst. "
          "Die effektive Fallzahl ist 1 / Summe der quadrierten normierten Fallgewichte; sie ist keine nominale Stichprobengröße.", "",
          markdown(diagnostics), "",
          "Die Gewichtskonzentration ist ein Befund dieses Gamma und dieser Daten; Verluste wurden weder "
          "abgeschnitten noch wurde das Gamma nachträglich verändert. Für eine Methodenbewertung sind weitere "
          "Trainings-/Gamma-Sensitivitätsanalysen erforderlich.", "",
          "## Historischer EV-Vergleich", "",
          ("Die neuen Bank-Läufe reproduzieren die Bank-Kennzahlen des früheren CATE-EV-Laufs auf 0,001 Tage genau. "
           "vergleich_ev_erm.csv enthält zusätzlich gepaarte EV-zu-ERM-Differenzen für Mittel, P90, P95 und CVaR."
           if all(old_bank_matches) else
           "Die neuen Bank-Läufe stimmen nicht vollständig mit dem früheren CATE-EV-Lauf überein. "
           "vergleich_ev_erm.csv ist deshalb nur ein beschreibender historischer Vergleich; keine gepaarten KIs ausgewiesen."), "",
          "Die früheren EV-CSVs enthalten keine Einzelfall-Durchlaufzeiten; ihr ERM lässt sich daraus nicht nachberechnen.", "",
          "## Dateien", "", "- bericht.html: Standardbericht mit Ergebnissen je Ansatz.",
          "- ergebnis_gesamt.csv / laeufe_gesamt.csv: alle vier Policies zusammengeführt.",
          "- zusammenfassung.csv / differenzen_zur_bank.csv / vergleich_ev_erm.csv: kompakte Auswertung.",
          "- effekte_testfaelle.csv / training_diagnostik.csv / training.log: Training und Diagnose.",
          "- catalog.yaml / laufparameter.json: Einstellungen, Modell- und Quellcode-Prüfsummen.",
          "- checkpoints/: atomar gespeicherte Einzelläufe einschließlich Durchlaufzeiten je Fall.",
          "- online_resume.log: vollständiges Log der in Batches ausgeführten Auswertung.",
          "- online_resume.log: alle neu berechneten Einzelläufe unter gelernter Bank-Zuteilung.", ""]
(OUT / "auswertung.md").write_text("\n".join(report))
print(summary.to_string(index=False))
print("\n" + "\n".join(findings))
print(f"\nHistorische Bank-Replikation stimmt überein: {all(old_bank_matches)}")
