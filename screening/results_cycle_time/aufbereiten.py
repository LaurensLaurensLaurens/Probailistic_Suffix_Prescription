"""Vorhandene Screening-Artefakte sammeln und lesbar aufbereiten; kein erneutes Screening.

Aufruf im Projektordner:
    process-simulation-engine/.venv/bin/python screening/results/aufbereiten.py
Originaldateien bleiben erhalten. Unterschiedliche Läufe werden getrennt archiviert und gekennzeichnet.
"""
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import textwrap
from urllib.parse import quote
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SCRIPTS = ROOT / "prescriptive/scripts"
DATA = ROOT / "prescriptive/data"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib_cache"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

RUNS = {"screening_alle": SCRIPTS / "results_100k/screening_alle",
        "screening_alt": SCRIPTS / "results/prescription/screening",
        "validierung_gesamt": SCRIPTS / "results_100k/screening_validierung",
        "validierung_neue_punkte": SCRIPTS / "results_100k/neue_punkte_validierung"}
ALIASES = {"Policy-−Durchlaufzeit": "Policy-Gewinn", "−Durchlaufzeit ggü. Bank": "Gewinn ggü. Bank"}
MANIFEST = []


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def read(path):
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def archive(source, destination, run):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == hashlib.sha256(destination.read_bytes()).hexdigest()
    row = {"Lauf": run, "Quelle": relative(source), "Kopie": destination.relative_to(OUT).as_posix(),
           "Bytes": source.stat().st_size, "SHA256": digest,
           "Geändert UTC": datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat()}
    if source.suffix == ".csv":
        table = read(source)
        row.update({"Zeilen": len(table), "Spalten": len(table.columns), "Status": "leer" if table.empty else "vorhanden"})
    MANIFEST.append(row)


def collect():
    for run, directory in RUNS.items():
        assert directory.is_dir(), directory
        for source in sorted(directory.rglob("*")):
            if source.is_file() and "__pycache__" not in source.parts:
                archive(source, OUT / "originale" / run / source.relative_to(directory), run)
    extras = {"screening_alle": ["screening_alle.log", "bpic17_screening_alle.sh", "catalog_screening_alle.yaml"],
              "screening_alt": ["screening_run.log"],
              "validierung_gesamt": ["screening_validierung.log", "bpic17_screening_validierung.sh", "catalog_simulator_kpis_gesamt.yaml"],
              "validierung_neue_punkte": ["neue_punkte_treue.log", "bpic17_neue_punkte_treue.sh", "catalog_neue_punkte.yaml"]}
    for run, names in extras.items():
        for name in names:
            archive(DATA / name, OUT / "provenienz" / run / name, run)
    pd.DataFrame(MANIFEST).to_csv(OUT / "quellenverzeichnis.csv", index=False)
    (OUT / "quellenverzeichnis.json").write_text(json.dumps(MANIFEST, indent=2, ensure_ascii=False) + "\n")


def screening_tables():
    summaries = []
    effects, propensities = [], []
    for run in ("screening_alle", "screening_alt"):
        source = RUNS[run] / "prescription/screening" if run == "screening_alle" else RUNS[run]
        for section, kind, stem in (("control_flow", "Control Flow", "control_flow"), ("variables", "Variable", "variablen")):
            path = source / section / f"bpic17_screening_{stem}.csv"
            table = read(path).rename(columns=ALIASES).set_index("Kandidat")
            table["Quelle"] = relative(path)
            table["Präzision"] = "Originalzusammenfassung (3 Nachkommastellen)"
            detailed_path = path.with_name(path.stem + "_detailliert.csv")
            if detailed_path.exists():
                detailed = read(detailed_path).rename(columns=ALIASES).set_index("Kandidat")
                assert detailed.index.is_unique and set(detailed.index) <= set(table.index)
                for name in detailed.index:
                    for column in detailed.columns:
                        if column not in table:
                            table[column] = np.nan if pd.api.types.is_numeric_dtype(detailed[column]) else None
                        value = detailed.loc[name, column]
                        if pd.notna(value):
                            existing = table.loc[name, column]
                            if isinstance(value, (float, int, np.number)) and pd.notna(existing):
                                assert np.isclose(float(value), float(existing), atol=.00051, rtol=0), (name, column, value, existing)
                            table.loc[name, column] = value
                    table.loc[name, "Quelle"] = relative(detailed_path)
                    table.loc[name, "Präzision"] = "Detaillierte Originaltabelle"
            table.insert(0, "Lauf", run)
            table.insert(1, "Art", ["Mitarbeiter" if str(name).startswith("Mitarbeiter:") else kind for name in table.index])
            summaries.append(table.reset_index())
        for path in sorted((source / "details").glob("*screening_Mitarbeiter*.csv")):
            table = read(path)
            table.insert(0, "Lauf", run)
            table.insert(1, "Detailtabelle", path.stem.removeprefix("bpic17_screening_"))
            table["Quelle"] = relative(path)
            if "Effekt ggü. Pool (AIPW)" in table:
                effects.append(table)
            elif "Propensity im Pool (Mittel)" in table:
                propensities.append(table)
            else:
                raise ValueError(f"Unbekannte Detailtabelle: {path}")
    all_screening = pd.concat(summaries, ignore_index=True, sort=False)
    assert not all_screening.duplicated(["Lauf", "Kandidat"]).any()
    current = all_screening[all_screening.Lauf == "screening_alle"].copy()
    assert len(current) == 20 and len(all_screening) == 28
    employees = pd.concat(effects, ignore_index=True, sort=False)
    old_propensities = pd.concat(propensities, ignore_index=True, sort=False)
    for name, table in (("screening_gesamt", all_screening), ("screening_aktuell", current),
                        ("mitarbeitereffekte", employees), ("mitarbeiter_propensities_historisch", old_propensities)):
        table.to_csv(OUT / "tabellen" / f"{name}.csv", index=False)
    return all_screening, current, employees, old_propensities


def validation_tables():
    tables, details = [], []
    for run in ("validierung_gesamt", "validierung_neue_punkte"):
        path = RUNS[run] / "validierung.csv"
        table = read(path)
        table.insert(0, "Lauf", run)
        table["Quelle"] = relative(path)
        tables.append(table)
        path = RUNS[run] / "validierung_mitarbeiter.csv"
        table = read(path)
        if len(table):
            table.insert(0, "Lauf", run)
            table["Quelle"] = relative(path)
            details.append(table)
    table = pd.concat(tables, ignore_index=True, sort=False)
    employee = pd.concat(details, ignore_index=True, sort=False)
    assert len(table) == 17 and len(employee) == 553
    table.to_csv(OUT / "tabellen/validierung_gesamt.csv", index=False)
    employee.to_csv(OUT / "tabellen/validierung_mitarbeiter.csv", index=False)
    return table, employee


def wrap(name, width=36):
    return "\n".join(textwrap.wrap(str(name).replace("_", " "), width=width))


def plots(current, validations):
    control = current[current.Art == "Control Flow"].copy()
    catalog = yaml.safe_load((DATA / "catalog_screening_alle.yaml").read_text())["bpic17"]
    tests = {item["name"]: item.get("test", {}) for item in catalog["interventionen"]}
    control["Referenz"] = [tests[name].get("action", ["keine Aktion", "Aktion"])[0] for name in control.Kandidat]
    control["Aktion"] = [tests[name].get("action", ["keine Aktion", "Aktion"])[1] for name in control.Kandidat]
    control.to_csv(OUT / "tabellen/aipw_control_flow.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 6.1), layout="constrained")
    for i, row in enumerate(control.itertuples(index=False)):
        data = control.iloc[i]
        mean, low, high = (float(data[key]) for key in ("Effekt (AIPW)", "Effekt KI unten", "Effekt KI oben"))
        good_overlap = data["Überlappung"] >= .5
        ax.errorbar(mean, i, xerr=[[mean-low], [high-mean]], fmt="o", color="#2a78d6" if good_overlap else "#999999", capsize=4)
    ax.set_yticks(range(len(control)), [wrap(name) for name in control.Kandidat])
    ax.invert_yaxis()
    ax.axvline(0, color="#555", linestyle="--", linewidth=1)
    ax.set_xlabel("AIPW-Effekt: Aktion minus Referenz (Tage); positiv = kürzere Durchlaufzeit\n95-%-Konfidenzintervalle; grau = Überlappung unter 50 %")
    ax.set_title("BPIC17: mittlere Aktionseffekte im umfassenden Screening")
    ax.grid(axis="x", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "diagramme/aipw_control_flow.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 11.5), sharey=True, layout="constrained", gridspec_kw={"width_ratios": [1.7, 1, 1]})
    y = np.arange(len(current))
    for i, (_, row) in enumerate(current.iterrows()):
        mean, low, high = (row.get(key, np.nan) for key in ("Gewinn ggü. Bank", "Bank KI unten", "Bank KI oben"))
        color = "#db7b27" if str(row.Ergebnis).startswith("unsicher") else "#999999" if row.get("Überlappung", 0) < .5 else "#2a78d6"
        if np.isfinite([mean, low, high]).all():
            axes[0].errorbar(mean, i, xerr=[[mean-low], [high-mean]], fmt="o", color=color, capsize=3)
        else:
            axes[0].text(.02, i, "zu wenige Fälle", transform=axes[0].get_yaxis_transform(), va="center", fontsize=8, color="#777")
    axes[0].axvline(0, color="#555", linestyle="--", linewidth=1)
    axes[0].set_title("Policy-Gewinn gegenüber Bank")
    axes[0].set_xlabel("Tage (95-%-KI); positiv = besser")
    axes[0].set_yticks(y, [wrap(name, 35) for name in current.Kandidat], fontsize=9)
    axes[1].barh(y, current["Überlappung"] * 100, color="#2a78d6", height=.6)
    axes[1].axvline(50, color="#db7b27", linestyle="--")
    axes[1].set_xlim(0, 105)
    axes[1].set_title("Überlappung")
    axes[1].set_xlabel("Anteil in %; Schwelle 50 %")
    axes[2].barh(y, current["Placebo d"].abs(), color="#2a78d6", height=.6)
    axes[2].axvline(.1, color="#db7b27", linestyle="--")
    axes[2].set_title("Placebo-Betrag")
    axes[2].set_xlabel("|d|; Schwelle 0,1")
    axes[0].set_ylim(len(current) - .5, -.5)
    for i, value in enumerate(current["Placebo d"]):
        if pd.isna(value):
            axes[2].text(.02, i, "nicht verfügbar", transform=axes[2].get_yaxis_transform(), va="center", fontsize=8, color="#777")
    for ax in axes:
        ax.grid(axis="x", alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("BPIC17: alle 20 Screening-Kandidaten\nMitarbeiterpool frei, mindestens 200 Fälle je eigener Mitarbeiteroption", fontsize=14)
    fig.savefig(OUT / "diagramme/screening_ueberblick.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.4), layout="constrained")
    labels = [("echt (AIPW)", "echt Std.-Fehler", "Echtes Log (AIPW)", "#2a78d6", -.19),
              ("Simulator beobachtet (AIPW)", "Simulator beobachtet Std.-Fehler", "Simulator beobachtet (AIPW)", "#db7b27", 0),
              ("Simulator wahr", "Simulator wahr Std.-Fehler", "Simulator randomisiert", "#35976a", .19)]
    for ax, run, title in zip(axes, ("validierung_gesamt", "validierung_neue_punkte"), ("Frühere Validierung", "Validierung der neuen Wahlpunkte")):
        frame = validations[(validations.Lauf == run) & (validations.Art == "Control Flow")]
        y = np.arange(len(frame))
        for value, se, label, color, offset in labels:
            ax.errorbar(frame[value], y + offset, xerr=1.96 * frame[se], fmt="o", color=color, label=label, capsize=3, markersize=4)
        ax.set_yticks(y, [wrap(name, 30) for name in frame.Kandidat], fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="#555", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Effekt (Tage), Intervall ±1,96 × Standardfehler\npositiv = kürzere Durchlaufzeit durch Aktion")
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Screening-Validierungen: getrennte Läufe, keine Mittelung zwischen Katalogen", fontsize=13)
    fig.savefig(OUT / "diagramme/validierung_control_flow.png", dpi=180)
    plt.close(fig)
    return control


def display(table):
    table = table.copy()
    for key in table.columns:
        if pd.api.types.is_numeric_dtype(table[key]):
            table[key] = table[key].map(lambda value: "—" if pd.isna(value) else f"{value:.3f}")
    return table.fillna("—")


def markdown(table):
    table = display(table)
    return "\n".join(["| " + " | ".join(table.columns) + " |", "| " + " | ".join(["---"] * len(table.columns)) + " |"] +
                     ["| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |" for row in table.itertuples(index=False, name=None)])


def intervention_table(current):
    """Eine Zeile je Interventionspunkt, als direkt lesbare gemeinsame Tabelle."""
    def estimate(row, mean, lower, upper):
        value = row.get(mean, np.nan)
        if pd.isna(value):
            return "—"
        low, high = row.get(lower, np.nan), row.get(upper, np.nan)
        if pd.isna(low) or pd.isna(high):
            return f"{value:.3f}"
        return f"{value:.3f} [{low:.3f}; {high:.3f}]"

    rows = []
    for _, row in current.iterrows():
        rows.append({"Interventionspunkt": row.Kandidat, "Art": row.Art,
                     "AIPW-Effekt [95-%-KI] (Tage)": estimate(row, "Effekt (AIPW)", "Effekt KI unten", "Effekt KI oben"),
                     "Policy vs. feste Option [95-%-KI] (Tage)": estimate(row, "Policy-Gewinn", "KI unten", "KI oben"),
                     "Policy vs. Bank [95-%-KI] (Tage)": estimate(row, "Gewinn ggü. Bank", "Bank KI unten", "Bank KI oben"),
                     "Überlappung (%)": "—" if pd.isna(row["Überlappung"]) else f"{100 * row['Überlappung']:.1f}",
                     "Placebo d": "—" if pd.isna(row["Placebo d"]) else f"{row['Placebo d']:.3f}",
                     "Screening-Ergebnis": row.Ergebnis})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "alle_interventionspunkte.csv", index=False)
    explanation = ("Eine Zeile je Interventionspunkt des umfassenden BPIC17-Screenings (20 Punkte). "
                   "Alle Effekt- und Gewinnwerte in Tagen, mit 95-%-Konfidenzintervall. Positiv bedeutet kürzere Durchlaufzeit. "
                   "Policy vs. feste Option ist der zusätzliche Nutzen einer individuellen Policy gegenüber der besten festen Option. "
                   "Ein einzelner AIPW-Aktionseffekt liegt hier nur für die binären Control-Flow-Punkte vor; "
                   "Mitarbeitereffekte stehen je Mitarbeiter in tabellen/mitarbeitereffekte.csv. "
                   "Ein Gedankenstrich bedeutet nicht anwendbar oder nicht verfügbar, nicht null.")
    (OUT / "alle_interventionspunkte.md").write_text("# Auswertung aller Interventionspunkte\n\n" + explanation + "\n\n" + markdown(table) + "\n")
    headers = ["Interventionspunkt", "Art", "AIPW-Effekt\n[95-%-KI], Tage", "Policy vs. feste Option\n[95-%-KI], Tage",
               "Policy vs. Bank\n[95-%-KI], Tage", "Überlappung\n(%)", "Placebo d", "Screening-Ergebnis"]
    cells = []
    for values in table.itertuples(index=False, name=None):
        cells.append([wrap(values[0], 37), wrap(values[1], 12),
                      *[str(value).replace(" [", "\n[") for value in values[2:5]],
                      str(values[5]), str(values[6]), "\n".join(textwrap.wrap(str(values[7]), width=27))])
    fig, ax = plt.subplots(figsize=(23, 15))
    fig.subplots_adjust(left=.015, right=.985, top=.925, bottom=.09)
    ax.axis("off")
    chart = ax.table(cellText=cells, colLabels=headers, colWidths=[.25, .07, .14, .13, .13, .06, .06, .16],
                     cellLoc="center", colLoc="center", bbox=[0, 0, 1, 1])
    chart.auto_set_font_size(False)
    chart.set_fontsize(9.5)
    for (row_index, column), cell in chart.get_celld().items():
        cell.set_edgecolor("#dce2e8")
        cell.set_linewidth(.5)
        if row_index == 0:
            cell.set_facecolor("#e2ecf8")
            cell.set_text_props(weight="bold", color="#17202a")
        else:
            cell.set_facecolor("#f3f7fb" if row_index % 2 == 0 else "white")
            if column in (0, 7):
                cell.set_text_props(ha="left")
            if column == 7 and str(table.iloc[row_index - 1, 7]).startswith("unsicher"):
                cell.set_facecolor("#fff0d8")
    fig.suptitle("BPIC17: Auswertung aller 20 Interventionspunkte", fontsize=22, y=.98)
    fig.text(.5, .945, "6 Control-Flow-Punkte · 10 Mitarbeiterpunkte · 4 weitere Variablen", ha="center", fontsize=13)
    fig.text(.02, .055, "Positive Effekt-/Gewinnwerte = kürzere Durchlaufzeit. Intervalle: 95-%-KI. Überlappung als Prozentanteil.", fontsize=11)
    fig.text(.02, .035, "— = kein einzelner binärer AIPW-Effekt bzw. keine Schätzung verfügbar. Mitarbeiter: individuelle Effekte in der Detailtabelle.", fontsize=11)
    fig.text(.02, .015, "Quelle: umfassender Screening-Lauf; Mitarbeiterpool frei, mindestens 200 Fälle je eigener Mitarbeiteroption. Werte unverändert übernommen und gerundet.", fontsize=10, color="#555")
    fig.savefig(OUT / "alle_interventionspunkte.png", dpi=180)
    plt.close(fig)
    return table


def write_reports(all_screening, current, employees, propensities, validations, validation_employees, control):
    count = current.Ergebnis.value_counts()
    approved = int(current.Ergebnis.str.startswith("ja").sum())
    uncertain = int(current.Ergebnis.str.startswith("unsicher").sum())
    compact = intervention_table(current)
    sources = pd.DataFrame([{"Lauf": "screening_alle", "Umfang": "20 Kandidaten: 6 Control Flow + 14 Variablen/Mitarbeiter", "Konfiguration": "Pool frei; mindestens 200 Fälle je Mitarbeiteroption"},
                            {"Lauf": "screening_alt", "Umfang": "8 Kandidaten: 1 Control Flow + 7 Mitarbeiter", "Konfiguration": "Historischer Teilstand; Pool kompetent, mindestens 1 Fall"},
                            {"Lauf": "validierung_gesamt", "Umfang": "13 Kandidaten: 5 Control Flow + 8 Mitarbeiter", "Konfiguration": "20.000 simulierte Fälle je Lauf; früherer Katalog"},
                            {"Lauf": "validierung_neue_punkte", "Umfang": "4 Control-Flow-Kandidaten", "Konfiguration": "20.000 simulierte Fälle je Lauf; Katalog der neuen Wahlpunkte"}])
    sections = [
        ("Umfang und Herkunft", "Alle gefundenen Ergebnisse dieses Screening-Pakets betreffen BPIC17. "
         "Die Originaldateien wurden kopiert und über SHA256 geprüft; die bisherigen Ergebnisordner bleiben erhalten. "
         "Aufbereitet werden vorhandene Ergebnisse, ohne neue Modellschätzungen oder Simulatorläufe.", sources),
        ("Ergebnis des umfassenden Screenings", f"Von 20 Kandidaten erfüllen {approved} alle Screening-Kriterien; {uncertain} ist als unsicher markiert. "
         "Kein bestandenes Screening bedeutet hier nicht, dass alle Aktionen wirkungslos sind: Eine feste Aktion kann "
         "vorteilhaft sein, ohne dass eine fallabhängige Policy die beste feste Regel verbessert.", compact),
        ("AIPW-Aktionseffekte: Control Flow", "Mittlerer Effekt der Aktion gegenüber der Referenz, mit den originalen Konfidenzintervallen. "
         "Das Outcome ist negative Durchlaufzeit in Tagen: positive Effekte bedeuten kürzere Durchlaufzeiten. "
         "Diese Mittelwerte sind keine individuellen CATEs und kein entropisches Risikomaß.",
         control[["Kandidat", "Referenz", "Aktion", "Entscheidungsfälle", "Effekt (AIPW)", "Effekt KI unten", "Effekt KI oben", "Überlappung"]]),
        ("Simulator-Validierung", "Die beiden Validierungen verwenden unterschiedliche Kataloge und bleiben getrennt. "
         "Echtes Log und beobachteter Simulator werden mittels AIPW ausgewertet; der Referenzeffekt im Simulator "
         "stammt aus randomisierten Entscheidungen. Die Validierung nutzt drei Cross-Fitting-Folds, das Screening standardmäßig fünf; "
         "daher müssen gleichnamige AIPW-Schätzungen nicht identisch sein.",
         validations[["Lauf", "Kandidat", "Art", "echt (AIPW)", "Simulator beobachtet (AIPW)", "Simulator wahr", "Überlappung echt", "Überlappung Simulator"]]),
    ]
    methodology = ["Policy-Gewinn: geschätzter Wert der fallabhängigen Policy minus beste feste Option (AIPW).",
                   "Gewinn gegenüber Bank: geschätzter Policy-Wert minus beobachtetes Outcome. Positiv bedeutet kürzere Durchlaufzeit.",
                   "Screening-Regel: Überlappung mindestens 0,5, untere Grenze des relevanten Gewinn-KI über 0 und |Placebo d| < 0,1. "
                   "Bei Mitarbeiterpools wird der Gewinn gegenüber Bank verwendet, sonst der Gewinn gegenüber der besten festen Regel. "
                   "Die Ergebnislabels wurden unverändert übernommen; ein fehlender Placebowert ist kein bestandener Placebotest.",
                   "Überlappung: Anteil der Fälle, in denen alle verfügbaren Optionen ausreichend Propensity haben; Anteil 0–1.",
                   "AIPW verwendet Propensity- und Outcome-Modelle mit Cross-Fitting; Propensities werden in der Formel auf [0,01; 0,99] begrenzt.",
                   "Unbeobachtete Confounder können AIPW-Schätzungen verzerren. Ein positives Ergebnis oder KI allein ersetzt keine Überlappungs- und Placeboprüfung.",
                   "Die sechs Control-Flow- und vier Nicht-Mitarbeiter-Variablen liegen zusätzlich in detaillierten Tabellen vor. "
                   "Deren Werte wurden nach Prüfung auf Rundungskonsistenz bevorzugt. Mitarbeiter-Zusammenfassungen liegen nur gerundet vor; "
                   "die Präzision und die Quelle stehen pro Zeile in den aufbereiteten CSVs.",
                   "183 Mitarbeiter-Effektschätzungen gehören zum umfassenden Screening; 511 historische Detailzeilen sind Propensities, keine AIPW-Effekte. "
                   "Die Simulator-Validierung enthält zusätzlich 553 Mitarbeiterzeilen. Fehlende Werte bleiben fehlend.",
                   "Die Mitarbeiterdatei der Validierung der neuen Wahlpunkte ist leer, weil dort ausschließlich Control Flow geprüft wurde."]
    created = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%d.%m.%Y %H:%M %Z")
    intro = "# Screening-Ergebnisse\n\nAufbereitung vorhandener BPIC17-Ergebnisse — " + created + ".\n\n" \
            "**Eine Tabelle für alle Interventionspunkte:** [CSV](alle_interventionspunkte.csv) · [Bild](alle_interventionspunkte.png) · [Markdown](alle_interventionspunkte.md)\n\n" \
            "[Lesbarer HTML-Bericht](bericht.html) · [Alle Quellen mit Prüfsummen](quellenverzeichnis.csv) · " \
            "[Originaldateien](originale/) · [Aufbereitete Tabellen](tabellen/)\n\n"
    body = [intro]
    for title, explanation, table in sections:
        body += [f"## {title}", "", explanation, "", markdown(table), ""]
    body += ["## Diagramme", "", "![AIPW-Effekte](diagramme/aipw_control_flow.png)", "",
             "![Screening-Überblick](diagramme/screening_ueberblick.png)", "",
             "![Validierungen](diagramme/validierung_control_flow.png)", "",
             "## Kennzahlen und Interpretation", ""] + ["- " + text for text in methodology]
    body += ["", "## Dateien", "", "- `tabellen/screening_aktuell.csv`: die 20 Kandidaten des umfassenden Screenings.",
             "- `tabellen/screening_gesamt.csv`: alle 28 Kandidatenzeilen beider Screening-Stände mit Herkunft.",
             "- `tabellen/aipw_control_flow.csv`: Aktion, Referenz und AIPW-Effekte mit KI.",
             "- `tabellen/mitarbeitereffekte.csv`: 183 Mitarbeiter-Effektschätzungen.",
             "- `tabellen/mitarbeiter_propensities_historisch.csv`: 511 historische Propensity-Zeilen.",
             "- `tabellen/validierung_gesamt.csv`: 17 Kandidatenzeilen beider Validierungen.",
             "- `tabellen/validierung_mitarbeiter.csv`: 553 Mitarbeiterzeilen der Validierung.",
             "- `originale/`: unveränderte CSVs, Grafiken, Beispiele und der vollständige Bericht des umfassenden Screening-Laufs.",
             "- `provenienz/`: zugehörige Logs, Aufrufskripte und verfügbare Katalogdateien. Katalogkopien zeigen den heutigen Dateiinhalt; "
             "ein historischer Dateistand lässt sich daraus allein nicht beweisen.",
             "- `quellenverzeichnis.csv` / `.json`: jeder kopierte Ursprungspfad, Ziel, Dateigröße, Zeilenzahl und SHA256.",
             "", "Neu erstellen: `process-simulation-engine/.venv/bin/python screening/results/aufbereiten.py` im Projektordner.", ""]
    (OUT / "README.md").write_text("\n".join(body))

    css = """body{font:15px/1.5 system-ui,sans-serif;color:#17202a;background:#fafafa;margin:0}main{max-width:1500px;margin:auto;padding:30px}h1,h2{line-height:1.2}h2{margin-top:42px}.scroll{overflow:auto;max-height:650px;background:white;border:1px solid #ddd;border-radius:6px}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:8px 10px;border-bottom:1px solid #eee;text-align:left;white-space:nowrap}th{background:#edf3fa;position:sticky;top:0}tr:nth-child(even){background:#f8fafc}img{width:100%;height:auto;background:white}a{color:#185fa5}.note{padding:14px 18px;background:#eaf2fb;border-left:4px solid #2a78d6}details{margin:15px 0}summary{cursor:pointer;font-weight:600}.index li{margin:7px 0}code{background:#eee;padding:2px 4px}"""
    page = ["<!doctype html><html lang='de'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
            "<title>Screening-Ergebnisse BPIC17</title><style>" + css + "</style><main>",
            "<h1>Screening-Ergebnisse BPIC17</h1>", f"<p>Aufbereitet am {html.escape(created)}. Vorhandene Daten, keine Neuberechnung.</p>",
            f"<p class='note'><strong>Umfassender Lauf:</strong> 20 Kandidaten, {approved} bestanden, {uncertain} unsicher. "
            "Aktionseffekt, Nutzen einer individuellen Policy und Simulator-Validierung werden getrennt ausgewiesen.</p>",
            "<p><a href='README.md'>Markdown-Übersicht</a> · <a href='quellenverzeichnis.csv'>Quellenverzeichnis</a> · "
            "<a href='alle_interventionspunkte.csv'>Eine Tabelle: alle Interventionspunkte (CSV)</a> · "
            "<a href='alle_interventionspunkte.png'>Tabelle als Bild</a></p>"]
    for index, (title, explanation, table) in enumerate(sections):
        page += ["<h2>" + html.escape(title) + "</h2><p>" + html.escape(explanation) + "</p>",
                 "<div class='scroll'>" + display(table).to_html(index=False, escape=True, border=0) + "</div>"]
        if index == 1:
            page += ["<img src='diagramme/screening_ueberblick.png' alt='Überblick aller 20 Screening-Kandidaten'>"]
        elif index == 2:
            page += ["<img src='diagramme/aipw_control_flow.png' alt='AIPW-Aktionseffekte mit Konfidenzintervallen'>"]
        elif index == 3:
            page += ["<img src='diagramme/validierung_control_flow.png' alt='Vergleich beider Simulator-Validierungen'>"]
    page += ["<h2>Kennzahlen und Interpretation</h2><ul>"] + ["<li>" + html.escape(text) + "</li>" for text in methodology] + ["</ul>"]
    for title, table in (("Historischer Screening-Teilstand", all_screening[all_screening.Lauf == "screening_alt"]),
                         ("Mitarbeiter-Effekte: alle 183 Zeilen", employees), ("Historische Mitarbeiter-Propensities: alle 511 Zeilen", propensities),
                         ("Mitarbeiter-Validierung: alle 553 Zeilen", validation_employees)):
        page += ["<details><summary>" + html.escape(title) + "</summary><div class='scroll'>" + display(table).to_html(index=False, escape=True, border=0) + "</div></details>"]
    page += ["<h2>Originaldateien und Herkunft</h2><p>Jede Datei liegt unverändert als Kopie vor. Logs, Kataloge und Startskripte sind ebenfalls enthalten.</p><ul class='index'>"]
    for run in RUNS:
        page += ["<li><details><summary>" + html.escape(run) + "</summary><ul>"]
        for item in MANIFEST:
            if item["Lauf"] == run:
                target = quote(item["Kopie"], safe="/")
                page += [f"<li><a href='{target}'>{html.escape(item['Kopie'])}</a></li>"]
        page += ["</ul></details></li>"]
    page += ["</ul></main></html>"]
    (OUT / "bericht.html").write_text("\n".join(page))
    (OUT / "aufbereitung.json").write_text(json.dumps({"created": created, "original_files": len(MANIFEST),
        "screening_rows": len(all_screening), "current_rows": len(current), "employee_effect_rows": len(employees),
        "historical_propensity_rows": len(propensities), "validation_rows": len(validations),
        "validation_employee_rows": len(validation_employees), "current_results": count.to_dict()}, indent=2, ensure_ascii=False) + "\n")


def main():
    for name in ("tabellen", "diagramme"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    collect()
    all_screening, current, employees, propensities = screening_tables()
    validations, validation_employees = validation_tables()
    assert len(employees) == 183 and len(propensities) == 511
    control = plots(current, validations)
    write_reports(all_screening, current, employees, propensities, validations, validation_employees, control)
    print((OUT / "aufbereitung.json").read_text())
    print(f"Fertig: {OUT / 'bericht.html'}")


if __name__ == "__main__":
    main()
