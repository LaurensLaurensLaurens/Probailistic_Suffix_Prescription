"""Readable tables for the complete real-log and randomized screening."""
from __future__ import annotations

from pathlib import Path
import html
import shutil
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NOTE = ("Outcome: negative Durchlaufzeit in Tagen; positive Effekte bedeuten kürzere Durchlaufzeit. "
        "Alle Punkte und alle Optionen werden geprüft. Bei mehreren Optionen zeigt die Übersicht einen vorab nach "
        "Katalogreihenfolge festgelegten Vergleich; bei Mitarbeitern den häufigsten Mitarbeiter gegenüber einer "
        "zufälligen Wahl aus seinem verfügbaren Pool. Diese Auswahl verwendet keine Outcomes. "
        "Sämtliche weiteren Vergleiche mit 95-%-Intervallen stehen in aipw_alle_optionen.csv und im HTML-Bericht. "
        "Intervalle sind punktweise, ohne Korrektur für multiples Testen. Seltene und nicht identifizierbare "
        "Vergleiche bleiben ausdrücklich gekennzeichnet.")


def number(value):
    return f"{value:.3f}" if pd.notna(value) and np.isfinite(value) else "nicht schätzbar"


def readable(summary):
    result = pd.DataFrame({"Interventionspunkt": summary["Interventionspunkt"], "Vergleich": summary["Vergleich"],
                           "Fälle / Optionen": summary["Entscheidungsfälle"].astype(str) + " / " + summary["Optionen"].astype(str)})
    result["AIPW [95-%-KI], Tage"] = [f"{number(r['Effekt (AIPW)'])} [{number(r['KI unten'])}; {number(r['KI oben'])}]"
                                                if pd.notna(r["Effekt (AIPW)"]) else "nicht schätzbar"
                                                for _, r in summary.iterrows()]
    result["Überlappung %"] = summary["Überlappung"].map(lambda x: f"{100*x:.1f}" if pd.notna(x) else "—") if "Überlappung" in summary else "—"
    result["Status"] = summary["Status"]
    return result


def markdown(table):
    clean = lambda value: str(value).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(clean, table.columns)) + " |",
                       "| " + " | ".join(["---"] * len(table.columns)) + " |"] +
                      ["| " + " | ".join(map(clean, values)) + " |" for values in table.itertuples(index=False, name=None)])


def render(out, summary, details, *, randomized, manifest=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "alle_interventionspunkte.csv", index=False)
    details.to_csv(out / "aipw_alle_optionen.csv", index=False)
    table = readable(summary)
    title = f"BPIC17: AIPW an allen {len(summary)} Interventionspunkten — " + ("30.000 RCT-Fälle" if randomized else "echtes Log")
    method = ("RCT: bekannte Zuteilungswahrscheinlichkeiten ohne Clipping; fünf Cross-Fitting-Folds nach Simulatorlauf. "
              "95-%-KI mit nach Simulatorlauf geclusterten Standardfehlern und t-Verteilung. "
              "Die Fälle teilen innerhalb eines Laufs Mitarbeiter; die zehn Läufe sind unabhängig. "
              "Reaktionszeit: Effekt der zugewiesenen Frist (Intention to treat), einschließlich der Fälle ohne späteren Arbeitsbeginn. "
              "Weitere Entscheidungen und spätere Besuche eines Punkts folgen weiterhin der randomisierten Studienregel."
              if randomized else
              "Echtes Log: fünf Cross-Fitting-Folds nach Fall, XGBoost für Outcome und Propensity; "
              "Propensity-Clipping [0,01; 0,99]. Keine Mindestfallzahl von 200 und keine Zusammenfassung seltener Mitarbeiter als other. "
              "Mitarbeiter werden nur in Fällen verglichen, in denen sie zum freien Pool gehören und mindestens zwei Personen wählbar sind. "
              "Der beobachtete Mitarbeiterpool enthält nur Personen, die diese Tätigkeit im Log selbst ausgeführt haben; "
              "für nie beobachtete Personen gibt es keine identifizierbare Wirkung. "
              "Sehr kleine Fallzahlen und fehlende Überlappung begrenzen die Aussagekraft; unbeobachtete Konfundierung bleibt möglich.")
    text = f"# {title}\n\n{NOTE}\n\n{method}\n\n" + markdown(table) + "\n"
    (out / "alle_interventionspunkte.md").write_text(text)
    body = [f"<h1>{html.escape(title)}</h1>", f"<p>{html.escape(NOTE)}</p>", f"<p>{html.escape(method)}</p>",
            "<p><a href='alle_interventionspunkte.csv'>Übersicht CSV</a> · <a href='aipw_alle_optionen.csv'>Alle Optionen CSV</a></p>",
            table.to_html(index=False, escape=True)]
    for _, row in summary.iterrows():
        name = row["Interventionspunkt"]
        part = details[details["Interventionspunkt"] == name] if len(details) else details
        body += [f"<h2>{html.escape(name)}</h2>", f"<p>{html.escape(row['Status'])}</p>"]
        body += [part.round(4).to_html(index=False, escape=True)] if len(part) else ["<p>Kein identifizierbarer Vergleich; siehe Status.</p>"]
    (out / "bericht.html").write_text("<!doctype html><html lang='de'><meta charset='utf-8'><title>AIPW alle Punkte</title>"
        "<style>body{font:15px system-ui;margin:32px;color:#17202a}table{border-collapse:collapse;margin:20px 0}"
        "td,th{border:1px solid #dce2e8;padding:7px}th{background:#e7eef8}tr:nth-child(even){background:#f6f8fa}</style>" + "\n".join(body) + "</html>")
    cells = []
    for values in table.itertuples(index=False, name=None):
        cells.append(["\n".join(textwrap.wrap(str(value), width)) for value, width in zip(values, [37, 50, 15, 33, 14, 37])])
    fig, ax = plt.subplots(figsize=(25, max(12, .56 * len(table) + 3)))
    fig.subplots_adjust(left=.015, right=.985, top=.88, bottom=.13)
    ax.axis("off")
    chart = ax.table(cellText=cells, colLabels=list(table.columns), colWidths=[.23, .29, .07, .19, .07, .15],
                     cellLoc="center", bbox=[0, 0, 1, 1])
    chart.auto_set_font_size(False)
    chart.set_fontsize(9)
    for (i, j), cell in chart.get_celld().items():
        cell.set_edgecolor("#dce2e8")
        cell.set_linewidth(.5)
        cell.set_facecolor("#e7eef8" if i == 0 else "#f5f8fc" if i % 2 == 0 else "white")
        if i == 0:
            cell.set_text_props(weight="bold")
        elif j in (0, 1, 5):
            cell.set_text_props(ha="left")
    fig.suptitle(title, fontsize=20, y=.975)
    fig.text(.02, .935, "Positive Werte = kürzere Durchlaufzeit. Alle weiteren Optionen mit Konfidenzintervallen: aipw_alle_optionen.csv", fontsize=12)
    fig.text(.02, .91, "Übersicht: festgelegte letzte vs. erste Option; Mitarbeiter: häufigster Mitarbeiter vs. zufälliger freier Mitarbeiter.", fontsize=11)
    fig.text(.02, .075, "\n".join(textwrap.wrap(method, 190)), fontsize=10, va="top")
    fig.savefig(out / "alle_interventionspunkte.png", dpi=160)
    plt.close(fig)


def publish_index(root):
    root = Path(root)
    archive = root / "archiv_vor_vollstaendig"
    archive.mkdir(exist_ok=True)
    for name in ("alle_interventionspunkte.csv", "alle_interventionspunkte.md", "alle_interventionspunkte.png", "README.md", "bericht.html"):
        if (root / name).exists() and not (archive / name).exists():
            shutil.copy2(root / name, archive / name)
    real, rct = root / "aipw_alle_real", root / "rct_30000"
    if (real / "alle_interventionspunkte.csv").exists():
        for suffix in ("csv", "md", "png"):
            shutil.copy2(real / f"alle_interventionspunkte.{suffix}", root / f"alle_interventionspunkte.{suffix}")
    text = ("# Screening aller Interventionspunkte\n\n"
            "Vollständiges AIPW-Screening aller 20 BPIC17-Kandidaten (6 Control Flow, 10 Mitarbeiter, 4 weitere Variablen). "
            "Bei mehreren Optionen stehen alle Kontraste mit expliziter Referenz und 95-%-Intervall in der Detailtabelle.\n\n"
            "- Echtes Log: [Bericht](aipw_alle_real/bericht.html), [Übersicht](aipw_alle_real/alle_interventionspunkte.csv), "
            "[alle Optionen](aipw_alle_real/aipw_alle_optionen.csv).\n"
            "- 30.000 RCT-Fälle: [Bericht](rct_30000/bericht.html), [Übersicht](rct_30000/alle_interventionspunkte.csv), "
            "[alle Optionen](rct_30000/aipw_alle_optionen.csv), [Daten und Design](rct_30000/daten/).\n"
            "- [Laufstatus](vollstaendig_lauf/status.json) und [Protokoll](vollstaendig_lauf/lauf.log).\n"
            "- Frühere Aufbereitung: [archivierter Bericht](archiv_vor_vollstaendig/bericht.html). "
            "Die ursprünglichen Ergebnisse und Provenienzdateien bleiben in originale/, tabellen/ und provenienz/.\n\n"
            + NOTE + "\n\nNeu starten/fortsetzen: `process-simulation-engine/.venv/bin/python -u screening/run_all_points.py`.\n")
    (root / "README.md").write_text(text)
    (root / "bericht.html").write_text("<!doctype html><html lang='de'><meta charset='utf-8'><title>Screening alle Punkte</title>"
        "<body style='font:17px system-ui;margin:40px'><h1>BPIC17: vollständiges AIPW-Screening</h1>"
        "<p>20 Interventionspunkte: 6 Control Flow, 10 Mitarbeiter, 4 weitere Variablen.</p>"
        "<ul><li><a href='aipw_alle_real/bericht.html'>Echtes Log: alle Punkte und Optionen</a></li>"
        "<li><a href='rct_30000/bericht.html'>30.000 RCT-Fälle: alle Punkte und Optionen</a></li>"
        "<li><a href='vollstaendig_lauf/status.json'>Laufstatus</a></li>"
        "<li><a href='archiv_vor_vollstaendig/bericht.html'>Bisheriger Bericht (Archiv)</a></li></ul>"
        f"<p>{html.escape(NOTE)}</p></body></html>")
