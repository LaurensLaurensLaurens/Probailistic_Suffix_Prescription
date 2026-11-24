"""Acceptance AIPW tables in percentage points, including all option contrasts."""
from __future__ import annotations

import html
from pathlib import Path
import shutil
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .acceptance import ACTIVITY_URL, SEMANTICS_URL, ENDPOINTS
from .all_points_report import markdown, named_actions


NOTE = ("Einheit: Prozentpunkte, keine relative prozentuale Änderung. Positive Effekte erhöhen die Wahrscheinlichkeit, "
        "dass mindestens ein Angebot im weiteren Fallverlauf das Zielereignis erreicht. Je Fall und Punkt zählt die erste "
        "Entscheidung; Fälle mit bereits erreichtem Ziel werden anhand des damaligen Prefix ausgeschlossen. "
        "Bei mehreren Optionen zeigt die Übersicht letzte vs. erste Katalogoption; bei Mitarbeitern den häufigsten "
        "Mitarbeiter vs. gleichverteilte Wahl aus seinem verfügbaren Pool. Diese Auswahl verwendet keine Outcomes. "
        "Die Spalte Hinweise zählt signifikante Kontraste nach Holm-Korrektur über sämtliche schätzbaren Optionen "
        "dieser Datenquelle und Zielgröße. Sie bezieht sich auch auf Optionen, die nicht in der Übersicht gezeigt sind. "
        "Kein gesicherter Hinweis bedeutet nicht nachgewiesene Wirkungslosigkeit.")


def readable(summary):
    rows = []
    for _, r in summary.iterrows():
        effect = (f"{r['Effekt (AIPW) (Prozentpunkte)']:+.2f} "
                  f"[{r['KI unten (Prozentpunkte)']:+.2f}; {r['KI oben (Prozentpunkte)']:+.2f}]"
                  if pd.notna(r["Effekt (AIPW) (Prozentpunkte)"]) else "nicht schätzbar")
        rows.append({"Interventionspunkt": r["Interventionspunkt"], "Vergleich": r["Vergleich"],
                     "Fälle": int(r["Entscheidungsfälle"]), "AIPW [95-%-KI], Prozentpunkte": effect,
                     "Hinweise / Kontraste (Holm)": f"{int(r['Kontraste mit Hinweis nach Holm'])} / {int(r['Schätzbare Kontraste'])}",
                     "Überlappung %": f"{100 * r['Überlappung']:.1f}" if pd.notna(r.get("Überlappung")) else "—",
                     "Einfluss": r["Einfluss"] + ("; geringe Überlappung" if r.get("Überlappung", 1) < .5 else "")})
    return pd.DataFrame(rows)


def render(out, summary, details, endpoint, source):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary, details = named_actions(summary, details)
    summary.to_csv(out / "alle_interventionspunkte.csv", index=False)
    details.to_csv(out / "aipw_alle_optionen.csv", index=False)
    table = readable(summary)
    spec = ENDPOINTS[endpoint]
    label = "30.000 RCT-Fälle" if source == "rct" else "echtes Log"
    title = f"BPIC17: {spec['label']} ({spec['event']}) — {label}"
    method = ("AIPW mit logistischen XGBoost-Outcome-Modellen und fünf Cross-Fitting-Folds. "
              + ("Bekannte Randomisierungswahrscheinlichkeiten ohne Clipping; ganze Simulatorläufe in einem Fold. "
                 "Standardfehler nach Simulatorlauf geclustert; t-Intervalle über bis zu zehn unabhängige Läufe. "
                 "Reaktionszeit: Effekt der zugewiesenen Frist (ITT), einschließlich der Fälle ohne späteren Arbeitsbeginn. "
                 "Alle weiteren Interventionen folgen der randomisierten Studienregel. " if source == "rct" else
                 "Propensity mit XGBoost geschätzt und auf den verfügbaren Pool normiert; Clipping [0,01; 0,99]. "
                 "Cross-Fitting nach Fall; Mitarbeiterpool frei, mit im Log beobachteten Mitarbeitern. "
                 "Kausale Interpretation setzt ausreichende Überlappung und keine unbeobachteten Confounder voraus. ")
              + "95-%-Intervalle sind punktweise. Die Holm-Korrektur der p-Werte prüft alle schätzbaren Kontraste "
                "dieser Datenquelle/Zielgröße gemeinsam bei 5 %. Sehr seltene Mitarbeiteroptionen sind in den Details markiert. "
                "Das Ziel wird bis zum Ende des vorhandenen Verlaufs beobachtet; spätere Annahmen außerhalb des Logs sind unbekannt.")
    interpretation = ("Hauptziel: O_Accepted im weiteren Fallverlauf. A_Accepted ist ein vorheriger Antragsstatus; "
                      "die Attributspalte Accepted beschreibt die Zulässigkeit eines Angebots. "
                      "Die ergänzende Auswertung mit O_Returned betrachtet die Rücksendung des unterschriebenen Angebots. "
                      "Bei mehreren Angeboten wird mindestens ein späterer Erfolg je Antrag betrachtet, nicht zwingend "
                      "die Annahme genau des ersten Angebots.")
    md = f"# {title}\n\n{NOTE}\n\n{method}\n\n{interpretation}\n\n" + markdown(table)
    md += (f"\n\n[Datensatzanbieter: Selected und Accepted]({SEMANTICS_URL}) · "
           f"[Datensatzanbieter: Prozessstatus]({ACTIVITY_URL})\n")
    (out / "alle_interventionspunkte.md").write_text(md)
    parts = [f"<h1>{html.escape(title)}</h1>", f"<p>{html.escape(NOTE)}</p>", f"<p>{html.escape(method)}</p>",
             f"<p>{html.escape(interpretation)}</p>",
             "<p><a href='alle_interventionspunkte.png'>Tabelle als Bild</a> · <a href='alle_interventionspunkte.csv'>Übersicht CSV</a> · "
             "<a href='aipw_alle_optionen.csv'>Alle Optionen CSV</a></p>", table.to_html(index=False, escape=True)]
    columns = ["Option", "Referenz", "Vergleichsfälle", "Gewählt in Vergleichsfällen", "Effekt (AIPW) (Prozentpunkte)",
               "KI unten (Prozentpunkte)", "KI oben (Prozentpunkte)", "p-Wert", "p Holm", "Überlappung", "Status"]
    for _, row in summary.iterrows():
        name = row["Interventionspunkt"]
        parts += [f"<h2>{html.escape(name)}</h2>",
                  f"<p>Entscheidungsfälle: {int(row['Entscheidungsfälle'])}; wegen bereits erreichtem Ziel ausgeschlossen: "
                  f"{int(row['Ziel bereits erreicht: ausgeschlossen'])}. {html.escape(row['Status'])}</p>"]
        part = details[details.Interventionspunkt.eq(name)] if len(details) else details
        parts += [part.reindex(columns=columns).round(5).to_html(index=False, escape=True)] if len(part) else ["<p>Kein identifizierbarer Vergleich.</p>"]
    parts.append(f"<p>Definitionen: <a href='{SEMANTICS_URL}'>Selected / Accepted</a> · <a href='{ACTIVITY_URL}'>Prozessstatus</a></p>")
    (out / "bericht.html").write_text("<!doctype html><html lang='de'><meta charset='utf-8'><title>AIPW Angebotsannahme</title>"
        "<style>body{font:15px system-ui;margin:32px;color:#17202a}table{border-collapse:collapse;margin:20px 0}"
        "td,th{border:1px solid #dce2e8;padding:7px}th{background:#e7eef8}tr:nth-child(even){background:#f6f8fa}</style>" + "\n".join(parts) + "</html>")
    cells = [["\n".join(textwrap.wrap(str(value), width)) for value, width in zip(values, [37, 46, 12, 30, 15, 12, 27])]
             for values in table.itertuples(index=False, name=None)]
    fig, ax = plt.subplots(figsize=(26, 15))
    fig.subplots_adjust(left=.015, right=.985, top=.88, bottom=.15)
    ax.axis("off")
    chart = ax.table(cellText=cells, colLabels=["Interventionspunkt", "Vergleich", "Fälle", "AIPW [95-%-KI]\nProzentpunkte",
                                               "Hinweise / Kontraste\nHolm, 5 %", "Überlappung\n%", "Einfluss"],
                     colWidths=[.23, .27, .05, .17, .08, .06, .14], cellLoc="center", bbox=[0, 0, 1, 1])
    chart.auto_set_font_size(False)
    chart.set_fontsize(9)
    for (i, j), cell in chart.get_celld().items():
        cell.set_edgecolor("#dce2e8")
        cell.set_linewidth(.5)
        cell.set_facecolor("#e7eef8" if i == 0 else "#f5f8fc" if i % 2 == 0 else "white")
        if i == 0:
            cell.set_text_props(weight="bold")
        elif j in (0, 1, 6):
            cell.set_text_props(ha="left")
    fig.suptitle(title, fontsize=19, y=.975)
    fig.text(.02, .94, "20 Interventionspunkte · Positive Werte = höhere Zielwahrscheinlichkeit · Alle Optionsvergleiche im HTML-Bericht / CSV", fontsize=12)
    fig.text(.02, .91, "Gezeigter Vergleich: Katalogoption bzw. häufigster Mitarbeiter. Hinweise: alle Optionen nach Holm-Korrektur.", fontsize=11)
    fig.text(.02, .11, "\n".join(textwrap.wrap(method, 205)), fontsize=10, va="top")
    fig.savefig(out / "alle_interventionspunkte.png", dpi=160)
    plt.close(fig)


def publish(root):
    root = Path(root)
    for suffix in ("csv", "md", "png"):
        source = root / "aipw_alle_real" / f"alle_interventionspunkte.{suffix}"
        if source.exists():
            shutil.copy2(source, root / source.name)
    note = ("# Einfluss auf die Angebotsannahme\n\n"
            "Alle 20 bisherigen BPIC17-Interventionspunkte, dieselben echten Fälle und dieselben vorhandenen 30.000 RCT-Fälle. "
            "Hauptziel ist ein späteres O_Accepted je Antrag; ergänzend wird O_Returned ausgewertet. "
            "Die Effekte sind Risikodifferenzen in Prozentpunkten, mit 95-%-Intervallen und Holm-korrigierten p-Werten.\n\n"
            "| Ziel | Echtes Log | 30.000 RCT-Fälle |\n| --- | --- | --- |\n"
            "| O_Accepted: erfolgreicher Angebotsabschluss | [Bericht](aipw_alle_real/bericht.html) · [Bild](aipw_alle_real/alle_interventionspunkte.png) "
            "| [Bericht](rct_30000/bericht.html) · [Bild](rct_30000/alle_interventionspunkte.png) |\n"
            "| O_Returned: unterschriebenes Angebot zurückgesandt | [Bericht](returned/aipw_alle_real/bericht.html) "
            "| [Bericht](returned/rct_30000/bericht.html) |\n\n"
            + NOTE + "\n\nBereits erreichte Zielereignisse werden vor der Schätzung ausgeschlossen. "
            "Damit wird die zeitliche Reihenfolge gewahrt. Die neue Stichprobe je Punkt kann von der Durchlaufzeitauswertung abweichen. "
            "Bei Angeboten mit bereits geschehener Rücksendung ist ein Effekt auf diese Rücksendung nicht mehr identifizierbar.\n\n"
            "[Zieldefinitionen und Grundquoten](zielgroessen.json) · [Status](status.json) · [Protokoll](lauf.log) · "
            "[Provenienz](manifest.json) · [Ausgangsdaten](../results_cycle_time/rct_30000/daten/README.md)\n\n"
            f"[Datensatzanbieter: Selected und Accepted]({SEMANTICS_URL}) · [Prozessstatus]({ACTIVITY_URL})\n\n"
            "Ausführen/fortsetzen: `process-simulation-engine/.venv/bin/python -u screening/run_acceptance.py`.\n")
    (root / "README.md").write_text(note)
    (root / "bericht.html").write_text("<!doctype html><html lang='de'><meta charset='utf-8'><title>Angebotsannahme</title>"
        "<body style='font:17px system-ui;margin:40px'><h1>Einfluss auf die Angebotsannahme</h1>"
        "<p>20 Interventionspunkte, echtes Log und vorhandene 30.000 RCT-Fälle. Effekte in Prozentpunkten.</p>"
        "<ul><li><a href='aipw_alle_real/bericht.html'>O_Accepted: echtes Log</a></li>"
        "<li><a href='rct_30000/bericht.html'>O_Accepted: RCT</a></li>"
        "<li><a href='returned/aipw_alle_real/bericht.html'>O_Returned: echtes Log</a></li>"
        "<li><a href='returned/rct_30000/bericht.html'>O_Returned: RCT</a></li></ul>"
        f"<p>{html.escape(NOTE)}</p></body></html>")
