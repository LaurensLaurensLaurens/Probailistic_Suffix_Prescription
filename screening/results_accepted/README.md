# Einfluss auf die Angebotsannahme

Alle 20 bisherigen BPIC17-Interventionspunkte, dieselben echten Fälle und dieselben vorhandenen 30.000 RCT-Fälle. Hauptziel ist ein späteres O_Accepted je Antrag; ergänzend wird O_Returned ausgewertet. Die Effekte sind Risikodifferenzen in Prozentpunkten, mit 95-%-Intervallen und Holm-korrigierten p-Werten.

| Ziel | Echtes Log | 30.000 RCT-Fälle |
| --- | --- | --- |
| O_Accepted: erfolgreicher Angebotsabschluss | [Bericht](aipw_alle_real/bericht.html) · [Bild](aipw_alle_real/alle_interventionspunkte.png) | [Bericht](rct_30000/bericht.html) · [Bild](rct_30000/alle_interventionspunkte.png) |
| O_Returned: unterschriebenes Angebot zurückgesandt | [Bericht](returned/aipw_alle_real/bericht.html) | [Bericht](returned/rct_30000/bericht.html) |

Einheit: Prozentpunkte, keine relative prozentuale Änderung. Positive Effekte erhöhen die Wahrscheinlichkeit, dass mindestens ein Angebot im weiteren Fallverlauf das Zielereignis erreicht. Je Fall und Punkt zählt die erste Entscheidung; Fälle mit bereits erreichtem Ziel werden anhand des damaligen Prefix ausgeschlossen. Bei mehreren Optionen zeigt die Übersicht letzte vs. erste Katalogoption; bei Mitarbeitern den häufigsten Mitarbeiter vs. gleichverteilte Wahl aus seinem verfügbaren Pool. Diese Auswahl verwendet keine Outcomes. Die Spalte Hinweise zählt signifikante Kontraste nach Holm-Korrektur über sämtliche schätzbaren Optionen dieser Datenquelle und Zielgröße. Sie bezieht sich auch auf Optionen, die nicht in der Übersicht gezeigt sind. Kein gesicherter Hinweis bedeutet nicht nachgewiesene Wirkungslosigkeit.

Bereits erreichte Zielereignisse werden vor der Schätzung ausgeschlossen. Damit wird die zeitliche Reihenfolge gewahrt. Die neue Stichprobe je Punkt kann von der Durchlaufzeitauswertung abweichen. Bei Angeboten mit bereits geschehener Rücksendung ist ein Effekt auf diese Rücksendung nicht mehr identifizierbar.

[Zieldefinitionen und Grundquoten](zielgroessen.json) · [Status](status.json) · [Protokoll](lauf.log) · [Provenienz](manifest.json) · [Ausgangsdaten](../results_cycle_time/rct_30000/daten/README.md)

[Datensatzanbieter: Selected und Accepted](https://promforum.win.tue.nl/discussion/749/meaning-of-columns-selected-and-accepted) · [Prozessstatus](https://promforum.win.tue.nl/discussion/comment/2126/)

Ausführen/fortsetzen: `process-simulation-engine/.venv/bin/python -u screening/run_acceptance.py`.
