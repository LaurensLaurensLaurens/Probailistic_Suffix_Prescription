# Screening aller Interventionspunkte

Vollständiges AIPW-Screening aller 20 BPIC17-Kandidaten (6 Control Flow, 10 Mitarbeiter, 4 weitere Variablen). Bei mehreren Optionen stehen alle Kontraste mit expliziter Referenz und 95-%-Intervall in der Detailtabelle.

- Echtes Log: [Bericht](aipw_alle_real/bericht.html), [Übersicht](aipw_alle_real/alle_interventionspunkte.csv), [alle Optionen](aipw_alle_real/aipw_alle_optionen.csv).
- 30.000 RCT-Fälle: [Bericht](rct_30000/bericht.html), [Übersicht](rct_30000/alle_interventionspunkte.csv), [alle Optionen](rct_30000/aipw_alle_optionen.csv), [Daten und Design](rct_30000/daten/).
- [Laufstatus](vollstaendig_lauf/status.json) und [Protokoll](vollstaendig_lauf/lauf.log).
- Frühere Aufbereitung: [archivierter Bericht](archiv_vor_vollstaendig/bericht.html). Die ursprünglichen Ergebnisse und Provenienzdateien bleiben in originale/, tabellen/ und provenienz/.

Outcome: negative Durchlaufzeit in Tagen; positive Effekte bedeuten kürzere Durchlaufzeit. Alle Punkte und alle Optionen werden geprüft. Bei mehreren Optionen zeigt die Übersicht einen vorab nach Katalogreihenfolge festgelegten Vergleich; bei Mitarbeitern den häufigsten Mitarbeiter gegenüber einer zufälligen Wahl aus seinem verfügbaren Pool. Diese Auswahl verwendet keine Outcomes. Sämtliche weiteren Vergleiche mit 95-%-Intervallen stehen in aipw_alle_optionen.csv und im HTML-Bericht. Intervalle sind punktweise, ohne Korrektur für multiples Testen. Seltene und nicht identifizierbare Vergleiche bleiben ausdrücklich gekennzeichnet.

Neu starten/fortsetzen: `process-simulation-engine/.venv/bin/python -u screening/run_all_points.py`.
