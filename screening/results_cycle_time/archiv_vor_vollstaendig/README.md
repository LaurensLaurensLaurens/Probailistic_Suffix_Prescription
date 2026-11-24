# Screening-Ergebnisse

Aufbereitung vorhandener BPIC17-Ergebnisse — 12.11.2026 13:21 CET.

**Eine Tabelle für alle Interventionspunkte:** [CSV](alle_interventionspunkte.csv) · [Bild](alle_interventionspunkte.png) · [Markdown](alle_interventionspunkte.md)

[Lesbarer HTML-Bericht](bericht.html) · [Alle Quellen mit Prüfsummen](quellenverzeichnis.csv) · [Originaldateien](originale/) · [Aufbereitete Tabellen](tabellen/)


## Umfang und Herkunft

Alle gefundenen Ergebnisse dieses Screening-Pakets betreffen BPIC17. Die Originaldateien wurden kopiert und über SHA256 geprüft; die bisherigen Ergebnisordner bleiben erhalten. Aufbereitet werden vorhandene Ergebnisse, ohne neue Modellschätzungen oder Simulatorläufe.

| Lauf | Umfang | Konfiguration |
| --- | --- | --- |
| screening_alle | 20 Kandidaten: 6 Control Flow + 14 Variablen/Mitarbeiter | Pool frei; mindestens 200 Fälle je Mitarbeiteroption |
| screening_alt | 8 Kandidaten: 1 Control Flow + 7 Mitarbeiter | Historischer Teilstand; Pool kompetent, mindestens 1 Fall |
| validierung_gesamt | 13 Kandidaten: 5 Control Flow + 8 Mitarbeiter | 20.000 simulierte Fälle je Lauf; früherer Katalog |
| validierung_neue_punkte | 4 Control-Flow-Kandidaten | 20.000 simulierte Fälle je Lauf; Katalog der neuen Wahlpunkte |

## Ergebnis des umfassenden Screenings

Von 20 Kandidaten erfüllen 0 alle Screening-Kriterien; 1 ist als unsicher markiert. Kein bestandenes Screening bedeutet hier nicht, dass alle Aktionen wirkungslos sind: Eine feste Aktion kann vorteilhaft sein, ohne dass eine fallabhängige Policy die beste feste Regel verbessert.

| Interventionspunkt | Art | AIPW-Effekt [95-%-KI] (Tage) | Policy vs. feste Option [95-%-KI] (Tage) | Policy vs. Bank [95-%-KI] (Tage) | Überlappung (%) | Placebo d | Screening-Ergebnis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| channel | Control Flow | 1.652 [0.366; 2.937] | 0.104 [-0.785; 0.993] | 1.688 [0.770; 2.606] | 6.3 | -0.035 | nein: kaum Überlappung |
| handle_leads | Control Flow | -2.217 [-2.779; -1.655] | -0.586 [-0.907; -0.264] | -0.176 [-0.469; 0.118] | 97.1 | -0.009 | nein: kein Policy-Gewinn gegenüber fester Regel |
| annehmen_nach_vervollstaendigung | Control Flow | 2.935 [2.445; 3.425] | -0.659 [-0.916; -0.402] | 1.483 [1.144; 1.822] | 91.5 | 0.022 | nein: kein Policy-Gewinn gegenüber fester Regel |
| angebot_nachlegen | Control Flow | -1.463 [-2.094; -0.832] | -0.511 [-0.863; -0.160] | -0.400 [-0.733; -0.068] | 80.4 | 0.017 | nein: kein Policy-Gewinn gegenüber fester Regel |
| neues_angebot_beim_nachfassen | Control Flow | 2.972 [2.109; 3.836] | -0.475 [-0.913; -0.037] | 2.340 [1.618; 3.061] | 42.7 | -0.061 | nein: kaum Überlappung |
| annehmen_bei_konzept | Control Flow | 5.366 [4.815; 5.918] | -0.596 [-0.825; -0.366] | 2.476 [2.014; 2.938] | 56.6 | -0.068 | nein: kein Policy-Gewinn gegenüber fester Regel |
| Mitarbeiter: W_Call incomplete files | Mitarbeiter | — | -0.555 [-1.150; 0.040] | 0.346 [-0.239; 0.931] | 4.5 | 0.223 | nein: kaum Überlappung |
| Mitarbeiter: W_Complete application | Mitarbeiter | — | -0.914 [-1.829; 0.001] | 0.867 [-0.027; 1.761] | 10.1 | 0.176 | nein: kaum Überlappung |
| Mitarbeiter: W_Handle leads | Mitarbeiter | — | -0.138 [-3.460; 3.184] | 0.487 [-2.069; 3.044] | 45.5 | 0.132 | nein: kaum Überlappung |
| Mitarbeiter: W_Validate application | Mitarbeiter | — | -0.318 [-0.793; 0.156] | 0.580 [0.117; 1.044] | 58.3 | 0.112 | unsicher: Placebo auffällig |
| Mitarbeiter: W_Call after offers (resume) | Mitarbeiter | — | 0.075 [-0.619; 0.769] | 1.138 [0.469; 1.806] | 19.0 | 0.261 | nein: kaum Überlappung |
| Mitarbeiter: W_Call incomplete files (resume) | Mitarbeiter | — | 0.550 [-0.200; 1.300] | 2.059 [1.358; 2.760] | 25.5 | 0.068 | nein: kaum Überlappung |
| Mitarbeiter: W_Complete application (resume) | Mitarbeiter | — | -0.358 [-3.129; 2.413] | 0.536 [-2.000; 3.072] | 32.2 | 0.211 | nein: kaum Überlappung |
| Mitarbeiter: W_Validate application (resume) | Mitarbeiter | — | -0.637 [-1.550; 0.277] | 0.110 [-0.695; 0.916] | 41.2 | 0.231 | nein: kaum Überlappung |
| angebotshoehe | Variable | — | -0.303 [-0.877; 0.271] | -0.135 [-0.685; 0.414] | 61.8 | 0.058 | nein: kein Policy-Gewinn gegenüber fester Regel |
| laufzeit | Variable | — | -0.096 [-0.521; 0.329] | 0.225 [-0.132; 0.582] | 96.6 | 0.008 | nein: kein Policy-Gewinn gegenüber fester Regel |
| erste_auszahlung | Variable | — | -0.434 [-0.722; -0.145] | 1.701 [1.383; 2.019] | 85.3 | 0.033 | nein: kein Policy-Gewinn gegenüber fester Regel |
| reaktionszeit_antrag | Variable | — | -0.079 [-0.217; 0.059] | 0.906 [0.727; 1.086] | 63.2 | — | nein: kein Policy-Gewinn gegenüber fester Regel |
| Mitarbeiter: W_Assess potential fraud | Mitarbeiter | — | — | — | — | — | nein: zu wenige Fälle (unter 200 je Option) |
| Mitarbeiter: W_Handle leads (resume) | Mitarbeiter | — | — | — | — | — | nein: zu wenige Fälle (unter 200 je Option) |

## AIPW-Aktionseffekte: Control Flow

Mittlerer Effekt der Aktion gegenüber der Referenz, mit den originalen Konfidenzintervallen. Das Outcome ist negative Durchlaufzeit in Tagen: positive Effekte bedeuten kürzere Durchlaufzeiten. Diese Mittelwerte sind keine individuellen CATEs und kein entropisches Risikomaß.

| Kandidat | Referenz | Aktion | Entscheidungsfälle | Effekt (AIPW) | Effekt KI unten | Effekt KI oben | Überlappung |
| --- | --- | --- | --- | --- | --- | --- | --- |
| channel | O_Sent (mail and online) | O_Sent (online only) | 30723.000 | 1.652 | 0.366 | 2.937 | 0.063 |
| handle_leads | W_Handle leads/withdraw | W_Handle leads/start | 20423.000 | -2.217 | -2.779 | -1.655 | 0.971 |
| annehmen_nach_vervollstaendigung | W_Complete application/resume | A_Accepted | 16423.000 | 2.935 | 2.445 | 3.425 | 0.915 |
| angebot_nachlegen | O_Sent (mail and online) | O_Create Offer | 30592.000 | -1.463 | -2.094 | -0.832 | 0.804 |
| neues_angebot_beim_nachfassen | W_Call after offers/resume | O_Create Offer | 25344.000 | 2.972 | 2.109 | 3.836 | 0.427 |
| annehmen_bei_konzept | W_Complete application/start | A_Accepted | 24296.000 | 5.366 | 4.815 | 5.918 | 0.566 |

## Simulator-Validierung

Die beiden Validierungen verwenden unterschiedliche Kataloge und bleiben getrennt. Echtes Log und beobachteter Simulator werden mittels AIPW ausgewertet; der Referenzeffekt im Simulator stammt aus randomisierten Entscheidungen. Die Validierung nutzt drei Cross-Fitting-Folds, das Screening standardmäßig fünf; daher müssen gleichnamige AIPW-Schätzungen nicht identisch sein.

| Lauf | Kandidat | Art | echt (AIPW) | Simulator beobachtet (AIPW) | Simulator wahr | Überlappung echt | Überlappung Simulator |
| --- | --- | --- | --- | --- | --- | --- | --- |
| validierung_gesamt | channel | Control Flow | 1.414 | 3.496 | 1.949 | 0.061 | 0.052 |
| validierung_gesamt | handle_leads | Control Flow | -2.158 | -1.529 | -1.384 | 0.969 | 0.999 |
| validierung_gesamt | annehmen_nach_vervollstaendigung | Control Flow | 2.955 | 2.691 | 1.912 | 0.916 | 0.914 |
| validierung_gesamt | angebot_nachlegen | Control Flow | -1.421 | -1.616 | -0.837 | 0.797 | 0.747 |
| validierung_gesamt | neues_angebot_beim_nachfassen | Control Flow | 2.965 | 1.280 | 1.185 | 0.425 | 0.315 |
| validierung_gesamt | Mitarbeiter: W_Call incomplete files | Mitarbeiter | — | — | — | 0.047 | 0.041 |
| validierung_gesamt | Mitarbeiter: W_Complete application | Mitarbeiter | — | — | — | 0.104 | 0.370 |
| validierung_gesamt | Mitarbeiter: W_Handle leads | Mitarbeiter | — | — | — | 0.448 | — |
| validierung_gesamt | Mitarbeiter: W_Validate application | Mitarbeiter | — | — | — | 0.586 | 0.147 |
| validierung_gesamt | Mitarbeiter: W_Call after offers (resume) | Mitarbeiter | — | — | — | 0.199 | 0.223 |
| validierung_gesamt | Mitarbeiter: W_Call incomplete files (resume) | Mitarbeiter | — | — | — | 0.261 | 0.168 |
| validierung_gesamt | Mitarbeiter: W_Complete application (resume) | Mitarbeiter | — | — | — | 0.320 | 0.198 |
| validierung_gesamt | Mitarbeiter: W_Validate application (resume) | Mitarbeiter | — | — | — | 0.413 | 0.415 |
| validierung_neue_punkte | handle_leads | Control Flow | -2.158 | -1.529 | -1.384 | 0.969 | 0.999 |
| validierung_neue_punkte | annehmen_nach_vervollstaendigung | Control Flow | 2.955 | 2.691 | 1.912 | 0.916 | 0.914 |
| validierung_neue_punkte | angebot_nachlegen | Control Flow | -1.421 | -1.616 | -1.645 | 0.797 | 0.747 |
| validierung_neue_punkte | annehmen_bei_konzept | Control Flow | 5.230 | 3.160 | 3.364 | 0.562 | 0.556 |

## Diagramme

![AIPW-Effekte](diagramme/aipw_control_flow.png)

![Screening-Überblick](diagramme/screening_ueberblick.png)

![Validierungen](diagramme/validierung_control_flow.png)

## Kennzahlen und Interpretation

- Policy-Gewinn: geschätzter Wert der fallabhängigen Policy minus beste feste Option (AIPW).
- Gewinn gegenüber Bank: geschätzter Policy-Wert minus beobachtetes Outcome. Positiv bedeutet kürzere Durchlaufzeit.
- Screening-Regel: Überlappung mindestens 0,5, untere Grenze des relevanten Gewinn-KI über 0 und |Placebo d| < 0,1. Bei Mitarbeiterpools wird der Gewinn gegenüber Bank verwendet, sonst der Gewinn gegenüber der besten festen Regel. Die Ergebnislabels wurden unverändert übernommen; ein fehlender Placebowert ist kein bestandener Placebotest.
- Überlappung: Anteil der Fälle, in denen alle verfügbaren Optionen ausreichend Propensity haben; Anteil 0–1.
- AIPW verwendet Propensity- und Outcome-Modelle mit Cross-Fitting; Propensities werden in der Formel auf [0,01; 0,99] begrenzt.
- Unbeobachtete Confounder können AIPW-Schätzungen verzerren. Ein positives Ergebnis oder KI allein ersetzt keine Überlappungs- und Placeboprüfung.
- Die sechs Control-Flow- und vier Nicht-Mitarbeiter-Variablen liegen zusätzlich in detaillierten Tabellen vor. Deren Werte wurden nach Prüfung auf Rundungskonsistenz bevorzugt. Mitarbeiter-Zusammenfassungen liegen nur gerundet vor; die Präzision und die Quelle stehen pro Zeile in den aufbereiteten CSVs.
- 183 Mitarbeiter-Effektschätzungen gehören zum umfassenden Screening; 511 historische Detailzeilen sind Propensities, keine AIPW-Effekte. Die Simulator-Validierung enthält zusätzlich 553 Mitarbeiterzeilen. Fehlende Werte bleiben fehlend.
- Die Mitarbeiterdatei der Validierung der neuen Wahlpunkte ist leer, weil dort ausschließlich Control Flow geprüft wurde.

## Dateien

- `tabellen/screening_aktuell.csv`: die 20 Kandidaten des umfassenden Screenings.
- `tabellen/screening_gesamt.csv`: alle 28 Kandidatenzeilen beider Screening-Stände mit Herkunft.
- `tabellen/aipw_control_flow.csv`: Aktion, Referenz und AIPW-Effekte mit KI.
- `tabellen/mitarbeitereffekte.csv`: 183 Mitarbeiter-Effektschätzungen.
- `tabellen/mitarbeiter_propensities_historisch.csv`: 511 historische Propensity-Zeilen.
- `tabellen/validierung_gesamt.csv`: 17 Kandidatenzeilen beider Validierungen.
- `tabellen/validierung_mitarbeiter.csv`: 553 Mitarbeiterzeilen der Validierung.
- `originale/`: unveränderte CSVs, Grafiken, Beispiele und der vollständige Bericht des umfassenden Screening-Laufs.
- `provenienz/`: zugehörige Logs, Aufrufskripte und verfügbare Katalogdateien. Katalogkopien zeigen den heutigen Dateiinhalt; ein historischer Dateistand lässt sich daraus allein nicht beweisen.
- `quellenverzeichnis.csv` / `.json`: jeder kopierte Ursprungspfad, Ziel, Dateigröße, Zeilenzahl und SHA256.

Neu erstellen: `process-simulation-engine/.venv/bin/python screening/results/aufbereiten.py` im Projektordner.
