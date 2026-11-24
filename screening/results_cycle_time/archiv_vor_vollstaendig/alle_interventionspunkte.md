# Auswertung aller Interventionspunkte

Eine Zeile je Interventionspunkt des umfassenden BPIC17-Screenings (20 Punkte). Alle Effekt- und Gewinnwerte in Tagen, mit 95-%-Konfidenzintervall. Positiv bedeutet kürzere Durchlaufzeit. Policy vs. feste Option ist der zusätzliche Nutzen einer individuellen Policy gegenüber der besten festen Option. Ein einzelner AIPW-Aktionseffekt liegt hier nur für die binären Control-Flow-Punkte vor; Mitarbeitereffekte stehen je Mitarbeiter in tabellen/mitarbeitereffekte.csv. Ein Gedankenstrich bedeutet nicht anwendbar oder nicht verfügbar, nicht null.

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
