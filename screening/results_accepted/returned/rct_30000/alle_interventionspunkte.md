# BPIC17: Rücksendung des unterschriebenen Angebots (O_Returned) — 30.000 RCT-Fälle

Einheit: Prozentpunkte, keine relative prozentuale Änderung. Positive Effekte erhöhen die Wahrscheinlichkeit, dass mindestens ein Angebot im weiteren Fallverlauf das Zielereignis erreicht. Je Fall und Punkt zählt die erste Entscheidung; Fälle mit bereits erreichtem Ziel werden anhand des damaligen Prefix ausgeschlossen. Bei mehreren Optionen zeigt die Übersicht letzte vs. erste Katalogoption; bei Mitarbeitern den häufigsten Mitarbeiter vs. gleichverteilte Wahl aus seinem verfügbaren Pool. Diese Auswahl verwendet keine Outcomes. Die Spalte Hinweise zählt signifikante Kontraste nach Holm-Korrektur über sämtliche schätzbaren Optionen dieser Datenquelle und Zielgröße. Sie bezieht sich auch auf Optionen, die nicht in der Übersicht gezeigt sind. Kein gesicherter Hinweis bedeutet nicht nachgewiesene Wirkungslosigkeit.

AIPW mit logistischen XGBoost-Outcome-Modellen und fünf Cross-Fitting-Folds. Bekannte Randomisierungswahrscheinlichkeiten ohne Clipping; ganze Simulatorläufe in einem Fold. Standardfehler nach Simulatorlauf geclustert; t-Intervalle über bis zu zehn unabhängige Läufe. Reaktionszeit: Effekt der zugewiesenen Frist (ITT), einschließlich der Fälle ohne späteren Arbeitsbeginn. Alle weiteren Interventionen folgen der randomisierten Studienregel. 95-%-Intervalle sind punktweise. Die Holm-Korrektur der p-Werte prüft alle schätzbaren Kontraste dieser Datenquelle/Zielgröße gemeinsam bei 5 %. Sehr seltene Mitarbeiteroptionen sind in den Details markiert. Das Ziel wird bis zum Ende des vorhandenen Verlaufs beobachtet; spätere Annahmen außerhalb des Logs sind unbekannt.

Hauptziel: O_Accepted im weiteren Fallverlauf. A_Accepted ist ein vorheriger Antragsstatus; die Attributspalte Accepted beschreibt die Zulässigkeit eines Angebots. Die ergänzende Auswertung mit O_Returned betrachtet die Rücksendung des unterschriebenen Angebots. Bei mehreren Angeboten wird mindestens ein späterer Erfolg je Antrag betrachtet, nicht zwingend die Annahme genau des ersten Angebots.

| Interventionspunkt | Vergleich | Fälle | AIPW [95-%-KI], Prozentpunkte | Hinweise / Kontraste (Holm) | Überlappung % | Einfluss |
| --- | --- | --- | --- | --- | --- | --- |
| channel | O_Sent (online only) vs. O_Sent (mail and online) | 29185 | -4.91 [-6.30; -3.51] | 1 / 1 | 100.0 | Hinweis nach Holm-Korrektur |
| handle_leads | W_Handle leads/start vs. W_Handle leads/withdraw | 19142 | -2.55 [-3.66; -1.44] | 0 / 1 | 100.0 | kein gesicherter Hinweis |
| annehmen_nach_vervollstaendigung | A_Accepted vs. W_Complete application/resume | 18150 | +2.09 [+0.43; +3.76] | 0 / 1 | 100.0 | kein gesicherter Hinweis |
| angebot_nachlegen | O_Create Offer vs. O_Sent (mail and online) | 23818 | -6.35 [-7.34; -5.35] | 1 / 1 | 100.0 | Hinweis nach Holm-Korrektur |
| neues_angebot_beim_nachfassen | O_Create Offer vs. W_Call after offers/resume | 25295 | +6.53 [+5.63; +7.43] | 1 / 1 | 100.0 | Hinweis nach Holm-Korrektur |
| Mitarbeiter: W_Call incomplete files | User_30 vs. zufälliger verfügbarer Mitarbeiter | 2801 | +3.13 [-3.09; +9.35] | 0 / 70 | 100.0 | kein gesicherter Hinweis |
| Mitarbeiter: W_Complete application | User_75 vs. zufälliger verfügbarer Mitarbeiter | 25303 | +3.24 [-0.93; +7.42] | 0 / 65 | 100.0 | kein gesicherter Hinweis |
| Mitarbeiter: W_Handle leads | User_2 vs. zufälliger verfügbarer Mitarbeiter | 9652 | +0.60 [-4.61; +5.80] | 0 / 31 | 100.0 | kein gesicherter Hinweis |
| Mitarbeiter: W_Validate application | User_87 vs. zufälliger verfügbarer Mitarbeiter | 21707 | -3.43 [-6.39; -0.48] | 6 / 63 | 100.0 | Hinweis nach Holm-Korrektur |
| Mitarbeiter: W_Call after offers (resume) | User_2 vs. zufälliger verfügbarer Mitarbeiter | 23570 | -2.41 [-5.46; +0.63] | 0 / 53 | 100.0 | kein gesicherter Hinweis |
| Mitarbeiter: W_Call incomplete files (resume) | User_116 vs. zufälliger verfügbarer Mitarbeiter | 2311 | -0.44 [-14.49; +13.62] | 0 / 65 | 100.0 | kein gesicherter Hinweis |
| Mitarbeiter: W_Complete application (resume) | User_2 vs. zufälliger verfügbarer Mitarbeiter | 9647 | +1.96 [-4.16; +8.07] | 0 / 61 | 100.0 | kein gesicherter Hinweis |
| Mitarbeiter: W_Validate application (resume) | User_87 vs. zufälliger verfügbarer Mitarbeiter | 2031 | +1.83 [-6.49; +10.15] | 0 / 62 | 100.0 | kein gesicherter Hinweis |
| annehmen_bei_konzept | A_Accepted vs. W_Complete application/start | 15844 | +2.49 [+1.24; +3.75] | 0 / 1 | 100.0 | kein gesicherter Hinweis |
| angebotshoehe | mehr_als_beantragt vs. weniger_als_beantragt | 27447 | +0.72 [-0.68; +2.12] | 0 / 2 | 100.0 | kein gesicherter Hinweis |
| laufzeit | ueber_120_monate vs. bis_60_monate | 29990 | +1.84 [+0.20; +3.47] | 0 / 2 | 100.0 | kein gesicherter Hinweis |
| erste_auszahlung | voll vs. keine | 29990 | +2.45 [+1.16; +3.74] | 0 / 2 | 100.0 | kein gesicherter Hinweis |
| reaktionszeit_antrag | langsam vs. schnell (zugewiesene Frist, ITT) | 30000 | -0.55 [-1.71; +0.60] | 0 / 1 | 100.0 | kein gesicherter Hinweis |
| Mitarbeiter: W_Assess potential fraud | User_100 vs. zufälliger verfügbarer Mitarbeiter | 34 | -38.33 [-115.58; +38.91] | 0 / 3 | 100.0 | kein gesicherter Hinweis |
| Mitarbeiter: W_Handle leads (resume) | User_2 vs. zufälliger verfügbarer Mitarbeiter | 2188 | +2.46 [-2.99; +7.91] | 0 / 31 | 100.0 | kein gesicherter Hinweis |

[Datensatzanbieter: Selected und Accepted](https://promforum.win.tue.nl/discussion/749/meaning-of-columns-selected-and-accepted) · [Datensatzanbieter: Prozessstatus](https://promforum.win.tue.nl/discussion/comment/2126/)
