# BPIC17: Erfolgreiche Angebotsannahme (O_Accepted) — 30.000 RCT-Fälle

Einheit: Prozentpunkte, keine relative prozentuale Änderung. Positive Effekte erhöhen die Wahrscheinlichkeit, dass mindestens ein Angebot im weiteren Fallverlauf das Zielereignis erreicht. Je Fall und Punkt zählt die erste Entscheidung; Fälle mit bereits erreichtem Ziel werden anhand des damaligen Prefix ausgeschlossen. Bei mehreren Optionen zeigt die Übersicht letzte vs. erste Katalogoption; bei Mitarbeitern den häufigsten Mitarbeiter vs. gleichverteilte Wahl aus seinem verfügbaren Pool. Diese Auswahl verwendet keine Outcomes. Die Spalte Hinweise zählt signifikante Kontraste nach Holm-Korrektur über sämtliche schätzbaren Optionen dieser Datenquelle und Zielgröße. Sie bezieht sich auch auf Optionen, die nicht in der Übersicht gezeigt sind. Kein gesicherter Hinweis bedeutet nicht nachgewiesene Wirkungslosigkeit.

AIPW mit logistischen XGBoost-Outcome-Modellen und fünf Cross-Fitting-Folds. Bekannte Randomisierungswahrscheinlichkeiten ohne Clipping; ganze Simulatorläufe in einem Fold. Standardfehler nach Simulatorlauf geclustert; t-Intervalle über bis zu zehn unabhängige Läufe. Reaktionszeit: Effekt der zugewiesenen Frist (ITT), einschließlich der Fälle ohne späteren Arbeitsbeginn. Alle weiteren Interventionen folgen der randomisierten Studienregel. 95-%-Intervalle sind punktweise. Die Holm-Korrektur der p-Werte prüft alle schätzbaren Kontraste dieser Datenquelle/Zielgröße gemeinsam bei 5 %. Sehr seltene Mitarbeiteroptionen sind in den Details markiert. Das Ziel wird bis zum Ende des vorhandenen Verlaufs beobachtet; spätere Annahmen außerhalb des Logs sind unbekannt.

Hauptziel: O_Accepted im weiteren Fallverlauf. A_Accepted ist ein vorheriger Antragsstatus; die Attributspalte Accepted beschreibt die Zulässigkeit eines Angebots. Die ergänzende Auswertung mit O_Returned betrachtet die Rücksendung des unterschriebenen Angebots. Bei mehreren Angeboten wird mindestens ein späterer Erfolg je Antrag betrachtet, nicht zwingend die Annahme genau des ersten Angebots.

| Interventionspunkt | Vergleich | Fälle | AIPW [95-%-KI], Prozentpunkte | Hinweise / Kontraste (Holm) | Überlappung % | Einfluss |
| --- | --- | --- | --- | --- | --- | --- |
| channel | O_Sent (online only) vs. O_Sent (mail and online) | 29219 | -3.02 [-4.11; -1.92] | 0 / 1 | 100.0 | kein gesicherter Hinweis |
| handle_leads | W_Handle leads/start vs. W_Handle leads/withdraw | 19142 | -4.20 [-5.71; -2.69] | 0 / 1 | 100.0 | kein gesicherter Hinweis |
| annehmen_nach_vervollstaendigung | A_Accepted vs. W_Complete application/resume | 18150 | +2.27 [+0.80; +3.75] | 0 / 1 | 100.0 | kein gesicherter Hinweis |
| angebot_nachlegen | O_Create Offer vs. O_Sent (mail and online) | 24036 | -4.55 [-5.85; -3.26] | 1 / 1 | 100.0 | Hinweis nach Holm-Korrektur |
| neues_angebot_beim_nachfassen | O_Create Offer vs. W_Call after offers/resume | 25297 | +6.82 [+5.80; +7.84] | 1 / 1 | 100.0 | Hinweis nach Holm-Korrektur |
| Mitarbeiter: W_Call incomplete files | User_87 vs. zufälliger verfügbarer Mitarbeiter | 14395 | +1.85 [-2.82; +6.53] | 0 / 72 | 100.0 | kein gesicherter Hinweis |
| Mitarbeiter: W_Complete application | User_75 vs. zufälliger verfügbarer Mitarbeiter | 25303 | +2.00 [-1.62; +5.61] | 0 / 65 | 100.0 | kein gesicherter Hinweis |
| Mitarbeiter: W_Handle leads | User_2 vs. zufälliger verfügbarer Mitarbeiter | 9652 | +0.36 [-4.32; +5.03] | 0 / 31 | 100.0 | kein gesicherter Hinweis |
| Mitarbeiter: W_Validate application | User_87 vs. zufälliger verfügbarer Mitarbeiter | 21709 | -0.59 [-4.46; +3.27] | 0 / 63 | 100.0 | kein gesicherter Hinweis |
| Mitarbeiter: W_Call after offers (resume) | User_2 vs. zufälliger verfügbarer Mitarbeiter | 23575 | -2.71 [-5.53; +0.10] | 0 / 53 | 100.0 | kein gesicherter Hinweis |
| Mitarbeiter: W_Call incomplete files (resume) | User_87 vs. zufälliger verfügbarer Mitarbeiter | 13563 | -2.19 [-5.81; +1.43] | 0 / 73 | 100.0 | kein gesicherter Hinweis |
| Mitarbeiter: W_Complete application (resume) | User_2 vs. zufälliger verfügbarer Mitarbeiter | 9647 | -0.42 [-5.99; +5.14] | 0 / 61 | 100.0 | kein gesicherter Hinweis |
| Mitarbeiter: W_Validate application (resume) | User_87 vs. zufälliger verfügbarer Mitarbeiter | 15756 | -0.61 [-6.77; +5.55] | 0 / 65 | 100.0 | kein gesicherter Hinweis |
| annehmen_bei_konzept | A_Accepted vs. W_Complete application/start | 15844 | +2.93 [+1.90; +3.96] | 0 / 1 | 100.0 | kein gesicherter Hinweis |
| angebotshoehe | mehr_als_beantragt vs. weniger_als_beantragt | 27447 | +0.38 [-0.74; +1.50] | 0 / 2 | 100.0 | kein gesicherter Hinweis |
| laufzeit | ueber_120_monate vs. bis_60_monate | 29990 | +1.71 [-0.30; +3.72] | 0 / 2 | 100.0 | kein gesicherter Hinweis |
| erste_auszahlung | voll vs. keine | 29990 | +0.67 [-0.46; +1.81] | 0 / 2 | 100.0 | kein gesicherter Hinweis |
| reaktionszeit_antrag | langsam vs. schnell (zugewiesene Frist, ITT) | 30000 | -0.68 [-2.22; +0.86] | 0 / 1 | 100.0 | kein gesicherter Hinweis |
| Mitarbeiter: W_Assess potential fraud | User_100 vs. zufälliger verfügbarer Mitarbeiter | 200 | -5.03 [-20.67; +10.62] | 0 / 5 | 100.0 | kein gesicherter Hinweis |
| Mitarbeiter: W_Handle leads (resume) | User_2 vs. zufälliger verfügbarer Mitarbeiter | 2188 | +1.09 [-4.05; +6.23] | 0 / 31 | 100.0 | kein gesicherter Hinweis |

[Datensatzanbieter: Selected und Accepted](https://promforum.win.tue.nl/discussion/749/meaning-of-columns-selected-and-accepted) · [Datensatzanbieter: Prozessstatus](https://promforum.win.tue.nl/discussion/comment/2126/)
