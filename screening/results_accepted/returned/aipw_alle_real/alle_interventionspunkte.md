# BPIC17: Rücksendung des unterschriebenen Angebots (O_Returned) — echtes Log

Einheit: Prozentpunkte, keine relative prozentuale Änderung. Positive Effekte erhöhen die Wahrscheinlichkeit, dass mindestens ein Angebot im weiteren Fallverlauf das Zielereignis erreicht. Je Fall und Punkt zählt die erste Entscheidung; Fälle mit bereits erreichtem Ziel werden anhand des damaligen Prefix ausgeschlossen. Bei mehreren Optionen zeigt die Übersicht letzte vs. erste Katalogoption; bei Mitarbeitern den häufigsten Mitarbeiter vs. gleichverteilte Wahl aus seinem verfügbaren Pool. Diese Auswahl verwendet keine Outcomes. Die Spalte Hinweise zählt signifikante Kontraste nach Holm-Korrektur über sämtliche schätzbaren Optionen dieser Datenquelle und Zielgröße. Sie bezieht sich auch auf Optionen, die nicht in der Übersicht gezeigt sind. Kein gesicherter Hinweis bedeutet nicht nachgewiesene Wirkungslosigkeit.

AIPW mit logistischen XGBoost-Outcome-Modellen und fünf Cross-Fitting-Folds. Propensity mit XGBoost geschätzt und auf den verfügbaren Pool normiert; Clipping [0,01; 0,99]. Cross-Fitting nach Fall; Mitarbeiterpool frei, mit im Log beobachteten Mitarbeitern. Kausale Interpretation setzt ausreichende Überlappung und keine unbeobachteten Confounder voraus. 95-%-Intervalle sind punktweise. Die Holm-Korrektur der p-Werte prüft alle schätzbaren Kontraste dieser Datenquelle/Zielgröße gemeinsam bei 5 %. Sehr seltene Mitarbeiteroptionen sind in den Details markiert. Das Ziel wird bis zum Ende des vorhandenen Verlaufs beobachtet; spätere Annahmen außerhalb des Logs sind unbekannt.

Hauptziel: O_Accepted im weiteren Fallverlauf. A_Accepted ist ein vorheriger Antragsstatus; die Attributspalte Accepted beschreibt die Zulässigkeit eines Angebots. Die ergänzende Auswertung mit O_Returned betrachtet die Rücksendung des unterschriebenen Angebots. Bei mehreren Angeboten wird mindestens ein späterer Erfolg je Antrag betrachtet, nicht zwingend die Annahme genau des ersten Angebots.

| Interventionspunkt | Vergleich | Fälle | AIPW [95-%-KI], Prozentpunkte | Hinweise / Kontraste (Holm) | Überlappung % | Einfluss |
| --- | --- | --- | --- | --- | --- | --- |
| channel | O_Sent (online only) vs. O_Sent (mail and online) | 30691 | -12.39 [-16.31; -8.48] | 1 / 1 | 5.7 | Hinweis nach Holm-Korrektur; geringe Überlappung |
| handle_leads | W_Handle leads/start vs. W_Handle leads/withdraw | 20423 | -2.79 [-4.92; -0.66] | 0 / 1 | 97.3 | kein gesicherter Hinweis |
| annehmen_nach_vervollstaendigung | A_Accepted vs. W_Complete application/resume | 16423 | +4.90 [+2.98; +6.83] | 1 / 1 | 91.8 | Hinweis nach Holm-Korrektur |
| angebot_nachlegen | O_Create Offer vs. O_Sent (mail and online) | 30570 | -2.08 [-4.38; +0.23] | 0 / 1 | 80.5 | kein gesicherter Hinweis |
| neues_angebot_beim_nachfassen | O_Create Offer vs. W_Call after offers/resume | 25344 | +14.08 [+11.16; +17.01] | 1 / 1 | 42.7 | Hinweis nach Holm-Korrektur; geringe Überlappung |
| Mitarbeiter: W_Call incomplete files | User_126 vs. zufälliger verfügbarer Mitarbeiter | 1374 | +2.18 [-0.11; +4.47] | 1 / 55 | 25.1 | Hinweis nach Holm-Korrektur; geringe Überlappung |
| Mitarbeiter: W_Complete application | User_49 vs. zufälliger verfügbarer Mitarbeiter | 29646 | +1.43 [-1.90; +4.77] | 5 / 116 | 5.4 | Hinweis nach Holm-Korrektur; geringe Überlappung |
| Mitarbeiter: W_Handle leads | User_14 vs. zufälliger verfügbarer Mitarbeiter | 3643 | -0.89 [-22.92; +21.14] | 0 / 82 | 51.5 | kein gesicherter Hinweis |
| Mitarbeiter: W_Validate application | User_123 vs. zufälliger verfügbarer Mitarbeiter | 21870 | +0.48 [+0.08; +0.88] | 3 / 66 | 0.2 | Hinweis nach Holm-Korrektur; geringe Überlappung |
| Mitarbeiter: W_Call after offers (resume) | User_5 vs. zufälliger verfügbarer Mitarbeiter | 25112 | +0.48 [-1.94; +2.91] | 1 / 88 | 1.4 | Hinweis nach Holm-Korrektur; geringe Überlappung |
| Mitarbeiter: W_Call incomplete files (resume) | User_2 vs. zufälliger verfügbarer Mitarbeiter | 518 | +6.08 [-18.95; +31.11] | 0 / 65 | 70.9 | kein gesicherter Hinweis |
| Mitarbeiter: W_Complete application (resume) | User_49 vs. zufälliger verfügbarer Mitarbeiter | 11842 | -2.91 [-11.60; +5.79] | 0 / 99 | 21.4 | kein gesicherter Hinweis; geringe Überlappung |
| Mitarbeiter: W_Validate application (resume) | User_30 vs. zufälliger verfügbarer Mitarbeiter | 218 | +1.22 [-17.01; +19.46] | 0 / 20 | 88.7 | kein gesicherter Hinweis |
| annehmen_bei_konzept | A_Accepted vs. W_Complete application/start | 24296 | +9.25 [+7.33; +11.17] | 1 / 1 | 56.6 | Hinweis nach Holm-Korrektur |
| angebotshoehe | mehr_als_beantragt vs. weniger_als_beantragt | 28429 | +11.79 [+8.15; +15.43] | 2 / 2 | 61.4 | Hinweis nach Holm-Korrektur |
| laufzeit | ueber_120_monate vs. bis_60_monate | 31509 | +1.89 [-0.24; +4.02] | 1 / 2 | 96.7 | Hinweis nach Holm-Korrektur |
| erste_auszahlung | voll vs. keine | 31509 | +8.37 [+6.76; +9.99] | 2 / 2 | 85.2 | Hinweis nach Holm-Korrektur |
| reaktionszeit_antrag | langsam vs. schnell | 29646 | -5.30 [-6.45; -4.15] | 1 / 1 | 63.3 | Hinweis nach Holm-Korrektur |
| Mitarbeiter: W_Assess potential fraud | User_138 vs. zufälliger verfügbarer Mitarbeiter | 51 | -132.81 [-355.43; +89.81] | 0 / 8 | 69.2 | kein gesicherter Hinweis |
| Mitarbeiter: W_Handle leads (resume) | User_46 vs. zufälliger verfügbarer Mitarbeiter | 780 | -26.21 [-61.79; +9.38] | 0 / 69 | 86.7 | kein gesicherter Hinweis |

[Datensatzanbieter: Selected und Accepted](https://promforum.win.tue.nl/discussion/749/meaning-of-columns-selected-and-accepted) · [Datensatzanbieter: Prozessstatus](https://promforum.win.tue.nl/discussion/comment/2126/)
