# BPIC17: Erfolgreiche Angebotsannahme (O_Accepted) — echtes Log

Einheit: Prozentpunkte, keine relative prozentuale Änderung. Positive Effekte erhöhen die Wahrscheinlichkeit, dass mindestens ein Angebot im weiteren Fallverlauf das Zielereignis erreicht. Je Fall und Punkt zählt die erste Entscheidung; Fälle mit bereits erreichtem Ziel werden anhand des damaligen Prefix ausgeschlossen. Bei mehreren Optionen zeigt die Übersicht letzte vs. erste Katalogoption; bei Mitarbeitern den häufigsten Mitarbeiter vs. gleichverteilte Wahl aus seinem verfügbaren Pool. Diese Auswahl verwendet keine Outcomes. Die Spalte Hinweise zählt signifikante Kontraste nach Holm-Korrektur über sämtliche schätzbaren Optionen dieser Datenquelle und Zielgröße. Sie bezieht sich auch auf Optionen, die nicht in der Übersicht gezeigt sind. Kein gesicherter Hinweis bedeutet nicht nachgewiesene Wirkungslosigkeit.

AIPW mit logistischen XGBoost-Outcome-Modellen und fünf Cross-Fitting-Folds. Propensity mit XGBoost geschätzt und auf den verfügbaren Pool normiert; Clipping [0,01; 0,99]. Cross-Fitting nach Fall; Mitarbeiterpool frei, mit im Log beobachteten Mitarbeitern. Kausale Interpretation setzt ausreichende Überlappung und keine unbeobachteten Confounder voraus. 95-%-Intervalle sind punktweise. Die Holm-Korrektur der p-Werte prüft alle schätzbaren Kontraste dieser Datenquelle/Zielgröße gemeinsam bei 5 %. Sehr seltene Mitarbeiteroptionen sind in den Details markiert. Das Ziel wird bis zum Ende des vorhandenen Verlaufs beobachtet; spätere Annahmen außerhalb des Logs sind unbekannt.

Hauptziel: O_Accepted im weiteren Fallverlauf. A_Accepted ist ein vorheriger Antragsstatus; die Attributspalte Accepted beschreibt die Zulässigkeit eines Angebots. Die ergänzende Auswertung mit O_Returned betrachtet die Rücksendung des unterschriebenen Angebots. Bei mehreren Angeboten wird mindestens ein späterer Erfolg je Antrag betrachtet, nicht zwingend die Annahme genau des ersten Angebots.

| Interventionspunkt | Vergleich | Fälle | AIPW [95-%-KI], Prozentpunkte | Hinweise / Kontraste (Holm) | Überlappung % | Einfluss |
| --- | --- | --- | --- | --- | --- | --- |
| channel | O_Sent (online only) vs. O_Sent (mail and online) | 30723 | -8.47 [-12.52; -4.41] | 1 / 1 | 6.2 | Hinweis nach Holm-Korrektur; geringe Überlappung |
| handle_leads | W_Handle leads/start vs. W_Handle leads/withdraw | 20423 | -11.83 [-13.98; -9.69] | 1 / 1 | 97.3 | Hinweis nach Holm-Korrektur |
| annehmen_nach_vervollstaendigung | A_Accepted vs. W_Complete application/resume | 16423 | +5.35 [+3.32; +7.37] | 1 / 1 | 91.8 | Hinweis nach Holm-Korrektur |
| angebot_nachlegen | O_Create Offer vs. O_Sent (mail and online) | 30592 | +0.07 [-2.45; +2.59] | 0 / 1 | 80.5 | kein gesicherter Hinweis |
| neues_angebot_beim_nachfassen | O_Create Offer vs. W_Call after offers/resume | 25344 | +12.57 [+8.85; +16.29] | 1 / 1 | 42.7 | Hinweis nach Holm-Korrektur; geringe Überlappung |
| Mitarbeiter: W_Call incomplete files | User_100 vs. zufälliger verfügbarer Mitarbeiter | 15003 | +8.41 [+7.50; +9.32] | 7 / 67 | 1.3 | Hinweis nach Holm-Korrektur; geringe Überlappung |
| Mitarbeiter: W_Complete application | User_49 vs. zufälliger verfügbarer Mitarbeiter | 29646 | +0.63 [-2.72; +3.98] | 9 / 116 | 5.4 | Hinweis nach Holm-Korrektur; geringe Überlappung |
| Mitarbeiter: W_Handle leads | User_14 vs. zufälliger verfügbarer Mitarbeiter | 3643 | -7.02 [-26.83; +12.78] | 0 / 82 | 51.5 | kein gesicherter Hinweis |
| Mitarbeiter: W_Validate application | User_123 vs. zufälliger verfügbarer Mitarbeiter | 21870 | +0.78 [-1.37; +2.92] | 4 / 66 | 0.2 | Hinweis nach Holm-Korrektur; geringe Überlappung |
| Mitarbeiter: W_Call after offers (resume) | User_5 vs. zufälliger verfügbarer Mitarbeiter | 25113 | +0.72 [-1.71; +3.16] | 0 / 88 | 1.4 | kein gesicherter Hinweis; geringe Überlappung |
| Mitarbeiter: W_Call incomplete files (resume) | User_2 vs. zufälliger verfügbarer Mitarbeiter | 14517 | -2.39 [-4.67; -0.12] | 0 / 92 | 6.0 | kein gesicherter Hinweis; geringe Überlappung |
| Mitarbeiter: W_Complete application (resume) | User_49 vs. zufälliger verfügbarer Mitarbeiter | 11842 | -2.43 [-10.29; +5.43] | 0 / 99 | 21.4 | kein gesicherter Hinweis; geringe Überlappung |
| Mitarbeiter: W_Validate application (resume) | User_87 vs. zufälliger verfügbarer Mitarbeiter | 11535 | -2.93 [-6.98; +1.13] | 1 / 52 | 5.0 | Hinweis nach Holm-Korrektur; geringe Überlappung |
| annehmen_bei_konzept | A_Accepted vs. W_Complete application/start | 24296 | +10.74 [+8.57; +12.91] | 1 / 1 | 56.6 | Hinweis nach Holm-Korrektur |
| angebotshoehe | mehr_als_beantragt vs. weniger_als_beantragt | 28429 | +17.59 [+13.76; +21.42] | 2 / 2 | 61.4 | Hinweis nach Holm-Korrektur |
| laufzeit | ueber_120_monate vs. bis_60_monate | 31509 | +1.48 [-0.77; +3.74] | 1 / 2 | 96.7 | Hinweis nach Holm-Korrektur |
| erste_auszahlung | voll vs. keine | 31509 | +3.85 [+2.17; +5.54] | 2 / 2 | 85.2 | Hinweis nach Holm-Korrektur |
| reaktionszeit_antrag | langsam vs. schnell | 29646 | -6.12 [-7.36; -4.89] | 1 / 1 | 63.3 | Hinweis nach Holm-Korrektur |
| Mitarbeiter: W_Assess potential fraud | User_68 vs. zufälliger verfügbarer Mitarbeiter | 301 | -28.28 [-58.89; +2.34] | 0 / 27 | 95.6 | kein gesicherter Hinweis |
| Mitarbeiter: W_Handle leads (resume) | User_46 vs. zufälliger verfügbarer Mitarbeiter | 780 | -17.57 [-50.72; +15.58] | 0 / 69 | 86.7 | kein gesicherter Hinweis |

[Datensatzanbieter: Selected und Accepted](https://promforum.win.tue.nl/discussion/749/meaning-of-columns-selected-and-accepted) · [Datensatzanbieter: Prozessstatus](https://promforum.win.tue.nl/discussion/comment/2126/)
