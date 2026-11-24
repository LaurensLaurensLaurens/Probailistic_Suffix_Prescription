# BPIC17: AIPW an allen 20 Interventionspunkten — echtes Log

Outcome: negative Durchlaufzeit in Tagen; positive Effekte bedeuten kürzere Durchlaufzeit. Alle Punkte und alle Optionen werden geprüft. Bei mehreren Optionen zeigt die Übersicht einen vorab nach Katalogreihenfolge festgelegten Vergleich; bei Mitarbeitern den häufigsten Mitarbeiter gegenüber einer zufälligen Wahl aus seinem verfügbaren Pool. Diese Auswahl verwendet keine Outcomes. Sämtliche weiteren Vergleiche mit 95-%-Intervallen stehen in aipw_alle_optionen.csv und im HTML-Bericht. Intervalle sind punktweise, ohne Korrektur für multiples Testen. Seltene und nicht identifizierbare Vergleiche bleiben ausdrücklich gekennzeichnet.

Echtes Log: fünf Cross-Fitting-Folds nach Fall, XGBoost für Outcome und Propensity; Propensity-Clipping [0,01; 0,99]. Keine Mindestfallzahl von 200 und keine Zusammenfassung seltener Mitarbeiter als other. Mitarbeiter werden nur in Fällen verglichen, in denen sie zum freien Pool gehören und mindestens zwei Personen wählbar sind. Der beobachtete Mitarbeiterpool enthält nur Personen, die diese Tätigkeit im Log selbst ausgeführt haben; für nie beobachtete Personen gibt es keine identifizierbare Wirkung. Sehr kleine Fallzahlen und fehlende Überlappung begrenzen die Aussagekraft; unbeobachtete Konfundierung bleibt möglich.

| Interventionspunkt | Vergleich | Fälle / Optionen | AIPW [95-%-KI], Tage | Überlappung % | Status |
| --- | --- | --- | --- | --- | --- |
| channel | O_Sent (online only) vs. O_Sent (mail and online) | 30723 / 2 | 1.631 [0.237; 3.025] | 6.2 | geschätzt; geringe Überlappung |
| handle_leads | W_Handle leads/start vs. W_Handle leads/withdraw | 20423 / 2 | -2.069 [-2.633; -1.505] | 97.3 | geschätzt |
| annehmen_nach_vervollstaendigung | A_Accepted vs. W_Complete application/resume | 16423 / 2 | 2.872 [2.379; 3.366] | 91.8 | geschätzt |
| angebot_nachlegen | O_Create Offer vs. O_Sent (mail and online) | 30592 / 2 | -1.377 [-1.997; -0.757] | 80.5 | geschätzt |
| neues_angebot_beim_nachfassen | O_Create Offer vs. W_Call after offers/resume | 25344 / 2 | 3.009 [2.162; 3.855] | 42.7 | geschätzt; geringe Überlappung |
| Mitarbeiter: W_Call incomplete files | User_100 vs. zufälliger verfügbarer Mitarbeiter | 15003 / 67 | 1.463 [1.049; 1.877] | 1.3 | geschätzt; geringe Überlappung |
| Mitarbeiter: W_Complete application | User_49 vs. zufälliger verfügbarer Mitarbeiter | 29646 / 118 | -0.623 [-1.530; 0.283] | 5.4 | geschätzt; geringe Überlappung |
| Mitarbeiter: W_Handle leads | User_14 vs. zufälliger verfügbarer Mitarbeiter | 3643 / 85 | -0.134 [-5.671; 5.404] | 51.5 | geschätzt |
| Mitarbeiter: W_Validate application | User_123 vs. zufälliger verfügbarer Mitarbeiter | 21870 / 66 | 0.374 [-0.041; 0.788] | 0.2 | geschätzt; geringe Überlappung |
| Mitarbeiter: W_Call after offers (resume) | User_5 vs. zufälliger verfügbarer Mitarbeiter | 25113 / 88 | 0.819 [-0.038; 1.675] | 1.4 | geschätzt; geringe Überlappung |
| Mitarbeiter: W_Call incomplete files (resume) | User_2 vs. zufälliger verfügbarer Mitarbeiter | 14517 / 92 | 0.330 [-0.263; 0.922] | 6.0 | geschätzt; geringe Überlappung |
| Mitarbeiter: W_Complete application (resume) | User_49 vs. zufälliger verfügbarer Mitarbeiter | 11842 / 100 | 0.291 [-1.636; 2.218] | 21.4 | geschätzt; geringe Überlappung |
| Mitarbeiter: W_Validate application (resume) | User_87 vs. zufälliger verfügbarer Mitarbeiter | 11535 / 53 | -0.832 [-1.769; 0.104] | 5.0 | geschätzt; geringe Überlappung |
| annehmen_bei_konzept | A_Accepted vs. W_Complete application/start | 24296 / 2 | 5.425 [4.865; 5.986] | 56.6 | geschätzt |
| angebotshoehe | mehr_als_beantragt vs. weniger_als_beantragt | 28429 / 3 | 1.153 [0.189; 2.117] | 61.4 | geschätzt |
| laufzeit | ueber_120_monate vs. bis_60_monate | 31509 / 3 | 0.173 [-0.399; 0.746] | 96.7 | geschätzt |
| erste_auszahlung | voll vs. keine | 31509 / 3 | 1.239 [0.814; 1.663] | 85.2 | geschätzt |
| reaktionszeit_antrag | langsam vs. schnell | 29646 / 2 | -2.562 [-2.852; -2.271] | 63.3 | geschätzt |
| Mitarbeiter: W_Assess potential fraud | User_68 vs. zufälliger verfügbarer Mitarbeiter | 301 / 36 | 4.361 [-4.819; 13.541] | 95.6 | geschätzt |
| Mitarbeiter: W_Handle leads (resume) | User_46 vs. zufälliger verfügbarer Mitarbeiter | 780 / 69 | -1.932 [-11.944; 8.079] | 86.7 | geschätzt |
