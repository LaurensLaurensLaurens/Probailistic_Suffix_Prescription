# BPIC17: AIPW an allen 20 Interventionspunkten — 30.000 RCT-Fälle

Outcome: negative Durchlaufzeit in Tagen; positive Effekte bedeuten kürzere Durchlaufzeit. Alle Punkte und alle Optionen werden geprüft. Bei mehreren Optionen zeigt die Übersicht einen vorab nach Katalogreihenfolge festgelegten Vergleich; bei Mitarbeitern den häufigsten Mitarbeiter gegenüber einer zufälligen Wahl aus seinem verfügbaren Pool. Diese Auswahl verwendet keine Outcomes. Sämtliche weiteren Vergleiche mit 95-%-Intervallen stehen in aipw_alle_optionen.csv und im HTML-Bericht. Intervalle sind punktweise, ohne Korrektur für multiples Testen. Seltene und nicht identifizierbare Vergleiche bleiben ausdrücklich gekennzeichnet.

RCT: bekannte Zuteilungswahrscheinlichkeiten ohne Clipping; fünf Cross-Fitting-Folds nach Simulatorlauf. 95-%-KI mit nach Simulatorlauf geclusterten Standardfehlern und t-Verteilung. Die Fälle teilen innerhalb eines Laufs Mitarbeiter; die zehn Läufe sind unabhängig. Reaktionszeit: Effekt der zugewiesenen Frist (Intention to treat), einschließlich der Fälle ohne späteren Arbeitsbeginn. Weitere Entscheidungen und spätere Besuche eines Punkts folgen weiterhin der randomisierten Studienregel.

| Interventionspunkt | Vergleich | Fälle / Optionen | AIPW [95-%-KI], Tage | Überlappung % | Status |
| --- | --- | --- | --- | --- | --- |
| channel | O_Sent (online only) vs. O_Sent (mail and online) | 29219 / 2 | 1.365 [1.058; 1.671] | 100.0 | geschätzt |
| handle_leads | W_Handle leads/start vs. W_Handle leads/withdraw | 19142 / 2 | -2.856 [-3.369; -2.344] | 100.0 | geschätzt |
| annehmen_nach_vervollstaendigung | A_Accepted vs. W_Complete application/resume | 18150 / 2 | 2.373 [1.701; 3.044] | 100.0 | geschätzt |
| angebot_nachlegen | O_Create Offer vs. O_Sent (mail and online) | 24036 / 2 | -1.231 [-1.797; -0.665] | 100.0 | geschätzt |
| neues_angebot_beim_nachfassen | O_Create Offer vs. W_Call after offers/resume | 25297 / 2 | 1.342 [0.718; 1.965] | 100.0 | geschätzt |
| Mitarbeiter: W_Call incomplete files | User_87 vs. zufälliger verfügbarer Mitarbeiter | 14396 / 87 | 0.565 [-0.971; 2.101] | 100.0 | geschätzt |
| Mitarbeiter: W_Complete application | User_75 vs. zufälliger verfügbarer Mitarbeiter | 25304 / 66 | 1.333 [-0.756; 3.422] | 100.0 | geschätzt |
| Mitarbeiter: W_Handle leads | User_2 vs. zufälliger verfügbarer Mitarbeiter | 9652 / 31 | 0.554 [-0.649; 1.757] | 100.0 | geschätzt |
| Mitarbeiter: W_Validate application | User_87 vs. zufälliger verfügbarer Mitarbeiter | 21709 / 77 | 0.703 [-0.301; 1.708] | 100.0 | geschätzt |
| Mitarbeiter: W_Call after offers (resume) | User_2 vs. zufälliger verfügbarer Mitarbeiter | 23575 / 57 | -0.507 [-1.321; 0.307] | 100.0 | geschätzt |
| Mitarbeiter: W_Call incomplete files (resume) | User_87 vs. zufälliger verfügbarer Mitarbeiter | 13564 / 87 | 0.207 [-0.318; 0.733] | 100.0 | geschätzt |
| Mitarbeiter: W_Complete application (resume) | User_2 vs. zufälliger verfügbarer Mitarbeiter | 9647 / 65 | -2.047 [-4.498; 0.403] | 100.0 | geschätzt |
| Mitarbeiter: W_Validate application (resume) | User_87 vs. zufälliger verfügbarer Mitarbeiter | 15756 / 73 | 0.321 [-0.490; 1.131] | 100.0 | geschätzt |
| annehmen_bei_konzept | A_Accepted vs. W_Complete application/start | 15844 / 2 | 2.362 [1.817; 2.908] | 100.0 | geschätzt |
| angebotshoehe | mehr_als_beantragt vs. weniger_als_beantragt | 27447 / 3 | -0.252 [-0.644; 0.139] | 100.0 | geschätzt |
| laufzeit | ueber_120_monate vs. bis_60_monate | 29990 / 3 | 0.402 [0.052; 0.751] | 100.0 | geschätzt |
| erste_auszahlung | voll vs. keine | 29990 / 3 | 0.637 [0.241; 1.034] | 100.0 | geschätzt |
| reaktionszeit_antrag | langsam vs. schnell (zugewiesene Frist, ITT) | 30000 / 2 | -1.580 [-1.964; -1.195] | 100.0 | geschätzt |
| Mitarbeiter: W_Assess potential fraud | User_100 vs. zufälliger verfügbarer Mitarbeiter | 200 / 5 | 0.927 [-4.368; 6.221] | 100.0 | geschätzt |
| Mitarbeiter: W_Handle leads (resume) | User_2 vs. zufälliger verfügbarer Mitarbeiter | 2188 / 31 | 0.370 [-2.050; 2.791] | 100.0 | geschätzt |
