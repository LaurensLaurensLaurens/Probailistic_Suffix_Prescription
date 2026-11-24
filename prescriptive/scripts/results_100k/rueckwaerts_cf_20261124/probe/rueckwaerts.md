# Causal Forest mit Rückwärtsinduktion (Konformität, Effekte v3), 2026-11-24 11:33

Je Stufe 4 × 600 Fälle; Stufenpunkt zufällig, spätere Punkte ihr Forest, frühere Bank. Abweichungen gegen BPIC17_ToBe_angepasst; Aktion = zweite Option des Punkts (Referenz = erste). Negativ = Aktion hat weniger Abweichungen.

```
                           Punkt                                                                                  Gruppe  Proben  Anteil Aktion  Abw. Aktion − Referenz (zufällig)    SE  Forest −τ (Abw. Aktion − Referenz)  Forest wählt Aktion
               angebot_nachlegen                                              dauer_ab_tage=0.84, nur_erster_besuch=True     613          0.481                              0.073 0.098                               0.074                0.330
               angebot_nachlegen dauer_ab_tage=0.84, nur_erster_besuch=True | dauer_ab_tage=1.71, nur_erster_besuch=True     875          0.523                             -0.290 0.091                              -0.221                0.949
               angebot_nachlegen                                           dauer_unter_tage=0.84, nur_erster_besuch=True     846          0.492                              1.024 0.112                               1.038                0.000
annehmen_nach_vervollstaendigung                                                                                   keine    1021          0.511                              0.007 0.069                               0.071                0.091
annehmen_nach_vervollstaendigung                                               ziel_in=['Car', 'Existing loan takeover']    1114          0.507                              0.032 0.088                               0.079                0.064
            annehmen_bei_konzept                                                                         betrag_ab=11000     635          0.494                              0.275 0.116                               0.166                0.236
            annehmen_bei_konzept                                                                      betrag_unter=11000     559          0.469                             -0.532 0.107                              -0.225                0.930
                    handle_leads                                                                      betrag_unter=11000     680          0.501                             -0.024 0.088                               0.069                0.207
                    handle_leads                                                                                   keine     800          0.478                              0.274 0.095                               0.162                0.056
```

Forest-Zusammenfassung:
```
{
 "angebot_nachlegen": {
  "Proben": 1865,
  "F\u00e4lle": 1865,
  "Anteil Aktion": 0.502949059009552,
  "Merkmale": 68,
  "B\u00e4ume": "4/5 (Hilfsmodelle), 500 (Wald)",
  "Wald": {
   "max_depth": null,
   "min_weight_fraction_leaf": 0.01,
   "min_var_fraction_leaf": 0.001
  },
  "Effekt Mittel (Validierung)": -0.3063195542743791,
  "Effekt Streuung": 0.5778078987386486,
  "Anteil Effekt > 0": 0.4502923976608187
 },
 "annehmen_nach_vervollstaendigung": {
  "Proben": 1735,
  "F\u00e4lle": 1169,
  "Anteil Aktion": 0.5118155479431152,
  "Merkmale": 46,
  "B\u00e4ume": "15/9 (Hilfsmodelle), 500 (Wald)",
  "Wald": {
   "max_depth": 3,
   "min_weight_fraction_leaf": 0.01,
   "min_var_fraction_leaf": 0.001
  },
  "Effekt Mittel (Validierung)": -0.07160913270591367,
  "Effekt Streuung": 0.05084367167964912,
  "Anteil Effekt > 0": 0.06832298136645963
 },
 "annehmen_bei_konzept": {
  "Proben": 953,
  "F\u00e4lle": 953,
  "Anteil Aktion": 0.48793283104896545,
  "Merkmale": 29,
  "B\u00e4ume": "195/4 (Hilfsmodelle), 500 (Wald)",
  "Wald": {
   "max_depth": 5,
   "min_weight_fraction_leaf": 0.0001,
   "min_var_fraction_leaf": 0.001
  },
  "Effekt Mittel (Validierung)": 0.031128785327646475,
  "Effekt Streuung": 0.2797114633911583,
  "Anteil Effekt > 0": 0.5978260869565217
 },
 "handle_leads": {
  "Proben": 1183,
  "F\u00e4lle": 1182,
  "Anteil Aktion": 0.49535080790519714,
  "Merkmale": 27,
  "B\u00e4ume": "18/18 (Hilfsmodelle), 500 (Wald)",
  "Wald": {
   "max_depth": 5,
   "min_weight_fraction_leaf": 0.0001,
   "min_var_fraction_leaf": 0.001
  },
  "Effekt Mittel (Validierung)": -0.10643334118799894,
  "Effekt Streuung": 0.10259019601724803,
  "Anteil Effekt > 0": 0.15454545454545454
 }
}
```
