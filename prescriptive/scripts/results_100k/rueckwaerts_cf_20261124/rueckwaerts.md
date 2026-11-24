# Causal Forest mit Rückwärtsinduktion (Konformität, Effekte v3), 2026-11-24 11:52

Je Stufe 8 × 6000 Fälle; Stufenpunkt zufällig, spätere Punkte ihr Forest, frühere Bank. Abweichungen gegen BPIC17_ToBe_angepasst; Aktion = zweite Option des Punkts (Referenz = erste). Negativ = Aktion hat weniger Abweichungen.

```
                           Punkt                                                                                  Gruppe  Proben  Anteil Aktion  Abw. Aktion − Referenz (zufällig)    SE  Forest −τ (Abw. Aktion − Referenz)  Forest wählt Aktion
               angebot_nachlegen                                              dauer_ab_tage=0.84, nur_erster_besuch=True   10765          0.485                              0.174 0.022                               0.190                0.000
               angebot_nachlegen dauer_ab_tage=0.84, nur_erster_besuch=True | dauer_ab_tage=1.71, nur_erster_besuch=True   15879          0.485                             -0.302 0.020                              -0.251                0.997
               angebot_nachlegen                                           dauer_unter_tage=0.84, nur_erster_besuch=True   19953          0.495                              1.162 0.020                               1.164                0.000
annehmen_nach_vervollstaendigung                                                                                   keine   20503          0.498                              0.007 0.017                               0.007                0.764
annehmen_nach_vervollstaendigung                                               ziel_in=['Car', 'Existing loan takeover']   22857          0.505                              0.015 0.016                               0.003                0.747
            annehmen_bei_konzept                                                                         betrag_ab=11000   13072          0.502                              0.198 0.022                               0.164                0.169
            annehmen_bei_konzept                                                                      betrag_unter=11000   12452          0.505                             -0.399 0.021                              -0.392                0.986
                    handle_leads                                                                      betrag_unter=11000   15107          0.513                              0.079 0.019                               0.054                0.002
                    handle_leads                                                                                   keine   15726          0.497                              0.034 0.020                               0.042                0.042
```

Forest-Zusammenfassung:
```
{
 "angebot_nachlegen": {
  "Proben": 37279,
  "F\u00e4lle": 37279,
  "Anteil Aktion": 0.48727166652679443,
  "Merkmale": 77,
  "B\u00e4ume": "124/82 (Hilfsmodelle), 500 (Wald)",
  "Wald": {
   "max_depth": 5,
   "min_weight_fraction_leaf": 0.01,
   "min_var_fraction_leaf": 0.001
  },
  "Effekt Mittel (Validierung)": -0.44426755742670504,
  "Effekt Streuung": 0.6358426613113379,
  "Anteil Effekt > 0": 0.35019136139967194
 },
 "annehmen_nach_vervollstaendigung": {
  "Proben": 34759,
  "F\u00e4lle": 22371,
  "Anteil Aktion": 0.5002157688140869,
  "Merkmale": 55,
  "B\u00e4ume": "159/8 (Hilfsmodelle), 500 (Wald)",
  "Wald": {
   "max_depth": null,
   "min_weight_fraction_leaf": 0.01,
   "min_var_fraction_leaf": 0.001
  },
  "Effekt Mittel (Validierung)": -0.002421272814215185,
  "Effekt Streuung": 0.14318352573174095,
  "Anteil Effekt > 0": 0.762043795620438
 },
 "annehmen_bei_konzept": {
  "Proben": 20352,
  "F\u00e4lle": 20352,
  "Anteil Aktion": 0.5000491142272949,
  "Merkmale": 29,
  "B\u00e4ume": "26/156 (Hilfsmodelle), 500 (Wald)",
  "Wald": {
   "max_depth": null,
   "min_weight_fraction_leaf": 0.0001,
   "min_var_fraction_leaf": 0.001
  },
  "Effekt Mittel (Validierung)": 0.11328398293804816,
  "Effekt Streuung": 0.32823364914530234,
  "Anteil Effekt > 0": 0.5773869346733669
 },
 "handle_leads": {
  "Proben": 24604,
  "F\u00e4lle": 24602,
  "Anteil Aktion": 0.5062591433525085,
  "Merkmale": 33,
  "B\u00e4ume": "46/10 (Hilfsmodelle), 500 (Wald)",
  "Wald": {
   "max_depth": 5,
   "min_weight_fraction_leaf": 0.01,
   "min_var_fraction_leaf": 0.001
  },
  "Effekt Mittel (Validierung)": -0.047533583681335013,
  "Effekt Streuung": 0.02333382217172109,
  "Anteil Effekt > 0": 0.026119402985074626
 }
}
```
