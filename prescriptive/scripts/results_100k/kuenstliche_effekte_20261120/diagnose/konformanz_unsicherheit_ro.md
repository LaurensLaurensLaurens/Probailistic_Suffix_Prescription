# Unsicherheit beim Conformance-Ansatz, psp_model_durchlaufzeit_option_learner_k3x4_schlank_crn.pkl, 100 Ziehungen je Option (2026-11-23)

Δ = erwartete Abweichungen andere − Regel-Option je Fall; nicht unterscheidbar: |Δ| < 1,96 · SE.

```
                           Punkt                                                                                  Gruppe  Zustände  Δ Mittel  SE Median  ρ Median  nicht unterscheidbar  sicher andere besser  sicher Regel besser  Wahrheit Bank sonst  Wahrheit Regel sonst
               angebot_nachlegen                                              dauer_ab_tage=0.84, nur_erster_besuch=True        90     0.211      0.171     0.381                 0.678                 0.000                0.322                0.392                 0.373
               angebot_nachlegen dauer_ab_tage=0.84, nur_erster_besuch=True | dauer_ab_tage=1.71, nur_erster_besuch=True       133    -0.234      0.166     0.371                 0.594                 0.406                0.000               -0.170                -0.235
               angebot_nachlegen                                           dauer_unter_tage=0.84, nur_erster_besuch=True       125     0.507      0.159     0.401                 0.216                 0.000                0.784                1.336                 0.835
            annehmen_bei_konzept                                                                         betrag_ab=11000       101    -0.004      0.160     0.354                 0.931                 0.020                0.050               -0.182                -0.173
            annehmen_bei_konzept                                                                      betrag_unter=11000        80     0.024      0.166     0.298                 0.962                 0.000                0.038                0.324                 0.381
annehmen_nach_vervollstaendigung                                                                                   keine        93    -0.026      0.146     0.483                 0.946                 0.032                0.022                0.078                -0.172
annehmen_nach_vervollstaendigung                                               ziel_in=['Car', 'Existing loan takeover']       127     0.104      0.144     0.487                 0.850                 0.016                0.134                0.033                -0.083
                    handle_leads                                                                      betrag_unter=11000       100     0.167      0.191     0.356                 0.870                 0.000                0.130               -0.246                 0.152
                    handle_leads                                                                                   keine       118     0.232      0.185     0.399                 0.737                 0.000                0.263                0.136                 0.151
```

Gesamt: {'Zustände': 967, 'nicht unterscheidbar': 0.7321613236814891, 'sicher andere besser': 0.06308169596690796, 'sicher Regel besser': 0.2047569803516029, 'SE Median': 0.16581296509603644, 'ρ Median': 0.397233830654987}
