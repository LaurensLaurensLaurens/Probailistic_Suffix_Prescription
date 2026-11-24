# Unsicherheit beim Conformance-Ansatz, psp_model_durchlaufzeit_option_learner_k3x4_schlank_crn.pkl, 100 Ziehungen je Option (2026-11-23)

Δ = erwartete Abweichungen andere − Regel-Option je Fall; nicht unterscheidbar: |Δ| < 1,96 · SE.

```
                           Punkt                                                                                  Gruppe  Zustände  Δ Mittel  SE Median  ρ Median  nicht unterscheidbar  sicher andere besser  sicher Regel besser  Wahrheit Bank sonst  Wahrheit Regel sonst
               angebot_nachlegen                                              dauer_ab_tage=0.84, nur_erster_besuch=True        90     0.193      0.168     0.391                 0.722                 0.000                0.278                0.392                 0.373
               angebot_nachlegen dauer_ab_tage=0.84, nur_erster_besuch=True | dauer_ab_tage=1.71, nur_erster_besuch=True       133    -0.248      0.165     0.373                 0.594                 0.406                0.000               -0.170                -0.235
               angebot_nachlegen                                           dauer_unter_tage=0.84, nur_erster_besuch=True       125     0.477      0.158     0.404                 0.280                 0.000                0.720                1.336                 0.835
            annehmen_bei_konzept                                                                         betrag_ab=11000       101     0.071      0.165     0.335                 0.911                 0.010                0.079               -0.182                -0.173
            annehmen_bei_konzept                                                                      betrag_unter=11000        80     0.094      0.173     0.292                 0.950                 0.000                0.050                0.324                 0.381
annehmen_nach_vervollstaendigung                                                                                   keine        93    -0.012      0.142     0.504                 0.914                 0.054                0.032                0.078                -0.172
annehmen_nach_vervollstaendigung                                               ziel_in=['Car', 'Existing loan takeover']       127     0.004      0.147     0.466                 0.882                 0.071                0.047                0.033                -0.083
                    handle_leads                                                                      betrag_unter=11000       100    -0.064      0.173     0.416                 0.980                 0.020                0.000               -0.246                 0.152
                    handle_leads                                                                                   keine       118     0.109      0.169     0.428                 0.890                 0.000                0.110                0.136                 0.151
```

Gesamt: {'Zustände': 967, 'nicht unterscheidbar': 0.7724922440537746, 'sicher andere besser': 0.0734229576008273, 'sicher Regel besser': 0.15408479834539815, 'SE Median': 0.16225681103507714, 'ρ Median': 0.40503366518743505}
