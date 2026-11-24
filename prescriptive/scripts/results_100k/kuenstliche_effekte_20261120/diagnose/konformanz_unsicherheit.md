# Unsicherheit beim Conformance-Ansatz, k3x4ol, 100 Ziehungen je Option (2026-11-23)

Δ = erwartete Abweichungen andere − Regel-Option je Fall; nicht unterscheidbar: |Δ| < 1,96 · SE.

```
                           Punkt                                                                                  Gruppe  Zustände  Δ Mittel  SE Median  ρ Median  nicht unterscheidbar  sicher andere besser  sicher Regel besser  Wahrheit Bank sonst  Wahrheit Regel sonst
               angebot_nachlegen                                              dauer_ab_tage=0.84, nur_erster_besuch=True        90     0.202      0.194     0.094                 0.767                 0.011                0.222                0.392                 0.373
               angebot_nachlegen dauer_ab_tage=0.84, nur_erster_besuch=True | dauer_ab_tage=1.71, nur_erster_besuch=True       133    -0.250      0.201     0.053                 0.737                 0.256                0.008               -0.170                -0.235
               angebot_nachlegen                                           dauer_unter_tage=0.84, nur_erster_besuch=True       125     0.499      0.196     0.096                 0.352                 0.000                0.648                1.336                 0.835
            annehmen_bei_konzept                                                                         betrag_ab=11000       101     0.102      0.203     0.062                 0.871                 0.030                0.099               -0.182                -0.173
            annehmen_bei_konzept                                                                      betrag_unter=11000        80     0.068      0.204     0.090                 0.925                 0.012                0.062                0.324                 0.381
annehmen_nach_vervollstaendigung                                                                                   keine        93    -0.036      0.190     0.107                 0.914                 0.043                0.043                0.078                -0.172
annehmen_nach_vervollstaendigung                                               ziel_in=['Car', 'Existing loan takeover']       127     0.014      0.188     0.085                 0.913                 0.047                0.039                0.033                -0.083
                    handle_leads                                                                      betrag_unter=11000       100     0.009      0.215     0.105                 0.960                 0.010                0.030               -0.246                 0.152
                    handle_leads                                                                                   keine       118     0.092      0.213     0.072                 0.941                 0.017                0.042                0.136                 0.151
```

Gesamt: {'Zustände': 967, 'nicht unterscheidbar': 0.8076525336091003, 'sicher andere besser': 0.05377456049638056, 'sicher Regel besser': 0.13857290589451912, 'SE Median': 0.2010678562781302, 'ρ Median': 0.08215240407753938}
