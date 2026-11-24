# Wahrheit der Conformance-Effekte in Aufbau F, Regel als Basis (2026-11-23)

basis = Regel an allen Punkten (12 Seeds; Abweichungen je Fall 0.586, konform 72.0 %; Kontrolle Seed 0: basis 0.550, Online-Arm regel 0.550).

## Stufe 1: andere Option an einem Punkt, Regel sonst – je Teilgruppe (Fälle, die den Punkt unter der Regel erreichen)

```
                           Punkt                                                                                  Gruppe  Fälle  andere−Regel Abw. (Regel sonst)  SE über Seeds  konform (pp)  DLZ (d)  Bank sonst (wahrheit_f)
            annehmen_bei_konzept                                                                         betrag_ab=11000    793                           -0.173          0.069        13.241   -1.089                   -0.182
            annehmen_bei_konzept                                                                      betrag_unter=11000    685                            0.381          0.070       -32.117    4.081                    0.324
annehmen_nach_vervollstaendigung                                                                                   keine    215                           -0.172          0.126        -0.930    3.912                    0.078
annehmen_nach_vervollstaendigung                                               ziel_in=['Car', 'Existing loan takeover']    156                           -0.083          0.069         4.487   -5.483                    0.033
               angebot_nachlegen                                              dauer_ab_tage=0.84, nur_erster_besuch=True    276                            0.373          0.076       -14.855   -1.114                    0.392
               angebot_nachlegen dauer_ab_tage=0.84, nur_erster_besuch=True | dauer_ab_tage=1.71, nur_erster_besuch=True     34                           -0.235          0.224         5.882   -8.585                   -0.170
               angebot_nachlegen                                           dauer_unter_tage=0.84, nur_erster_besuch=True   2027                            0.835          0.040       -49.729    1.194                    1.336
                    handle_leads                                                                      betrag_unter=11000    679                            0.152          0.077        -6.922    2.030                   -0.246
                    handle_leads                                                                                   keine    780                            0.151          0.072         0.513    0.100                    0.136
```

## Über alle bewerteten Fälle (Lauf minus basis; kombi = alle Teilgruppen mit Effekt < 0 umgeschaltet)

```
                            Lauf  Seeds  Abw. je bewertetem Fall    SE  konform (pp)  SE konform (pp)  DLZ (d)  SE DLZ
            annehmen_bei_konzept     12                    0.058 0.035        -4.625            1.284    0.795   0.336
annehmen_nach_vervollstaendigung     12                    0.017 0.030        -0.458            0.729    0.156   0.249
               angebot_nachlegen     12                    0.746 0.038       -43.708            1.218    0.751   0.484
                    handle_leads     12                    0.108 0.036        -1.125            1.281    0.476   0.542
                           kombi     12                   -0.059 0.028         4.250            0.914   -0.763   0.384
```

Summe der negativen Einzeleffekte je bewertetem Fall: -0.081; kombi direkt gemessen: -0.059 ± 0.028. Umgeschaltet: angebot_nachlegen: ['dauer_ab_tage=0.84, nur_erster_besuch=True | dauer_ab_tage=1.71, nur_erster_besuch=True'], annehmen_bei_konzept: ['betrag_ab=11000'], annehmen_nach_vervollstaendigung: ['keine', "ziel_in=['Car', 'Existing loan takeover']"].
