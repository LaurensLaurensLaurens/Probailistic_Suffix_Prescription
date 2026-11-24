# Wahrheit der Conformance-Effekte in Aufbau F (2026-11-23)

Option nur für die 200 bewerteten Fälle an einem Punkt erzwungen, sonst Bank; 12 Seeds; gepaart je Fall.

## Je Teilgruppe (Fälle, die den Punkt erreichen)

```
                           Punkt                                                                                  Gruppe  Fälle  je Seed  Gruppe gleich in beiden Läufen  andere−Regel Abw. (F)  SE über Seeds  konform (pp)  DLZ (d)  alte Tabelle (alle Fälle erzwungen)
            annehmen_bei_konzept                                                                         betrag_ab=11000    647     53.9                           0.998                 -0.182          0.054         7.573   -1.985                               -0.064
            annehmen_bei_konzept                                                                      betrag_unter=11000    553     46.1                           0.995                  0.324          0.097       -26.944    5.038                                0.529
annehmen_nach_vervollstaendigung                                                                                   keine    643     53.6                           0.932                  0.078          0.073        -7.776    5.403                                0.048
annehmen_nach_vervollstaendigung                                               ziel_in=['Car', 'Existing loan takeover']    751     62.6                           0.967                  0.033          0.064        -2.530   -5.625                                0.085
               angebot_nachlegen                                              dauer_ab_tage=0.84, nur_erster_besuch=True    699     58.2                           0.845                  0.392          0.084       -18.598    0.657                                0.280
               angebot_nachlegen dauer_ab_tage=0.84, nur_erster_besuch=True | dauer_ab_tage=1.71, nur_erster_besuch=True    477     39.8                           0.811                 -0.170          0.086        15.094   -1.034                               -0.259
               angebot_nachlegen                                           dauer_unter_tage=0.84, nur_erster_besuch=True   1162     96.8                           0.904                  1.336          0.076       -49.053    2.684                                1.381
                    handle_leads                                                                      betrag_unter=11000    679     56.6                           1.000                 -0.246          0.040        20.324   -3.318                               -0.434
                    handle_leads                                                                                   keine    780     65.0                           1.000                  0.136          0.056        -3.205    2.169                                0.079
```

## Über alle bewerteten Fälle je Punkt

```
                           Punkt  Seeds  Fälle  erreicht  andere−Regel Abw. je bewertetem Fall    SE  konform (pp)  DLZ (d)
            annehmen_bei_konzept     12   2400      1200                                 0.018 0.033        -3.250    0.432
annehmen_nach_vervollstaendigung     12   2400      1394                                 0.066 0.031        -3.250    0.295
               angebot_nachlegen     12   2400      2338                                 0.734 0.048       -26.458    1.417
                    handle_leads     12   2400      1459                                -0.037 0.025         4.208   -0.175
```

Erwarteter Vorsprung eines Orakels vor der Regel (Abweichungen je bewertetem Fall): nach F-Wahrheit 0.152, nach alter Tabelle 0.192; online gemessen Orakel − Regel 0,009 (online_konformanz.md).
