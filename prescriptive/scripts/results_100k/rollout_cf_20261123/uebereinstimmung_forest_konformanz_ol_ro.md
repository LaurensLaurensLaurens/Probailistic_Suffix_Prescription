# Übereinstimmung PSP-Wahl mit Forest-Wahl am selben Online-Zustand (erste Besuche, bewertete Fälle)

Forest: cate_model_cf_konformanz_simv3.pkl. psp_andere / forest_andere = Anteil „andere Option“ (nicht Regel); gleich = gleiche Wahl.

## Gesamt
```
                     n  gleich
arm                           
konformanz_ol_ro  6179   0.655
```

## Je Punkt
```
                                                      n  psp_andere  forest_andere  gleich
arm              punkt                                                                    
konformanz_ol_ro angebot_nachlegen                 2304       0.115          0.140   0.931
                 annehmen_bei_konzept              1442       0.595          0.199   0.445
                 annehmen_nach_vervollstaendigung   980       0.444          0.160   0.508
                 handle_leads                      1453       0.017          0.474   0.524
```

## Je Teilgruppe
```
                                                                                                                                              n  psp_andere  forest_andere  gleich
arm              punkt                            gruppe                                                                                                                          
konformanz_ol_ro angebot_nachlegen                dauer_ab_tage=0.84, nur_erster_besuch=True                                                623       0.083          0.026   0.900
                                                  dauer_ab_tage=0.84, nur_erster_besuch=True | dauer_ab_tage=1.71, nur_erster_besuch=True   307       0.687          0.997   0.691
                                                  dauer_unter_tage=0.84, nur_erster_besuch=True                                            1374       0.001          0.000   0.999
                 annehmen_bei_konzept             betrag_ab=11000                                                                           774       0.598          0.368   0.473
                                                  betrag_unter=11000                                                                        668       0.591          0.003   0.412
                 annehmen_nach_vervollstaendigung keine                                                                                     527       0.571          0.133   0.440
                                                  ziel_in=['Car', 'Existing loan takeover']                                                 453       0.296          0.192   0.587
                 handle_leads                     betrag_unter=11000                                                                        677       0.016          1.000   0.016
                                                  keine                                                                                     776       0.018          0.015   0.966
```

## Nach Abstand der PSP-Schätzungen (|Δ erwartete Abweichungen|)
```
                                    n  gleich
arm              abstand_klasse              
konformanz_ol_ro <0,02            350   0.503
                 0,02–0,05        614   0.513
                 0,05–0,1         970   0.513
                 0,1–0,2         1273   0.582
                 ≥0,2            2972   0.779
```
