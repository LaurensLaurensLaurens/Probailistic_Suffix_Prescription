# Übereinstimmung PSP-Wahl mit Forest-Wahl am selben Online-Zustand (erste Besuche, bewertete Fälle)

Forest: cate_model_cf_konformanz_simv3.pkl. psp_andere / forest_andere = Anteil „andere Option“ (nicht Regel); gleich = gleiche Wahl.

## Gesamt
```
                     n  gleich
arm                           
konformanz_ol300  5940    0.74
```

## Je Punkt
```
                                                      n  psp_andere  forest_andere  gleich
arm              punkt                                                                    
konformanz_ol300 angebot_nachlegen                 2302       0.069          0.076   0.962
                 annehmen_bei_konzept              1195       0.423          0.233   0.551
                 annehmen_nach_vervollstaendigung   993       0.400          0.156   0.583
                 handle_leads                      1450       0.192          0.472   0.651
```

## Je Teilgruppe
```
                                                                                                                                              n  psp_andere  forest_andere  gleich
arm              punkt                            gruppe                                                                                                                          
konformanz_ol300 angebot_nachlegen                dauer_ab_tage=0.84, nur_erster_besuch=True                                                504       0.067          0.010   0.927
                                                  dauer_ab_tage=0.84, nur_erster_besuch=True | dauer_ab_tage=1.71, nur_erster_besuch=True   171       0.713          0.994   0.719
                                                  dauer_unter_tage=0.84, nur_erster_besuch=True                                            1627       0.002          0.000   0.998
                 annehmen_bei_konzept             betrag_ab=11000                                                                           731       0.436          0.379   0.518
                                                  betrag_unter=11000                                                                        464       0.403          0.004   0.601
                 annehmen_nach_vervollstaendigung keine                                                                                     542       0.406          0.133   0.572
                                                  ziel_in=['Car', 'Existing loan takeover']                                                 451       0.392          0.184   0.596
                 handle_leads                     betrag_unter=11000                                                                        672       0.332          1.000   0.332
                                                  keine                                                                                     778       0.071          0.015   0.927
```

## Nach Abstand der PSP-Schätzungen (|Δ erwartete Abweichungen|)
```
                                    n  gleich
arm              abstand_klasse              
konformanz_ol300 <0,02            425   0.492
                 0,02–0,05        715   0.564
                 0,05–0,1        1019   0.612
                 0,1–0,2         1396   0.674
                 ≥0,2            2385   0.930
```
