# Online-Wahlen je Teilgruppe (bewertete Fälle, erste Besuche; 2026-11-23)

```
                                                                                                                                              n  wahrheit  anteil_andere  anteil_bank  falsch  verlust_je_entscheidung
punkt                            gruppe                                                                                  arm                                                                                          
angebot_nachlegen                dauer_ab_tage=0.84, nur_erster_besuch=True                                              konformanz_ol      491     0.280          0.173        0.008   0.173                    0.048
                                                                                                                         konformanz_ol300   505     0.280          0.067        0.002   0.067                    0.019
                                                                                                                         konformanz_ol_ro   624     0.280          0.083        0.002   0.083                    0.023
                                                                                                                         konformanz_x4      577     0.280          0.087        0.000   0.087                    0.024
                                 dauer_ab_tage=0.84, nur_erster_besuch=True | dauer_ab_tage=1.71, nur_erster_besuch=True konformanz_ol      203    -0.259          0.635        0.025   0.340                    0.088
                                                                                                                         konformanz_ol300   171    -0.259          0.713        0.000   0.287                    0.074
                                                                                                                         konformanz_ol_ro   310    -0.259          0.681        0.010   0.310                    0.080
                                                                                                                         konformanz_x4      297    -0.259          0.337        0.010   0.653                    0.169
                                 dauer_unter_tage=0.84, nur_erster_besuch=True                                           konformanz_ol     1595     1.381          0.031        0.001   0.031                    0.043
                                                                                                                         konformanz_ol300  1627     1.381          0.002        0.000   0.002                    0.003
                                                                                                                         konformanz_ol_ro  1374     1.381          0.001        0.000   0.001                    0.001
                                                                                                                         konformanz_x4     1409     1.381          0.011        0.002   0.011                    0.015
annehmen_bei_konzept             betrag_ab=11000                                                                         konformanz_ol      615    -0.064          0.447        0.018   0.535                    0.034
                                                                                                                         konformanz_ol300   739    -0.064          0.432        0.011   0.558                    0.036
                                                                                                                         konformanz_ol_ro   779    -0.064          0.594        0.006   0.399                    0.026
                                                                                                                         konformanz_x4      640    -0.064          0.662        0.008   0.330                    0.021
                                 betrag_unter=11000                                                                      konformanz_ol      422     0.529          0.445        0.026   0.445                    0.236
                                                                                                                         konformanz_ol300   470     0.529          0.398        0.013   0.398                    0.210
                                                                                                                         konformanz_ol_ro   675     0.529          0.585        0.010   0.585                    0.309
                                                                                                                         konformanz_x4      546     0.529          0.559        0.011   0.559                    0.295
annehmen_nach_vervollstaendigung keine                                                                                   konformanz_ol      562     0.048          0.440        0.027   0.440                    0.021
                                                                                                                         konformanz_ol300   551     0.048          0.399        0.016   0.399                    0.019
                                                                                                                         konformanz_ol_ro   530     0.048          0.568        0.006   0.568                    0.027
                                                                                                                         konformanz_x4      523     0.048          0.669        0.019   0.669                    0.032
                                 ziel_in=['Car', 'Existing loan takeover']                                               konformanz_ol      556     0.085          0.453        0.014   0.453                    0.039
                                                                                                                         konformanz_ol300   461     0.085          0.384        0.022   0.384                    0.033
                                                                                                                         konformanz_ol_ro   457     0.085          0.293        0.009   0.293                    0.025
                                                                                                                         konformanz_x4      611     0.085          0.568        0.018   0.568                    0.048
handle_leads                     betrag_unter=11000                                                                      konformanz_ol      679    -0.434          0.405        0.018   0.577                    0.251
                                                                                                                         konformanz_ol300   679    -0.434          0.328        0.010   0.661                    0.287
                                                                                                                         konformanz_ol_ro   679    -0.434          0.016        0.003   0.981                    0.426
                                                                                                                         konformanz_x4      679    -0.434          0.212        0.019   0.769                    0.334
                                 keine                                                                                   konformanz_ol      780     0.079          0.241        0.010   0.241                    0.019
                                                                                                                         konformanz_ol300   780     0.079          0.071        0.003   0.071                    0.006
                                                                                                                         konformanz_ol_ro   780     0.079          0.018        0.005   0.018                    0.001
                                                                                                                         konformanz_x4      780     0.079          0.201        0.010   0.201                    0.016
```

```
                  entscheidungen  falsch  verlust_summe
arm                                                    
konformanz_ol               5903   0.305        449.541
konformanz_ol300            5983   0.265        376.620
konformanz_ol_ro            6208   0.317        585.800
konformanz_x4               6062   0.355        545.115
```
