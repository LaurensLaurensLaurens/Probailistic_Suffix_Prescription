# MC-Dropout an/aus beim Ziehen (Diagnose, 2026-11-19)

```
            Modell Dropout                           Gruppe  Zustände  SD je Verlauf (d)  ρ Optionen  Rauschen je Entscheidung bei 100 (d)  Rauschen bei 300 (d)  Vorteil Regel-Option (d)  richtig (100 Ziehungen)  rauschfreie Streuung zw. Zuständen (d)
T-Learner (ZZ-TF1)      an                     handle_leads        60             13.508       0.172                                 1.739                 1.004                     0.804                    0.717                                   0.159
T-Learner (ZZ-TF1)     aus                     handle_leads        60             13.767       0.156                                 1.792                 1.035                     0.715                    0.583                                   1.075
 ZZ-TF1 (ein Netz)      an                     handle_leads        60             14.740       0.233                                 1.828                 1.055                    -0.034                    0.517                                   0.000
 ZZ-TF1 (ein Netz)     aus                     handle_leads        60             14.913       0.234                                 1.807                 1.043                     0.281                    0.550                                   0.000
T-Learner (ZZ-TF1)      an annehmen_nach_vervollstaendigung        60             13.760       0.127                                 1.816                 1.048                     2.470                    0.933                                   0.672
T-Learner (ZZ-TF1)     aus annehmen_nach_vervollstaendigung        60             13.653       0.148                                 1.740                 1.005                     2.431                    0.850                                   1.312
 ZZ-TF1 (ein Netz)      an annehmen_nach_vervollstaendigung        60             14.848       0.136                                 1.932                 1.116                     2.289                    0.767                                   0.751
 ZZ-TF1 (ein Netz)     aus annehmen_nach_vervollstaendigung        60             15.188       0.119                                 1.984                 1.145                     2.862                    0.933                                   0.918
T-Learner (ZZ-TF1)      an              nachlegen 1. Besuch        60             12.505       0.142                                 1.699                 0.981                     0.279                    0.683                                   0.000
T-Learner (ZZ-TF1)     aus              nachlegen 1. Besuch        60             12.820       0.113                                 1.702                 0.983                     0.053                    0.533                                   0.000
 ZZ-TF1 (ein Netz)      an              nachlegen 1. Besuch        60             13.509       0.152                                 1.750                 1.010                     0.119                    0.517                                   0.622
 ZZ-TF1 (ein Netz)     aus              nachlegen 1. Besuch        60             13.504       0.142                                 1.780                 1.028                     0.011                    0.433                                   1.113
T-Learner (ZZ-TF1)      an           nachlegen ab 2. Besuch        60             13.101       0.110                                 1.716                 0.991                     0.771                    0.533                                   1.740
T-Learner (ZZ-TF1)     aus           nachlegen ab 2. Besuch        60             13.255       0.091                                 1.818                 1.049                     1.036                    0.633                                   2.034
 ZZ-TF1 (ein Netz)      an           nachlegen ab 2. Besuch        60             14.156       0.130                                 1.832                 1.058                     0.542                    0.683                                   2.192
 ZZ-TF1 (ein Netz)     aus           nachlegen ab 2. Besuch        60             14.825       0.147                                 1.923                 1.110                     1.044                    0.633                                   2.318
T-Learner (ZZ-TF1)      an             annehmen_bei_konzept        60             12.798       0.124                                 1.709                 0.987                     1.469                    0.750                                   0.445
T-Learner (ZZ-TF1)     aus             annehmen_bei_konzept        60             13.159       0.084                                 1.731                 1.000                     1.505                    0.833                                   0.000
 ZZ-TF1 (ein Netz)      an             annehmen_bei_konzept        60             14.393       0.121                                 1.850                 1.068                     1.362                    0.800                                   0.000
 ZZ-TF1 (ein Netz)     aus             annehmen_bei_konzept        60             14.695       0.134                                 1.942                 1.121                     1.513                    0.817                                   0.599

            Modell                           Gruppe  Dropout-Anteil an Var. je Verlauf
T-Learner (ZZ-TF1)                     handle_leads                             -0.039
 ZZ-TF1 (ein Netz)                     handle_leads                             -0.024
T-Learner (ZZ-TF1) annehmen_nach_vervollstaendigung                              0.015
 ZZ-TF1 (ein Netz) annehmen_nach_vervollstaendigung                             -0.046
T-Learner (ZZ-TF1)              nachlegen 1. Besuch                             -0.051
 ZZ-TF1 (ein Netz)              nachlegen 1. Besuch                              0.001
T-Learner (ZZ-TF1)           nachlegen ab 2. Besuch                             -0.024
 ZZ-TF1 (ein Netz)           nachlegen ab 2. Besuch                             -0.097
T-Learner (ZZ-TF1)             annehmen_bei_konzept                             -0.057
 ZZ-TF1 (ein Netz)             annehmen_bei_konzept                             -0.042
```
