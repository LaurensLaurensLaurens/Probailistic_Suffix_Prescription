# Unterscheidbarkeit der Optionen je Entscheidung (2026-11-23)

Anteil der Zustände; nicht unterscheidbar = |Δ| < 1,96 · SE.

## Je Wahlpunkt

```
                                           Modell                           Gruppe  Zustände  Ziehungen  nicht unterscheidbar (100)  nicht unterscheidbar (300)  sicher andere besser (300)  sicher Regel besser (300)  Δ Median (d)  SE(300) Median (d)
tlsvk (T-Learner + Suffix-Verlust + Kalibrierung)             annehmen_bei_konzept        60        100                       0.217                       0.083                       0.000                      0.917         2.806               0.603
tlsvk (T-Learner + Suffix-Verlust + Kalibrierung) annehmen_nach_vervollstaendigung        60        100                       0.150                       0.033                       0.000                      0.967         3.532               0.636
tlsvk (T-Learner + Suffix-Verlust + Kalibrierung)                     handle_leads        60        100                       0.717                       0.367                       0.000                      0.633         1.545               0.626
tlsvk (T-Learner + Suffix-Verlust + Kalibrierung)              nachlegen 1. Besuch        60        100                       0.750                       0.433                       0.000                      0.567         1.986               0.934
tlsvk (T-Learner + Suffix-Verlust + Kalibrierung)           nachlegen ab 2. Besuch        60        100                       0.400                       0.183                       0.000                      0.817         4.108               1.061
                                   olsv (8 Netze)             annehmen_bei_konzept        60        100                       0.450                       0.217                       0.000                      0.783         2.406               0.615
                                   olsv (8 Netze) annehmen_nach_vervollstaendigung        60        100                       0.333                       0.233                       0.000                      0.767         3.020               0.705
                                   olsv (8 Netze)                     handle_leads        60        100                       0.533                       0.183                       0.000                      0.817         2.539               0.713
                                   olsv (8 Netze)              nachlegen 1. Besuch        60        100                       0.533                       0.267                       0.000                      0.733         2.143               0.716
                                   olsv (8 Netze)           nachlegen ab 2. Besuch        60        100                       0.767                       0.533                       0.100                      0.367         1.535               1.006
                tlsv (T-Learner + Suffix-Verlust)             annehmen_bei_konzept        60        100                       0.850                       0.617                       0.000                      0.383         0.857               0.609
                tlsv (T-Learner + Suffix-Verlust) annehmen_nach_vervollstaendigung        60        100                       0.650                       0.383                       0.000                      0.617         1.880               0.673
                tlsv (T-Learner + Suffix-Verlust)                     handle_leads        60        100                       0.650                       0.200                       0.000                      0.800         1.897               0.632
                tlsv (T-Learner + Suffix-Verlust)              nachlegen 1. Besuch        60        100                       0.867                       0.533                       0.017                      0.450         0.914               0.621
                tlsv (T-Learner + Suffix-Verlust)           nachlegen ab 2. Besuch        60        100                       0.833                       0.500                       0.133                      0.367         1.053               0.897
                    plsv (S-Learner je Wahlpunkt)             annehmen_bei_konzept        60        100                       0.600                       0.300                       0.000                      0.700         1.730               0.591
                    plsv (S-Learner je Wahlpunkt) annehmen_nach_vervollstaendigung        60        100                       0.300                       0.117                       0.000                      0.883         3.072               0.688
                    plsv (S-Learner je Wahlpunkt)                     handle_leads        60        100                       0.917                       0.533                       0.067                      0.400         0.964               0.612
                    plsv (S-Learner je Wahlpunkt)              nachlegen 1. Besuch        60        100                       0.967                       0.750                       0.133                      0.117        -0.022               0.568
                    plsv (S-Learner je Wahlpunkt)           nachlegen ab 2. Besuch        60        100                       0.783                       0.550                       0.183                      0.267        -0.057               1.020
          T-Learner (ZZ-TF1, ohne Suffix-Verlust)             annehmen_bei_konzept        60        100                       0.617                       0.350                       0.000                      0.650         1.377               0.571
          T-Learner (ZZ-TF1, ohne Suffix-Verlust) annehmen_nach_vervollstaendigung        60        100                       0.483                       0.233                       0.000                      0.767         2.125               0.603
          T-Learner (ZZ-TF1, ohne Suffix-Verlust)                     handle_leads        60        100                       0.900                       0.533                       0.000                      0.467         0.967               0.593
          T-Learner (ZZ-TF1, ohne Suffix-Verlust)              nachlegen 1. Besuch        60        100                       0.950                       0.783                       0.067                      0.150         0.141               0.596
          T-Learner (ZZ-TF1, ohne Suffix-Verlust)           nachlegen ab 2. Besuch        60        100                       0.850                       0.533                       0.183                      0.283         0.596               0.821
```

## Gewichtet nach Online-Wahlen je Fall (ohne spätere nachlegen-Besuche)

```
                                           Modell  nicht unterscheidbar (100)  nicht unterscheidbar (300)  sicher andere besser (300)  sicher Regel besser (300)
tlsvk (T-Learner + Suffix-Verlust + Kalibrierung)                       0.556                       0.294                       0.000                      0.706
                                   olsv (8 Netze)                       0.494                       0.230                       0.000                      0.770
                tlsv (T-Learner + Suffix-Verlust)                       0.786                       0.453                       0.007                      0.540
                    plsv (S-Learner je Wahlpunkt)                       0.803                       0.526                       0.071                      0.403
          T-Learner (ZZ-TF1, ohne Suffix-Verlust)                       0.813                       0.563                       0.027                      0.410
```
