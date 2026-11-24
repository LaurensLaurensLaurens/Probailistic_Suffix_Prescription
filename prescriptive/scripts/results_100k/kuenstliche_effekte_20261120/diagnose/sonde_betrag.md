# Lineare Sonde „Betrag < 11.000 €“ in den Zuständen von k3x4ol an konzept (2026-11-23 23:27)

800 frische Zustände (Bank-Läufe), 5 Ziehungen je Fall und Netz; logistische Regression, 5-fach GroupKFold nach Fall; AUC 0,5 = nicht ablesbar, 1,0 = perfekt. Decoder Schritt k: Zustand, aus dem Event k nach der Option gezogen wird (nur Ziehungen, die so lang sind).

```
                        Netz             Zustand  Zeilen  Fälle  Anteil < 11k  AUC Sonde
W_Complete application|start Encoder nach Option    4000    800         0.500      0.689
W_Complete application|start   Decoder Schritt 0    4000    800         0.500      0.646
W_Complete application|start   Decoder Schritt 1    4000    800         0.500      0.591
W_Complete application|start   Decoder Schritt 2    4000    800         0.500      0.581
W_Complete application|start   Decoder Schritt 3    4000    800         0.500      0.559
W_Complete application|start   Decoder Schritt 4    3997    800         0.500      0.538
W_Complete application|start   Decoder Schritt 6    3996    800         0.501      0.504
W_Complete application|start   Decoder Schritt 8    3982    800         0.501      0.533
W_Complete application|start  Decoder Schritt 10    3950    800         0.499      0.511
W_Complete application|start  Decoder Schritt 12    3928    800         0.498      0.505
W_Complete application|start  Decoder Schritt 15    3740    800         0.498      0.483
W_Complete application|start  Decoder Schritt 20    3304    799         0.495      0.488
                  A_Accepted Encoder nach Option    4000    800         0.500      0.931
                  A_Accepted   Decoder Schritt 0    4000    800         0.500      0.918
                  A_Accepted   Decoder Schritt 1    4000    800         0.500      0.917
                  A_Accepted   Decoder Schritt 2    3998    800         0.500      0.911
                  A_Accepted   Decoder Schritt 3    3998    800         0.500      0.907
                  A_Accepted   Decoder Schritt 4    3996    800         0.500      0.899
                  A_Accepted   Decoder Schritt 6    3985    800         0.501      0.890
                  A_Accepted   Decoder Schritt 8    3972    800         0.500      0.856
                  A_Accepted  Decoder Schritt 10    3946    800         0.499      0.744
                  A_Accepted  Decoder Schritt 12    3784    800         0.499      0.632
                  A_Accepted  Decoder Schritt 15    2943    800         0.491      0.511
                  A_Accepted  Decoder Schritt 20    2466    795         0.491      0.504
```

Wo im gezogenen Suffix der erste A_Complete steht (Schritte nach der Option):

```
                              Suffixe  Länge_Median  Anteil_mit_A_Complete  Schritt_erster_A_Complete_Median
option                                                                                                      
A_Accepted                       4000          28.0                   0.99                               6.0
W_Complete application|start     4000          33.0                   1.00                               8.0
```

Laufzeit 484 s
