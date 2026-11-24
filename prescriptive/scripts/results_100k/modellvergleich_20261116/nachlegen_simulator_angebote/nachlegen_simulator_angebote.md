# Späte Zusatzangebote im Simulator: Zuteilung, natürliche Wahl, echter Log (2026-11-19)

Erster Besuch angebot_nachlegen 50/50 gelost, natürliche Wahl des Simulator-Modells festgehalten; 10 Seeds × 2.000 gewertete Fälle je Zuteilung. Abschnitt ab der Entscheidung bis W_Validate application|start (sonst Fallende); Restzeit bis Fallende. Echt = beobachtet (natürlich = gewählt).

## Zellen (Mittel je Fall)

```
                               bis Validierung/Ende (d)  Validierung erreicht  späte Angebote  Zeit vor späten Angeboten (d)  O_Sent  Nachfass-suspend  Storno (A_Cancelled)  Restzeit bis Fallende (d)      n
zuteilung natuerlich gewaehlt                                                                                                                                                                                 
echt      extra      extra                       17.740                 0.621           0.262                          1.240   2.214             2.131                 0.400                     22.515   3221
          senden     senden                      15.152                 0.704           0.137                          0.734   1.154             1.939                 0.322                     20.231  26907
learned   extra      extra                       19.252                 0.617           0.420                          2.345   2.392             2.137                 0.416                     24.057    849
                     senden                      17.357                 0.667           0.253                          1.464   1.265             2.073                 0.377                     23.291    853
          senden     extra                       17.881                 0.671           0.398                          2.203   2.289             2.123                 0.359                     23.237   8691
                     senden                      15.811                 0.715           0.227                          1.381   1.239             2.041                 0.324                     21.400   8850
random    extra      extra                       19.348                 0.637           0.397                          2.557   2.386             2.189                 0.407                     24.369    779
                     senden                      16.880                 0.693           0.236                          1.432   1.254             2.109                 0.349                     22.706    850
          senden     extra                       18.125                 0.659           0.397                          2.285   2.295             2.187                 0.365                     23.024   8705
                     senden                      15.853                 0.714           0.238                          1.374   1.252             2.097                 0.320                     21.134   8814
```

## Kontraste (± Standardfehler)

```
                                                                       n Restzeit bis Fallende (d) bis Validierung/Ende (d)  späte Angebote Zeit vor späten Angeboten (d) Storno (A_Cancelled)          O_Sent
quelle  vergleich                                                                                                                                                                                             
echt    beobachtet: unerzwungen extra − unerzwungen senden  3221 / 26907              +2.28 ± 0.24             +2.59 ± 0.23  +0.124 ± 0.010                  +0.51 ± 0.08       +0.078 ± 0.009  +1.060 ± 0.016
learned beobachtet: unerzwungen extra − unerzwungen senden    849 / 8850              +2.66 ± 0.54             +3.44 ± 0.47  +0.193 ± 0.025                  +0.96 ± 0.20       +0.092 ± 0.018  +1.153 ± 0.036
        randomisiert gesamt: extra − senden                  9540 / 9703              +1.74 ± 0.22             +2.06 ± 0.18  +0.171 ± 0.009                  +0.83 ± 0.07       +0.035 ± 0.007  +1.057 ± 0.011
        bei natürlich senden: erzwungen extra − senden       8691 / 8850              +1.84 ± 0.22             +2.07 ± 0.19  +0.171 ± 0.009                  +0.82 ± 0.08       +0.035 ± 0.007  +1.050 ± 0.012
        bei natürlich extra: extra − erzwungen senden          849 / 853              +0.77 ± 0.74             +1.90 ± 0.62  +0.167 ± 0.031                  +0.88 ± 0.25       +0.038 ± 0.024  +1.127 ± 0.041
random  beobachtet: unerzwungen extra − unerzwungen senden    779 / 8814              +3.24 ± 0.59             +3.50 ± 0.53  +0.159 ± 0.025                  +1.18 ± 0.25       +0.087 ± 0.018  +1.134 ± 0.035
        randomisiert gesamt: extra − senden                  9484 / 9664              +1.86 ± 0.22             +2.28 ± 0.18  +0.160 ± 0.009                  +0.93 ± 0.08       +0.046 ± 0.007  +1.050 ± 0.011
        bei natürlich senden: erzwungen extra − senden       8705 / 8814              +1.89 ± 0.23             +2.27 ± 0.19  +0.160 ± 0.010                  +0.91 ± 0.08       +0.045 ± 0.007  +1.043 ± 0.012
        bei natürlich extra: extra − erzwungen senden          779 / 850              +1.66 ± 0.76             +2.47 ± 0.67  +0.160 ± 0.031                  +1.13 ± 0.29       +0.058 ± 0.024  +1.132 ± 0.040
```

## Ganze Fälle ohne Eingriff: was folgt auf W_Call after offers|suspend

Anteil:
```
quelle                          echt  learned  random
folgt                                                
A_Cancelled|complete           0.124    0.120   0.116
O_Cancelled|complete           0.003    0.003   0.003
O_Create Offer|complete        0.059    0.101   0.100
W_Call after offers|ate_abort  0.344    0.307   0.298
W_Call after offers|resume     0.451    0.459   0.473
W_Call after offers|suspend    0.016    0.009   0.009
```
Abstand davor (Tage):
```
quelle                          echt  learned  random
folgt                                                
A_Cancelled|complete           23.05    23.01   22.89
O_Cancelled|complete            3.20     3.32    3.22
O_Create Offer|complete         5.53     6.26    6.08
W_Call after offers|ate_abort   4.82     5.02    5.03
W_Call after offers|resume      3.85     3.88    3.68
W_Call after offers|suspend     0.00     0.02    0.17
```
Je Fall:
```
quelle                                           echt    learned     random
suspend_je_fall                                 1.943      2.036      2.088
neue_angebote_nach_1_sent_je_fall               0.230      0.327      0.322
faelle                                      31509.000  10456.000  10445.000
neues_angebot_nach_suspend_bei_1_angeboten      0.055      0.097      0.096
neues_angebot_nach_suspend_bei_2_angeboten      0.076      0.119      0.118
neues_angebot_nach_suspend_bei_3_angeboten      0.102      0.136      0.124
```
