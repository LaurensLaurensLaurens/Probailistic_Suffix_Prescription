# Verläufe nach dem 1. nachlegen: extra − senden je Quelle (2026-11-19)

Bis zur ersten W_Validate application|start (sonst Fallende). Echt = beobachtet, Simulator = randomisiert mit Bank-Fortsetzung.

## Kennzahlen (extra − senden)

```
quelle                            echt  Simulator  T-Learner  Steigung  β-NLL 1
bis Validierung bzw. Ende (d)     2.59       2.28       0.21     -0.15     0.40
erreicht Validierung             -0.08      -0.05      -0.02     -0.02    -0.03
Events bis dahin                  4.61       4.45       3.96      3.89     3.88
O_Sent                            1.06       1.06       0.92      0.91     0.90
O_Created                         1.27       1.28       1.20      1.20     1.18
Nachfass-Schleifen                0.38       0.18       0.18      0.10     0.15
bis 2. O_Sent (d)                -6.80      -7.46      -6.14     -5.38    -6.63
bis letztes O_Sent (d)            0.83       1.46       0.40      0.24     0.23
letztes O_Sent → Validierung (d)  0.64       0.50      -0.49     -0.43    -0.44
Restzeit gesamt (d)               2.28       1.55      -0.07     -0.38    -0.30
```

## Kennzahlen je Option

```
                  bis Validierung bzw. Ende (d)  erreicht Validierung  Events bis dahin  O_Sent  O_Created  Nachfass-Schleifen  bis 2. O_Sent (d)  bis letztes O_Sent (d)  letztes O_Sent → Validierung (d)  Restzeit gesamt (d)      n
quelle    opt                                                                                                                                                                                                                          
Simulator extra                           18.28                  0.66             16.46    2.31       1.53                3.35               0.42                    3.30                              9.23                22.98   3753
          senden                          15.99                  0.71             12.01    1.24       0.25                3.17               7.88                    1.85                              8.73                21.42   3894
Steigung  extra                           14.98                  0.69             15.73    2.11       1.39                2.97               0.25                    1.32                              8.45                21.03   6000
          senden                          15.13                  0.71             11.84    1.20       0.19                2.87               5.63                    1.07                              8.88                21.40   6000
T-Learner extra                           11.94                  0.76             15.82    2.13       1.40                3.04               0.34                    1.66                              8.21                18.21   6000
          senden                          11.73                  0.78             11.86    1.21       0.20                2.86               6.48                    1.27                              8.71                18.28   6000
echt      extra                           17.74                  0.62             16.21    2.21       1.43                3.23               0.20                    1.88                              9.39                22.52   3221
          senden                          15.15                  0.70             11.60    1.15       0.16                2.85               6.99                    1.05                              8.75                20.23  26907
β-NLL 1   extra                           16.25                  0.68             15.70    2.11       1.38                2.99               0.32                    1.62                              9.08                22.71   6000
          senden                          15.85                  0.72             11.82    1.21       0.20                2.84               6.95                    1.39                              9.53                23.00   6000
```

## Zeit vor Token je Verlauf, extra − senden (Tage; die 14 Token mit dem größten Unterschied echt + β-NLL)

```
                                   echt  Simulator  T-Learner  Steigung  β-NLL 1
token                                                                           
A_Cancelled|complete               1.70       0.79       0.11     -0.07     0.63
O_Create Offer|complete            0.51       1.08       0.15      0.12     0.08
W_Call after offers|resume         0.27       0.23       0.23      0.35     0.05
W_Call after offers|ate_abort     -0.02       0.11      -0.25     -0.44    -0.23
W_Call after offers|withdraw       0.04      -0.01      -0.15     -0.17    -0.20
O_Cancelled|complete               0.04       0.03       0.07      0.05     0.06
W_Complete application|resume      0.02       0.01       0.05      0.04     0.04
O_Sent (mail and online)|complete  0.01       0.00       0.01     -0.00     0.03
A_Denied|complete                  0.01       0.01       0.03      0.01     0.01
W_Call incomplete files|resume     0.00       0.00      -0.01     -0.02    -0.02
W_Call after offers|start          0.00      -0.01      -0.00     -0.01    -0.02
W_Call after offers|suspend        0.00       0.01      -0.01     -0.03    -0.02
W_Personal Loan collection|resume  0.00       0.00       0.00      0.00    -0.02
A_Complete|complete                0.00       0.00      -0.00      0.00    -0.01
```

## Zahl je Verlauf, extra − senden (gleiche Token)

```
                                   echt  Simulator  T-Learner  Steigung  β-NLL 1
token                                                                           
A_Cancelled|complete               0.08       0.05       0.03      0.02     0.03
O_Create Offer|complete            1.27       1.28       1.20      1.20     1.18
W_Call after offers|resume         0.19       0.08       0.09      0.06     0.08
W_Call after offers|ate_abort     -0.00      -0.00      -0.00     -0.01    -0.01
W_Call after offers|withdraw      -0.00      -0.00      -0.03     -0.02    -0.03
O_Cancelled|complete               0.68       0.65       0.43      0.48     0.49
W_Complete application|resume      0.02       0.01       0.02      0.03     0.03
O_Sent (mail and online)|complete  1.02       1.01       0.86      0.85     0.84
A_Denied|complete                  0.00       0.00       0.00      0.00     0.00
W_Call incomplete files|resume     0.00       0.00       0.00     -0.00    -0.00
W_Call after offers|start         -0.00       0.00      -0.00     -0.00    -0.00
W_Call after offers|suspend        0.19       0.10       0.09      0.04     0.07
W_Personal Loan collection|resume  0.00       0.00       0.00      0.00    -0.00
A_Complete|complete               -0.00      -0.00      -0.00     -0.00    -0.00
```
