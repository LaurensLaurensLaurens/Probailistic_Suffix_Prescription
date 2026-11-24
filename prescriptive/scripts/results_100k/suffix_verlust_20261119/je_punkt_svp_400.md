# Gezogene Restzeit je Punkt und Arm, vorher / nachher (svp)

400 Testfälle je Zelle, 30 Ziehungen

```
    Punkt                            Option      Arm  Fälle  echt (d)  gezogen vorher (d)  gezogen nachher (d)  Fehler vorher (d)  Fehler nachher (d)  Verschiebung (d)  Rang vorher  Rang nachher  Streuung Fallmittel vorher (d)  Streuung Fallmittel nachher (d)  SD Ziehungen vorher (d)  SD Ziehungen nachher (d)
nachlegen O_Sent (mail and online)|complete referenz    400     20.01               18.15                21.86              -1.85                1.85              3.70        -0.04          0.07                            0.00                             1.41                    13.04                     18.65
nachlegen           O_Create Offer|complete   aktion    400     22.21               18.34                20.96              -3.88               -1.26              2.62         0.02         -0.03                            0.18                             1.70                    12.91                     17.80
  konzept      W_Complete application|start referenz    400     22.46               19.78                23.65              -2.68                1.19              3.87         0.05          0.01                            0.00                             0.00                    13.53                     19.09
  konzept               A_Accepted|complete   aktion    400     19.25               18.44                19.56              -0.81                0.32              1.12         0.02          0.08                            0.00                             0.00                    12.80                     16.18
  nach V.     W_Complete application|resume referenz    400     21.50               20.60                24.19              -0.89                2.69              3.59         0.02          0.09                            0.82                             0.75                    13.87                     19.26
  nach V.               A_Accepted|complete   aktion    400     20.61               18.67                20.28              -1.94               -0.32              1.61         0.00          0.02                            0.00                             0.69                    13.44                     16.90
    leads           W_Handle leads|withdraw referenz    400     22.77               20.80                24.69              -1.96                1.92              3.89         0.02          0.05                            0.62                             0.00                    13.72                     19.53
    leads              W_Handle leads|start   aktion    400     23.79               21.51                23.82              -2.28                0.03              2.31        -0.08         -0.10                            0.78                             0.54                    13.93                     18.34
```

Proben im ganzen Log:

```
    punkt      arm  Proben im Log (alle Besuche)
nachlegen referenz                         35604
nachlegen   aktion                          3887
  konzept referenz                         15065
  konzept   aktion                          9231
  nach V. referenz                         22347
  nach V.   aktion                          9457
    leads referenz                         16802
    leads   aktion                          3644
```
