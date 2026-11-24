# Gezogene Restzeit je Punkt und Arm, vorher / nachher (svh)

400 Testfälle je Zelle, 30 Ziehungen

```
    Punkt                            Option      Arm  Fälle  echt (d)  gezogen vorher (d)  gezogen nachher (d)  Fehler vorher (d)  Fehler nachher (d)  Verschiebung (d)  Rang vorher  Rang nachher  Streuung Fallmittel vorher (d)  Streuung Fallmittel nachher (d)  SD Ziehungen vorher (d)  SD Ziehungen nachher (d)
nachlegen O_Sent (mail and online)|complete referenz    400     20.01               18.15                20.49              -1.85                0.48              2.34        -0.04          0.06                            0.00                             0.00                    13.04                     16.19
nachlegen           O_Create Offer|complete   aktion    400     22.21               18.34                22.08              -3.88               -0.14              3.74         0.02         -0.03                            0.18                             1.38                    12.91                     19.80
  konzept      W_Complete application|start referenz    400     22.46               19.78                22.38              -2.68               -0.08              2.60         0.05          0.04                            0.00                             0.00                    13.53                     17.22
  konzept               A_Accepted|complete   aktion    400     19.25               18.44                20.20              -0.81                0.96              1.76         0.02          0.04                            0.00                             0.38                    12.80                     17.53
  nach V.     W_Complete application|resume referenz    400     21.50               20.60                23.11              -0.89                1.62              2.51         0.02          0.02                            0.82                             1.50                    13.87                     17.94
  nach V.               A_Accepted|complete   aktion    400     20.61               18.67                20.80              -1.94                0.20              2.13         0.00          0.03                            0.00                             1.51                    13.44                     18.12
    leads           W_Handle leads|withdraw referenz    400     22.77               20.80                23.15              -1.96                0.39              2.35         0.02          0.03                            0.62                             1.03                    13.72                     17.31
    leads              W_Handle leads|start   aktion    400     23.79               21.51                25.14              -2.28                1.35              3.63        -0.08         -0.11                            0.78                             0.00                    13.93                     20.31
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
