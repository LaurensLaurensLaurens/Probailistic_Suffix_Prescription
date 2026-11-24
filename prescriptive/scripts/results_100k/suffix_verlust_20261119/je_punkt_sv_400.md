# Gezogene Restzeit je Punkt und Arm, vorher / nachher (sv)

400 Testfälle je Zelle, 30 Ziehungen

```
    Punkt                            Option      Arm  Fälle  echt (d)  gezogen vorher (d)  gezogen nachher (d)  Fehler vorher (d)  Fehler nachher (d)  Verschiebung (d)  Rang vorher  Rang nachher  Streuung Fallmittel vorher (d)  Streuung Fallmittel nachher (d)  SD Ziehungen vorher (d)  SD Ziehungen nachher (d)
nachlegen O_Sent (mail and online)|complete referenz    400     20.01               18.15                20.02              -1.85                0.01              1.87        -0.04         -0.07                            0.00                             0.33                    13.04                     13.98
nachlegen           O_Create Offer|complete   aktion    400     22.21               18.34                20.68              -3.88               -1.54              2.34         0.02          0.03                            0.18                             0.74                    12.91                     14.22
  konzept      W_Complete application|start referenz    400     22.46               19.78                21.84              -2.68               -0.62              2.06         0.05         -0.00                            0.00                             0.42                    13.53                     14.59
  konzept               A_Accepted|complete   aktion    400     19.25               18.44                20.99              -0.81                1.74              2.55         0.02         -0.05                            0.00                             0.00                    12.80                     14.73
  nach V.     W_Complete application|resume referenz    400     21.50               20.60                22.86              -0.89                1.36              2.26         0.02          0.01                            0.82                             1.26                    13.87                     15.25
  nach V.               A_Accepted|complete   aktion    400     20.61               18.67                20.88              -1.94                0.27              2.21         0.00          0.03                            0.00                             0.00                    13.44                     14.61
    leads           W_Handle leads|withdraw referenz    400     22.77               20.80                22.98              -1.96                0.21              2.18         0.02          0.06                            0.62                             0.81                    13.72                     14.79
    leads              W_Handle leads|start   aktion    400     23.79               21.51                24.90              -2.28                1.10              3.39        -0.08         -0.09                            0.78                             0.00                    13.93                     16.03
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
