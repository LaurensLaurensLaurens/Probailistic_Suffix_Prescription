# Gezogene Restzeit je Punkt und Arm, vorher / nachher (svk)

400 Testfälle je Zelle, 30 Ziehungen

```
    Punkt                            Option      Arm  Fälle  echt (d)  gezogen vorher (d)  gezogen nachher (d)  Fehler vorher (d)  Fehler nachher (d)  Verschiebung (d)  Rang vorher  Rang nachher  Streuung Fallmittel vorher (d)  Streuung Fallmittel nachher (d)  SD Ziehungen vorher (d)  SD Ziehungen nachher (d)
nachlegen O_Sent (mail and online)|complete referenz    400     20.01               18.15                20.18              -1.85                0.17              2.02        -0.04         -0.03                            0.00                             0.45                    13.04                     11.69
nachlegen           O_Create Offer|complete   aktion    400     22.21               18.34                22.01              -3.88               -0.20              3.67         0.02         -0.05                            0.18                             0.46                    12.91                     15.49
  konzept      W_Complete application|start referenz    400     22.46               19.78                22.52              -2.68                0.06              2.74         0.05         -0.01                            0.00                             0.00                    13.53                     12.08
  konzept               A_Accepted|complete   aktion    400     19.25               18.44                19.99              -0.81                0.75              1.55         0.02          0.04                            0.00                             0.00                    12.80                     12.66
  nach V.     W_Complete application|resume referenz    400     21.50               20.60                23.34              -0.89                1.85              2.74         0.02         -0.01                            0.82                             0.75                    13.87                     12.36
  nach V.               A_Accepted|complete   aktion    400     20.61               18.67                20.07              -1.94               -0.54              1.40         0.00          0.09                            0.00                             0.00                    13.44                     12.52
    leads           W_Handle leads|withdraw referenz    400     22.77               20.80                23.52              -1.96                0.75              2.72         0.02          0.04                            0.62                             0.63                    13.72                     12.05
    leads              W_Handle leads|start   aktion    400     23.79               21.51                24.76              -2.28                0.96              3.24        -0.08         -0.03                            0.78                             0.00                    13.93                     13.55
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
