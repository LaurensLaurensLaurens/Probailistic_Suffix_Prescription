# Gezogene Restzeit je Punkt und Arm, vorher / nachher (sv)

100 Testfälle je Zelle, 100 Ziehungen

```
    Punkt                            Option      Arm  Fälle  echt (d)  gezogen vorher (d)  gezogen nachher (d)  Fehler vorher (d)  Fehler nachher (d)  Verschiebung (d)  Rang vorher  Rang nachher
nachlegen O_Sent (mail and online)|complete referenz    100     20.78               18.14                20.00              -2.64               -0.78              1.85         0.03          0.02
nachlegen           O_Create Offer|complete   aktion    100     22.50               18.18                20.67              -4.32               -1.83              2.49        -0.03         -0.09
  konzept      W_Complete application|start referenz    100     23.21               19.74                21.87              -3.47               -1.34              2.12        -0.08         -0.12
  konzept               A_Accepted|complete   aktion    100     19.54               18.55                20.93              -0.99                1.39              2.38        -0.05         -0.08
  nach V.     W_Complete application|resume referenz    100     21.20               20.59                22.78              -0.61                1.57              2.19         0.02          0.01
  nach V.               A_Accepted|complete   aktion    100     20.62               18.66                21.00              -1.96                0.38              2.34         0.01          0.08
    leads           W_Handle leads|withdraw referenz    100     24.04               20.88                22.99              -3.15               -1.05              2.11        -0.06         -0.03
    leads              W_Handle leads|start   aktion    100     23.63               21.64                24.91              -1.99                1.28              3.27        -0.05         -0.12
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
