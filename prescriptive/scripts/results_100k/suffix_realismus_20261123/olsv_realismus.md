# Wie realistisch zieht olsv die Suffixe online? (2026-11-23)

Zustände: erste Besuche in Bank-Läufen Aufbau F (Seeds 0–11), je Punkt bis 100; je Zustand 30 Ziehungen mit der Option der Bank; echt = Simulator.

```
                           Punkt  Quelle     n  Restzeit Mittel    SD  Q10   Q50   Q90  Events  Angebote  O_Sent  Anrufe  A_Pending %  A_Denied %  A_Cancelled %  kein Ende %  unbekannte Übergänge je 100
               angebot_nachlegen    echt   100            20.89 14.81 7.16 17.88 33.87   29.04      0.37    1.31    1.01        59.00        7.00          34.00         0.00                         0.00
               angebot_nachlegen gezogen  3000            20.56 14.31 6.30 16.90 39.38   30.28      0.42    1.32    1.01        60.83       12.93          24.63         1.60                         0.68
            annehmen_bei_konzept    echt   100            21.35 12.46 7.68 17.61 37.54   35.02      1.41    1.33    1.01        54.00        7.00          39.00         0.00                         0.00
            annehmen_bei_konzept gezogen  3000            22.21 14.72 7.43 18.85 41.77   35.32      1.43    1.32    1.01        59.47       13.13          26.27         1.13                         0.65
annehmen_nach_vervollstaendigung    echt   100            20.91 11.75 7.88 18.51 32.39   39.19      1.38    1.28    1.00        69.00        6.00          25.00         0.00                         0.00
annehmen_nach_vervollstaendigung gezogen  3000            22.55 14.84 7.66 19.18 42.21   34.56      1.36    1.27    1.00        55.90       15.80          26.53         1.77                         0.59
                    handle_leads    echt   100            23.58 14.81 7.94 21.37 41.20   38.29      1.42    1.29    0.98        53.00        6.00          40.00         1.00                         0.00
                    handle_leads gezogen  3000            22.88 14.70 7.78 19.70 42.22   39.06      1.38    1.28    1.00        52.87       16.90          28.97         1.27                         0.70
                            alle    echt   400            21.68 13.52 7.34 18.95 36.61   35.38      1.14    1.30    1.00        58.75        6.50          34.50         0.25                         0.00
                            alle gezogen 12000            22.05 14.67 7.28 18.61 41.38   34.81      1.15    1.30    1.00        57.27       14.69          26.60         1.44                         0.66
```

## Restzeit je Zustand: Fehler des Ziehungsmittels

```
                                  MAE olsv (d)  MAE naiv (d)  Korrelation
punkt                                                                    
angebot_nachlegen                        11.44         11.41        -0.01
annehmen_bei_konzept                     11.32         11.06         0.04
annehmen_nach_vervollstaendigung          9.84          9.71         0.14
handle_leads                             11.54         12.07         0.23
```
