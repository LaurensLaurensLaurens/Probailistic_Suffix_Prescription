# Prüfung ep45 (2026-11-18 05:03)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),1.7320252329262666,3.0171389393482246,-0.009555831300634603,0.7712991971523998
SE gepaart Median,1.5720535337429773,1.6405415895455744,1.6066291491187008,1.2969626019475502
Korrelation der Optionen Median,0.2627701334451727,0.22137425911005176,0.19433920675832517,0.5420093471631298
Vorzeichen wie AIPW (alle),0.855,0.95,0.465,0.32
Durchlaufzeit gezogen (Mittel),19.467740885488716,21.655089848336278,20.296471836703166,21.535009625886634
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe            Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept         Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept ZZ-TF1-WP150 ep45                  1.52 0.19                  2.51              0.32/0.32                                     0.87                   1.57             1.44/1.37
annehmen_nach_vervollstaendigung         Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung ZZ-TF1-WP150 ep45                  2.63 0.22                  2.48              0.34/0.34                                     0.92                   1.64             1.42/1.34
             nachlegen 1. Besuch         Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch ZZ-TF1-WP150 ep45                  0.02 0.20                  2.28              0.32/0.31                                     0.47                   1.59             0.31/0.39
          nachlegen ab 2. Besuch         Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch ZZ-TF1-WP150 ep45                 -0.31 0.40                  1.49              0.37/0.33                                     0.55                   1.70             0.34/0.43
                    handle_leads         Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads ZZ-TF1-WP150 ep45                 -0.87 0.17                 -0.05              0.34/0.33                                     0.27                   1.27             1.38/1.33
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZ-TF1-WP150 ep45
Restzeit                                       1.55              -0.07
Events                                         3.84               1.94
Angebote                                       1.29               1.09
versendet                                      1.07               0.17
O_Cancelled                                    1.25               0.01
Nachfass-Schleifen                             0.18               0.01
bis Nachfassen                                 0.02               0.15
Nachfassen bis Validierung                     1.92               0.39
Validierung bis Ende                          -0.53              -0.35
erreicht Validierung                          -0.05               0.00
A_Pending                                     -0.03               0.01
A_Cancelled                                    0.04              -0.00
A_Denied                                      -0.01              -0.00
ohne                                           0.00              -0.00
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
                   angebote  events  pending   rest  storniert  SE rest  Prefixe
modell                                                                          
ZZ-TF1-WP150 ep45    -0.016  -0.538   -0.001 -0.224      0.026    0.076      241
Restzeit ab erstem O_Sent (Tage):
gruppe             Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log                   22.88               20.05             2.83
ZZ-TF1-WP150 ep45            18.77               18.60             0.17
SE der echten Differenz: 1.11
FERTIG
```
