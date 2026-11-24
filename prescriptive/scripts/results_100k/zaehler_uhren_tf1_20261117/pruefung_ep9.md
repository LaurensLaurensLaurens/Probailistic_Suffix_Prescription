# Prüfung ep9 (2026-11-17 22:34)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),1.162718317102342,2.424513772408548,0.06743441439469511,0.398779850811835
SE gepaart Median,1.7477784682913338,1.787437250439695,1.673587194170851,1.6801265797745784
Korrelation der Optionen Median,0.11533534998046571,0.13446057748951612,0.12662933576325852,0.23229715960929917
Vorzeichen wie AIPW (alle),0.77,0.93,0.49,0.41
Durchlaufzeit gezogen (Mittel),19.304050125945974,21.09066297840447,19.47024084394781,20.513263157907776
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe      Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept   Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept ZZU-TF1 ep9                  1.24 0.23                  2.85              0.34/0.34                                     0.75                   1.81             1.36/1.38
annehmen_nach_vervollstaendigung   Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung ZZU-TF1 ep9                  2.04 0.25                  2.45              0.33/0.35                                     0.87                   1.84             1.38/1.35
             nachlegen 1. Besuch   Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch ZZU-TF1 ep9                  0.01 0.25                  2.71              0.30/0.32                                     0.48                   1.71             0.30/0.43
          nachlegen ab 2. Besuch   Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch ZZU-TF1 ep9                 -1.64 0.35                  2.81              0.41/0.36                                     0.28                   1.63             0.54/0.63
                    handle_leads   Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads ZZU-TF1 ep9                 -0.12 0.21                  4.05              0.35/0.34                                     0.50                   1.70             1.39/1.39
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZU-TF1 ep9
Restzeit                                       1.55         0.35
Events                                         3.84         2.84
Angebote                                       1.29         1.13
versendet                                      1.07         0.32
O_Cancelled                                    1.25         0.78
Nachfass-Schleifen                             0.18         0.13
bis Nachfassen                                 0.02         0.16
Nachfassen bis Validierung                     1.92         0.57
Validierung bis Ende                          -0.53        -0.48
erreicht Validierung                          -0.05        -0.03
A_Pending                                     -0.03        -0.01
A_Cancelled                                    0.04         0.02
A_Denied                                      -0.01        -0.01
ohne                                           0.00        -0.00
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
             angebote  events  pending   rest  storniert  SE rest  Prefixe
modell                                                                    
ZZU-TF1 ep9     0.033   1.208   -0.009 -0.001       0.82    0.075      241
Restzeit ab erstem O_Sent (Tage):
gruppe       Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log             22.88               20.05             2.83
ZZU-TF1 ep9            18.18               17.80             0.37
SE der echten Differenz: 1.11
FERTIG
```
