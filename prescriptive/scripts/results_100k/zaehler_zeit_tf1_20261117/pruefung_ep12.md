# Prüfung ep12 (2026-11-17 16:07)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),1.446419860944338,3.50752633419207,0.17795518260731932,-0.30673982980902
SE gepaart Median,1.797761707013266,1.8654492184482654,1.7755873409142535,1.7916545742230023
Korrelation der Optionen Median,0.1446966350077342,0.13302031740443757,0.10886755925255645,0.20237780276954792
Vorzeichen wie AIPW (alle),0.765,0.97,0.495,0.6
Durchlaufzeit gezogen (Mittel),20.07034918957843,22.00247333942247,20.0493997406982,21.830281427029306
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe      Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept   Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept ZZ-TF1 ep12                  1.61 0.20                  2.56              0.36/0.36                                     0.83                   1.81             1.39/1.38
annehmen_nach_vervollstaendigung   Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung ZZ-TF1 ep12                  3.04 0.27                  2.80              0.35/0.37                                     0.93                   1.89             1.39/1.28
             nachlegen 1. Besuch   Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch ZZ-TF1 ep12                 -0.13 0.23                  3.22              0.32/0.35                                     0.47                   1.70             0.26/0.41
          nachlegen ab 2. Besuch   Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch ZZ-TF1 ep12                  1.44 0.26                  2.62              0.34/0.38                                     0.77                   1.76             0.35/0.48
                    handle_leads   Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads ZZ-TF1 ep12                  0.32 0.22                  3.50              0.37/0.35                                     0.62                   1.80             1.40/1.35
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZ-TF1 ep12
Restzeit                                       1.55         0.42
Events                                         3.84         3.75
Angebote                                       1.29         1.17
versendet                                      1.07         0.68
O_Cancelled                                    1.25         0.98
Nachfass-Schleifen                             0.18         0.17
bis Nachfassen                                 0.02         0.13
Nachfassen bis Validierung                     1.92         0.53
Validierung bis Ende                          -0.53        -0.25
erreicht Validierung                          -0.05        -0.03
A_Pending                                     -0.03        -0.02
A_Cancelled                                    0.04         0.02
A_Denied                                      -0.01        -0.00
ohne                                           0.00        -0.00
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
             angebote  events  pending  rest  storniert  SE rest  Prefixe
modell                                                                   
ZZ-TF1 ep12     0.024   0.733    -0.01 -0.06      0.887    0.077      241
Restzeit ab erstem O_Sent (Tage):
gruppe       Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log             22.88               20.05             2.83
ZZ-TF1 ep12            18.76               18.32             0.45
SE der echten Differenz: 1.11
FERTIG
```
