# Prüfung ep5 (2026-11-17 21:08)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),0.007211271762842371,0.1977613017383085,0.36416848014004666,-0.26662300109595555
SE gepaart Median,0.8452693151616406,0.996107577891141,1.1242633309730152,0.7576990453322527
Korrelation der Optionen Median,0.8090343050860879,0.7210623127267699,0.6564918991397217,0.8510653362217047
Vorzeichen wie AIPW (alle),0.5,0.585,0.355,0.625
Durchlaufzeit gezogen (Mittel),21.988131835864596,22.47654193013657,22.273472079269737,22.072568986372755
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe        Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept     Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept ZZ-TF1-WP ep5                  0.19 0.13                  0.24              0.29/0.29                                     0.58                   0.79             1.73/1.48
annehmen_nach_vervollstaendigung     Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung ZZ-TF1-WP ep5                  0.46 0.11                 -0.04              0.28/0.28                                     0.73                   0.92             1.71/1.43
             nachlegen 1. Besuch     Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch ZZ-TF1-WP ep5                 -0.49 0.15                 -0.79              0.29/0.28                                     0.30                   1.12             1.30/1.21
          nachlegen ab 2. Besuch     Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch ZZ-TF1-WP ep5                 -3.58 0.39                 -0.74              0.26/0.17                                     0.08                   1.28             0.46/0.47
                    handle_leads     Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads ZZ-TF1-WP ep5                  0.10 0.09                  0.49              0.26/0.28                                     0.57                   0.76             1.49/1.52
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZ-TF1-WP ep5
Restzeit                                       1.55          -0.31
Events                                         3.84          -0.34
Angebote                                       1.29           0.95
versendet                                      1.07          -0.78
O_Cancelled                                    1.25          -0.00
Nachfass-Schleifen                             0.18          -0.18
bis Nachfassen                                 0.02           0.83
Nachfassen bis Validierung                     1.92          -1.50
Validierung bis Ende                          -0.53           0.34
erreicht Validierung                          -0.05           0.01
A_Pending                                     -0.03           0.01
A_Cancelled                                    0.04          -0.01
A_Denied                                      -0.01          -0.00
ohne                                           0.00           0.00
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
               angebote  events  pending   rest  storniert  SE rest  Prefixe
modell                                                                      
ZZ-TF1-WP ep5     -0.07  -1.334   -0.004 -0.254     -0.001    0.092      241
Restzeit ab erstem O_Sent (Tage):
gruppe         Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log               22.88               20.05             2.83
ZZ-TF1-WP ep5            20.05               20.28            -0.23
SE der echten Differenz: 1.11
FERTIG
```
