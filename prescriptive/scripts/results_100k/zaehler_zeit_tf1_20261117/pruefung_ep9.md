# Prüfung ep9 (2026-11-17 12:20)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),1.6576152296434947,3.6083920351502625,0.227378008205484,0.27035392314884654
SE gepaart Median,1.793717082535239,1.8765632679929514,1.8044792283112026,1.7586398605041669
Korrelation der Optionen Median,0.14491296258704522,0.1595906833755385,0.11770915539113466,0.2424942027645749
Vorzeichen wie AIPW (alle),0.805,0.97,0.455,0.41
Durchlaufzeit gezogen (Mittel),20.231330169617276,22.20374028184816,20.191972674023894,21.68508459297438
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe     Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept  Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept ZZ-TF1 ep9                  2.02 0.22                  3.21              0.35/0.35                                     0.87                   1.87             1.39/1.39
annehmen_nach_vervollstaendigung  Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung ZZ-TF1 ep9                  2.86 0.26                  3.01              0.34/0.36                                     0.92                   1.88             1.39/1.34
             nachlegen 1. Besuch  Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch ZZ-TF1 ep9                 -0.30 0.21                  2.77              0.32/0.36                                     0.47                   1.76             0.25/0.41
          nachlegen ab 2. Besuch  Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch ZZ-TF1 ep9                  0.35 0.27                  2.29              0.39/0.40                                     0.57                   1.75             0.40/0.52
                    handle_leads  Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads ZZ-TF1 ep9                 -0.27 0.22                  2.97              0.36/0.34                                     0.53                   1.73             1.40/1.37
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZ-TF1 ep9
Restzeit                                       1.55        0.39
Events                                         3.84        3.23
Angebote                                       1.29        1.15
versendet                                      1.07        0.55
O_Cancelled                                    1.25        0.78
Nachfass-Schleifen                             0.18        0.12
bis Nachfassen                                 0.02        0.14
Nachfassen bis Validierung                     1.92        0.50
Validierung bis Ende                          -0.53       -0.04
erreicht Validierung                          -0.05       -0.04
A_Pending                                     -0.03       -0.03
A_Cancelled                                    0.04        0.03
A_Denied                                      -0.01        0.00
ohne                                           0.00       -0.00
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
            angebote  events  pending   rest  storniert  SE rest  Prefixe
modell                                                                   
ZZ-TF1 ep9     0.022   0.546   -0.023 -0.132      0.808    0.078      241
Restzeit ab erstem O_Sent (Tage):
gruppe      Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log            22.88               20.05             2.83
ZZ-TF1 ep9            19.19               18.65             0.53
SE der echten Differenz: 1.11
FERTIG
```
