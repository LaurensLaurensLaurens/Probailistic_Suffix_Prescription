# Prüfung ep12 (2026-11-18 03:31)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),1.5271550392077862,2.84219895236893,0.03763042768730132,-0.34848715637016264
SE gepaart Median,1.861169690966281,1.9090812509971906,1.7693456261994605,1.8048652400851073
Korrelation der Optionen Median,0.1325305939471786,0.11377781369429106,0.1267043079273854,0.18732543195578458
Vorzeichen wie AIPW (alle),0.78,0.925,0.5,0.585
Durchlaufzeit gezogen (Mittel),19.856637325391418,21.434412788415568,19.694677913932306,21.41971718085293
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe       Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept    Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept ZZU-TF1 ep12                  1.81 0.22                  3.11              0.36/0.37                                     0.90                   1.89             1.34/1.35
annehmen_nach_vervollstaendigung    Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung ZZU-TF1 ep12                  2.12 0.26                  2.11              0.35/0.39                                     0.87                   1.86             1.34/1.30
             nachlegen 1. Besuch    Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch ZZU-TF1 ep12                 -0.09 0.27                  2.59              0.32/0.34                                     0.50                   1.76             0.29/0.40
          nachlegen ab 2. Besuch    Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch ZZU-TF1 ep12                 -0.81 0.30                  2.38              0.39/0.35                                     0.38                   1.72             0.46/0.53
                    handle_leads    Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads ZZU-TF1 ep12                  0.29 0.22                  4.46              0.38/0.37                                     0.58                   1.84             1.38/1.38
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZU-TF1 ep12
Restzeit                                       1.55          0.69
Events                                         3.84          3.40
Angebote                                       1.29          1.10
versendet                                      1.07          0.34
O_Cancelled                                    1.25          0.82
Nachfass-Schleifen                             0.18          0.11
bis Nachfassen                                 0.02          0.11
Nachfassen bis Validierung                     1.92          0.60
Validierung bis Ende                          -0.53          0.01
erreicht Validierung                          -0.05         -0.02
A_Pending                                     -0.03         -0.01
A_Cancelled                                    0.04          0.01
A_Denied                                      -0.01         -0.00
ohne                                           0.00          0.00
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
              angebote  events  pending  rest  storniert  SE rest  Prefixe
modell                                                                    
ZZU-TF1 ep12     0.029   1.257   -0.009  0.24      0.834    0.073      241
Restzeit ab erstem O_Sent (Tage):
gruppe        Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log              22.88               20.05             2.83
ZZU-TF1 ep12            18.35               17.96             0.39
SE der echten Differenz: 1.11
FERTIG
```
