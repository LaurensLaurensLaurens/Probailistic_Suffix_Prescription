# Prüfung ep6 (2026-11-17 08:47)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),1.7574064293265519,3.450304374586127,0.14831238359371748,0.6374518627336966
SE gepaart Median,1.6909375274341372,1.7427400624734073,1.6651501712057672,1.5907113367328432
Korrelation der Optionen Median,0.13877314489356424,0.15261045076408467,0.1530697021207512,0.3003607930175799
Vorzeichen wie AIPW (alle),0.875,0.955,0.465,0.32
Durchlaufzeit gezogen (Mittel),19.508219518501523,21.440193456018143,19.69328505577617,20.71417354572739
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe     Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept  Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept ZZ-TF1 ep6                  1.74 0.20                  2.78              0.35/0.34                                     0.87                   1.71             1.41/1.38
annehmen_nach_vervollstaendigung  Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung ZZ-TF1 ep6                  3.43 0.25                  2.67              0.34/0.35                                     0.97                   1.79             1.42/1.30
             nachlegen 1. Besuch  Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch ZZ-TF1 ep6                 -0.27 0.22                  2.78              0.32/0.33                                     0.43                   1.59             0.27/0.42
          nachlegen ab 2. Besuch  Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch ZZ-TF1 ep6                 -0.10 0.30                  2.57              0.39/0.36                                     0.50                   1.57             0.36/0.45
                    handle_leads  Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads ZZ-TF1 ep6                 -0.96 0.19                  1.82              0.35/0.34                                     0.27                   1.62             1.40/1.37
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZ-TF1 ep6
Restzeit                                       1.55        0.30
Events                                         3.84        2.90
Angebote                                       1.29        1.14
versendet                                      1.07        0.25
O_Cancelled                                    1.25        0.49
Nachfass-Schleifen                             0.18        0.10
bis Nachfassen                                 0.02        0.14
Nachfassen bis Validierung                     1.92        0.29
Validierung bis Ende                          -0.53       -0.09
erreicht Validierung                          -0.05       -0.02
A_Pending                                     -0.03       -0.01
A_Cancelled                                    0.04        0.02
A_Denied                                      -0.01       -0.01
ohne                                           0.00       -0.00
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
            angebote  events  pending   rest  storniert  SE rest  Prefixe
modell                                                                   
ZZ-TF1 ep6     0.018    0.63   -0.012 -0.169      0.619    0.064      241
Restzeit ab erstem O_Sent (Tage):
gruppe      Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log            22.88               20.05             2.83
ZZ-TF1 ep6            18.58               17.93             0.65
SE der echten Differenz: 1.11
FERTIG
```
