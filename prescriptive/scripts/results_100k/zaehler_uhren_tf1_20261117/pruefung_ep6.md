# Prüfung ep6 (2026-11-17 18:30)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),1.2795606507275905,3.1724351262941832,0.1368092784467007,0.8571120674868056
SE gepaart Median,1.6986122732967304,1.8086429927480407,1.6708194203638436,1.5991342400606001
Korrelation der Optionen Median,0.14209906044135345,0.15467233944386333,0.15688685481004394,0.29840201616536344
Vorzeichen wie AIPW (alle),0.785,0.97,0.465,0.295
Durchlaufzeit gezogen (Mittel),19.173415853433344,21.342313257971146,19.548952451925267,20.45134738179138
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe      Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept   Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept ZZU-TF1 ep6                  1.36 0.20                  2.82              0.35/0.36                                     0.82                   1.71             1.38/1.37
annehmen_nach_vervollstaendigung   Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung ZZU-TF1 ep6                  2.71 0.25                  2.69              0.34/0.37                                     0.95                   1.83             1.37/1.32
             nachlegen 1. Besuch   Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch ZZU-TF1 ep6                 -0.15 0.22                  2.73              0.31/0.33                                     0.47                   1.59             0.27/0.43
          nachlegen ab 2. Besuch   Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch ZZU-TF1 ep6                 -1.66 0.28                  3.03              0.41/0.34                                     0.27                   1.52             0.52/0.68
                    handle_leads   Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads ZZU-TF1 ep6                 -0.86 0.26                  2.57              0.37/0.35                                     0.35                   1.59             1.38/1.37
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZU-TF1 ep6
Restzeit                                       1.55         0.59
Events                                         3.84         2.98
Angebote                                       1.29         1.15
versendet                                      1.07         0.28
O_Cancelled                                    1.25         0.53
Nachfass-Schleifen                             0.18         0.12
bis Nachfassen                                 0.02         0.16
Nachfassen bis Validierung                     1.92         0.63
Validierung bis Ende                          -0.53        -0.22
erreicht Validierung                          -0.05        -0.01
A_Pending                                     -0.03        -0.01
A_Cancelled                                    0.04         0.01
A_Denied                                      -0.01         0.00
ohne                                           0.00        -0.00
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
             angebote  events  pending   rest  storniert  SE rest  Prefixe
modell                                                                    
ZZU-TF1 ep6      0.01   0.987   -0.001 -0.168      0.631    0.067      241
Restzeit ab erstem O_Sent (Tage):
gruppe       Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log             22.88               20.05             2.83
ZZU-TF1 ep6            18.35               17.76             0.59
SE der echten Differenz: 1.11
FERTIG
```
