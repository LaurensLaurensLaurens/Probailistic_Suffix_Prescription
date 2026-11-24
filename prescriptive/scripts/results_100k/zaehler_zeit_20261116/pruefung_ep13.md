# Prüfung ep13 (2026-11-17 04:49)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),3.454171206927688,5.5180908146867536,-2.1457388776524393,-1.9599455589502646
SE gepaart Median,3.9816895581323197,4.018017540569636,4.00725921844314,4.204534742266375
Korrelation der Optionen Median,0.40324614734104536,0.41712694235374703,0.35363480202145803,0.4425124348123839
Vorzeichen wie AIPW (alle),0.78,0.925,0.685,0.67
Durchlaufzeit gezogen (Mittel),26.1104002617484,28.053109728113785,25.85837320091943,30.76552182664009
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe    Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept   ZZ ep13                  3.79 0.63                  3.26              0.24/0.28                                     0.78                   4.11             1.66/1.90
annehmen_nach_vervollstaendigung Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung   ZZ ep13                  6.71 0.61                  3.47              0.24/0.30                                     0.93                   4.01             1.56/1.73
             nachlegen 1. Besuch Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch   ZZ ep13                  1.43 0.49                  3.07              0.22/0.23                                     0.70                   3.76             0.45/0.53
          nachlegen ab 2. Besuch Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch   ZZ ep13                 -5.19 0.86                 -0.99              0.26/0.24                                     0.22                   4.91             0.41/0.40
                    handle_leads Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads   ZZ ep13                  2.17 0.59                  4.75              0.27/0.29                                     0.70                   3.92             1.88/1.83
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZ ep13
Restzeit                                       1.55     1.31
Events                                         3.84     3.04
Angebote                                       1.29     1.07
versendet                                      1.07     0.46
O_Cancelled                                    1.25     0.41
Nachfass-Schleifen                             0.18     0.11
bis Nachfassen                                 0.02     0.25
Nachfassen bis Validierung                     1.92     1.25
Validierung bis Ende                          -0.53    -0.09
erreicht Validierung                          -0.05    -0.02
A_Pending                                     -0.03     0.01
A_Cancelled                                    0.04     0.00
A_Denied                                      -0.01    -0.00
ohne                                           0.00    -0.01
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
         angebote  events  pending   rest  storniert  SE rest  Prefixe
modell                                                                
ZZ ep13     0.023   1.226    0.019  0.708      0.306    0.215      241
Restzeit ab erstem O_Sent (Tage):
gruppe      Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log            22.88               20.05             2.83
ZZ ep13               27.17               24.39             2.78
SE der echten Differenz: 1.11
FERTIG
```
