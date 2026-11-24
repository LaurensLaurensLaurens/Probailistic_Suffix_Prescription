# Prüfung ep4 (2026-11-17 11:18)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),0.3788439067783196,3.5926265360736314,-0.48283726301402086,0.34643323112817315
SE gepaart Median,2.0505975073968115,2.2379773891990267,2.2940222469962546,2.0176175865945853
Korrelation der Optionen Median,0.3926528995560593,0.4197891513010736,0.25874137814602405,0.5015613506535841
Vorzeichen wie AIPW (alle),0.585,0.955,0.6,0.425
Durchlaufzeit gezogen (Mittel),16.136507828466648,19.3594743682074,16.826078174021887,17.027444199971548
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe    Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept    ZZ ep4                  0.61 0.27                  1.68              0.28/0.27                                     0.60                   1.91             1.50/1.37
annehmen_nach_vervollstaendigung Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung    ZZ ep4                  3.13 0.30                  1.40              0.29/0.30                                     0.93                   2.16             1.47/1.26
             nachlegen 1. Besuch Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch    ZZ ep4                  0.69 0.29                  3.33              0.26/0.27                                     0.58                   2.32             0.39/0.59
          nachlegen ab 2. Besuch Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch    ZZ ep4                 -1.27 0.58                  0.73              0.32/0.28                                     0.42                   2.70             0.38/0.38
                    handle_leads Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads    ZZ ep4                 -0.38 0.29                  1.33              0.28/0.28                                     0.45                   1.96             1.36/1.36
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZ ep4
Restzeit                                       1.55    0.72
Events                                         3.84    3.19
Angebote                                       1.29    1.20
versendet                                      1.07    0.47
O_Cancelled                                    1.25    0.04
Nachfass-Schleifen                             0.18   -0.09
bis Nachfassen                                 0.02    0.45
Nachfassen bis Validierung                     1.92    0.18
Validierung bis Ende                          -0.53    0.19
erreicht Validierung                          -0.05    0.01
A_Pending                                     -0.03    0.01
A_Cancelled                                    0.04    0.00
A_Denied                                      -0.01    0.00
ohne                                           0.00   -0.01
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
        angebote  events  pending   rest  storniert  SE rest  Prefixe
modell                                                               
ZZ ep4    -0.009  -0.447      0.0 -0.214     -0.011    0.118      241
Restzeit ab erstem O_Sent (Tage):
gruppe      Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log            22.88               20.05             2.83
ZZ ep4                15.49               15.24             0.25
SE der echten Differenz: 1.11
FERTIG
```
