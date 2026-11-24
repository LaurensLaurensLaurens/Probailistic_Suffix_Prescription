# Prüfung ep9 (2026-11-16 23:46)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),3.698349017364562,7.124882022698338,-5.1810784967057835,-3.5638226101014276
SE gepaart Median,3.9470220243656446,4.0996640739019945,4.274079682702483,3.888862356938624
Korrelation der Optionen Median,0.4604774620691633,0.45428273773444944,0.30425930160002757,0.5407852980514367
Vorzeichen wie AIPW (alle),0.815,0.96,0.87,0.77
Durchlaufzeit gezogen (Mittel),28.241935855728897,31.492845818342907,26.48748163020436,34.34824211578259
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe    Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept    ZZ ep9                  3.61 0.52                  4.85              0.29/0.33                                     0.80                   3.92             1.91/2.21
annehmen_nach_vervollstaendigung Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung    ZZ ep9                  6.71 0.63                  4.69              0.30/0.35                                     0.95                   4.09             1.82/2.01
             nachlegen 1. Besuch Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch    ZZ ep9                  3.76 0.68                  5.89              0.23/0.27                                     0.75                   4.05             0.56/0.80
          nachlegen ab 2. Besuch Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch    ZZ ep9                 -3.01 0.80                  0.38              0.30/0.29                                     0.28                   5.00             0.45/0.48
                    handle_leads Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads    ZZ ep9                  3.52 0.49                  4.90              0.33/0.36                                     0.80                   3.82             2.26/2.22
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZ ep9
Restzeit                                       1.55    4.55
Events                                         3.84    5.85
Angebote                                       1.29    1.22
versendet                                      1.07    0.82
O_Cancelled                                    1.25    0.20
Nachfass-Schleifen                             0.18    0.84
bis Nachfassen                                 0.02    0.38
Nachfassen bis Validierung                     1.92    2.37
Validierung bis Ende                          -0.53    2.11
erreicht Validierung                          -0.05    0.01
A_Pending                                     -0.03   -0.00
A_Cancelled                                    0.04    0.04
A_Denied                                      -0.01    0.01
ohne                                           0.00   -0.04
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
        angebote  events  pending   rest  storniert  SE rest  Prefixe
modell                                                               
ZZ ep9    -0.003   0.015    0.002 -0.508      0.061    0.234      241
Restzeit ab erstem O_Sent (Tage):
gruppe      Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log            22.88               20.05             2.83
ZZ ep9                27.03               23.32             3.70
SE der echten Differenz: 1.11
FERTIG
```
