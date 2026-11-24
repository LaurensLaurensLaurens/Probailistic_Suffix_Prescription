# Prüfung ep15 (2026-11-18 01:09)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),1.3170716534191809,2.3170778736530626,-0.06456556234963846,0.3141615276321426
SE gepaart Median,1.4647198157189694,1.4946038711875689,1.5738078263335256,0.8947453194723574
Korrelation der Optionen Median,0.35667176938602274,0.3212524175178251,0.2629641996573888,0.7788196528705522
Vorzeichen wie AIPW (alle),0.805,0.93,0.555,0.335
Durchlaufzeit gezogen (Mittel),19.550034052979562,21.045852680702083,20.29321404305339,21.02015081033476
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe            Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept         Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept ZZ-TF1-WP150 ep15                  1.12 0.24                  1.81              0.32/0.33                                     0.70                   1.53             1.45/1.35
annehmen_nach_vervollstaendigung         Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung ZZ-TF1-WP150 ep15                  1.81 0.18                  1.52              0.33/0.32                                     0.90                   1.56             1.43/1.32
             nachlegen 1. Besuch         Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch ZZ-TF1-WP150 ep15                  0.01 0.24                  2.29              0.32/0.31                                     0.53                   1.58             0.37/0.48
          nachlegen ab 2. Besuch         Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch ZZ-TF1-WP150 ep15                 -1.76 0.38                 -0.46              0.35/0.28                                     0.27                   1.56             0.32/0.34
                    handle_leads         Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads ZZ-TF1-WP150 ep15                 -0.24 0.13                 -0.39              0.31/0.32                                     0.40                   0.90             1.33/1.31
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZ-TF1-WP150 ep15
Restzeit                                       1.55               0.33
Events                                         3.84               2.23
Angebote                                       1.29               1.09
versendet                                      1.07               0.01
O_Cancelled                                    1.25               0.02
Nachfass-Schleifen                             0.18              -0.03
bis Nachfassen                                 0.02               0.17
Nachfassen bis Validierung                     1.92               0.21
Validierung bis Ende                          -0.53               0.20
erreicht Validierung                          -0.05              -0.00
A_Pending                                     -0.03               0.00
A_Cancelled                                    0.04              -0.01
A_Denied                                      -0.01               0.00
ohne                                           0.00               0.00
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
                   angebote  events  pending  rest  storniert  SE rest  Prefixe
modell                                                                         
ZZ-TF1-WP150 ep15    -0.002  -1.093   -0.007 -0.27      0.021     0.08      241
Restzeit ab erstem O_Sent (Tage):
gruppe             Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log                   22.88               20.05             2.83
ZZ-TF1-WP150 ep15            18.34               18.19             0.15
SE der echten Differenz: 1.11
FERTIG
```
