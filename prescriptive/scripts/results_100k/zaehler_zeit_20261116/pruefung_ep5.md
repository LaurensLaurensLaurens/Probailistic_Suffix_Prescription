# Prüfung ep5 (2026-11-17 11:28)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),0.545264986730377,1.7036920396690882,-0.40363009899034175,-0.4070061767695239
SE gepaart Median,2.5523033692512,2.6820902633805073,2.881452036326369,2.6372865536930865
Korrelation der Optionen Median,0.42692587327447457,0.4269964140919214,0.2722347991823646,0.4166490104776689
Vorzeichen wie AIPW (alle),0.575,0.75,0.585,0.56
Durchlaufzeit gezogen (Mittel),17.537850592138444,19.48029211134483,18.798881449369443,18.257916563261006
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe    Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept    ZZ ep5                  0.79 0.31                  1.72              0.26/0.26                                     0.70                   2.35             1.57/1.39
annehmen_nach_vervollstaendigung Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung    ZZ ep5                  1.84 0.38                  0.96              0.28/0.28                                     0.75                   2.61             1.51/1.24
             nachlegen 1. Besuch Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch    ZZ ep5                  1.01 0.34                  3.14              0.24/0.25                                     0.67                   2.78             0.39/0.55
          nachlegen ab 2. Besuch Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch    ZZ ep5                 -0.06 0.48                  1.30              0.26/0.25                                     0.50                   2.55             0.37/0.41
                    handle_leads Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads    ZZ ep5                  0.44 0.33                  2.65              0.26/0.27                                     0.57                   2.56             1.39/1.41
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZ ep5
Restzeit                                       1.55    0.78
Events                                         3.84    3.25
Angebote                                       1.29    1.15
versendet                                      1.07    0.48
O_Cancelled                                    1.25    0.05
Nachfass-Schleifen                             0.18   -0.07
bis Nachfassen                                 0.02    0.32
Nachfassen bis Validierung                     1.92   -0.16
Validierung bis Ende                          -0.53    0.31
erreicht Validierung                          -0.05    0.01
A_Pending                                     -0.03    0.00
A_Cancelled                                    0.04    0.01
A_Denied                                      -0.01    0.00
ohne                                           0.00   -0.02
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
        angebote  events  pending   rest  storniert  SE rest  Prefixe
modell                                                               
ZZ ep5     0.001  -0.118      0.0 -0.067      0.019    0.116      241
Restzeit ab erstem O_Sent (Tage):
gruppe      Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log            22.88               20.05             2.83
ZZ ep5                17.23               16.70             0.53
SE der echten Differenz: 1.11
FERTIG
```
