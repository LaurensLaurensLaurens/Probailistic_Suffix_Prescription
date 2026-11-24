# Prüfung ep6 (2026-11-17 11:38)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),0.500699796955667,1.2129892573081396,-0.2659705961978359,-0.6380563899199069
SE gepaart Median,2.0272144731449533,2.1016768588994723,2.416358950961013,2.2434735736428335
Korrelation der Optionen Median,0.4600018424845661,0.46086180577733193,0.27055198188509755,0.4185058003030428
Vorzeichen wie AIPW (alle),0.59,0.73,0.53,0.57
Durchlaufzeit gezogen (Mittel),15.357323288728278,16.85735331997595,16.656228987088255,16.564710293776056
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe    Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept    ZZ ep6                  0.06 0.27                  1.79              0.23/0.23                                     0.50                   1.99             1.60/1.44
annehmen_nach_vervollstaendigung Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung    ZZ ep6                  0.95 0.27                  0.57              0.25/0.24                                     0.65                   2.08             1.59/1.25
             nachlegen 1. Besuch Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch    ZZ ep6                 -0.22 0.29                  2.51              0.22/0.23                                     0.43                   2.36             0.47/0.60
          nachlegen ab 2. Besuch Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch    ZZ ep6                 -1.92 0.65                  1.38              0.30/0.26                                     0.37                   3.10             0.34/0.41
                    handle_leads Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads    ZZ ep6                  1.48 0.27                  3.47              0.23/0.25                                     0.78                   2.26             1.40/1.37
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZ ep6
Restzeit                                       1.55    0.15
Events                                         3.84    3.00
Angebote                                       1.29    1.11
versendet                                      1.07    0.39
O_Cancelled                                    1.25    0.07
Nachfass-Schleifen                             0.18   -0.10
bis Nachfassen                                 0.02    0.28
Nachfassen bis Validierung                     1.92   -0.29
Validierung bis Ende                          -0.53    0.09
erreicht Validierung                          -0.05    0.01
A_Pending                                     -0.03    0.01
A_Cancelled                                    0.04   -0.00
A_Denied                                      -0.01    0.00
ohne                                           0.00   -0.01
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
        angebote  events  pending   rest  storniert  SE rest  Prefixe
modell                                                               
ZZ ep6    -0.007  -0.383    0.003 -0.326       0.04    0.123      241
Restzeit ab erstem O_Sent (Tage):
gruppe      Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log            22.88               20.05             2.83
ZZ ep6                15.73               14.75             0.98
SE der echten Differenz: 1.11
FERTIG
```
