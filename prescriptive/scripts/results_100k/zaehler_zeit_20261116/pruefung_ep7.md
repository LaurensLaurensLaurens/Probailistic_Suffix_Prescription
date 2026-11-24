# Prüfung ep7 (2026-11-16 21:34)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),0.2506818094346117,1.5479340277179807,-0.35524599517027844,-2.094452371270722
SE gepaart Median,2.3469538091535362,2.364491113894171,2.88852766518786,2.8283872176435008
Korrelation der Optionen Median,0.5175874575436088,0.5377945232657907,0.3365416222774005,0.40318009788270764
Vorzeichen wie AIPW (alle),0.55,0.695,0.555,0.765
Durchlaufzeit gezogen (Mittel),17.585644686000677,19.910015214540323,18.651968105384935,19.352904860127225
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe    Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept    ZZ ep7                  0.05 0.32                  1.44              0.23/0.23                                     0.52                   2.26             1.79/1.69
annehmen_nach_vervollstaendigung Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung    ZZ ep7                  1.21 0.41                  0.95              0.24/0.24                                     0.67                   2.54             1.78/1.60
             nachlegen 1. Besuch Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch    ZZ ep7                  0.15 0.37                  3.47              0.22/0.22                                     0.58                   2.95             0.49/0.72
          nachlegen ab 2. Besuch Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch    ZZ ep7                 -3.10 0.65                  0.83              0.29/0.25                                     0.28                   3.28             0.35/0.44
                    handle_leads Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads    ZZ ep7                  1.22 0.33                  4.78              0.23/0.24                                     0.67                   2.71             1.65/1.62
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZ ep7
Restzeit                                       1.55   -0.10
Events                                         3.84    3.20
Angebote                                       1.29    1.19
versendet                                      1.07    0.68
O_Cancelled                                    1.25    0.14
Nachfass-Schleifen                             0.18   -0.13
bis Nachfassen                                 0.02    0.49
Nachfassen bis Validierung                     1.92   -1.18
Validierung bis Ende                          -0.53    0.37
erreicht Validierung                          -0.05   -0.00
A_Pending                                     -0.03    0.00
A_Cancelled                                    0.04   -0.01
A_Denied                                      -0.01    0.00
ohne                                           0.00    0.00
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
        angebote  events  pending  rest  storniert  SE rest  Prefixe
modell                                                              
ZZ ep7    -0.018  -0.538   -0.001 -0.55      0.056    0.156      241
Restzeit ab erstem O_Sent (Tage):
gruppe      Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log            22.88               20.05             2.83
ZZ ep7                17.74               16.25             1.49
SE der echten Differenz: 1.11
FERTIG
```
