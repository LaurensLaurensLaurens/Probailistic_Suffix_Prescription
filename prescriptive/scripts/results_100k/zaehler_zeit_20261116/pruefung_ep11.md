# Prüfung ep11 (2026-11-17 02:18)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),3.2635297411928423,7.814882740635621,-3.8961986088178717,-3.2067388367085203
SE gepaart Median,4.075361416475529,4.406498660502335,4.277242103348996,4.5665681138756575
Korrelation der Optionen Median,0.4502280972072056,0.4321583834383649,0.3400108310432861,0.4349170860565087
Vorzeichen wie AIPW (alle),0.74,0.96,0.82,0.775
Durchlaufzeit gezogen (Mittel),27.40574115634,31.015992194119782,26.149744320813557,32.77347223618518
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe    Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept   ZZ ep11                  3.54 0.59                  3.56              0.27/0.30                                     0.80                   4.02             1.72/1.96
annehmen_nach_vervollstaendigung Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung   ZZ ep11                  8.67 0.58                  4.42              0.27/0.34                                     0.92                   4.63             1.55/1.78
             nachlegen 1. Besuch Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch   ZZ ep11                  3.81 0.64                  3.87              0.23/0.26                                     0.72                   4.01             0.50/0.62
          nachlegen ab 2. Besuch Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch   ZZ ep11                 -7.60 1.02                 -0.13              0.32/0.30                                     0.15                   5.72             0.44/0.44
                    handle_leads Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads   ZZ ep11                  2.75 0.56                  5.50              0.30/0.33                                     0.77                   4.26             1.95/1.85
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZ ep11
Restzeit                                       1.55     1.84
Events                                         3.84     3.47
Angebote                                       1.29     1.09
versendet                                      1.07     0.35
O_Cancelled                                    1.25     0.25
Nachfass-Schleifen                             0.18     0.38
bis Nachfassen                                 0.02     0.40
Nachfassen bis Validierung                     1.92     1.27
Validierung bis Ende                          -0.53     0.03
erreicht Validierung                          -0.05     0.00
A_Pending                                     -0.03     0.01
A_Cancelled                                    0.04     0.02
A_Denied                                      -0.01     0.01
ohne                                           0.00    -0.03
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
         angebote  events  pending   rest  storniert  SE rest  Prefixe
modell                                                                
ZZ ep11     0.009   0.477      0.0  0.294      0.138    0.204      241
Restzeit ab erstem O_Sent (Tage):
gruppe      Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log            22.88               20.05             2.83
ZZ ep11               26.62               23.85             2.77
SE der echten Differenz: 1.11
FERTIG
```
