# Prüfung final (2026-11-17 20:08)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),1.350045001198022,3.3102317849339067,0.14952571101687828,-0.06429934452133886
SE gepaart Median,1.8546800322550303,1.9071889919017433,1.83592043854179,1.8002537108909464
Korrelation der Optionen Median,0.15324144627699116,0.1361110139221257,0.13510866641323338,0.21092753482345844
Vorzeichen wie AIPW (alle),0.74,0.96,0.49,0.505
Durchlaufzeit gezogen (Mittel),20.70480992474684,22.5525461074314,20.74140709812052,21.922539585030925
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe       Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept    Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept ZZ-TF1 final                  1.36 0.23                  2.84              0.35/0.34                                     0.80                   1.85             1.35/1.36
annehmen_nach_vervollstaendigung    Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung ZZ-TF1 final                  2.29 0.27                  1.75              0.32/0.36                                     0.77                   1.93             1.36/1.23
             nachlegen 1. Besuch    Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch ZZ-TF1 final                  0.12 0.24                  3.26              0.30/0.33                                     0.52                   1.75             0.27/0.39
          nachlegen ab 2. Besuch    Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch ZZ-TF1 final                  0.54 0.38                  1.94              0.38/0.40                                     0.68                   1.83             0.36/0.47
                    handle_leads    Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads ZZ-TF1 final                 -0.03 0.24                  3.01              0.35/0.34                                     0.52                   1.83             1.38/1.35
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZ-TF1 final
Restzeit                                       1.55          0.34
Events                                         3.84          3.09
Angebote                                       1.29          1.14
versendet                                      1.07          0.55
O_Cancelled                                    1.25          0.98
Nachfass-Schleifen                             0.18          0.19
bis Nachfassen                                 0.02          0.15
Nachfassen bis Validierung                     1.92          0.49
Validierung bis Ende                          -0.53         -0.26
erreicht Validierung                          -0.05         -0.04
A_Pending                                     -0.03         -0.03
A_Cancelled                                    0.04          0.03
A_Denied                                      -0.01         -0.00
ohne                                           0.00         -0.00
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
              angebote  events  pending   rest  storniert  SE rest  Prefixe
modell                                                                     
ZZ-TF1 final     0.035   0.981   -0.023 -0.011      0.928    0.085      241
Restzeit ab erstem O_Sent (Tage):
gruppe        Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log              22.88               20.05             2.83
ZZ-TF1 final            19.16               18.95             0.21
SE der echten Differenz: 1.11
FERTIG
```
