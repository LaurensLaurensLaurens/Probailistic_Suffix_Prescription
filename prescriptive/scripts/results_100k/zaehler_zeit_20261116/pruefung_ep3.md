# Prüfung ep3 (2026-11-16 16:55)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),0.8981818793237448,2.759538362185744,-0.3784564117185299,0.34655824821770026
SE gepaart Median,1.5624398799734442,1.66614855345079,1.632746860248694,1.271106474508577
Korrelation der Optionen Median,0.3532996095378315,0.3293747007865007,0.28665555480095717,0.5790098069855443
Vorzeichen wie AIPW (alle),0.69,0.96,0.545,0.4
Durchlaufzeit gezogen (Mittel),15.455382959680206,17.531449482989032,16.27519149315837,16.344625805519687
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe    Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept    ZZ ep3                  1.30 0.21                  2.57              0.29/0.30                                     0.78                   1.53             1.53/1.44
annehmen_nach_vervollstaendigung Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung    ZZ ep3                  2.45 0.23                  2.00              0.30/0.31                                     0.90                   1.55             1.49/1.34
             nachlegen 1. Besuch Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch    ZZ ep3                  0.30 0.23                  2.65              0.28/0.28                                     0.58                   1.68             0.37/0.50
          nachlegen ab 2. Besuch Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch    ZZ ep3                 -2.44 0.45                  0.79              0.38/0.29                                     0.20                   1.99             0.40/0.41
                    handle_leads Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads    ZZ ep3                 -0.37 0.18                  0.96              0.29/0.28                                     0.43                   1.28             1.45/1.46
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZ ep3
Restzeit                                       1.55    0.11
Events                                         3.84    2.49
Angebote                                       1.29    1.11
versendet                                      1.07    0.15
O_Cancelled                                    1.25    0.04
Nachfass-Schleifen                             0.18   -0.01
bis Nachfassen                                 0.02    0.21
Nachfassen bis Validierung                     1.92    0.24
Validierung bis Ende                          -0.53    0.10
erreicht Validierung                          -0.05   -0.01
A_Pending                                     -0.03    0.00
A_Cancelled                                    0.04   -0.00
A_Denied                                      -0.01   -0.00
ohne                                           0.00   -0.00
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
        angebote  events  pending  rest  storniert  SE rest  Prefixe
modell                                                              
ZZ ep3    -0.015  -1.355   -0.028 -1.12     -0.007    0.097      241
Restzeit ab erstem O_Sent (Tage):
gruppe      Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log            22.88               20.05             2.83
ZZ ep3                14.75               14.25             0.51
SE der echten Differenz: 1.11
FERTIG
```
