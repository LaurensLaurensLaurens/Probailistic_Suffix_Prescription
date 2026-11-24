# Prüfung ep3 (2026-11-17 05:13)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),1.6121567133592856,3.4378825790807763,0.15664085447963275,0.7934624176365336
SE gepaart Median,1.8462563440908872,1.9202051503315243,1.8184566295791573,1.5840051015872878
Korrelation der Optionen Median,0.1798513907742999,0.2130985111951901,0.17942325654061528,0.4254618055974202
Vorzeichen wie AIPW (alle),0.815,0.965,0.505,0.29
Durchlaufzeit gezogen (Mittel),21.07677158147101,23.106137510420055,21.11341983409657,22.218884415081533
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe     Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept  Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept ZZ-TF1 ep3                  1.77 0.21                  3.13              0.34/0.34                                     0.82                   1.83             1.42/1.41
annehmen_nach_vervollstaendigung  Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung ZZ-TF1 ep3                  2.92 0.23                  2.47              0.33/0.34                                     0.98                   1.88             1.43/1.35
             nachlegen 1. Besuch  Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch ZZ-TF1 ep3                 -0.36 0.25                  2.14              0.31/0.32                                     0.48                   1.80             0.30/0.38
          nachlegen ab 2. Besuch  Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch ZZ-TF1 ep3                 -1.64 0.33                  1.18              0.36/0.28                                     0.25                   1.77             0.31/0.30
                    handle_leads  Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads ZZ-TF1 ep3                 -1.02 0.21                  0.66              0.33/0.33                                     0.27                   1.62             1.43/1.41
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZ-TF1 ep3
Restzeit                                       1.55        0.09
Events                                         3.84        2.43
Angebote                                       1.29        1.10
versendet                                      1.07        0.26
O_Cancelled                                    1.25        0.05
Nachfass-Schleifen                             0.18       -0.01
bis Nachfassen                                 0.02        0.13
Nachfassen bis Validierung                     1.92        0.37
Validierung bis Ende                          -0.53        0.10
erreicht Validierung                          -0.05       -0.01
A_Pending                                     -0.03        0.00
A_Cancelled                                    0.04       -0.00
A_Denied                                      -0.01       -0.00
ohne                                           0.00        0.00
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
            angebote  events  pending   rest  storniert  SE rest  Prefixe
modell                                                                   
ZZ-TF1 ep3    -0.013  -0.458   -0.013 -0.552      0.024    0.092      241
Restzeit ab erstem O_Sent (Tage):
gruppe      Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log            22.88               20.05             2.83
ZZ-TF1 ep3            19.62               19.18             0.44
SE der echten Differenz: 1.11
FERTIG
```
