# Prüfung ep3 (2026-11-17 14:35)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),1.6119554178596964,3.6953576550756133,0.19626988916311908,0.7161831545296792
SE gepaart Median,1.7897582420594493,1.8283855055474363,1.7212928535852776,1.5550400897700771
Korrelation der Optionen Median,0.19500645391067198,0.19089628466838204,0.19912990037009737,0.41522516002648724
Vorzeichen wie AIPW (alle),0.85,0.985,0.42,0.345
Durchlaufzeit gezogen (Mittel),20.236828812174345,22.43364281129173,20.29085100296212,21.475612807246026
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe      Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept   Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept ZZU-TF1 ep3                  1.53 0.20                  3.05              0.33/0.33                                     0.85                   1.79             1.40/1.41
annehmen_nach_vervollstaendigung   Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung ZZU-TF1 ep3                  3.91 0.24                  3.62              0.32/0.33                                     1.00                   1.79             1.39/1.36
             nachlegen 1. Besuch   Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch ZZU-TF1 ep3                 -0.33 0.20                  1.96              0.31/0.31                                     0.43                   1.70             0.30/0.38
          nachlegen ab 2. Besuch   Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch ZZU-TF1 ep3                 -0.88 0.28                  2.07              0.34/0.29                                     0.35                   1.65             0.39/0.41
                    handle_leads   Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads ZZU-TF1 ep3                 -0.97 0.22                  0.84              0.33/0.32                                     0.32                   1.53             1.43/1.41
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZU-TF1 ep3
Restzeit                                       1.55        -0.14
Events                                         3.84         2.03
Angebote                                       1.29         1.10
versendet                                      1.07         0.19
O_Cancelled                                    1.25         0.05
Nachfass-Schleifen                             0.18         0.02
bis Nachfassen                                 0.02         0.11
Nachfassen bis Validierung                     1.92         0.26
Validierung bis Ende                          -0.53        -0.31
erreicht Validierung                          -0.05         0.01
A_Pending                                     -0.03         0.00
A_Cancelled                                    0.04        -0.01
A_Denied                                      -0.01         0.01
ohne                                           0.00         0.00
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
             angebote  events  pending   rest  storniert  SE rest  Prefixe
modell                                                                    
ZZU-TF1 ep3    -0.004   0.122      0.0 -0.196      0.056    0.047      241
Restzeit ab erstem O_Sent (Tage):
gruppe       Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log             22.88               20.05             2.83
ZZU-TF1 ep3            18.91               18.39             0.51
SE der echten Differenz: 1.11
FERTIG
```
