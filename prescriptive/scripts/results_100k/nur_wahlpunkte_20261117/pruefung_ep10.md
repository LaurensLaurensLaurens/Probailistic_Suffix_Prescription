# Prüfung ep10 (2026-11-17 21:51)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),0.790417010810479,1.12268579399481,0.005821394453762237,0.14335940624509963
SE gepaart Median,1.2532483122118707,1.3400079775176825,1.4224106320019685,0.7833498999672586
Korrelation der Optionen Median,0.4964787417611635,0.40952268585322094,0.34161598652779124,0.8048527527290767
Vorzeichen wie AIPW (alle),0.745,0.8,0.475,0.43
Durchlaufzeit gezogen (Mittel),19.43031538315193,20.467894466133306,19.928583569096773,20.2872892553283
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe         Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept      Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept ZZ-TF1-WP ep10                  0.98 0.19                  1.28              0.31/0.29                                     0.70                   1.32             1.64/1.43
annehmen_nach_vervollstaendigung      Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung ZZ-TF1-WP ep10                  1.18 0.15                  1.15              0.30/0.30                                     0.83                   1.32             1.60/1.36
             nachlegen 1. Besuch      Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch ZZ-TF1-WP ep10                 -0.08 0.18                  0.93              0.31/0.31                                     0.47                   1.45             0.64/0.75
          nachlegen ab 2. Besuch      Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch ZZ-TF1-WP ep10                 -3.95 0.40                 -1.11              0.32/0.19                                     0.10                   1.41             0.31/0.32
                    handle_leads      Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads ZZ-TF1-WP ep10                 -0.17 0.11                 -0.23              0.30/0.30                                     0.45                   0.80             1.41/1.41
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZ-TF1-WP ep10
Restzeit                                       1.55           -0.04
Events                                         3.84            1.10
Angebote                                       1.29            1.10
versendet                                      1.07           -0.22
O_Cancelled                                    1.25           -0.02
Nachfass-Schleifen                             0.18           -0.08
bis Nachfassen                                 0.02            0.36
Nachfassen bis Validierung                     1.92           -0.39
Validierung bis Ende                          -0.53            0.01
erreicht Validierung                          -0.05           -0.01
A_Pending                                     -0.03            0.00
A_Cancelled                                    0.04           -0.01
A_Denied                                      -0.01           -0.01
ohne                                           0.00            0.01
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
                angebote  events  pending   rest  storniert  SE rest  Prefixe
modell                                                                       
ZZ-TF1-WP ep10    -0.092  -2.986   -0.022 -0.921     -0.004    0.121      241
Restzeit ab erstem O_Sent (Tage):
gruppe          Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log                22.88               20.05             2.83
ZZ-TF1-WP ep10            17.84               18.08            -0.24
SE der echten Differenz: 1.11
FERTIG
```
