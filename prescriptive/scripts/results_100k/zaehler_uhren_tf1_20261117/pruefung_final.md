# Prüfung final (2026-11-18 07:37)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),1.352869662360637,2.660370881650518,0.23215769554049082,-0.20545720959268457
SE gepaart Median,1.8074567778289763,1.8763505394074809,1.7493690381171283,1.787147900373197
Korrelation der Optionen Median,0.12838687919540542,0.11826797733183479,0.12044241023889624,0.2094000239345426
Vorzeichen wie AIPW (alle),0.77,0.925,0.445,0.54
Durchlaufzeit gezogen (Mittel),19.890966378987613,21.458361125273278,19.750757567228483,21.460050554181223
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe        Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept     Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept ZZU-TF1 final                  2.06 0.23                  3.34              0.34/0.35                                     0.87                   1.83             1.33/1.35
annehmen_nach_vervollstaendigung     Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung ZZU-TF1 final                  2.06 0.26                  1.96              0.34/0.36                                     0.83                   1.89             1.36/1.31
             nachlegen 1. Besuch     Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch ZZU-TF1 final                 -0.11 0.25                  3.10              0.31/0.32                                     0.43                   1.70             0.26/0.38
          nachlegen ab 2. Besuch     Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch ZZU-TF1 final                 -1.19 0.31                  2.34              0.37/0.32                                     0.32                   1.70             0.40/0.47
                    handle_leads     Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads ZZU-TF1 final                  0.41 0.23                  4.49              0.35/0.36                                     0.57                   1.78             1.38/1.37
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZU-TF1 final
Restzeit                                       1.55           0.29
Events                                         3.84           3.45
Angebote                                       1.29           1.11
versendet                                      1.07           0.45
O_Cancelled                                    1.25           0.89
Nachfass-Schleifen                             0.18           0.12
bis Nachfassen                                 0.02           0.16
Nachfassen bis Validierung                     1.92           0.28
Validierung bis Ende                          -0.53          -0.20
erreicht Validierung                          -0.05          -0.02
A_Pending                                     -0.03          -0.01
A_Cancelled                                    0.04           0.01
A_Denied                                      -0.01           0.00
ohne                                           0.00          -0.00
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
               angebote  events  pending   rest  storniert  SE rest  Prefixe
modell                                                                      
ZZU-TF1 final     0.025   1.151   -0.003 -0.139      0.862     0.08      241
Restzeit ab erstem O_Sent (Tage):
gruppe         Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log               22.88               20.05             2.83
ZZU-TF1 final            18.52               18.13             0.39
SE der echten Differenz: 1.11
FERTIG
```
