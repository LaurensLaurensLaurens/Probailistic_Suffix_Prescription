# Prüfung ep90 (2026-11-18 10:43)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),1.7120355083945276,3.0084463405006137,0.02310952014973048,0.7767394380403109
SE gepaart Median,1.5849982173982171,1.6335117731023232,1.6154259171517187,1.3120508292806632
Korrelation der Optionen Median,0.2529529832062548,0.2267056620138989,0.19540185293312323,0.5517880418344905
Vorzeichen wie AIPW (alle),0.85,0.95,0.465,0.295
Durchlaufzeit gezogen (Mittel),19.45974617860918,21.68140152679578,20.270340999318346,21.5069607226282
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe            Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept         Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept ZZ-TF1-WP150 ep90                  1.45 0.19                  2.45              0.32/0.32                                     0.82                   1.58             1.44/1.38
annehmen_nach_vervollstaendigung         Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung ZZ-TF1-WP150 ep90                  2.56 0.21                  2.53              0.34/0.34                                     0.93                   1.63             1.42/1.34
             nachlegen 1. Besuch         Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch ZZ-TF1-WP150 ep90                  0.04 0.20                  2.39              0.32/0.32                                     0.48                   1.58             0.31/0.39
          nachlegen ab 2. Besuch         Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch ZZ-TF1-WP150 ep90                 -0.25 0.40                  1.62              0.37/0.32                                     0.52                   1.69             0.33/0.43
                    handle_leads         Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads ZZ-TF1-WP150 ep90                 -0.82 0.17                 -0.00              0.34/0.33                                     0.32                   1.27             1.38/1.33
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZ-TF1-WP150 ep90
Restzeit                                       1.55              -0.09
Events                                         3.84               1.96
Angebote                                       1.29               1.08
versendet                                      1.07               0.17
O_Cancelled                                    1.25               0.02
Nachfass-Schleifen                             0.18               0.01
bis Nachfassen                                 0.02               0.15
Nachfassen bis Validierung                     1.92               0.33
Validierung bis Ende                          -0.53              -0.36
erreicht Validierung                          -0.05               0.00
A_Pending                                     -0.03               0.01
A_Cancelled                                    0.04              -0.00
A_Denied                                      -0.01               0.00
ohne                                           0.00              -0.00
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
                   angebote  events  pending   rest  storniert  SE rest  Prefixe
modell                                                                          
ZZ-TF1-WP150 ep90    -0.017  -0.509    0.001 -0.234      0.025    0.074      241
Restzeit ab erstem O_Sent (Tage):
gruppe             Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log                   22.88               20.05             2.83
ZZ-TF1-WP150 ep90            18.77               18.55             0.22
SE der echten Differenz: 1.11
FERTIG
```
