# Prüfung final (2026-11-17 07:25)

## Optionssignal (200 Testentscheidungen je Punkt, 100 Ziehungen)
```
,annehmen_bei_konzept,annehmen_nach_vervollstaendigung,angebot_nachlegen,handle_leads
Diff Mittel (Tage),4.251396889760685,5.8105682216069985,-3.6571465909439604,-2.4674836615590063
SE gepaart Median,4.792181270948934,4.965473862758492,4.929640829885832,4.941072505326558
Korrelation der Optionen Median,0.4514426683397885,0.4321462485910935,0.3666672619529794,0.45361120662290855
Vorzeichen wie AIPW (alle),0.76,0.855,0.78,0.67
Durchlaufzeit gezogen (Mittel),33.68843863496723,36.01725439785585,32.97519350228954,37.56311034433313
```

## Gegen die Simulator-Wahrheit (60 Zustände je Gruppe)
```
                          Gruppe    Quelle  Vorteil Regel-Option   SE  Events mehr (andere) Cancelled Regel/andere  Anteil Zustände richtig (100 Ziehungen)  MC-SE je Entscheidung Angebote Regel/andere
            annehmen_bei_konzept Simulator                  2.79 0.32                  0.91              0.32/0.36                                      NaN                    NaN                   NaN
            annehmen_bei_konzept  ZZ final                  4.24 0.75                  4.33              0.29/0.34                                     0.78                   4.87             1.79/2.05
annehmen_nach_vervollstaendigung Simulator                  2.40 0.30                  2.56              0.32/0.35                                      NaN                    NaN                   NaN
annehmen_nach_vervollstaendigung  ZZ final                  6.07 0.64                  3.51              0.31/0.36                                     0.88                   4.85             1.67/1.83
             nachlegen 1. Besuch Simulator                  2.11 0.23                  6.91              0.30/0.35                                      NaN                    NaN                   NaN
             nachlegen 1. Besuch  ZZ final                  2.74 0.66                  3.86              0.28/0.29                                     0.72                   4.46             0.53/0.67
          nachlegen ab 2. Besuch Simulator                  0.42 0.22                  5.84              0.34/0.36                                      NaN                    NaN                   NaN
          nachlegen ab 2. Besuch  ZZ final                 -3.18 0.75                 -0.19              0.30/0.28                                     0.22                   5.45             0.41/0.41
                    handle_leads Simulator                  1.32 0.29                  5.10              0.35/0.35                                      NaN                    NaN                   NaN
                    handle_leads  ZZ final                  3.40 0.61                  5.05              0.34/0.36                                     0.87                   5.12             2.09/2.10
FERTIG
```

## Nachlegen, 1. Besuch: zusätzliches Angebot minus senden (Simulator mit Bank danach)
```
Zusätzliches Angebot minus senden:
quelle                      Simulator (Bank danach)  ZZ final
Restzeit                                       1.55      2.31
Events                                         3.84      3.56
Angebote                                       1.29      1.11
versendet                                      1.07      0.58
O_Cancelled                                    1.25      0.50
Nachfass-Schleifen                             0.18      0.04
bis Nachfassen                                 0.02      0.32
Nachfassen bis Validierung                     1.92      2.63
Validierung bis Ende                          -0.53      0.18
erreicht Validierung                          -0.05     -0.02
A_Pending                                     -0.03      0.00
A_Cancelled                                    0.04     -0.01
A_Denied                                      -0.01      0.00
ohne                                           0.00     -0.00
```

## Gedächtnis für Angebote / Leads
```
Mit eingefügtem zweitem Angebot minus ohne (Mittel über Prefixe):
          angebote  events  pending   rest  storniert  SE rest  Prefixe
modell                                                                 
ZZ final     0.016   1.225    0.016  1.117      0.384    0.296      241
Restzeit ab erstem O_Sent (Tage):
gruppe      Lead bearbeitet  Lead zurückgezogen  Lead minus ohne
echter Log            22.88               20.05             2.83
ZZ final              33.50               31.30             2.19
SE der echten Differenz: 1.11
FERTIG
```
