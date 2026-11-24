# Teacher Forcing bis zum ersten A_Complete an konzept – k3x4ol, simbasis (2026-11-24 08:46)

Bank-Läufe (Simulator v3), erste Besuche, Bank-Option; wahre Fortsetzung nach der Option bis einschließlich zum ersten A_Complete erzwungen, 10 Ziehungen je Fall. P_AC_naechstes: Softmax der mittleren Logits für A_Complete als nächstes Event (ohne Logit-Rauschen, Mittel über Dropout-Masken); gezogen: mit Rauschen und Übergangsregeln; wahr: Bank-Lauf. validate_fehlt: kein W_Validate application im freien Rest. Effekt v3: vervollständigen & < 11k → 30 % A_Complete doppelt (+1 ms, also direkt danach); annehmen & ≥ 11k → 40 % W_Validate ausgelassen.

```
                      option gruppe   huelle  Faelle  P_AC_naechstes  gezogen_AC_naechstes  wahr_AC_naechstes  validate_fehlt_gezogen  validate_fehlt_wahr
         A_Accepted|complete  < 11k   k3x4ol     120           0.001                 0.000              0.000                   0.369                0.267
         A_Accepted|complete  < 11k simbasis     120           0.024                 0.026              0.000                   0.250                0.267
         A_Accepted|complete  ≥ 11k   k3x4ol     120           0.002                 0.000              0.000                   0.361                0.708
         A_Accepted|complete  ≥ 11k simbasis     120           0.030                 0.035              0.000                   0.256                0.708
W_Complete application|start  < 11k   k3x4ol     120           0.109                 0.112              0.275                   0.162                0.333
W_Complete application|start  < 11k simbasis     120           0.113                 0.123              0.275                   0.224                0.333
W_Complete application|start  ≥ 11k   k3x4ol     120           0.107                 0.105              0.000                   0.158                0.167
W_Complete application|start  ≥ 11k simbasis     120           0.108                 0.109              0.000                   0.226                0.167
```

Laufzeit 317 s
