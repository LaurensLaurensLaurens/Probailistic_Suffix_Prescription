# Teacher Forcing bis zum ersten A_Complete an konzept (2026-11-24 01:30)

Bank-Läufe (Simulator v3), erste Besuche, Bank-Option; wahre Fortsetzung nach der Option bis einschließlich zum ersten A_Complete erzwungen, 10 Ziehungen je Fall. P_AC_naechstes: Softmax der mittleren Logits für A_Complete als nächstes Event (ohne Logit-Rauschen, Mittel über Dropout-Masken); gezogen: mit Rauschen und Übergangsregeln; wahr: Bank-Lauf. validate_fehlt: kein W_Validate application im freien Rest. Effekt v3: vervollständigen & < 11k → 30 % A_Complete doppelt (+1 ms, also direkt danach); annehmen & ≥ 11k → 40 % W_Validate ausgelassen.

```
                      option gruppe huelle  Faelle  P_AC_naechstes  gezogen_AC_naechstes  wahr_AC_naechstes  validate_fehlt_gezogen  validate_fehlt_wahr
         A_Accepted|complete  < 11k k3x4ol     120           0.001                 0.000              0.000                   0.369                0.267
         A_Accepted|complete  < 11k k3x4sd     120           0.001                 0.000              0.000                   0.342                0.267
         A_Accepted|complete  ≥ 11k k3x4ol     120           0.002                 0.000              0.000                   0.361                0.708
         A_Accepted|complete  ≥ 11k k3x4sd     120           0.001                 0.000              0.000                   0.348                0.708
W_Complete application|start  < 11k k3x4ol     120           0.109                 0.112              0.275                   0.162                0.333
W_Complete application|start  < 11k k3x4sd     120           0.102                 0.105              0.275                   0.176                0.333
W_Complete application|start  ≥ 11k k3x4ol     120           0.107                 0.105              0.000                   0.158                0.167
W_Complete application|start  ≥ 11k k3x4sd     120           0.098                 0.100              0.000                   0.170                0.167
```

Laufzeit 306 s
