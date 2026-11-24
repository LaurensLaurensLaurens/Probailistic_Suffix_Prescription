# Konzept-Effekte in gezogenen Suffixen (k3x4ol, keyed_noise) je Betragsgruppe (2026-11-23 23:09)

Effekt v3: vervollständigen & < 11k → 30 % A_Complete doppelt; annehmen & ≥ 11k → 40 % W_Validate application ausgelassen.
240 Zustände (erste Besuche, Bank-Läufe online_konformanz/bank_seed*), 30 Ziehungen je Option. Anteile je Fall ab dem Wahlpunkt.

```
                 quelle                       option gruppe     n  a_complete_doppelt  validate_fehlt
         gezogen k3x4ol                   A_Accepted  < 11k  3600               0.005           0.338
         gezogen k3x4ol                   A_Accepted  ≥ 11k  3600               0.003           0.345
         gezogen k3x4ol W_Complete application|start  < 11k  3600               0.101           0.160
         gezogen k3x4ol W_Complete application|start  ≥ 11k  3600               0.081           0.150
Trainings-Log x4 (Bank)                   A_Accepted  < 11k 18135               0.000           0.276
Trainings-Log x4 (Bank)                   A_Accepted  ≥ 11k 13114               0.000           0.682
Trainings-Log x4 (Bank) W_Complete application|start  < 11k 29127               0.299           0.328
Trainings-Log x4 (Bank) W_Complete application|start  ≥ 11k 31329               0.000           0.166
    Bank-Lauf simuliert                   A_Accepted  < 11k   322               0.000           0.245
    Bank-Lauf simuliert                   A_Accepted  ≥ 11k   225               0.000           0.702
    Bank-Lauf simuliert W_Complete application|start  < 11k   519               0.297           0.329
    Bank-Lauf simuliert W_Complete application|start  ≥ 11k   615               0.000           0.166
```

Laufzeit 243 s
