# Gradienten-Zerlegung an konzept / vervollständigen (k3x4ol, ohne Training), 2026-11-24 11:54

20000 Fälle des x4-Logs (ohne Testfälle), 20 Batches à 256, Teacher Forcing 1, train-Modus, GradNorm-Gewichte 1; Statik-Abbildung mit 0 begonnen (Gradient = Richtung des ersten Schritts). Fokus = Ziel direkt nach dem ersten A_Complete.

## Überblick
```
{
 "Proben": 5120,
 "gültige Token": 167887,
 "Fokus-Schritte": 5105,
 "Anteil Fokus an Token": 0.0304,
 "Gewinn umgerechnet auf Event-Verlust je Token (nats)": 0.00368,
 "Event-Verlust je Token (Mittel)": 0.3836,
 "Dämpfung durch Abschwächung (|grad att| / |grad ohne Rauschen|, Fokus-Logits)": 1.0,
 "Kosinus grad att / ohne Rauschen": 1.0,
 "σ Ziel-Klasse an Fokus (Median)": 0.05,
 "|grad gesamt Betragsspalte|": 1.4900158643722534,
 "|grad gesamt Abbildung|": 10.513245582580566,
 "|grad gesamt Event-Kopf|": 0.2292497754096985
}
```

## Fokus-Schritte je Gruppe
```
       Fokus-Schritte  wahr A_Complete  P(AC) Netz  Logit AC  logvar AC (Mittel)  σ AC (Median)  CE jetzt  Gewinn mit Gruppenrate (nats je Fokus-Token)
< 11k          2490.0            0.296       0.163     9.970                -6.0           0.05    1.0127                                        0.0996
≥ 11k          2615.0            0.000       0.128    10.093                -6.0           0.05    0.4057                                        0.1413
```

## Gradient je Verlustteil
(Anteil in Fokus-Richtung: Projektion des Teils auf die Richtung des Fokus-Gradienten, relativ zum Gesamtgradienten)
```
                  Teil  Verlust (Mittel je Batch)  |grad Betragsspalte|  Kosinus zu gesamt (Betrag)  Anteil in Fokus-Richtung  |grad Abbildung|  |grad Event-Kopf|
      kat:concept:name                    0.36220               0.13779                     0.09090                   0.35745           0.42938            0.22454
 num:case_elapsed_time                   -2.37498               1.37898                     0.94873                  -1.45439          10.55354            0.00000
num:event_elapsed_time                   -2.02363               0.36334                     0.42355                   0.52953           0.98364            0.00000
             kat:fokus                    0.02137               0.18913                     0.08098                   1.56742           0.20751            0.03891
```

Laufzeit 786 s
