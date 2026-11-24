# PSP-Verlustverläufe

Stand: 2026-11-14T17:17:53+01:00. Protokolle wurden als unveränderliche Kopien unter logs/ gesichert.

Training verwendet den attenuierten, mit GradNorm gewichteten Trainingsverlust; die Tabelle zeigt den Standard-Validierungsverlust. Beide sind unterschiedliche Größen. Absolute Verluste verschiedener Modellvarianten sind wegen verschiedener Aufgaben nicht direkt zu vergleichen.

## Minima und Trainingsende

| Modell | Fertige Epochen | Rohminimum (Epoche) | Minimum bei TF wie am Ende (Epoche) | Letzter Val-Verlust | Letzte bis zu fünf Val-Werte bei gleicher TF |
|---|---:|---:|---:|---:|---|
| Früher Nachtlauf | 25/25 | 2,6963 (24) | 2,6963 (24) | 2,7188 | 2,7273 → 2,7324 → 2,7301 → 2,6963 → 2,7188 |
| Echtlog, alte Wahlpunkte | 100/100 | 2,7919 (99) | 2,7919 (99) | 2,7998 | 2,7982 → 2,7960 → 2,8004 → 2,7919 → 2,7998 |
| Echtlog, neue Wahlpunkte | 100/100 | 2,8057 (95) | 2,8057 (95) | 2,8194 | 2,8251 → 2,8413 → 2,8346 → 2,8279 → 2,8194 |
| s40 | 20/20 | 7,8365 (18) | 7,8365 (18) | 7,8563 | 7,8532 → 7,8649 → 7,8365 → 7,8808 → 7,8563 |
| (a) | 15/15 | 7,7328 (7) | 8,1946 (9) | 8,4708 | 8,4793 → 8,4258 → 8,2077 → 8,2337 → 8,4708 |
| (b) | 15/15 | 7,7549 (7) | 7,8952 (9) | 8,1602 | 7,9803 → 8,1085 → 8,2276 → 8,1625 → 8,1602 |
| (c) | 15/15 | 8,8471 (9) | 8,8471 (9) | 9,2013 | 9,0194 → 8,8711 → 9,1101 → 9,0271 → 9,2013 |
| (d) | 15/15 | 10,2198 (8) | 10,4628 (9) | 11,6088 | 10,8304 → 11,2113 → 11,4975 → 11,2802 → 11,6088 |
| OE | 15/15 | 9,2697 (8) | 9,4106 (9) | 9,6006 | 9,5882 → 9,6309 → 9,6828 → 9,7090 → 9,6006 |
| (c), TF 1 | 15/15 | 9,5887 (1) | 9,5887 (1) | 13,1290 | 12,7781 → 12,8906 → 13,0484 → 12,8595 → 13,1290 |
| (e) | 15/15 | 9,3692 (9) | 9,3692 (9) | 9,6419 | 9,5037 → 9,5825 → 9,5107 → 9,6291 → 9,6419 |
| Neues D, TF 1 (echter Log) | 5/15 | 12,4713 (1) | 12,4713 (1) | 16,1234 | 12,4713 → 13,9320 → 14,9268 → 15,6413 → 16,1234 |
| Neues D, TF 1 (RCT) | 1/15 | 12,5956 (1) | 12,5956 (1) | 12,5956 | 12,5956 |

## Einordnung

Die Validierung übernimmt im Trainer die aktuelle Teacher-Forcing-Quote des Trainings. Bei den geplanten 15-Epochen-Läufen ohne TF-Untergrenze fällt sie von 1 auf 0 und bleibt ab Epoche 9 bei 0. Ein Rohminimum vor dieser Phase wurde daher unter anderen Bedingungen gemessen. Die zusätzliche Spalte vergleicht nur Epochen mit derselben Quote wie am Ende.

Die gezeigten Endwerte stammen aus den tatsächlich gespeicherten Modellen: Die vorhandene Pipeline übernimmt die letzte Epoche und lädt nicht automatisch den besten Validierungsstand. Der Trainer überschreibt last.pt beim Speichern. Die laufenden Trainings wurden durch diese Analyse nicht verändert.

Aus den Protokollen lässt sich der beobachtete Trend bis zum Ende beurteilen. Ob zusätzliche, bisher nicht trainierte Epochen danach besser würden, ist daraus nicht feststellbar.

[Standard-Validierungsverluste](validation_curves.png) · [Attenuierte Trainingsverluste](training_curves.png) · [Alle Epochen als CSV](epochs.csv) · [Zusammenfassung als CSV](summary.csv)
