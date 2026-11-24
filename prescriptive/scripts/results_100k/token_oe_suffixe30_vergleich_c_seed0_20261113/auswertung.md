# OE im Vergleich mit (c): Mini-Test

Seed 0; 30 Ziehungen pro Option; 200 gewertete Fälle (700–899), 700 Warmup- und 1.300 Cooldown-Fälle.
Alle Varianten verwenden zufällige Mitarbeiterzuteilung und keine Übergangssperre. Die Modelle und die Simulationskonfiguration des C-Vergleichs sind unverändert. Die Bank-Durchlaufzeiten aller 200 Fälle stimmen in allen drei Läufen exakt überein.

| Variante | Mittlere Durchlaufzeit (Tage) |
|---|---:|
| Bank | 20.374 |
| (c), ursprüngliche 1 ms | 20.198 |
| (c), ein gezogener Normal-Abstand je Option | 21.212 |
| token_oe, ohne Übergangssperre | 21.195 |

OE minus ursprüngliches (c): +0.997 Tage (+4.94 %). Positiv bedeutet längere Durchlaufzeit.
OE minus (c) mit gezogenem Normal-Abstand: -0.016 Tage.

## Wahlhäufigkeiten

| Kennzahl | (c), 1 ms | (c), Normal-Abstand | OE |
|---|---:|---:|---:|
| Anteil O_Create Offer: angebot_nachlegen | 55.01 % | 44.92 % | 47.78 % |
| Anteil A_Accepted: annehmen_bei_konzept | 81.95 % | 83.94 % | 65.35 % |
| Anteil A_Accepted: annehmen_nach_vervollstaendigung | 74.71 % | 71.11 % | 70.92 % |
| Anteil W_Handle leads|start: handle_leads | 28.12 % | 25.00 % | 54.69 % |

OE-Laufzeit: 14.1 Minuten; 797 Entscheidungen.
Reproduktion des früheren OE-Laufs mit demselben Seed: Entscheidungen identisch = True; einzelne Falldauern identisch = True.

Die Durchlaufzeiten werden aus den Simulator-Ereignissen gemessen, nicht aus den PSP-Prognosen. Der erneute Lauf mit demselben Seed ist eine Reproduktionskontrolle und kein zusätzlicher unabhängiger Seed. Ein Seed und 200 Fälle erlauben keine abgesicherte Aussage über andere Seeds und erklären nicht, welcher Modellunterschied die Ergebnisdifferenz verursacht.
