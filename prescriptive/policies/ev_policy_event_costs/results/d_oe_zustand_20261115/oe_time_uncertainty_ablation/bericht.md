**OE mit und ohne Unsicherheitsrückfall: Durchlaufzeitziel**

Identischer aktueller OE-Checkpoint, unveränderte Gewichte und Simulatorquellen. Seeds 0–2, je 200 bewertete Fälle, 700 Warmup und 1.300 Cooldown. Je erreichter Entscheidung dieselbe adaptive Regel: zunächst 30, bei unsicherer Differenz 100 Ziehungen je Option. Nur nach einer weiterhin unsicheren Schätzung wird die Bankfortsetzung durch die Option mit dem besten Mittelwert ersetzt. Bei fehlgeschlagenen vollständigen Suffixziehungen bleibt der Bank-Rückfall erhalten.

Abgeschlossen: 3 von 3 gepaarten Seeds. Die Referenzen mit Rückfall sind die vorhandenen OE-Zeitläufe; sie werden mit Prüfsummen gesichert und durch eine exakte Seed-0-Wiedergabe des Simulators geprüft.

| Seed | Policy | Arbeitskosten €/Fall | Gesamtkosten €/Fall | Durchlaufzeit Tage |
|---|---|---:|---:|---:|
| 0 | Bank | 41.70 | 178.64 | 20.374 |
| 0 | XGBoost-Zeit | 32.53 | 175.42 | 19.798 |
| 0 | OE-mit-Rueckfall | 52.88 | 204.15 | 20.633 |
| 0 | OE-ohne-Unsicherheitsrueckfall | 36.84 | 185.71 | 21.471 |
| 1 | Bank | 40.59 | 174.41 | 20.605 |
| 1 | XGBoost-Zeit | 36.48 | 184.05 | 19.975 |
| 1 | OE-mit-Rueckfall | 57.70 | 200.10 | 21.708 |
| 1 | OE-ohne-Unsicherheitsrueckfall | 59.39 | 208.29 | 20.777 |
| 2 | Bank | 53.62 | 199.49 | 22.448 |
| 2 | XGBoost-Zeit | 39.31 | 187.44 | 20.119 |
| 2 | OE-mit-Rueckfall | 39.22 | 184.48 | 22.535 |
| 2 | OE-ohne-Unsicherheitsrueckfall | 38.93 | 183.31 | 21.339 |

**Gepaarte Änderung: ohne minus mit Unsicherheitsrückfall. Negative Werte bedeuten eine Verbesserung.**

| Seed | Arbeitskosten Δ € | Gesamtkosten Δ € | Durchlaufzeit Δ Tage |
|---|---:|---:|---:|
| 0 | -16.04 | -18.44 | +0.838 |
| 1 | +1.69 | +8.19 | -0.931 |
| 2 | -0.28 | -1.17 | -1.196 |
| Mittel | -4.88 | -3.80 | -0.429 |

Das Trainingsziel bleibt die Durchlaufzeit. Arbeitskosten und Gesamtkosten sind zusätzlich gemessene Ergebnisse. Die Policies können unterschiedliche spätere Zustände und Entscheidungshäufigkeiten erzeugen; deshalb sind die Gesamtsummen ihrer Ziehungen trotz gleicher Regel je Entscheidung nicht identisch. Der Vergleich isoliert eine Policyregel bei diesem OE-Modell und beweist keinen allgemeinen Architektureffekt.
