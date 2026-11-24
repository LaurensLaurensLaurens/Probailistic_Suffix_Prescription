**D mit und ohne Unsicherheitsrückfall: Durchlaufzeitziel**

Identischer aktueller D-Checkpoint, unveränderte Gewichte und Simulatorquellen. Seeds 0–2, je 200 bewertete Fälle, 700 Warmup und 1.300 Cooldown. Je erreichter Entscheidung dieselbe adaptive Regel: zunächst 30, bei unsicherer Differenz 100 Ziehungen je Option. Nur nach einer weiterhin unsicheren Schätzung wird die Bankfortsetzung durch die Option mit dem besten Mittelwert ersetzt. Bei fehlgeschlagenen vollständigen Suffixziehungen bleibt der Bank-Rückfall erhalten.

Abgeschlossen: 3 von 3 gepaarten Seeds. Die Referenzen mit Rückfall sind die vorhandenen D-Zeitläufe; sie werden mit Prüfsummen gesichert und durch eine exakte Seed-0-Wiedergabe des Simulators geprüft.

| Seed | Policy | Arbeitskosten €/Fall | Gesamtkosten €/Fall | Durchlaufzeit Tage |
|---|---|---:|---:|---:|
| 0 | Bank | 41.70 | 178.64 | 20.374 |
| 0 | XGBoost-Zeit | 32.53 | 175.42 | 19.798 |
| 0 | D-mit-Rueckfall | 37.24 | 179.04 | 21.308 |
| 0 | D-ohne-Unsicherheitsrueckfall | 39.99 | 190.74 | 19.278 |
| 1 | Bank | 40.59 | 174.41 | 20.605 |
| 1 | XGBoost-Zeit | 36.48 | 184.05 | 19.975 |
| 1 | D-mit-Rueckfall | 40.97 | 182.57 | 21.339 |
| 1 | D-ohne-Unsicherheitsrueckfall | 58.07 | 202.79 | 20.877 |
| 2 | Bank | 53.62 | 199.49 | 22.448 |
| 2 | XGBoost-Zeit | 39.31 | 187.44 | 20.119 |
| 2 | D-mit-Rueckfall | 44.20 | 188.26 | 20.579 |
| 2 | D-ohne-Unsicherheitsrueckfall | 58.10 | 206.96 | 20.804 |

**Gepaarte Änderung: ohne minus mit Unsicherheitsrückfall. Negative Werte bedeuten eine Verbesserung.**

| Seed | Arbeitskosten Δ € | Gesamtkosten Δ € | Durchlaufzeit Δ Tage |
|---|---:|---:|---:|
| 0 | +2.75 | +11.71 | -2.030 |
| 1 | +17.10 | +20.22 | -0.462 |
| 2 | +13.90 | +18.70 | +0.225 |
| Mittel | +11.25 | +16.88 | -0.756 |

Das Trainingsziel bleibt die Durchlaufzeit. Arbeitskosten und Gesamtkosten sind zusätzlich gemessene Ergebnisse. Die Policies können unterschiedliche spätere Zustände und Entscheidungshäufigkeiten erzeugen; deshalb sind die Gesamtsummen ihrer Ziehungen trotz gleicher Regel je Entscheidung nicht identisch. Der Vergleich isoliert eine Policyregel bei diesem D-Modell und beweist keinen allgemeinen Architektureffekt.
