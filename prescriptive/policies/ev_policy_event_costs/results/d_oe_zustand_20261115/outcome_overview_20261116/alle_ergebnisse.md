**Vollständige Outcome-Ergebnisse der aktuellen Vergleichsreihe**

Alle Läufe abgeschlossen: Seeds 0–2, jeweils 200 bewertete Fälle, 700 Warmup und 1.300 Cooldown. Alle Mittelwerte verwenden dieselben drei Seeds. Bank bezeichnet Bankverhalten im korrigierten Simulator. Kosten entsprechen den konfigurierten Beispieltarifen. Die PSP-Ergebnisse verwenden korrigierte bisherige Checkpoints; das separate 100-Epochen-OE-Training ist darin nicht enthalten.

| Modell | Optimierungsziel | Unsicherheitsrückfall | Tage/Fall | Arbeitskosten €/Fall | Gesamtkosten €/Fall | Arbeitskostenersparnis ggü. Bank |
|---|---|---|---:|---:|---:|---:|
| Bank | Bankverhalten | — | 21,14 | 45,30 | 184,18 | 0,00 % |
| XGBoost | Durchlaufzeit | — | 19,96 | 36,11 | 182,30 | 20,30 % |
| Causal Forest | Gesamtkosten | — | 18,53 | 46,47 | 185,36 | -2,57 % |
| PSP D | Gesamtkosten | ja | 20,58 | 42,50 | 177,87 | 6,18 % |
| PSP OE | Gesamtkosten | ja | 20,98 | 40,62 | 178,61 | 10,33 % |
| PSP D | Durchlaufzeit | ja | 21,08 | 40,80 | 183,29 | 9,93 % |
| PSP D | Durchlaufzeit | nein | 20,32 | 52,05 | 200,16 | -14,90 % |
| PSP OE | Durchlaufzeit | ja | 21,63 | 49,93 | 196,24 | -10,22 % |
| PSP OE | Durchlaufzeit | nein | 21,20 | 45,05 | 192,44 | 0,55 % |

Die Prozentwerte vergleichen die aggregierten Mittelwerte mit dem aggregierten Bankmittel; sie sind nicht der Mittelwert seedweiser Prozente. Arbeitskosten sind ein zusätzlich gemessenes Ergebnis. Kein Modell in dieser Tabelle wurde ausschließlich auf Arbeitskosten optimiert.

**Mit versus ohne Unsicherheitsrückfall: ohne minus mit, Mittel über die drei gepaarten Seeds**

| PSP | Durchlaufzeit Δ Tage | Arbeitskosten Δ €/Fall | Gesamtkosten Δ €/Fall |
|---|---:|---:|---:|
| D | -0,756 | 11,25 | 16,88 |
| OE | -0,429 | -4,88 | -3,80 |

Negative Änderungen bedeuten eine Verbesserung. Die Effekte schwanken zwischen Seeds; beispielsweise senkt OE ohne Rückfall die Arbeitskosten in Seed 0 um 16,04 €, erhöht sie in Seed 1 um 1,69 € und senkt sie in Seed 2 um 0,28 €.

**Ergebnisse je Seed**

| Seed | Modell | Ziel | Unsicherheitsrückfall | Tage/Fall | Arbeitskosten €/Fall | Gesamtkosten €/Fall |
|---|---|---|---|---:|---:|---:|
| 0 | Bank | Bankverhalten | — | 20,37 | 41,70 | 178,64 |
| 1 | Bank | Bankverhalten | — | 20,60 | 40,59 | 174,41 |
| 2 | Bank | Bankverhalten | — | 22,45 | 53,62 | 199,49 |
| 0 | XGBoost | Durchlaufzeit | — | 19,80 | 32,53 | 175,42 |
| 1 | XGBoost | Durchlaufzeit | — | 19,98 | 36,48 | 184,05 |
| 2 | XGBoost | Durchlaufzeit | — | 20,12 | 39,31 | 187,44 |
| 0 | Causal Forest | Gesamtkosten | — | 18,32 | 47,00 | 189,29 |
| 1 | Causal Forest | Gesamtkosten | — | 18,64 | 50,43 | 184,93 |
| 2 | Causal Forest | Gesamtkosten | — | 18,62 | 41,97 | 181,85 |
| 0 | PSP D | Gesamtkosten | ja | 19,93 | 37,78 | 174,10 |
| 1 | PSP D | Gesamtkosten | ja | 19,94 | 34,95 | 163,53 |
| 2 | PSP D | Gesamtkosten | ja | 21,89 | 54,78 | 195,96 |
| 0 | PSP OE | Gesamtkosten | ja | 21,56 | 42,83 | 184,81 |
| 1 | PSP OE | Gesamtkosten | ja | 20,24 | 37,84 | 172,00 |
| 2 | PSP OE | Gesamtkosten | ja | 21,15 | 41,21 | 179,01 |
| 0 | PSP D | Durchlaufzeit | ja | 21,31 | 37,24 | 179,04 |
| 1 | PSP D | Durchlaufzeit | ja | 21,34 | 40,97 | 182,57 |
| 2 | PSP D | Durchlaufzeit | ja | 20,58 | 44,20 | 188,26 |
| 0 | PSP D | Durchlaufzeit | nein | 19,28 | 39,99 | 190,74 |
| 1 | PSP D | Durchlaufzeit | nein | 20,88 | 58,07 | 202,79 |
| 2 | PSP D | Durchlaufzeit | nein | 20,80 | 58,10 | 206,96 |
| 0 | PSP OE | Durchlaufzeit | ja | 20,63 | 52,88 | 204,15 |
| 1 | PSP OE | Durchlaufzeit | ja | 21,71 | 57,70 | 200,10 |
| 2 | PSP OE | Durchlaufzeit | ja | 22,54 | 39,22 | 184,48 |
| 0 | PSP OE | Durchlaufzeit | nein | 21,47 | 36,84 | 185,71 |
| 1 | PSP OE | Durchlaufzeit | nein | 20,78 | 59,39 | 208,29 |
| 2 | PSP OE | Durchlaufzeit | nein | 21,34 | 38,93 | 183,31 |

Quellen und Prüfsummen stehen in provenienz.json. Tabellen: ergebnisse_mittel.csv und ergebnisse_je_seed.csv. Modelle, Trainingsumfang und Optimierungsziele unterscheiden sich; die Ergebnistabelle isoliert keinen Architektureffekt.
