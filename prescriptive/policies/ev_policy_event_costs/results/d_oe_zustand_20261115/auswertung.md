**D und OE mit Zustandsprüfung und bedingter Zeitkalibrierung**

3 gepaarte Seeds, je 200 bewertete Fälle; 700 Warmup und 1.300 Cooldown. Je Option zunächst 30, bei unsicherer Differenz 100 Ziehungen. Rückfall auf Bankverhalten bei nicht ausreichend getrennter Monte-Carlo-Schätzung. Die vorhandenen Eventnetze wurden nicht neu trainiert. Beide Modelle verwenden dasselbe train-only Zeitmodell.

| Modell | Ziel | Kosten/Fall (€) | Tage/Fall | Kostenersparnis ggü. Bank (€) | Zeitersparnis ggü. Bank (Tage) |
|---|---|---:|---:|---:|---:|
| D | cost | 177.87 | 20.584 | +6.31 | +0.559 |
| D | time | 183.29 | 21.075 | +0.89 | +0.067 |
| OE | cost | 178.61 | 20.982 | +5.57 | +0.160 |
| OE | time | 196.24 | 21.626 | -12.06 | -0.483 |

Alle Verfahren verwenden die korrigierte Optionsprüfung des Simulators. Die historische Bank ist deshalb keine unverändert übernommene Referenz. Seedweise Ergebnisse und alle Rückfälle stehen in paired_results.csv und den Fortschrittsdateien.
