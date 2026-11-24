**XGBoost im aktuellen Simulatorvergleich**

Vorhandenes auf Durchlaufzeit trainiertes XGBoost-Modell, 10.000 echte Trainingsfälle. Seeds 0–2, je 200 bewertete Fälle, 700 Warmup und 1.300 Cooldown. Dieselbe korrigierte Optionsprüfung, Zufallszuteilung, Fall-IDs, Tarife und Bankreferenz wie beim laufenden D/OE-Vergleich. XGBoost verwendet keine Suffixziehungen und wurde hier nicht neu trainiert.

| Seed | Bank (Tage) | XGBoost (Tage) | Zeitersparnis (%) | XGBoost-Kosten/Fall (€) |
|---|---:|---:|---:|---:|
| 0 | 20.374 | 19.798 | +2.83 | 175.42 |
| 1 | 20.605 | 19.975 | +3.06 | 184.05 |
| 2 | 22.448 | 20.119 | +10.38 | 187.44 |

Mittel über die drei Seeds: Bank 21.142 Tage, XGBoost 19.964 Tage. Mittlerer gepaarter relativer Zeitgewinn: 5.42 %.

Die Kosten sind ein zusätzlich gemessener Ausgang dieser Zeitpolicy. Ein auf Eventkosten optimiertes XGBoost wurde mit dieser Referenz nicht getestet. Die vorhandenen XGBoost- und PSP-Modellstände unterscheiden sich in Trainingsumfang und Verfahren; der Vergleich isoliert keinen Architektureffekt.
