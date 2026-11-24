# C mit Teacher Forcing 1: Online-Vergleich

200 bewertete Fälle, 30 freie Suffixziehungen je Option, Seed 0. 700 Warmup und 1.300 Cooldown, zufällige Ressourcenvergabe, vier bisherige Wahlpunkte. Das gespeicherte C-TF1-Modell wird unverändert verwendet, mit dem ursprünglichen Optionsabstand von 1 ms. Ziel: minimale mittlere Durchlaufzeit.

| Variante | Durchlaufzeit (Tage) ↓ | Verbesserung ggü. Bank |
|---|---:|---:|
| C, TF 1 | 21.664 | -6.33 % |
| C bisher | 20.198 | +0.86 % |
| D | 20.087 | +1.41 % |
| E | 22.046 | -8.21 % |
| OE | 21.195 | -4.03 % |
| Bank | 20.374 | 0,00 % |

C mit TF 1 minus bisheriges C: +1.466 Tage. Negative Werte bedeuten kürzere Durchlaufzeit.

Bank-Falldauern aller Vergleichsläufe identisch: ja. Teacher Forcing 1 betrifft das Training; die Auswertung zieht die Fortsetzungen frei.

Ein Seed mit 200 Fällen; kein Konfidenzintervall über mehrere Seeds. Einzelne Laufwerte und Falldauern stehen unter c_tf1/.
