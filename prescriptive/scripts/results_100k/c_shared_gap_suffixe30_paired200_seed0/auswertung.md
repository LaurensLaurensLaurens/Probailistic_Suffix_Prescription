# (c): ein geteilter Normal-Abstand je Option

Seed 0, je 30 Ziehungen pro Option, 700 Warmup-, 200 gewertete und 1.300 Cooldown-Fälle.

Beide Läufe verwenden dieselben trainierten Gewichte, vier Wahlpunkte, zufällige Zuteilung und den gleichen Seed. Die Bank-Durchlaufzeiten aller 200 Fälle stimmen exakt überein.

| Variante | Mittlere Durchlaufzeit (Tage) |
|---|---:|
| Bank | 20.374 |
| (c), ursprüngliche 1 ms | 20.198 |
| (c), ein Normal-Abstand je Option | 21.212 |

Verbesserung durch die Änderung: -1.013 Tage (-5.02 %). Positiv bedeutet kürzer.

Je Entscheidung wird für jede Option unabhängig ein Abstand aus ihrer bei 0 trunkierten Normalverteilung gezogen. Alle 30 Suffixe dieser Option teilen diesen Abstand und dasselbe Encoder-Prefix. Die Normalparameter sind die empirischen Mittelwerte und Standardabweichungen der beobachteten Abstände im äußeren Trainingssplit; durch die Trunkierung ist der tatsächliche Erwartungswert größer als der rohe Lageparameter. Beide Varianten laufen ohne Übergangssperre. Die Erstabstandsunsicherheit wird bei dieser Variante nicht über 30 Zeitziehungen gemittelt.

## Wahlen an den vier Punkten

| Kennzahl | (c), ursprüngliche 1 ms | (c), ein Normal-Abstand je Option |
|---|---:|---:|
| Wahlen der Policy: angebot_nachlegen | 509.0000 | 443.0000 |
| Anteil O_Create Offer: angebot_nachlegen | 0.5501 | 0.4492 |
| Wahlen der Policy: annehmen_bei_konzept | 133.0000 | 137.0000 |
| Anteil A_Accepted: annehmen_bei_konzept | 0.8195 | 0.8394 |
| Wahlen der Policy: annehmen_nach_vervollstaendigung | 87.0000 | 90.0000 |
| Anteil A_Accepted: annehmen_nach_vervollstaendigung | 0.7471 | 0.7111 |
| Wahlen der Policy: handle_leads | 128.0000 | 128.0000 |
| Anteil W_Handle leads|start: handle_leads | 0.2812 | 0.2500 |

Ein einzelner Seed mit 200 gewerteten Fällen und 30 Ziehungen ist ein Vortest. Daraus folgt keine abgesicherte Aussage für andere Seeds oder Ziehungszahlen.

Die JSON-Dateien enthalten vollständige Kennzahlen und einzelne Falldauern. `choices.jsonl` protokolliert Entscheidungen; `progress.json` den Verlauf. In `run_manifest.json` stehen Parameter und Dateifingerabdrücke.

## Technische Prüfung

13 gezielte Tests bestanden. Am echten Modell erhalten sämtliche 30 Suffixe einer Option denselben ersten Abstand. Alle Entscheidungen des neuen Online-Laufs wurden auf genau eine nichtnegative Zeitziehung je Option geprüft; `choices.jsonl` enthält die gezogenen Werte.

Als Kontrolle wurde die neue Variante mit einem festen Abstand von 1 ms geprüft: Alle 60 Ereignisfolgen und die gewählte Option waren identisch mit dem ursprünglichen Modell. Durch die numerische Neukodierung einzelner Zeitmerkmale gab es Abweichungen der berechneten Suffixdauern von höchstens etwa 2,14 Sekunden. Details stehen in `validation.json`.

Dieser Test bewertet die konkrete Variante mit trainingsbasierten, pro Wahlpunkt und Option geschätzten Normalparametern und einer Ziehung je Entscheidung. Es ist kein Test eines zusätzlich auf den individuellen Prefix konditionierten Zeitmodells.
