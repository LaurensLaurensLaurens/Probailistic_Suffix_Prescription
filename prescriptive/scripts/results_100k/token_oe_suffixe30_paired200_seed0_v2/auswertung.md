# token_oe: kurzer gepaarter Test der Übergangssperre

Seed 0, je 30 Ziehungen pro Option, 700 Warmup-, 200 gewertete und 1.300 Cooldown-Fälle.

Beide Läufe verwenden dieselben trainierten Gewichte, vier Wahlpunkte, zufällige Zuteilung und den gleichen Seed. Die Bank-Durchlaufzeiten aller 200 Fälle stimmen in beiden Läufen exakt überein.

| Variante | Mittlere Durchlaufzeit (Tage) |
|---|---:|
| Bank | 20.374 |
| Ohne Übergangssperre | 21.195 |
| Mit Übergangssperre | 21.737 |

Verbesserung durch die Sperre: -0.542 Tage (-2.56 %). Positiv bedeutet kürzer mit Sperre.

## Wahlen an den vier Punkten

| Kennzahl | Ohne Sperre | Mit Sperre |
|---|---:|---:|
| Wahlen der Policy: angebot_nachlegen | 427.0000 | 491.0000 |
| Anteil O_Create Offer: angebot_nachlegen | 0.4778 | 0.5336 |
| Wahlen der Policy: annehmen_bei_konzept | 101.0000 | 101.0000 |
| Anteil A_Accepted: annehmen_bei_konzept | 0.6535 | 0.7030 |
| Wahlen der Policy: annehmen_nach_vervollstaendigung | 141.0000 | 143.0000 |
| Anteil A_Accepted: annehmen_nach_vervollstaendigung | 0.7092 | 0.6294 |
| Wahlen der Policy: handle_leads | 128.0000 | 128.0000 |
| Anteil W_Handle leads|start: handle_leads | 0.5469 | 0.5391 |

Ein einzelner Seed mit 200 gewerteten Fällen und 30 Ziehungen ist ein Vortest. Daraus folgt keine abgesicherte Aussage für den laufenden 300er-Test oder andere Seeds. Die beiden Optionenbewertungen sind mit 30 Ziehungen stärker verrauscht.

Die JSON-Dateien in den Variantenordnern enthalten vollständige Kennzahlen und die einzelnen Falldauern. `choices.jsonl` protokolliert jede Entscheidung; `progress.json` den Verlauf. In `run_manifest.json` stehen Parameter und Dateifingerabdrücke.

## Korrektur während des Tests

Der erste Sperrenlauf brach ab, weil eine bekannte Aktivität mit einer im Encoder unbekannten Zeitklasse kodiert war. Die Übergangsprüfung verwendet jetzt ausdrücklich das tatsächliche letzte Aktivität-/Lifecycle-Paar. Dadurch bleibt sie unabhängig von der Zeitklasse. Modellgewichte und Eingabeskalierung wurden nicht verändert. Neun gezielte Tests sowie der reproduzierte betroffene Prefix bestehen. Der korrigierte Lauf reproduziert alle 423 Entscheidungen vor der früheren Abbruchstelle exakt.

Der vollständige Referenzlauf ohne Sperre wurde wiederverwendet; der Sperrenlauf wurde vollständig mit Seed 0 neu simuliert. Die Ergebnisse dieses Berichts stammen ausschließlich aus vollständigen Läufen. Der abgebrochene Versuch liegt im vorherigen Ordner `token_oe_suffixe30_paired200_seed0`.

Auch der separate 300er-Lauf wurde mit der Korrektur am 13.11.2026 um 11:18 Uhr neu gestartet. Sein neuer Ordner ist `neue_punkte_online_token_oe_ue_suffixe300_seed0_v2`.
