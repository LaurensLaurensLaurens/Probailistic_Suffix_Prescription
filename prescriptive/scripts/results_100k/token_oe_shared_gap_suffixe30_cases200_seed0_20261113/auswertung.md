**OE: ein gezogener Optionsabstand für alle 30 Suffixe**

Seed 0; 30 Suffixziehungen pro Option; 200 gewertete Fälle (700–899); 700 Warmup- und 1.300 Cooldown-Fälle. Zufällige Mitarbeiterzuteilung, keine Übergangssperre.

| Variante | Mittlere Durchlaufzeit (Tage) |
|---|---:|
| OE, ein Modell-Abstand je Option für alle 30 Suffixe | 20.828 |
| OE, ein Modell-Abstand je Suffixziehung | 21.195 |
| (d), korrigierter Optionsabstand je Suffixziehung | 20.087 |
| (c), ursprüngliche 1 ms | 20.198 |
| (c), ein Normal-Abstand je Option für alle 30 Suffixe | 21.212 |
| Bank | 20.374 |

OE mit einem Abstand minus bisheriges OE: -0.368 Tage; minus korrigiertes (d): +0.740 Tage. Positive Differenzen bedeuten längere Durchlaufzeit.

Laufzeit: 11.3 Minuten; 758 Entscheidungen.

**Umsetzung und Prüfung**

Jede Entscheidung beginnt mit der beobachteten Vorgeschichte. Pro Option wird einmal aus dem vorhandenen OE-Modell gezogen: gemeinsames Event/Zeitklassen-Token unter der erzwungenen Option, dann ein Abstand aus der auf diese Zeitklasse trunkierten Normalverteilung. Dieser Abstand wird für alle 30 Fortsetzungen derselben Option wiederholt. Es wird kein neuer Abstand für jede Fortsetzung gezogen.

Das erweiterte Encoder-Prefix enthält die Option mit diesem Abstand, Fallalter, Wochentag, Tageszeit und passender Zeitklasse. Alle 30 Fortsetzungen einer Option sehen dieses identische erweiterte Prefix. Die gemeinsame Ausgangsbasis zwischen den Optionen ist die beobachtete Vorgeschichte; die erweiterten Prefixe enthalten jeweils das eigene Options-Event. Der Optionsabstand zählt genau einmal zur bewerteten Dauer.

Das OE-Modell wurde kopiert und ausschließlich shared_option_gap = true ergänzt. Kein neues Training; Originalmodell und gelernte Gewichte sind unverändert. Die Abstände stammen weiterhin aus dem OE-Decoder. Der frühere (c)-Versuch verwendete dagegen empirisch angepasste Normalverteilungen je Wahlpunkt und Option. Die Analogie zu (c) betrifft das Teilen eines Abstands durch die 30 Suffixe.

Die bisherigen gemeinsamen Zufallszahlen zwischen den Optionen bleiben erhalten. Verschiedene Optionen können dadurch denselben Zahlenwert erhalten; pro Option erfolgt dennoch genau eine Modellziehung.

Die Prüfung am echten Modell bestätigte die Aufrufgrößen 1/30/1/30, identische erweiterte Prefixe je Option, unveränderte beobachtete Historie und passende Zeitmerkmale. Mit einer einzigen Fortsetzung stimmen die Ergebnisse exakt mit dem bisherigen OE überein. Neun gezielte Unit-Tests bestanden. Alle 758 protokollierten Online-Entscheidungen enthalten genau einen endlichen, nichtnegativen Abstand je Option. Die Bank-Dauern aller 200 Fälle stimmen in sämtlichen Vergleichsläufen exakt überein. Die Dateiprüfsummen aus run_manifest.json stimmen nach dem Lauf unverändert.

**Wahlhäufigkeiten**

| Kennzahl | OE, ein Abstand | OE, pro Suffix | (d), korrigiert | (c), 1 ms | (c), ein Abstand |
|---|---:|---:|---:|---:|---:|
| Anteil O_Create Offer: angebot_nachlegen | 41.82 % | 47.78 % | 61.46 % | 55.01 % | 44.92 % |
| Anteil A_Accepted: annehmen_bei_konzept | 61.29 % | 65.35 % | 67.74 % | 81.95 % | 83.94 % |
| Anteil A_Accepted: annehmen_nach_vervollstaendigung | 69.74 % | 70.92 % | 73.73 % | 74.71 % | 71.11 % |
| Anteil W_Handle leads\|start: handle_leads | 61.72 % | 54.69 % | 33.59 % | 28.12 % | 25.00 % |

**Einordnung**

Dies ist ein Mini-Test mit einem Seed. Unterschiede zwischen den Varianten sind damit nicht statistisch abgesichert. Die Durchlaufzeiten stammen aus den tatsächlichen Simulator-Ereignissen, nicht aus den vorhergesagten Suffixzeiten. Details der Prüfung stehen in sampling_validation.json, die gezogenen Abstände je Entscheidung in oe_ein_abstand_neustart/choices.jsonl, die Konfiguration in run_manifest.json. Der erste Versuch unter oe_ein_abstand blieb nach 147 Entscheidungen ohne Abschluss stehen. Er ist aufbewahrt; ausgewertet wird ausschließlich der vollständige Neustart mit identischer Konfiguration.
Die ersten 147 Entscheidungen einschließlich der gezogenen Abstände wurden im Neustart exakt reproduziert.
