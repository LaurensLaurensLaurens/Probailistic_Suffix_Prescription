**(d): gezogener Optionsabstand je Suffixziehung**

Seed 0; 30 Ziehungen pro Option; 200 gewertete Fälle (700–899); 700 Warmup- und 1.300 Cooldown-Fälle. Zufällige Mitarbeiterzuteilung, keine Übergangssperre. Die Bank-Dauern aller 200 Fälle stimmen mit den drei früheren Vergleichsläufen exakt überein.

| Variante | Mittlere Durchlaufzeit (Tage) |
|---|---:|
| (d), Optionsabstand pro Suffix gezogen | 20.087 |
| (c), ursprüngliche 1 ms | 20.198 |
| (c), ein Normal-Abstand je Option für alle Suffixe | 21.212 |
| OE, Optionsabstand pro Suffix gezogen | 21.195 |
| Bank | 20.374 |

(d) minus OE: -1.108 Tage; (d) minus ursprüngliches (c): -0.111 Tage. Positive Differenzen bedeuten längere Durchlaufzeit.

Laufzeit: 15.8 Minuten; 933 Entscheidungen.

**Umsetzung und Prüfung**

Das vorhandene (d)-Modell wurde kopiert. Die einzige Konfigurationsänderung ist option_as = encoder. Es gab kein neues Training; gelernte Gewichte und das Originalmodell sind unverändert. Event und Zeitklasse bleiben getrennte Ausgaben von (d). Die vorhandene Ziehungslogik wird auch für OE verwendet.

Alle Ziehungen starten mit derselben beobachteten Vorgeschichte. Für jede Option werden 30 erste Abstände aus dem Modell gezogen: Zeitklasse und dann ein Abstand aus der auf diese Klasse trunkierten Normalverteilung. Für jede Ziehung wird die Option samt Abstand, Fallalter, Wochentag, Tageszeit und Zeitklasse im erweiterten Encoder-Prefix eingesetzt. Von dort folgt der restliche Suffix. Der Optionsabstand zählt genau einmal zur bewerteten Dauer.

Die Optionen verwenden gemeinsame Zufallszahlen, wie bei den bisherigen PSP-Vergleichen. Bei (d) hängen die erste Zeitklasse und die Gaussian-Parameter vor dem Options-Event nicht vom erzwungenen Kandidaten ab. Die gepaarten ersten Abstände können deshalb zwischen den Optionen gleich sein; über die 30 Suffixziehungen variieren sie. Der nachfolgende Encoder sieht die jeweilige Option und ihren Abstand.

Die Prüfung am echten Modell bestätigte 30 unterschiedliche Abstände pro Option, unveränderte beobachtete Vorgeschichte, passende Zeitmerkmale und Zeitklassen sowie die einmalige Berücksichtigung des Optionsabstands. Durch das Anhängen des Events verschiebt sich lediglich die linke Auffüllung im festen Encoderfenster. Details: sampling_validation.json; Parameter und Dateiprüfsummen: run_manifest.json.

**Wahlhäufigkeiten**

| Kennzahl | (d), korrigiert | (c), 1 ms | (c), ein Abstand | OE |
|---|---:|---:|---:|---:|
| Anteil O_Create Offer: angebot_nachlegen | 61.46 % | 55.01 % | 44.92 % | 47.78 % |
| Anteil A_Accepted: annehmen_bei_konzept | 67.74 % | 81.95 % | 83.94 % | 65.35 % |
| Anteil A_Accepted: annehmen_nach_vervollstaendigung | 73.73 % | 74.71 % | 71.11 % | 70.92 % |
| Anteil W_Handle leads|start: handle_leads | 33.59 % | 28.12 % | 25.00 % | 54.69 % |

**Einordnung**

Ein Seed mit 200 gewerteten Fällen und 30 Ziehungen ist ein Mini-Test. Die ursprüngliche (d)-Variante mit 1 ms wurde hier nicht zusätzlich online ausgeführt. Der Vergleich mit (c) und OE isoliert deshalb nicht die Wirkung der Zeitkorrektur bei (d). Die Online-Dauern stammen aus den tatsächlichen Simulator-Ereignissen, nicht aus den vorhergesagten Suffixzeiten.
