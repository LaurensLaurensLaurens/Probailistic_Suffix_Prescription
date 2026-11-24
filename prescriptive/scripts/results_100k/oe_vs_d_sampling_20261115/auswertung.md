# Zieht OE besser als D?

Verglichen wurden die vorhandenen Modelle auf denselben zwölf Entscheidungszuständen des früheren D-Laufs: je drei verschiedene Fälle pro Wahlpunkt, dreimal 500 Suffixe je Option, also je Modell 36.000 Suffixe. Gleiche Tarife, Testfälle, Wahlpunkte und Übergangssperre. Zusätzlich: Fall 750, fünfmal 500 Suffixe je Option. Der Simulatorvergleich verwendet die bereits geprüften 72 Fortsetzungen (eine erzwungene Option, anschließend Bankverhalten). Kein neuer Online-Benchmark.

| Kennzahl | D | OE |
|---|---:|---:|
| Verbleibende Kosten (€) | 6654.039 | 3330.800 |
| Aktive Arbeitsstunden | 139.534 | 69.399 |
| Verbleibende Tage | 18.143 | 20.511 |
| Suffix-Events | 29.773 | 31.360 |
| Ohne geschäftlichen Endmarker (%) | 34.453 | 28.647 |
| Offene Arbeitssitzung am Ende (%) | 26.597 | 14.389 |
| Erneute Beginne bei offener Sitzung je Suffix | 1.066 | 0.683 |
| An der Längengrenze (%) | 0.006 | 0.006 |

Simulator-Fortsetzungen derselben Zustände: 192.26 € und 18.60 Tage im Mittel. Nur drei Fortsetzungen pro Option; für individuelle Optionsrangfolgen ist diese Stichprobe klein.

OE: identische erste Optionsabstände in 0 von 36 Ziehungsaufrufen. Bei D waren es 36 von 36.

OE: 0.0000 % der gezogenen Events haben eine Kombination aus Event und aus dem tatsächlichen Abstand abgeleiteter Zeitklasse, die im gesamten echten Log nicht vorkommt. Bei D: 8,06 % in der ersten Wiederholung. OE zieht ein gemeinsames Token für Event und Klasse; Randwert-Rundungen können beim nachträglichen Ableiten der Klasse aus Sekunden Unterschiede erzeugen.

## Fall 750

- A_Accepted: 2267.40 €; erster Abstand im Mittel 0.162 Stunden.
- W_Complete application|start: 3473.42 €; erster Abstand im Mittel 2.521 Stunden.

OE wählt in 0 von fünf Wiederholungen den erneuten Start. D wählte ihn in einer von fünf neuen Wiederholungen und im ursprünglichen 500er-Lauf. Die fehlende Sitzungsprüfung des Simulators wird durch den Modellwechsel nicht behoben.

## Modellunterschiede

{
  "checkpoints": {
    "D": "/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/models/psp_training_real_alle_s60_klassen",
    "OE": "/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/models/psp_training_real_alle_s60_token_oe"
  },
  "gap_tokens": {
    "D": null,
    "OE": true
  },
  "gap_classes": {
    "D": true,
    "OE": null
  }
}

Die Wiederholungen verwenden identische Start-Seeds je Zustand. Wegen unterschiedlicher Modellausgaben sind einzelne Suffixe zwischen D und OE keine gepaarten identischen Zufallspfade. Gemeinsame Zufallszahlen zwischen den Optionen werden innerhalb jedes Modells verwendet.

Artefakte: comparison.csv, by_option.csv, draws.csv, estimates.csv, work_intervals.csv, uncertainty.csv, sampling_validation.json und manifest.json. Die Originalmodelle und laufenden Trainings wurden nicht verändert.

## Ergebnis und verbleibende Fehler

OE zieht in dieser Diagnose deutlich plausibler als das bisherige D. Die mittleren vorhergesagten Kosten sinken um 49.9 %, liegen aber mit 3330.80 € weiterhin etwa 17.3-mal über dem Mittel der Simulator-Fortsetzungen (192.26 €). In 24 von 24 Optionszuständen liegt die OE-Kostenschätzung näher am Simulator-Mittel. Drei Simulatorfortsetzungen je Option reichen für keine präzise Aussage über die wahre individuelle Optionsrangfolge.

Der Median der gezogenen Kosten beträgt bei OE 204.47 € gegenüber 2385.04 € bei D. Die teuersten 10 % der OE-Suffixe verursachen 78.5 % der gesamten vorhergesagten Kosten. 60.8 % der Arbeitszeitkosten der ersten OE-Wiederholung entfallen auf Sitzungen, die bis zum gezogenen Fallende offen bleiben. Das erklärt, warum der Median wesentlich besser aussieht als der für die Policy verwendete Mittelwert.

### Konkretes OE-Beispiel: Application_703, Option start, Ziehung 279

Dieser Pfad liegt ungefähr am 90. Kostenperzentil dieser Option und Wiederholung. Er enthält ausschließlich beobachtete Event-/Zeitklassenkombinationen und erlaubte aufeinanderfolgende Eventpaare. Trotzdem ist die vollständige Sitzungsfolge inkonsistent:

- Schritt 8: `W_Complete application|resume` öffnet die Arbeit bei Tag 14,8045.
- Danach werden Angebote verschickt und andere Aufgaben begonnen; `W_Complete application` erhält keinen abschließenden Lifecycle mehr.
- Schritt 13: `W_Call after offers|start` folgt erneut auf einen `start` derselben bereits offenen Aufgabe.
- Schritt 16: `W_Call after offers|resume` folgt nach 96,68 Stunden, obwohl seit dem letzten Start kein `suspend` gekommen ist. Diese Zeit wird als aktive Arbeit gezählt.
- Am Ende bei Tag 30,1771 bleibt `W_Complete application` seit 368,94 Stunden offen, obwohl der geschäftliche Endmarker `A_Pending` vorkommt.

Ergebnis dieses Suffixes: 466.87 aktive Arbeitsstunden und 20666.16 € verbleibende Kosten. Das Problem ist hier nicht mehr ein für sich ungesehener langer `suspend`-Abstand. Es entsteht aus dem Zusammensetzen einzeln beobachteter Events und Zeitklassen bei falschem Sitzungszustand.

### Schlussfolgerung

OE behebt den bei D nachgewiesenen identischen ersten Optionsabstand und unterbindet unbeobachtete Event-/Zeitklassenkombinationen. Es verbessert die typischen Kostenziehungen deutlich. Für belastbare Eventkostenentscheidungen fehlen weiterhin Zustandsregeln für offene/pausierte Arbeit, konsistente Fallenden und eine Prüfung der langen Kosten-Ausläufer. Eine pauschale Kappung der Kosten wäre keine Validierung dieser Regeln. Ein besserer Online-Outcome von OE wurde mit dieser Diagnose noch nicht gemessen.

### Einschränkung des Architekturvergleichs

Beide Modelle: 20,482 Trainingsfälle, 4,726 Validierungsfälle, 15 Epochen, Lernrate 0,0001, Decoderlänge 60 und Wahlpunktgewicht 3. D hat 933,392 gewichtete Trainingsproben pro Epoche, OE 1,084,154. Beim vorhandenen OE waren zusätzliche Proben am Entscheidungs-Prefix bereits im Training gewichtet; bei D wurde der Optionsmodus nach dem Training aktiviert. Der Vergleich zeigt damit die Qualität der vorhandenen Modellstände, nicht ausschließlich den isolierten Effekt der gemeinsamen Tokenarchitektur.
