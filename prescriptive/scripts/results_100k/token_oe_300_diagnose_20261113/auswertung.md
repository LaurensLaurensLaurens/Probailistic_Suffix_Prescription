# Warum token_oe mit 300 Ziehungen hinter dem Causal Forest liegt

Stand: 13.11.2026 10:31 Uhr (Europe/Berlin).

Bei token_oe kommen ein schwaches bzw. unpassendes Optionssignal und erhebliche Schwankungen durch das Suffixziehen zusammen. Besonders deutlich ist der Unterschied am Wahlpunkt Handle leads. Die gezogenen Pfade verletzen häufig die im Training beobachteten direkten Eventfolgen. Diese Befunde erklären die Schwäche plausibel; eine kausale Aufteilung des Online-Rückstands auf einzelne Fehlerursachen wurde damit nicht gemessen.

## Bisherige Online-Ergebnisse auf denselben Seeds

Fertige token_oe-Seeds: 0, 1, 3, 4, 6, 8. Je Seed 3.000 simulierte Ankünfte; gewertet werden die 1.000 Fälle im Fenster 700 bis 1700. Die token_oe-Werte wurden aus dem auf 0,1 Tage gerundeten Protokoll gelesen. Die Referenzwerte stammen aus den früheren CSV-Ergebnissen derselben Seeds mit zufälliger Mitarbeiterzuteilung.

| Seed | Bank (Tage) | token_oe 300 (Tage) | Causal Forest (Tage) | token_oe länger als CF |
|---|---:|---:|---:|---:|
| 0 | 21,71 | 21,3 | 19,41 | ≈1,89 |
| 1 | 22,35 | 21,4 | 19,18 | ≈2,21 |
| 3 | 22,59 | 21,6 | 19,76 | ≈1,84 |
| 4 | 22,84 | 20,8 | 19,34 | ≈1,46 |
| 6 | 22,57 | 21,2 | 20,10 | ≈1,10 |
| 8 | 21,76 | 21,5 | 19,20 | ≈2,31 |
| Mittel | 22,30 | ≈21,30 | 19,50 | ≈1,80 |

Auf diesen Seeds wählt der Causal Forest praktisch die feste Regel: an beiden Annahmepunkten annehmen, kein zusätzliches Angebot, Handle leads fast immer überspringen. Das sind beobachtete Policy-Wahlen im Simulator; die unten untersuchten PSP-Wahlen stammen dagegen aus separaten Testprefixen des echten Logs.

## Vergleich auf 800 identischen Testprefixen

Je Wahlpunkt 200 erste Entscheidungen aus zurückgehaltenen echten Fällen, hälftig mit tatsächlich gewählter Aktion bzw. Referenz. Für genau diese Prefixe wurden die vorhandenen PSP-Bewertungen mit 100 Ziehungen den neu berechneten Causal-Forest-Effekten gegenübergestellt. Die Testfallmengen der Modelle wurden auf Gleichheit geprüft. Positiv bedeutet: Die genannte Aktion wird als schneller eingeschätzt. Ein Causal-Forest-Effekt ist hier eine Modellvorhersage, keine bekannte individuelle Wahrheit.

| Aktion | token_oe, 100 Ziehungen: Vorteil (Tage) | CF: Vorteil (Tage) | Übereinstimmung der gewählten Option |
|---|---:|---:|---:|
| Annehmen bei Konzept | 1,02 | 4,20 | 73,0 % |
| Annehmen nach Vervollständigung | 1,11 | 2,35 | 72,0 % |
| Angebot nachlegen | -0,67 | -1,18 | 65,5 % |
| Handle leads bearbeiten | 0,21 | -2,31 | 47,0 % |

Der Vorteil des Annehmens wird im PSP stark abgeschwächt. Beim Bearbeiten von Handle leads erkennt das PSP den vom CF geschätzten Nachteil nicht. Beim zusätzlichen Angebot stimmt die mittlere Richtung, aber häufig wird die andere Option gewählt.

## Neue Wiederholungsprüfung mit 300 Ziehungen

Zufällig ausgewählte 12 der obigen Prefixe je Punkt (6 je beobachtetem Arm), zwei unabhängige Zufallsstarts pro Prefix, 300 Suffixe je Option. Insgesamt 48 Prefixe, 96 Bewertungsaufrufe und 57.600 gezogene Suffixe. Alle Modell-, Policy- und Katalogdateien blieben unverändert. Die Stichprobe ist klein und erlaubt keine präzise Hochrechnung ihrer Wahlquoten auf die Online-Läufe.

| Aktion | token_oe: Vorteil (Tage) | CF auf denselben 12 Prefixen | PSP wählt Aktion | Wechsel bei Wiederholung | Standardfehler der Differenz, Median (Tage) |
|---|---:|---:|---:|---:|---:|
| Annehmen bei Konzept | 1,25 | 4,22 | 91,7 % | 2/12 | 1,02 |
| Annehmen nach Vervollständigung | 1,38 | 2,34 | 91,7 % | 2/12 | 1,08 |
| Angebot nachlegen | -1,21 | -1,33 | 25,0 % | 4/12 | 1,18 |
| Handle leads bearbeiten | 0,56 | -2,20 | 79,2 % | 3/12 | 1,08 |

Insgesamt kippt die Entscheidung bei 11/48 unveränderten Prefixen (22,9 %), obwohl jeweils 300 Ziehungen verwendet werden. Die gemessenen Standardfehler liegen um einen Tag und damit in der Größenordnung der PSP-Optionsunterschiede. Mehr Ziehungen reduzieren dieses Rauschen, korrigieren aber keinen falschen oder abgeschwächten mittleren Effekt.

Bei Angebot nachlegen sind die mittleren Effekte in dieser Probe ähnlich (PSP −1,21, CF −1,33 Tage), die Entscheidung kippt aber bei 4/12 Prefixen. Handle leads hat dagegen zusätzlich eine abweichende mittlere Richtung (PSP +0,56, CF −2,20 Tage). An beiden Annahmepunkten bleibt der PSP-Vorteil kleiner als beim CF.

## Was an den Suffixen auffällt

Die erlaubten Paare und Fallend-Events wurden aus den 25.208 echten Fällen außerhalb von model.test_cases abgeleitet, wie im vorbereiteten Kopierskript: Aktivität plus Lifecycle, ohne Einschränkung der Zeitklasse. Es gibt 414 beobachtete Paare und 22 Tokens, nach denen ein Trainingsfall endet. Dabei werden auch die Validierungsfälle innerhalb dieses äußeren Trainingssplits verwendet.

| Wahlpunkt | Suffix mit mindestens einem unbeobachteten Übergang | Ohne A_Pending / A_Denied / A_Cancelled im Suffix | Vor der Längengrenze beendet, obwohl letztes Token nie Fallende im Trainingslog |
|---|---:|---:|---:|
| Annehmen bei Konzept | 99,5 % | 36,0 % | 51,1 % |
| Annehmen nach Vervollständigung | 99,3 % | 35,4 % | 50,9 % |
| Angebot nachlegen | 98,3 % | 35,2 % | 50,8 % |
| Handle leads bearbeiten | 100,0 % | 35,6 % | 51,1 % |

Die erste Spalte zählt ganze Suffixe mit mindestens einem Verstoß, nicht den Anteil ungültiger Einzelschritte. Die dritte Spalte kennzeichnet einen vorzeitigen Stop unter der aus dem Trainingsfenster abgeleiteten Längengrenze; sie ist nicht gleichbedeutend mit der zweiten Spalte. Ein Suffix kann einen fachlichen Abschluss enthalten und danach noch weitere Events ziehen.

Die Übergangsprüfung wurde an einem zusätzlichen konkreten Prefix mit 100 Suffixen je Option kontrolliert: 3.689 von 8.168 gezogenen Übergängen (45,2 %) waren in dessen Trainingssplit unbeobachtet. Beispiele sind A_Accepted|complete → A_Accepted|complete und A_Cancelled|complete → O_Created|complete. Diese 45,2 % gelten für diesen Kontrollprefix, nicht als repräsentativer Gesamtwert.

Die frühere größere Auswertung mit 160.000 Suffixen auf 800 Prefixen zeigt dieselbe fehlende Abschlussstruktur: etwa 34–37 % der Suffixe enthalten keinen der drei fachlichen Fallabschlüsse. Diese Suffixe dauern im Mittel 12,66 Tage, gegenüber etwa 20 Tagen bei A_Pending/A_Denied und 35,55 Tagen bei A_Cancelled. Fehlerhafte Enden können dadurch die Optionserwartungswerte deutlich verändern. Dass gerade diese Fehler den Online-Rückstand verursachen, muss ein Vergleich mit eingeschränktem Ziehen erst zeigen.

## Einordnung und Grenzen

- Der aktuelle Online-Rückstand ist auf gepaarten Seeds sichtbar. Die Diagnose untersucht jedoch echte Testprefixe und keine aus den laufenden Online-Prozessen rekonstruierten Entscheidungsverläufe. Die genauen Wahlquoten der bereits fertigen token_oe-Online-Seeds sind bisher nicht auf Platte gespeichert; die laufende Auswertung schreibt die vollständigen Tabellen erst am Ende.
- Die Suffixbewertung liefert an den Annahmepunkten ein schwächeres Signal und bei Handle leads ein unpassendes mittleres Signal. Beim zusätzlichen Angebot ist Rauschen besonders gut sichtbar. Diese Ursachen lassen sich anhand dieser Daten unterscheiden, aber nicht in konkrete verlorene Online-Tage aufteilen.
- Der Causal Forest bereinigt seine Effektmodelle um prognostische und Treatment-Einflüsse. PSP lernt die beobachtete Fortsetzung. Das ist ein methodischer Unterschied; seine alleinige Ursachewirkung wurde hier nicht isoliert. Beide verwenden Beobachtungsdaten, und CF-Vorhersagen sind kein individuelles Ground Truth.
- Das künstliche Einfügen der Option mit 1 ms ist bei token_oe bereits ersetzt: Der Optionsabstand wird zunächst gezogen und passend im Encoder eingesetzt. Der gefundene Rückstand ist deshalb kein Beleg für dieses frühere Eingabeproblem.
- Übergangsregeln sind ein begründeter nächster Test für die fehlerhaften Suffixe. Ihre Wirkung auf die Online-Policy bleibt offen; mehr Monte-Carlo-Ziehungen allein beheben die Strukturfehler nicht.

## Daten und Reproduktion

- `online_matched_seeds.csv`: gepaarter Online-Zwischenstand aus den Protokollen und früheren CSV-Dateien.
- `cf_online_choice_rates_matched_seeds.csv`: Online-Wahlquoten des CF auf den passenden Seeds.
- `offline800_same_prefix_effects.csv`, `offline800_summary.csv`: identische Testprefixe für PSP und kausale Schätzer.
- `probe300_all.csv`, `probe300_summary.csv`: Wiederholungen mit 300 Ziehungen und Übergangsprüfung.
- `existing_suffix_summary.csv`, `existing_suffix_end_states.csv`: größere vorhandene Suffixauswertung.
- `transition_examples_validation.json`: unabhängig kontrollierte Pfade und unbeobachtete Paare.
- `observed_first_option_gaps.csv`: ergänzende tatsächlich beobachtete erste Optionsabstände der 800 Entscheidungen; daraus wurde keine kausale Effektbehauptung abgeleitet.
- `settings.json`, `metadata.json`: Parameter und Modellfingerabdrücke.

Hauptprüfung aus `process-simulation-engine/` mit einem neuen Ausgabeordner wiederholen:

```bash
.venv/bin/python -u ../prescriptive/scripts/diagnose_token_oe_choices_20261113.py \
  --out ../prescriptive/scripts/results_100k/token_oe_300_diagnose_replikation \
  --prefixes-per-point 12 --samples 300 --repeats 2 --workers 2
```

Das Hauptskript erzeugt die Online-, 800-Prefix- und 300er-Probetabellen. Die ergänzenden vorhandenen Suffixaggregationen, die unabhängige Pfadkontrolle, die beobachteten Abstände und dieser Bericht wurden separat aus den angegebenen Quellen erstellt.
