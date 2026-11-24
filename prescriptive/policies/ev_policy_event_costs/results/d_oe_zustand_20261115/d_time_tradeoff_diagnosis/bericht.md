**Warum D mit Zeitziel Arbeitskosten spart, aber länger dauert**

Verglichen werden die vorhandenen abgeschlossenen Läufe für Seed 0 mit denselben 200 Fall-IDs. Die als D-Zeit bezeichnete Policy wählt anhand der im D-Modell geschätzten mittleren Suffixdauer. Die neuronalen Gewichte wurden für diese Auswahlregel nicht neu trainiert. Eine optimale geschätzte Wahl garantiert keine optimale tatsächliche Simulatorfortsetzung.

| Kennzahl je Fall | Bank | D mit Zeitziel | D minus Bank |
|---|---:|---:|---:|
| Gesamte Kalenderzeit | 20,37368 Tage | 21,30802 Tage | +0,93435 Tage |
| Zeit mit mindestens einer offenen aktiven Arbeitssitzung | 50,10558 Minuten | 44,57829 Minuten | −5,52729 Minuten |
| Kalenderzeit ohne aktive Arbeitssitzung | 488,13313 Stunden | 510,64955 Stunden | +22,51642 Stunden |
| Arbeitskosten | 41,69578 € | 37,23965 € | −4,45613 € |

Die Zeitdifferenz entsteht damit in diesem Lauf aus der inaktiven Kalenderzeit. Weniger aktive Arbeit ist mit längeren Pausen oder Wartezeiten vereinbar. Die Zuordnung erfolgt aus den Lifecycle-Events, nicht durch eine nachträgliche Vermutung über Ursachen: Zwischen zwei Events zählt die Zeit als aktiv, wenn mindestens ein Arbeitsabschnitt offen ist. Parallele Abschnitte werden für diese Kalenderzeitzerlegung nur einmal gezählt.

Unter den inaktiven Zeitabschnitten steigt die Summe vor `O_Create Offer` um 12,08 Stunden pro Fall und vor `W_Call after offers|resume` um 6,07 Stunden pro Fall. Andere Abschnitte steigen oder sinken ebenfalls. Diese Aufteilung zeigt, wo die Zeitdifferenz in den beobachteten Verläufen liegt; sie isoliert nicht den kausalen Effekt einer einzelnen Policyentscheidung.

Die tatsächlich ausgeführten Optionen unterscheiden sich: Zusätzliche Angebote am Punkt `angebot_nachlegen` treten bei D-Zeit 31-mal und bei der Bank 21-mal auf. D-Zeit führt am Leads-Punkt 26 Starts aus, die Bank 24. Bei D mit Gesamtkostenziel sind es 18 zusätzliche Angebote und 16 Leads-Starts. Die Häufigkeiten werden aus den tatsächlichen Eventfolgen gezählt und enthalten auch Bankfortsetzungen nach einem Policy-Rückfall.

Für die verbleibende Ursachenanalyse sind zwei Modellgrenzen relevant. Erstens gibt die aktuelle Zeitschätzung nur bei einem Teil der Vergleiche ein ausreichend getrenntes Monte-Carlo-Signal: Bei 478 von 659 adaptiv bewerteten Zweierwahlen behält die Zeitpolicy den Bankvorschlag bei. Zweitens ersetzt die Korrektur die numerischen Zeitabstände durch empirische Trainingsverteilungen, bedingt auf vorheriges Event, aktuelles Event und Arbeitszustand. Kalendermerkmale und etwa das genaue Alter des letzten Angebots gehen nicht direkt in diese zusätzliche Abstandsschätzung ein. Das ist eine plausible Quelle ungenauer Wartezeitprognosen; ihr Anteil am gemessenen Rückstand wurde nicht durch eine isolierte Ablation bestimmt. Das Eventnetz selbst sieht weiterhin seinen Prefix.

Dateien: summary.csv, case_time_components.csv, paired_d_time_minus_bank.csv und inactive_gap_hours_by_next_event.csv. Die laufenden weiteren Seeds und die Modelle wurden für diese Auswertung nicht verändert.
