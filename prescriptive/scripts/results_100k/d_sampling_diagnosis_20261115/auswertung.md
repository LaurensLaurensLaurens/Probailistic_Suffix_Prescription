# Warum D nicht besser abschneidet: Ziehverhalten und Simulator

Untersucht wurde die bisherige D-Variante mit Übergangssperre und einzeln gezogenem Optionsabstand. Referenz: Eventkosten, 200 gewertete Fälle, 500 Ziehungen je Option, Seed 0. Das aktuell trainierende D mit konstantem TF 1 ist ein anderes, hier nicht bewertetes Modell.

## Befund

Beim Eventkosten-Ziel liegt das klarste systematische Problem im PSP: D kombiniert lange Zeitabstände mit laufenden Arbeitssitzungen und erzeugt dadurch massiv überhöhte Arbeitskosten. Hinzu kommt eine konkrete Schwäche der Simulator-/Policy-Schnittstelle: Optionen werden ohne Prüfung bereits offener Sitzungen angeboten. Monte-Carlo-Rauschen kann dann eine seltene, besonders teure Wahl auslösen. Beim Durchlaufzeit-Ziel sind die Kostenfehler weniger sichtbar; dort bleiben schwache und verrauschte Optionsunterschiede sowie unvollständige Pfade als gemessene Probleme.

## Wie geprüft wurde

Alle 582 historischen Entscheidungen wurden mit ihren gespeicherten Wahlen erneut simuliert. Alle 7.954 gewerteten Events, Mitarbeiter, Zeitstempel und Fallkosten stimmen überein. Aus jedem der vier Wahlpunkte wurden drei verschiedene Fälle mit Seed 20261115 ausgewählt: 12 Zustände. Je Zustand und Option wurden dreimal 500 D-Suffixe gezogen, insgesamt 36.000 Suffixe. Zusätzlich wurde die Simulator-Engine am exakten Wahlzustand gesichert. Jede der zwölf Kopien reproduziert bei gespeicherter Fortsetzung den ursprünglichen Ziel-Fall exakt.

Für den Optionsvergleich wurde danach jeweils genau eine Aktion erzwungen und die gesamte Engine mit normalem Bankverhalten fortgesetzt, einschließlich anderer Fälle, Dienstplan und Mitarbeiterkonkurrenz. Je Option gibt es drei Fortsetzungen: einmal mit dem gesicherten Zufallszustand, zweimal mit neu gezogenen künftigen Zufallszahlen. Die drei Fortsetzungen sind eine kleine Diagnose, keine präzise Schätzung individueller kausaler Effekte und kein neuer 200-Fälle-Online-Benchmark.

Außerdem wurden für 13 Entscheidungen die ursprünglichen D-Zufallszustände rekonstruiert. Alle 13 neu berechneten Wahlen stimmen mit dem historischen 500er-Lauf überein. Modell, Simulator, Trainingscode und laufende Trainings wurden nicht verändert.

## 1. Kosten werden durch falsche Arbeitszeiten verzerrt

Über beide Optionen der zwölf Zustände schätzt D im Mittel **6654.04 €** verbleibende Kosten. Die 72 Simulator-Fortsetzungen ergeben im Mittel **192.26 €**. D zieht durchschnittlich **139.5 aktive Arbeitsstunden**; **98.45 %** seiner geschätzten Kosten sind Zeitkosten. Die Kostenrechnung wurde je Suffix gegen die vorhandene Outcome-Bewertung geprüft. Der Fehler entsteht in den gezogenen Ereignissen und Zeiten, bevor daraus Kosten berechnet werden.

Zum Größenvergleich: Die unveränderten Tarife ergeben im gesamten echten Log 168,18 € und 0,782 aktive Stunden je vollständigem Fall. Die Bank im Online-Lauf liegt bei 174,03 € und 0,844 Stunden. Diese beiden Gesamtfall-Mittel sind Kontextwerte, keine mit den zwölf Entscheidungszuständen gepaarte Stichprobe.

Konkreter D-Pfad: Application_703, Option `W_Handle leads|start`, Wiederholung 0, Ziehung 77. Dieser Pfad liegt ungefähr bei den medianen Kosten der 500 Suffixe dieser Option. Nach `W_Call after offers|start` und einem fast sofortigen `A_Complete` folgen bis zum `W_Call after offers|suspend` **103,39 Stunden**. Allein diese Arbeitssitzung kostet **4.135,65 €** bei 40 €/h. Der gesamte reale Log hat in dieser Kosten-/Sitzungsdefinition keinen Arbeitsabschnitt über 19,12 Stunden; Median 1,63 Minuten, 99. Perzentil 45,59 Minuten.

Die Gesamt-Restzeit wirkt dabei weniger auffällig: D 18.14 Tage gegenüber 18.60 Tagen im kleinen Simulatorvergleich. Eine ungefähr plausible Gesamtdauer garantiert keine plausible Aufteilung in Arbeitszeit und Pause.

## 2. Getrennte Köpfe: Die erzwungene Option ändert ihren eigenen ersten Abstand nicht

In allen 36 geprüften D-Ziehungsaufrufen sind die 500 ersten Optionsabstände für beide Optionen elementweise identisch. Das ist durch den D-Code erklärbar: Event und Zeitklasse werden aus demselben bisherigen Zustand getrennt gezogen; die Zeitklasse und numerische Zeitverteilung sehen das gerade erzwungene aktuelle Event nicht. Mit gemeinsamen Zufallszahlen ergibt das dieselben Abstände. Erst beim erneuten Encodieren des Prefixes beeinflusst die Option die folgenden Ereignisse.

Beispiel: ID 123 (Application_762, `annehmen_bei_konzept`). Die Simulator-Verteilung des ersten reinen Modellabstands liegt für `start` bei durchschnittlich 15,76 Stunden, für `A_Accepted` bei 5,90 Minuten (je 10.000 Ziehungen, vor zusätzlicher Mitarbeiterwartezeit). D zieht für beide Optionen exakt dieselbe Abstandsfolge. Damit fehlt schon im ersten Schritt ein Teil des echten Optionsunterschieds des ausgewerteten Simulators.

## 3. Die Übergangssperre erfüllt ihre Aufgabe, aber keine vollständige Ablaufprüfung

In 80.9 % der 12.000 gespeicherten Suffixe der ersten Wiederholung kommt mindestens eine Kombination aus Event und Zeitklasse vor, die im gesamten echten Log nie vorkommt (8.1 % aller gezogenen Events). Die Sperre prüft Aktivität|Lifecycle → Aktivität|Lifecycle sowie erlaubte Endtokens. Sie prüft weder die Zeitklasse noch den Zustand aller Arbeitssitzungen.

**34.45 %** der 36.000 Suffixe enden ohne `A_Pending`, `A_Denied` oder `A_Cancelled`, obwohl auch der Prefix kein solches Event enthält. Im gesamten echten Log fehlen diese geschäftlichen Abschlussmarker bei 0,31 % der Fälle. **26.60 %** der Suffixe lassen eine Arbeitssitzung offen; im echten Gesamtlog sind am Fallende keine geöffneten Abschnitte übrig. Nur **0.0056 %** der Suffixe erreichen die Längengrenze. Das hier gemessene vorzeitige Ende liegt damit überwiegend an gezogenem EOS, nicht am Längenlimit.

`withdraw` oder `ate_abort` ohne vorherigen Beginn sind im echten Log häufig und können das Zurückziehen einer nur geplanten Aufgabe bedeuten. Die Gesamtzahl solcher Endevents allein wurde deshalb nicht als Beweis ungültiger Pfade verwendet.

## 4. Fall 750: Ein konkretes Zusammenspiel von PSP-Rauschen und Simulatorzustand

Historischer Prefix: `W_Complete application|start` → `A_Concept`. Die Arbeitssitzung ist bereits offen. Trotzdem erlaubt der Wahlpunkt wieder `W_Complete application|start`. Dieser Zustand tritt bei 36 der 162 Annahmeentscheidungen auf. D wählt in 35 davon `A_Accepted`; in Fall 750 wählt es erneut `start`.

Die rekonstruierten Originalschätzungen mit 500 Ziehungen sind: erneuter Start **5222.64 €**, Annahme **5507.13 €**. Der vermeintliche Startvorteil ist **284.50 €**, sein gepaarter Monte-Carlo-Standardfehler aber **644.71 €**. In fünf neuen 500er-Ziehungen gewinnt der Start nur 1-mal. Die ursprüngliche Wahl ist also instabil.

Der Simulator legt zwischen `A_Concept` und erneutem `start` 20,99 Stunden. Da die alte Sitzung offen ist, werden dafür 944,56 € aktive Arbeit berechnet. Die Engine behandelt den erneuten Beginn nicht als neue Mitarbeiterzuordnung und beendet die alte Sitzung nicht vor dieser Wartezeit. Das passt zur aktuellen Kostenlogik, aber zeigt eine fehlende Zustandsprüfung an der Interventionsschnittstelle.

Die erste Zeitverteilung lässt sich hier getrennt prüfen: In 10.000 Ziehungen aus dem gesicherten Simulatorzustand liegt der erste Abstand für erneuten Start im Mittel bei 19.13 Stunden, für Annahme bei 4.65 Minuten. Da die Arbeit schon offen ist, entspricht dieser Unterschied allein **857.25 €** zusätzlichen Arbeitskosten für den Start. D zieht den ersten Abstand für beide Optionen gleich und lässt diesen Unterschied weg. Hält man alle späteren D-Events und Abstände fest und ersetzt nur diesen ersten erwarteten Kostendifferenzterm durch den Simulatorwert, kippt die ursprüngliche D-Wahl: Start minus Annahme wird von -284.50 € zu **572.75 €**. Das ist eine rechnerische Zerlegung dieses Zustands, kein getesteter neuer Policy-Lauf.

Im gesicherten Zustand und mit anschließender Bankfortsetzung kosten die beiden Varianten **1034.46 €** (Start) und **540.12 €** (Annahme). Auch in den zwei weiteren Fortsetzungen ist Annahme günstiger; die einzelnen Fortsetzungen stehen in `case750_branches.csv`. Die 944,56 € dieses einen Arbeitsabschnitts sind größer als der gesamte Kostennachteil des ursprünglichen 200-Fälle-Laufs (590.20 €). Das zeigt die Empfindlichkeit des Benchmarks, beweist aber nicht, dass eine einzelne Änderung alle anderen Verläufe unverändert ließe.

## 5. Mehr Ziehungen lösen verschiedene Probleme unterschiedlich gut

Kosten: Bei den zwölf zufällig ausgewählten Zuständen ist die Kostenrangfolge in drei Wiederholungen jeweils stabil. Die massive Verzerrung bleibt dabei bestehen. Mehr Ziehungen reduzieren Zufallsrauschen, reparieren aber weder fehlende Sitzungszustände noch die falsche Kopplung von Event und Zeit. Fall 750 ist ein zusätzlicher gezielt untersuchter Ausreißer und zeigt, dass einzelne wichtige Kostenentscheidungen trotzdem stark rauschen.

Durchlaufzeit: Bei gleicher D-Suffixbasis wechselt die bevorzugte Zeitoption in **25 %** der zwölf Zustände zwischen den drei 500er-Ziehungen. Der mediane gepaarte Standardfehler eines Zeitunterschieds beträgt **0.74 Tage**. Nur **27.8 %** der 36 Vergleiche liegen weiter als zwei Monte-Carlo-Standardfehler von null entfernt. Das ist Unsicherheit der PSP-Schätzung, kein Konfidenzintervall des echten Interventionseffekts.

## Einordnung und nächste Schritte

Der Simulator ist kein gesicherter Wahrheitsmaßstab für reale Interventionen: Auch er lernt aus Beobachtungsdaten. Seine durchschnittlichen Kosten liegen näher am echten Log als die D-Prognosen, doch er erzeugt ebenfalls seltene unplausibel lange Arbeitsabschnitte, etwa 30,29 Stunden in Bank-Fall 766. Die 200-Fälle-Auswertung mit einem Seed und 176,98 € gegenüber 174,03 € reicht nicht für die Aussage, D sei grundsätzlich schlechter als die Bank. Der kleine Branch-Vergleich misst zudem genau eine Intervention mit Bankfortsetzung; die spätere wiederholte Anwendung der PSP-Policy ist ein weiterer Unterschied zum vollen Online-Lauf.

Priorität 1: Wahloptionen gegen den aktuellen Sitzungszustand prüfen und die Bedeutung eines erneuten `start` bei bereits offener Arbeit festlegen. Priorität 2: Zeitklassen und Abstände an aktuelles Event/Lifecycle und offenen Arbeitszustand koppeln, anschließend aktive Stunden und Kosten auf zurückgehaltenen realen Prefixen kalibrieren. E hat bereits die Eventbedingung für die Zeitklasse; es sollte genau dieselbe Arbeitszeitdiagnose bestehen, bevor sein Nutzen für das Eventkosten-Ziel beurteilt wird. Priorität 3: EOS und Lifecycle-Konsistenz über den ganzen Suffix prüfen. Priorität 4: unsichere Optionsunterschiede mit gezielt zusätzlichen Ziehungen und einer begründeten Rückfallregel behandeln. Danach einen Vergleich über mehr Fälle und mehrere Seeds durchführen.

TF 1 und Training auf Simulator-RCT-Daten können andere Fehlerquellen adressieren; sie garantieren nicht, dass die hier nachgewiesene Zeitkopplung und Zustandslücke verschwinden.

## Konkrete Stellen im Code

- [PSP: getrennte Event-/Zeitklassenziehung](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/estimators/psp_model.py:891) und [Optionsabstand vor erneutem Encodieren](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/estimators/psp_model.py:1262). Die erzwungene aktuelle Option konditioniert bei D ihren eigenen Abstand nicht.
- [Simulator: Auslösen der Control-Flow-Wahl](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/process-simulation-engine/simulation/case_engine_lifecycle_aware.py:304) prüft die vorhergesagte Option, aber nicht, ob die betreffende Sitzung schon geöffnet ist.
- [Simulator: Mitarbeiterzuordnung bei start/resume](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/process-simulation-engine/simulation/case_engine_lifecycle_aware.py:359) überspringt eine neue Zuordnung, wenn die Aktivität schon in `case.sessions` steht. Die zuvor gezogene Zeit läuft trotzdem ab.
- [Kosten: erneuter Beginn bei offener Sitzung](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/policies/ev_policy_event_costs/costs.py:178) bucht die bisher offene Arbeitszeit und eröffnet den Abschnitt erneut.
- [Policy: reine Wahl nach Mittelwert](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/policies/ev_policy/psp_policy.py:46) beachtet keinen Standardfehler oder Mindestabstand zur anderen Option.

Artefakte: `summary.json`, `prediction_vs_simulator.csv`, `psp_estimates.csv`, `original_seed_psp_estimates.csv`, `psp_draws.csv`, `simulator_branches.csv`, `choice_session_context.csv`, `case750_branches.csv`, `sampled_work_intervals.csv`, `real_lifecycle_summary.json`, `simulator_first_gap_distributions.csv`, `diagnose.png`.
