# BPIC17: 30.000 randomisierte Simulatorfälle

Erzeugt am 12.11.2026 mit dem gespeicherten BPIC17-Simulator. Zehn unabhängig gestartete Läufe mit jeweils 3.000 Fällen ergeben **genau 30.000 eindeutige Fälle**, 1.458.352 Events und 736.142 protokollierte Entscheidungen. Die 100 Pilotfälle gehören nicht zu diesem Datensatz.

- `log.pkl`: Event-Log als pandas DataFrame; XES-Spalten, `outcome` (negative Durchlaufzeit in Tagen), `rct=True`, `simulation_run` (0 bis 9). Fall-IDs beginnen mit `R00_` bis `R09_`.
- `entscheidungen.pkl`: vollständiges Entscheidungsprotokoll. `case`, `point`, `kind`, `options` (tatsächlich angebotene Optionen), `choice`, `probability`, `time` (UTC-Sekunden seit Epoch), `event_pos` (Position im Fall, ab 0), `assigned_value`, `simulation_run`.
- `design.json`: vor Erzeugung festgelegte Kategorien und empirische Verteilungen innerhalb der Kategorien; Reaktionszeitgrenze aus dem echten Log.
- `rct.json`: Parameter, Seeds, Umfang, SHA256 der Daten und Quell-/Modellartefakte, Zusammenfassung je Lauf.
- `laeufe/run00/` bis `run09/`: atomar gespeicherte Daten, Entscheidungen und Kennzahlen je Lauf.

## Randomisierung

Alle sechs Control-Flow-Kandidaten, zehn Mitarbeiterpunkte und vier weiteren Variablen sind Teil der Studie. Die BPIC17-Produktionskonfiguration wurde dafür nicht verändert; die Studie verwendet `screening/configs/bpic17_alle.yaml`. Als Basiszuteilung an übrigen Schritten bleibt das gelernte Bankverhalten aktiv. Es gibt keine Beimischung nicht randomisierter Bankfälle (`delta=0`).

Die Policy wählt bei jedem Besuch eines Control-Flow- oder Mitarbeiterpunkts gleichverteilt aus den angebotenen Optionen. Mitarbeiter müssen kompetent, im Dienst und frei sein. Ein Pool mit einer Person hat Wahrscheinlichkeit eins und bietet keinen Mitarbeitervergleich. Ankünfte und Fallattribute werden weiterhin vom normalen Simulator erzeugt; jeder Lauf startet am Beginn des Logs. Die zehn Läufe sind keine aufeinanderfolgenden Kalenderabschnitte.

`channel` und `angebot_nachlegen` teilen sich dasselbe Entscheidungs-Event. Im Simulator wird aus der Vereinigung ihrer drei Optionen gezogen. Für den jeweiligen binären AIPW-Vergleich werden die zwei zugehörigen Optionen betrachtet; bedingt auf diese beiden Optionen beträgt die bekannte Propensity jeweils 1/2.

An jedem `O_Create Offer` werden Angebotsverhältnis, Laufzeit und Auszahlungsverhältnis unabhängig einer von jeweils drei Kategorien zugewiesen. Der konkrete Wert innerhalb der Kategorie stammt aus deren empirischer Verteilung im echten Log. Die Monatsrate wird passend zu zufällig gewähltem Betrag und Laufzeit neu berechnet. Bei einem beantragten Betrag von null ist das Angebotsverhältnis nicht definiert; diese Fälle erhalten an diesem einen Punkt keine Kategorienzuweisung. Das ist kein fehlender oder mit null geschätzter Effekt.

Bei `A_Create Application` wird eine Zielwartezeit zufällig der Kategorie `schnell` oder `langsam` zugewiesen. Die vorab am echten Log bestimmte Grenze beträgt **3,597911 Stunden**. Die Zielzeit ersetzt die geplante Fälligkeit des ersten `W_Complete application|start`, soweit der vorherige Prozessverlauf dies zulässt. Vorherige Events und die Verfügbarkeit von Mitarbeitern können den tatsächlichen Beginn weiter verzögern. Die Auswertung verwendet deshalb die **Zuweisung (Intention to treat)**, einschließlich der Fälle ohne späteren Arbeitsbeginn, und kategorisiert nicht nach der nachträglich erreichten Wartezeit. Dieser Effekt unterscheidet sich vom beobachteten Lag-Kategorienvergleich im echten Log.

## Auswertung und Grenzen

AIPW verwendet die bekannten Zuteilungswahrscheinlichkeiten ohne Clipping und Outcome-Modelle mit fünf Cross-Fitting-Folds. Ein Simulatorlauf bleibt vollständig innerhalb eines Folds. Die 95-%-Intervalle verwenden nach Simulatorlauf geclusterte Standardfehler und eine t-Verteilung; innerhalb eines Laufs konkurrieren Fälle um gemeinsame Mitarbeiter. Die Intervalle sind punktweise, ohne Korrektur für multiples Testen.

Pro Fall und Interventionspunkt wird die erste protokollierte Entscheidung ausgewertet. Alle weiteren Entscheidungen folgen weiterhin der randomisierten Studienregel. Mitarbeiter werden nur verglichen, wenn sie verfügbar sind und mindestens zwei Personen zur Wahl stehen. Die Anzahl geeigneter Fälle unterscheidet sich daher je Punkt; 30.000 Gesamtfälle bedeuten nicht 30.000 Vergleichsfälle an jedem Punkt.

Der unveränderte Simulator beendet sehr lange Verläufe an seiner bestehenden maximalen Eventzahl. Dies betraf **9 von 30.000 Fällen**. Diese Fälle bleiben in der Auswertung, damit nicht nach einem Treatment-Ergebnis selektiert wird. Zwangszuteilungen außerhalb des Dienstplans traten nicht auf. Die Daten beschreiben Wirkungen innerhalb des gelernten Simulators; sie ersetzen keinen RCT in der tatsächlichen Bank.

Berichte: [alle Punkte](../bericht.html), [Übersicht CSV](../alle_interventionspunkte.csv), [sämtliche Optionen und Intervalle](../aipw_alle_optionen.csv).
