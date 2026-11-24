# Auswertung der drei CATE-ERM-Policies

BPIC17, γ = 0,1 / Tag. Alle Modelle auf denselben 25.208 Fällen trainiert, 6.301 echte Fälle zurückgehalten. Training-Seed 42. Online: Seeds 0–9, je 3.000 Ankünfte, Warmup 700, Cooldown 1.300; je 1.000 gewertete Fälle, insgesamt 10.000 pro Policy. Vier parallele Prozesse. Vier Control-Flow-Punkte, keine first_only-Einschränkung, Mitarbeiterzuteilung random. Bestehendes Simulator-Modell models/next_event/bpic17; kein neues Simulator-Training.

## Ergebnisse

Alle Größen in Tagen; kleiner ist besser. ERM ist das empirische entropische Risiko der simulierten Durchlaufzeiten bei γ = 0,1. Ausgewiesen ist das Mittel der zehn je Seed berechneten Kennzahlen; die Daten werden nicht vor der Risikoberechnung über Seeds zusammengeworfen.

| Methode | Durchlaufzeit (Tage) | Durchlaufzeit P90 (Tage) | Durchlaufzeit P95 (Tage) | CVaR 90 % (Tage) | ERM γ=0.1 (Tage) | ERM γ=0.1 (Tage) ggü. Bank | ERM γ=0.1 (Tage) ggü. Bank 95-%-KI unten | ERM γ=0.1 (Tage) ggü. Bank 95-%-KI oben |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bank | 22.328 | 38.580 | 47.415 | 51.265 | 59.058 | 0.000 | 0.000 | 0.000 |
| S-Learner ERM | 22.346 | 39.484 | 47.394 | 52.159 | 62.784 | 3.726 | -17.092 | 24.544 |
| X-Learner ERM | 22.086 | 41.003 | 50.126 | 54.963 | 79.535 | 20.478 | -38.681 | 79.636 |
| Causal Forest ERM | 20.745 | 37.908 | 47.798 | 52.221 | 60.319 | 1.261 | -12.610 | 15.132 |

## Gepaarter Vergleich zur Bank

- S-Learner ERM: ERM 62.784 Tage; Differenz zur Bank +3.726 Tage (95-%-KI [-17.092; +24.544]), im KI nicht klar von der Bank getrennt.
- X-Learner ERM: ERM 79.535 Tage; Differenz zur Bank +20.478 Tage (95-%-KI [-38.681; +79.636]), im KI nicht klar von der Bank getrennt.
- Causal Forest ERM: ERM 60.319 Tage; Differenz zur Bank +1.261 Tage (95-%-KI [-12.610; +15.132]), im KI nicht klar von der Bank getrennt.

Die 95-%-Konfidenzintervalle beruhen auf den zehn gepaarten Seed-Differenzen (t-Intervall). Sie beschreiben die Streuung über Simulationswiederholungen, nicht die Unsicherheit der kausalen Schätzer oder die Übertragbarkeit auf den echten Betrieb. Die neuen CSVs enthalten volle Rechengenauigkeit; die Tabellen im HTML-Bericht und die historischen EV-CSVs sind auf drei Nachkommastellen gerundet.

![Vergleich zur Bank](vergleich_zur_bank.png)

## Auffälligkeit des Trainings

Der S-Learner schätzt auf sämtlichen Entscheidungsproben der zurückgehaltenen Fälle an allen vier Wahlpunkten Nulleffekte; die Policy löst diese Gleichstände zufällig auf. Ihre Ergebnisse belegen damit keine gelernte risikosensitive Unterscheidung der Aktionen. Frühes Stoppen wählte je Punkt nur ein bis zwei Bäume. Auch mehrere Outcome-Hilfsmodelle des X-Learners und Causal Forest stoppten nach ein bis zwei Bäumen.

![Aktionsanteile auf echten Testfällen](aktionsanteile_testfaelle.png)

Bei diesem Gamma konzentriert sich das exponentielle Trainingsgewicht extrem auf einzelne lange Fälle. Die folgende Diagnose bezieht sich auf den inneren Trainingsteil je Wahlpunkt. Wiederholte Entscheidungsproben desselben Falls wurden vor der Gewichtsdiagnose zusammengefasst. Die effektive Fallzahl ist 1 / Summe der quadrierten normierten Fallgewichte; sie ist keine nominale Stichprobengröße.

| Wahlpunkt | Innere Trainingsproben | Innere Trainingsfälle | Längste Durchlaufzeit (Tage) | Exp-Gewicht größter Fall | Exp-Gewicht Top-5-Fälle | Effektive Fallzahl exp-Gewichte |
| --- | --- | --- | --- | --- | --- | --- |
| angebot_nachlegen | 28463 | 22066 | 286.072 | 1.000 | 1.000 | 1.000 |
| annehmen_bei_konzept | 17536 | 17536 | 286.072 | 0.932 | 1.000 | 1.145 |
| annehmen_nach_vervollstaendigung | 23154 | 11876 | 169.072 | 0.733 | 0.986 | 1.691 |
| handle_leads | 14770 | 14751 | 259.887 | 1.000 | 1.000 | 1.001 |

Die Gewichtskonzentration ist ein Befund dieses Gamma und dieser Daten; Verluste wurden weder abgeschnitten noch wurde das Gamma nachträglich verändert. Für eine Methodenbewertung sind weitere Trainings-/Gamma-Sensitivitätsanalysen erforderlich.

## Historischer EV-Vergleich

Die neuen Bank-Läufe reproduzieren die Bank-Kennzahlen des früheren CATE-EV-Laufs auf 0,001 Tage genau. vergleich_ev_erm.csv enthält zusätzlich gepaarte EV-zu-ERM-Differenzen für Mittel, P90, P95 und CVaR.

Die früheren EV-CSVs enthalten keine Einzelfall-Durchlaufzeiten; ihr ERM lässt sich daraus nicht nachberechnen.

## Dateien

- bericht.html: Standardbericht mit Ergebnissen je Ansatz.
- ergebnis_gesamt.csv / laeufe_gesamt.csv: alle vier Policies zusammengeführt.
- zusammenfassung.csv / differenzen_zur_bank.csv / vergleich_ev_erm.csv: kompakte Auswertung.
- effekte_testfaelle.csv / training_diagnostik.csv / training.log: Training und Diagnose.
- catalog.yaml / laufparameter.json: Einstellungen, Modell- und Quellcode-Prüfsummen.
- checkpoints/: atomar gespeicherte Einzelläufe einschließlich Durchlaufzeiten je Fall.
- online_resume.log: vollständiges Log der in Batches ausgeführten Auswertung.
- online.log: erster, mit SIGTERM beendeter Versuch; seine nicht gespeicherten Teilergebnisse wurden nicht verwendet.
