# Auswertung der drei CATE-ERM-Policies

BPIC17, γ = 0,1 / Tag. Alle Modelle auf denselben 25.208 Fällen trainiert, 6.301 echte Fälle zurückgehalten. Training-Seed 42. Online: Seeds 0–9, je 3.000 Ankünfte, Warmup 700, Cooldown 1.300; je 1.000 gewertete Fälle, insgesamt 10.000 pro Policy. Vier parallele Prozesse. Vier Control-Flow-Punkte, keine first_only-Einschränkung, Mitarbeiterzuteilung nach gelerntem Bankverhalten (learned). Bestehendes Simulator-Modell models/next_event/bpic17; kein neues Simulator-Training.

## Ergebnisse

Alle Größen in Tagen; kleiner ist besser. ERM ist das empirische entropische Risiko der simulierten Durchlaufzeiten bei γ = 0,1. Ausgewiesen ist das Mittel der zehn je Seed berechneten Kennzahlen; die Daten werden nicht vor der Risikoberechnung über Seeds zusammengeworfen.

| Methode | Durchlaufzeit (Tage) | Durchlaufzeit P90 (Tage) | Durchlaufzeit P95 (Tage) | CVaR 90 % (Tage) | ERM γ=0.1 (Tage) | ERM γ=0.1 (Tage) ggü. Bank | ERM γ=0.1 (Tage) ggü. Bank 95-%-KI unten | ERM γ=0.1 (Tage) ggü. Bank 95-%-KI oben |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bank | 22.600 | 39.065 | 47.876 | 52.021 | 62.364 | 0.000 | 0.000 | 0.000 |
| S-Learner ERM | 22.646 | 39.953 | 48.295 | 52.995 | 57.743 | -4.621 | -15.594 | 6.352 |
| X-Learner ERM | 22.195 | 41.169 | 50.315 | 55.700 | 72.996 | 10.632 | -4.740 | 26.004 |
| Causal Forest ERM | 20.810 | 38.602 | 47.240 | 52.123 | 65.505 | 3.141 | -14.020 | 20.303 |

## Gepaarter Vergleich zur Bank

- S-Learner ERM: ERM 57.743 Tage; Differenz zur Bank -4.621 Tage (95-%-KI [-15.594; +6.352]), im KI nicht klar von der Bank getrennt.
- X-Learner ERM: ERM 72.996 Tage; Differenz zur Bank +10.632 Tage (95-%-KI [-4.740; +26.004]), im KI nicht klar von der Bank getrennt.
- Causal Forest ERM: ERM 65.505 Tage; Differenz zur Bank +3.141 Tage (95-%-KI [-14.020; +20.303]), im KI nicht klar von der Bank getrennt.

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

Die neuen Bank-Läufe stimmen nicht vollständig mit dem früheren CATE-EV-Lauf überein. vergleich_ev_erm.csv ist deshalb nur ein beschreibender historischer Vergleich; keine gepaarten KIs ausgewiesen.

Die früheren EV-CSVs enthalten keine Einzelfall-Durchlaufzeiten; ihr ERM lässt sich daraus nicht nachberechnen.

## Dateien

- bericht.html: Standardbericht mit Ergebnissen je Ansatz.
- ergebnis_gesamt.csv / laeufe_gesamt.csv: alle vier Policies zusammengeführt.
- zusammenfassung.csv / differenzen_zur_bank.csv / vergleich_ev_erm.csv: kompakte Auswertung.
- effekte_testfaelle.csv / training_diagnostik.csv / training.log: Training und Diagnose.
- catalog.yaml / laufparameter.json: Einstellungen, Modell- und Quellcode-Prüfsummen.
- checkpoints/: atomar gespeicherte Einzelläufe einschließlich Durchlaufzeiten je Fall.
- online_resume.log: vollständiges Log der in Batches ausgeführten Auswertung.
- online_resume.log: alle neu berechneten Einzelläufe unter gelernter Bank-Zuteilung.
