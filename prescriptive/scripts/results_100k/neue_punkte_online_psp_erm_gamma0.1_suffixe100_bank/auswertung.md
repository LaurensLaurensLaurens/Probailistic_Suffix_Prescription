# PSP-ERM-Auswertung

BPIC17, γ=0,1 / Tag; zehn Seeds (0–9), je 3.000 Ankünfte, Warmup 700, Cooldown 1.300, je 1.000 gewertete Fälle. Vier Control-Flow-Punkte wie bei CATE-ERM, Gelernte Bank-Zuteilung (learned), keine first_only-Einschränkung. Das vorhandene PSP real_alle_s60_normal bewertet jede Option mit 100 Suffixziehungen; das Modell wurde nicht neu trainiert.

## Ergebnisse

Alle Werte in Tagen; kleiner ist besser. Mittel über die je Seed berechneten Kennzahlen.

| Policy | Durchlaufzeit (Tage) | Durchlaufzeit P90 (Tage) | Durchlaufzeit P95 (Tage) | CVaR 90 % (Tage) | ERM γ=0.1 (Tage) |
| --- | --- | --- | --- | --- | --- |
| Bank | 22.600 | 39.065 | 47.876 | 52.021 | 62.364 |
| S-Learner ERM | 22.646 | 39.953 | 48.295 | 52.995 | 57.743 |
| X-Learner ERM | 22.195 | 41.169 | 50.315 | 55.700 | 72.996 |
| Causal Forest ERM | 20.810 | 38.602 | 47.240 | 52.123 | 65.505 |
| PSP-ERM | 22.437 | 39.775 | 49.205 | 54.170 | 63.487 |

PSP-ERM: Differenz des empirischen ERM zur Bank +1.124 Tage, 95-%-KI [-16.038; +18.285]: kein statistisch gesicherter ERM-Unterschied.

Bei 14,250 von 43,572 Entscheidungen (32.70 %) wählte ERM anders als die Mittelwertbewertung derselben gezogenen Suffixe. Dies ist kein separater EV-Simulationslauf: Nach einer anderen Wahl würde sich der weitere Verlauf ändern.

![Gepaarter Vergleich](vergleich_alle_erm.png)

## Vergleichbarkeit und Grenzen

Die Bank- und CATE-ERM-Rohdaten stammen aus dem vorherigen identischen Lauf. Katalog und Simulator-Prüfsummen wurden vor dem Start geprüft; die Bank wurde nicht unnötig neu simuliert. Die KIs beruhen auf zehn gepaarten Seed-Differenzen (t-Intervall), nicht auf unabhängigen Einzelentscheidungen. Das empirische ERM reagiert stark auf seltene lange Fälle; die Modellschätzungen und die Übertragbarkeit auf echte Prozesse sind durch diese KIs nicht abgesichert.

CATE-ERM: Der S-Learner schätzte auf allen Testproben Nulleffekte und entscheidet bei Gleichstand zufällig. Bei γ=0,1 konzentrierte sich das exponentielle CATE-Trainingsgewicht je Wahlpunkt zu 73–100 % auf einen einzelnen Fall. Diese Einschränkung gilt bei der Interpretation der Vergleichsmethoden.

## Dateien

- bericht.html / ergebnis.csv / laeufe.csv: PSP-ERM und Bank.
- vergleich_alle_erm.csv / vergleich_alle_erm_laeufe.csv: Vergleich mit allen drei CATE-ERM-Verfahren.
- zusammenfassung.csv / differenzen_zur_bank.csv: kompakte Tabellen und Konfidenzintervalle.
- checkpoints/: Durchlaufzeiten je Fall, einschließlich der übernommenen Bank-Läufe.
- laufparameter.json / catalog.yaml / logs/: Parameter, Prüfsummen und Seed-Logs.
