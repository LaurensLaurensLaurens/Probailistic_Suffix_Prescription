**Warum PSP D hier weniger Arbeitskosten spart als XGBoost**

Grundlage: vorhandene Läufe im korrigierten Simulator, jeweils dieselben 200 Fall-IDs je Seed. Arbeitskosten sind das Zeitentgelt aktiver Sitzungen nach den konfigurierten Beispieltarifen. Weder D mit Gesamtkostenziel noch die beiden Zeitpolicies sind auf reine Arbeitskosten optimiert.

| Seed | Verfahren | Arbeitskosten €/Fall | Gesamtkosten €/Fall | Durchlaufzeit Tage |
|---|---|---:|---:|---:|
| 0 | Bank | 41.70 | 178.64 | 20.374 |
| 0 | D-Kosten | 37.78 | 174.10 | 19.925 |
| 0 | D-Zeit | 37.24 | 179.04 | 21.308 |
| 0 | XGB-Zeit | 32.53 | 175.42 | 19.798 |
| 1 | Bank | 40.59 | 174.41 | 20.605 |
| 1 | D-Kosten | 34.95 | 163.53 | 19.935 |
| 1 | D-Zeit | 40.97 | 182.57 | 21.339 |
| 1 | XGB-Zeit | 36.48 | 184.05 | 19.975 |

D mit Gesamtkostenziel ist in Seed 0 bei den Gesamtkosten günstiger als XGBoost: 174,10 € gegenüber 175,42 €. Gegenüber XGBoost fallen bei D 5,25 € mehr Arbeitskosten, zugleich 6,58 € weniger Eventgebühren an. Das ergibt 1,33 € niedrigere Gesamtkosten. Für das reine Arbeitskostenziel bleibt XGBoost in diesem Seed vorn. In Seed 1 liegt D mit Gesamtkostenziel auch bei den Arbeitskosten vor XGBoost: 34,95 € gegenüber 36,48 €. Eine universelle Rangfolge ist daraus nicht ableitbar.

**Der auffälligste Unterschied ist die Entscheidungsausführung.** Im vollständigen Seed-0-Lauf D-Zeit gibt es 659 adaptiv bewertete Zweierwahlen. Bei 478 davon (72,53 %) bleibt wegen unsicherer Differenz der Bankvorschlag erhalten; zwei weitere Rückfälle entstehen durch fehlgeschlagene vollständige Suffixziehungen. XGBoost wählt anhand seines direkten Outcome-Modells ohne diese Monte-Carlo-Schranke.

Am Konzeptpunkt gibt es bei D-Zeit 105 echte Zweierwahlen: 91 Bank-Rückfälle, sechs ausdrückliche Annahmen und acht Starts der Vervollständigung. Bei XGBoost sind es 123 Zweierwahlen: 120 Annahmen und drei Starts. Die Policies erzeugen unterschiedliche spätere Verläufe; die Nenner bezeichnen deshalb nicht dieselben erreichten Zustände. Einschließlich der Stellen mit nur einer zulässigen Option werden 52 gegenüber 159 Annahmen ausgeführt.

**Neue Prüfung derselben Zustände:** Zwölf Bankprefixe aus dem aktuellen Seed-0-Lauf, drei je Entscheidungspunkt, wurden vor einer Intervention ausgewählt. Beide Modelle erhalten jeweils exakt denselben Prefix. Für D wurden je Option 100 Suffixe gezogen. In neun von zwölf Zuständen stimmt Ds Rohentscheidung (kleinste mittlere Dauer) mit XGBoost überein. Ebenfalls neun von zwölf D-Schätzungen bestehen bei 100 Ziehungen die vorhandene Zwei-Standardfehler-Schranke nicht. Am Konzeptpunkt und nach der Vervollständigung stimmen alle sechs Rohentscheidungen überein; vier davon bleiben im 100er-Test unsicher. Das spricht für einen wesentlichen Einfluss der Unsicherheitsregel zusätzlich zur Prognosequalität. Es misst nicht den kausalen Anteil der Regel am Online-Rückstand.

Beispiel Application_772 am Konzeptpunkt: D prognostiziert im Mittel 26,51 Tage Restzeit nach dem Start der Vervollständigung und 24,28 Tage nach direkter Annahme. Die Differenz von 2,23 Tagen hat einen gepaarten Monte-Carlo-Standardfehler von 2,09 Tagen. Am 100er-Prüfschritt wäre die Differenz damit zu klein für einen Eingriff. XGBoost bevorzugt am selben Prefix ebenfalls die Annahme. Die zwölf unabhängigen 100er-Prüfungen reproduzieren nicht die vollständige adaptive 30→100-Entscheidungsfolge; eine frühe Entscheidung nach 30 Ziehungen wird hier nicht simuliert.

**Wo die 4,71 € Arbeitskostendifferenz zwischen D-Zeit und XGBoost in Seed 0 entstehen:**

| Arbeitsschritt | D-Zeit €/Fall | XGBoost €/Fall |
|---|---:|---:|
| Antragsvervollständigung | 5,82 | 2,48 |
| Nachfassanrufe nach Angeboten | 6,02 | 2,91 |
| Antragsvalidierung | 20,73 | 22,78 |

Die übrigen Arbeitsschritte tragen netto 0,31 € zur Differenz bei. XGBoost spart vor allem bei Vervollständigung und Nachfassen, bei gleichzeitig teurerer Validierung. Diese Summen beschreiben ganze Verläufe und isolieren keinen einzelnen Interventionseffekt.

**Weitere Modellgrenzen:** XGBoost wird direkt auf den beobachteten Durchlaufzeit-Outcome trainiert. D prognostiziert vollständige Eventfolgen und addiert ihre Zeitabstände. Im aktuellen D ersetzt zudem ein empirisches Zeitmodell die neuronalen Abstände; seine Pools hängen von vorherigem Event, aktuellem Event und Arbeitszustand ab. Exakte Prefixhistorie, Kalender und Angebotsalter gehen nicht direkt in diese zusätzliche Abstandsschätzung ein; das Eventnetz sieht weiterhin Prefix und Fallattribute. Das ist eine plausible zusätzliche Fehlerquelle, deren Beitrag hier nicht isoliert wurde. Beide Modelle lernen aus Beobachtungsdaten; der Vergleich beweist weder kausale Korrektheit von XGBoost noch einen reinen Architektureffekt.

Die zusätzliche Verlängerung der Suffixobergrenze auf 1.000 Events ergibt auf denselben zwölf Prefixen zehn Rohübereinstimmungen und zehn unsichere 100er-Schätzungen. Eine D-Rohentscheidung wechselt. Dies liefert hier keinen Hinweis darauf, dass allein ein größeres Ziehungsfenster die Unsicherheit beseitigt.

Der nächste trennende Online-Test wäre derselbe D-Checkpoint mit identischem Ziel und Ziehungsbudget, jeweils mit und ohne Unsicherheitsrückfall. Eine separate Optimierung ausschließlich auf Arbeitskosten wäre ein zusätzlicher Zielvergleich. Solche neuen Online-Policyläufe wurden für diese Diagnose nicht gestartet.

Geprüft: alle geschützten Quellen und vorbereiteten Modelle des laufenden Benchmarks unverändert; 24 Zustands-/Horizontvergleiche, keine Ziehungsfehler, alle zurückgegebenen Suffixe zustandskonsistent und vollständig. Die zwölf Zustände sind eine explorative Diagnose. Die zusätzlichen Fall-Bootstrapintervalle in paired_labor_differences.csv erfassen weder Ressourcenkopplung zwischen Fällen noch die Unsicherheit zwischen Seeds.
