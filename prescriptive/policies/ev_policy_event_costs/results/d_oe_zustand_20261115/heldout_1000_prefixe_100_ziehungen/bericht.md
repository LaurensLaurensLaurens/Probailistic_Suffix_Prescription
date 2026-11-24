**Restzeitprüfung: 1.000 ursprüngliche Testprefixe, je 100 Ziehungen**

Alle 1.000 Prefixe wurden je Modell geprüft. Beide korrigierten Modelle liefern für denselben einen Prefix innerhalb des bestehenden Ziehungslimits keine vollständige Fortsetzung. Für die vollständigen 1.000 Prefixe ist deshalb keine neue Restzeit-MAE verfügbar. Die nachfolgende Haupttabelle verwendet auf beiden Seiten dieselben 999 erfolgreich ausgewerteten Prefixe; der Ausfall wird gesondert ausgewiesen.

| Modell | Ursprüngliche Referenz, 1.000 Prefixe | Bisher, dieselben 999 Prefixe | Korrigiert, dieselben 999 Prefixe | Änderung neu minus alt |
|---|---:|---:|---:|---:|
| D | 7.81438 Tage | 7.79593 Tage | 8.10418 Tage | +0.30826 Tage |
| OE | 7.98412 Tage | 7.96837 Tage | 8.02388 Tage | +0.05551 Tage |

Die Restzeit-MAE verbessert sich nicht. D verschlechtert sich auf den gemeinsamen Prefixen um 0,30826 Tage; OE um 0,05551 Tage. Die Intervalle beschreiben Unsicherheit über die Testfälle, keinen Interventionseffekt.

| Modell | Gepaarte Änderung | 95-%-Bootstrapintervall |
|---|---:|---:|
| D | +0.30826 Tage | [+0.14322; +0.47787] Tage |
| OE | +0.05551 Tage | [-0.11732; +0.22971] Tage |

Damit ist D in dieser Auswertung klar schlechter; bei OE ist die kleine Änderung nicht eindeutig von null getrennt.

**Sonderfall und Abdeckung.** Application_513583760, Prefixlänge 7, endet im Prefix bei W_Shortened completion|schedule. Beide Modelle starten die Tätigkeit, erzeugen anschließend aber nur selten deren suspend. In einer unabhängigen Prüfung von 5.000 zweiten Suffixschritten wird W_Shortened completion|suspend bei D zweimal und bei OE siebenmal gewählt. Viele Fortsetzungen lassen die Arbeit aktiv, erreichen einen geschäftlichen Endmarker und können unter den Zustands- und Zeitunterstützungsregeln nicht gültig enden. Für D scheiterten auch drei zusätzliche unabhängige vollständige Ziehungsaufrufe mit je 100 angeforderten Suffixen. Die neuronalen Gewichte, die Gültigkeitsregeln und die laufende Online-Auswertung wurden während dieser Prüfung nicht verändert.

Diese Abdeckung von 999/1.000 ist Teil des Ergebnisses. Der problematische Prefix wurde auch aus den alten Werten der gepaarten Tabelle entfernt; die ursprünglichen vollständigen Referenzen 7,81438/7,98412 Tage bleiben separat dokumentiert.

**Vergleichsbedingungen.** Dieselben ursprünglichen Fall-/Prefixschlüssel; je 100 Ziehungen; identische neuronale Gewichte, per SHA-256 der Gewichtstensoren geprüft. Die tatsächliche Restzeit wurde für jeden Prefix gegen die historische Referenz geprüft. Die neue Modellvariante umfasst Zustands-/EOS-Regeln, ausschließlich aus Trainingsfällen abgeleitete Übergangsregeln und das zusätzliche bedingte empirische Zeitmodell. Die ursprünglichen Referenzmodelle hatten keine Übergangssperre. Die gepaarten Intervalle beruhen auf 5.000 Bootstrap-Wiederholungen über Testfälle.

Dateien: paired_resttime_comparison.csv; comparison.csv mit ausschließlich gepaarten 999-Prefix-Kennzahlen; population_summary_unmatched.csv mit den getrennten ursprünglichen 1.000- und neuen 999-Prefix-Mitteln; D_paired_prefixes.csv und OE_paired_prefixes.csv; D/failures.json und OE/failures.json; rare_second_event_probe.json; neural_weight_validation.json; manifest.json.
