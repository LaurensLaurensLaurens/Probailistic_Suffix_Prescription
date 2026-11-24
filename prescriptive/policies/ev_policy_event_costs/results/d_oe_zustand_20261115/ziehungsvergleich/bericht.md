**Ziehungsqualität vor und nach der Korrektur**

Identische 100 zurückgehaltene Testprefixe je Modell, je 30 Suffixe und dieselben Start-Seeds. Die neuronalen Gewichte sind nachweislich identisch. Die bisherigen Modelle verwenden bereits ihre alte Übergangssperre. Die korrigierten Kopien ergänzen Zustands-/EOS-Prüfung und ein gemeinsames, ausschließlich auf Trainingsfällen gelerntes bedingtes Zeitmodell; ihre Übergangsregeln schließen auch die Validierungsfälle aus.

| Kennzahl | D bisher | D korrigiert | OE bisher | OE korrigiert |
|---|---:|---:|---:|---:|
| Treffer Schritt 1 (%) | 89.000 | 89.000 | 85.000 | 84.000 |
| Treffer Schritt 5 (%) | 46.667 | 47.778 | 46.667 | 48.889 |
| Treffer Schritt 10 (%) | 21.333 | 38.667 | 26.667 | 29.333 |
| Sequenzähnlichkeit DLS | 0.334 | 0.373 | 0.354 | 0.363 |
| Suffixlängen-MAE (Events) | 11.433 | 12.400 | 11.844 | 11.663 |
| Erster Abstand MAE (h) | 17.327 | 16.952 | 16.595 | 17.563 |
| Restzeit-MAE (Tage) | 10.075 | 10.026 | 10.080 | 10.165 |
| Restkosten-MAE (€) | 3926.779 | 64.417 | 2072.946 | 56.586 |

Die korrigierte Prüfung liefert auf allen 6.000 Suffixen keine offenen Sitzungen am Ende, keine fehlenden geschäftlichen Endmarker und keine nach ihren Regeln ungültigen Öffnungen. Diese Eigenschaften werden durch die Ziehungsregeln erzwungen; bessere Vorhersagefehler sind separat zu beurteilen. Ein vollständiger Vergleich des Online-Outcomes über alle Seeds läuft unabhängig weiter.

Weitere Dateien: matched_summary.csv, baseline_structure.csv, baseline_prefixes.csv, baseline_draws.csv und validation.json.
