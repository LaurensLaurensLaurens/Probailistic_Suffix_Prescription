# PSP-D: Eventkosten mit 30 und 300 Ziehungen

Jeweils 200 bewertete Fälle, Seed 0, 700 Warmup und 1.300 Cooldown. Identische Modellgewichte, Beispieltarife, Wahlpunkte und Ressourcenstrategie. Optionsabstand je Suffix gezogen. Einziger geänderter PSP-Parameter: Ziehungen je Option.

| Verfahren | Kosten/Fall (€) | Ersparnis ggü. Bank (€) | Ersparnis (%) | Durchlaufzeit (Tage) |
|---|---:|---:|---:|---:|
| Bank | 174.03 | +0.00 | +0.00 | 19.41 |
| PSP-D, 30 Ziehungen | 209.99 | -35.96 | -20.66 | 19.95 |
| PSP-D, 300 Ziehungen | 198.18 | -24.16 | -13.88 | 19.93 |

Niedrigere Kosten sind besser. Positive Ersparnisse bedeuten geringere Kosten als die Bank.
300 minus 30 Ziehungen: -11.80 €/Fall.

Die Bank wurde neu simuliert. Alle Kostenbestandteile und Arbeitszeiten der 200 Bankfälle stimmen mit der Referenz überein. Die Fall-IDs stimmen auch zwischen beiden PSP-Läufen überein.

Ein Seed mit 200 Fällen; kein Konfidenzintervall über mehrere Seeds. Kosten stammen aus den tatsächlich simulierten Events.

Kostenbestandteile und weitere Kennzahlen: comparison_30_vs_300.csv. Einzelkosten: paired_case_costs.csv; Events und Entscheidungen: online/.
