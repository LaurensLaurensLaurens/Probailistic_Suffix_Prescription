**Warum die Kostenersparnis gegenüber der Bank klein bleibt**

Alle nachfolgenden Komponenten beziehen sich auf dieselben 200 Fälle, Seed 0, im aktuellen korrigierten Simulator. Es handelt sich um die konfigurierten Beispieltarife. Kalenderwartezeit hat keinen eigenen Kostensatz; aktive Arbeitszeit und jedes Event werden berechnet.

| Verfahren | Basis + Lifecycle (€) | Aktive Zeit (€) | Gesamt (€) |
|---|---:|---:|---:|
| Bank | 136.94 | 41.70 | 178.64 |
| D-Kosten | 136.32 | 37.78 | 174.10 |
| OE-Kosten | 141.99 | 42.83 | 184.81 |
| D-Zeit | 141.80 | 37.24 | 179.04 |
| OE-Zeit | 151.27 | 52.88 | 204.15 |
| XGB-Zeit | 142.89 | 32.53 | 175.42 |
| Forest-Kosten | 142.29 | 47.00 | 189.29 |

Die Basis-/Lifecycle-Gebühren sind keine bewiesenen unvermeidbaren Fixkosten: Sie können durch weniger Events sinken. Eine Verkürzung reiner Wartezeit spart in dieser Kostenfunktion jedoch direkt nichts. Bei unveränderter Eventfolge bringen 10 % niedrigere Arbeitskosten nur rund 2,33 % weniger Gesamtkosten.

D-Kosten: 3.92 € weniger Arbeitskosten und 0.63 € weniger Eventgebühren, zusammen 4.54 € Ersparnis pro Fall.

XGB-Zeit: 9.17 € weniger Arbeitskosten bei 5.95 € zusätzlichen Eventgebühren. XGBoost optimiert hier die Durchlaufzeit, nicht Eventkosten.

Die PSP-Kostenpolicies fallen im abgeschlossenen Seed 0 bei etwa 60 % ihrer adaptiv bewerteten Zweierwahlen wegen nicht ausreichend getrennter Monte-Carlo-Schätzungen auf den Bankvorschlag zurück. Das begrenzt die Änderungen gegenüber der Bank; der tatsächlich andere Aktionsanteil kann daraus allein nicht abgeleitet werden. Die Netzgewichte lernen beobachtete Fortsetzungen und identifizieren damit nicht automatisch kausale Kostenunterschiede.

**Kostenoptimierte Referenz, dieselben drei Seeds:**

| Seed | Bank €/Fall | Causal Forest €/Fall | Ersparnis (%) |
|---|---:|---:|---:|
| 0 | 178.64 | 189.29 | -5.96 |
| 1 | 174.41 | 184.93 | -6.03 |
| 2 | 199.49 | 181.85 | +8.84 |

Mittel über Seeds: Bank 184.18 €, Forest 185.36 €. Mittlere seedweise Ersparnis: -1.05 %. Das ist eine Referenz mit vorhandenen Gewichten und kein Nachweis des optimal erreichbaren Kostenniveaus.

Dateien: cost_components_seed0.csv, activity_costs_seed0.csv, actual_options_seed0.csv, paired_case_distribution_seed0.csv, bank_first_decision_costs.csv, decomposition_summary.json und forest_runs.csv.
