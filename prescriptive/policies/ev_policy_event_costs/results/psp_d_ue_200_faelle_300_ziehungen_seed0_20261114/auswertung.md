# PSP-D: Eventkosten mit Übergangssperre

200 Fälle, Seed 0, 300 Ziehungen je Option; 700 Warmup und 1.300 Cooldown. Modellgewichte, Tarife, Wahlpunkte und Ressourcenstrategie entsprechen dem unbeschränkten 300er-Lauf. Die Sperre erlaubt ausschließlich direkt beobachtete Aktivität/Lifecycle-Paare und beobachtete Fallenden aus den 25.208 echten Fällen außerhalb der 6.301 Testfälle.

| Verfahren | Kosten/Fall (€) ↓ | Ersparnis ggü. Bank (€) | Ersparnis (%) | Durchlaufzeit (Tage) |
|---|---:|---:|---:|---:|
| Bank | 174.03 | +0.00 | +0.00 | 19.41 |
| PSP-D, 30 ohne Sperre | 209.99 | -35.96 | -20.66 | 19.95 |
| PSP-D, 300 ohne Sperre | 198.18 | -24.16 | -13.88 | 19.93 |
| PSP-D, 300 mit Sperre | 179.27 | -5.24 | -3.01 | 20.02 |

Mit minus ohne Sperre bei 300 Ziehungen: -18.91 €/Fall. Negative Werte bedeuten niedrigere Kosten.

Laufende Prüfung: 366,000 Suffixe und 10,146,285 Übergänge; 0 unbeobachtete Übergänge. Auch Prefix→Options-Event wird geprüft. Die Masken erlauben EOS nur nach beobachteten Endtokens; das unveränderte Längenlimit bleibt bestehen.

Bankkosten und Arbeitszeiten je Fall stimmen mit dem unbeschränkten Lauf überein. Alle verglichenen Verfahren haben dieselben 200 Fall-IDs. Ein Seed; kein Konfidenzintervall über mehrere Seeds.
