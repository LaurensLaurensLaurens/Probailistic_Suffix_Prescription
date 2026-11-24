# Alle vorhandenen BPIC17-PSP-Auswertungen

Stand: 2026-11-14T02:47:15+01:00.

Dies ist eine Zusammenstellung vorhandener Ergebnisse. Es wurden keine neuen Modelle trainiert und keine neuen Simulationen gestartet. Vollmodelle, Pilotmodelle und ältere Stichproben sind getrennt.

## Definitionen

- Längen-/Event-MAE: mittlerer absoluter Fehler der mittleren gezogenen Zahl verbleibender Events.
- Schritt k: häufigstes gezogenes Aktivität|Lifecycle-Token an Position k trifft das echte Token; nur Prefixe mit einem echten Event an dieser Position zählen.
- DLS: mittlere normierte Damerau-Levenshtein-Ähnlichkeit über die einzelnen gezogenen Suffixe, 1 = identisch.
- Restzeit-MAE, Bias und CRPS beziehen sich auf die Zeit ab dem letzten Prefix-Event. Bias = geschätzt minus echt. Kleinere MAE/CRPS sind besser; 80-%-Abdeckung sollte etwa 80 % erreichen.
- Online-Outcome bei Durchlaufzeit: Y = −Tage. Zur besseren Lesbarkeit wird T positiv gezeigt, kleiner ist besser. Positive Verbesserungsprozente bedeuten kürzer als die jeweilige Bankreferenz.

## Suffixgüte: Vollmodelle, gleiche 1.000 Testprefixe, je 100 Suffixziehungen

Die Übereinstimmung der Prefixmengen wurde anhand der einzelnen Fall-ID/Prefixlängen-Paare überprüft.

| Modell | Längen-MAE ↓ | DLS ↑ | S1 ↑ | S2 ↑ | S3 ↑ | S5 ↑ | S10 ↑ | S20 ↑ |
|---|---|---|---|---|---|---|---|---|
| Semi-Markov (Referenz) | 13,28 | 0,264 | 76,4 % | 63,1 % | 48,9 % | 30,4 % | 9,4 % | 0,0 % |
| s40, Gaps | 13,12 | 0,234 | 83,8 % | 70,6 % | 58,0 % | 40,7 % | 8,5 % | 2,3 % |
| (a), Log-Gaps | 8,99 | 0,278 | 85,5 % | 74,2 % | 62,0 % | 42,3 % | 12,7 % | 0,5 % |
| (c), normal | 10,57 | 0,249 | 83,4 % | 71,2 % | 60,6 % | 38,5 % | 9,4 % | 0,8 % |
| (d), Zeitklassen | 12,12 | 0,281 | 85,4 % | 74,7 % | 65,2 % | 49,2 % | 21,1 % | 4,1 % |
| OE, Event+Zeitklasse als Token | 12,12 | 0,274 | 84,1 % | 72,4 % | 64,1 % | 49,8 % | 21,2 % | 3,9 % |
| (c), Teacher Forcing 1 | 10,65 | 0,404 | 85,8 % | 75,7 % | 65,2 % | 50,3 % | 25,2 % | 3,6 % |

| Modell | Erster Abstand MAE (h) ↓ | Restzeit-MAE (d) ↓ | Bias (d) | CRPS (d) ↓ | 80-%-Abdeckung |
|---|---|---|---|---|---|
| Semi-Markov (Referenz) | 15,38 | 9,51 | 2,36 | 5,81 | 88,3 % |
| s40, Gaps | 13,83 | 9,69 | 6,41 | 6,08 | 87,6 % |
| (a), Log-Gaps | 17,44 | 15,57 | 13,64 | 9,21 | 95,4 % |
| (c), normal | 13,13 | 8,13 | -0,05 | 5,40 | 87,8 % |
| (d), Zeitklassen | 11,71 | 7,81 | -0,16 | 5,23 | 75,3 % |
| OE, Event+Zeitklasse als Token | 11,89 | 7,98 | 0,36 | 5,31 | 77,4 % |
| (c), Teacher Forcing 1 | 12,35 | 8,62 | 0,96 | 5,49 | 83,6 % |

(d) bezeichnet hier das Basisnetz mit getrennten Event-/Zeitklassen-Köpfen. Die freie Suffixdiagnose wurde nicht erneut als eigene Diagnose an Interventionsprefixen der späteren Optionsabstand-Variante gemessen. OE mit Übergangssperren hat keine eigene solche Suffixdiagnose; dessen Werte werden nicht vom unbeschränkten OE übernommen.

## Ältere Modelle und (b): 200 Prefixe aus dem gesamten Log

Diese Auswertung kann Trainingsfälle enthalten und ist nicht direkt mit der oberen Testtabelle vergleichbar. Für (b) fehlt die entsprechende 1.000-Testprefix-Auswertung.

| Modell | Längen-MAE ↓ | DLS ↑ | S1 ↑ | S2 ↑ | S3 ↑ | S5 ↑ | S10 ↑ | S20 ↑ |
|---|---|---|---|---|---|---|---|---|
| Markov-Kette | 18,53 | 0,240 | 74,5 % | 56,7 % | 44,6 % | 26,0 % | 9,0 % | 0,0 % |
| PSP Nachtlauf (5.000 Fälle, 1e-4) | 50,87 | 0,146 | 79,0 % | 63,4 % | 53,8 % | 29,6 % | 7,6 % | 0,0 % |
| PSP echter Log (10.000 Fälle, 1e-5, neue Punkte 3-fach) | 18,89 | 0,246 | 79,0 % | 68,0 % | 63,0 % | 37,3 % | 11,8 % | 0,0 % |
| PSP-Variante (Decoder 40, ohne Zukunftswissen, 1e-4) | 13,20 | 0,239 | 83,0 % | 70,1 % | 65,2 % | 39,1 % | 6,2 % | 1,3 % |
| Stufe 2 (a): alle Fälle, Decoder 60, log-normale Abstände | 8,69 | 0,277 | 83,5 % | 70,6 % | 64,7 % | 42,0 % | 13,2 % | 0,0 % |
| Stufe 2 (b): wie (a) mit Restzeit-Kopf | 8,84 | 0,266 | 84,5 % | 68,0 % | 61,4 % | 40,2 % | 13,2 % | 1,3 % |

## Online-Outcomes: abgeschlossene Vergleiche mit zehn Seeds

Je Seed 1.000 bewertete Fälle, 700 Warmup, 1.300 Cooldown; zufällige Mitarbeiterzuteilung. Bank: 22,328 Tage. Prozentwerte sind Mittel der gepaarten Seed-Verbesserungen. 95-%-Intervalle und Tail-Kennzahlen stehen in online_10_seeds.csv.

| Variante | Ziehungen/Option | Seeds × Fälle | Tage ↓ | Bank (Tage) | Verbesserung ggü. Bank |
|---|---|---|---|---|---|
| Ursprüngliches Echtlog-PSP | 30 | 10 × 1000 | 22,573 | 22,328 | -1,15 % |
| Auf neue Wahlpunkte neu trainiert | 30 | 10 × 1000 | 22,420 | 22,328 | -0,44 % |
| s40, Zeit aus vorhergesagtem Fallalter | 30 | 10 × 1000 | 21,806 | 22,328 | 2,28 % |
| s40, Summe der Zeitabstände | 30 | 10 × 1000 | 21,482 | 22,328 | 3,75 % |
| s40, Summe der Zeitabstände | 100 | 10 × 1000 | 20,938 | 22,328 | 6,19 % |
| s40, Angebot nur beim ersten Besuch wählen | 100 | 10 × 1000 | 21,024 | 22,328 | 5,81 % |
| (a), logarithmierte Abstände | 100 | 10 × 1000 | 22,819 | 22,328 | -2,25 % |
| (b), mit Restzeitkopf, Suffixbewertung | 100 | 10 × 1000 | 22,626 | 22,328 | -1,36 % |
| (b), direkte Restzeitbewertung | 30 | 10 × 1000 | 19,830 | 22,328 | 11,15 % |
| (c), normale Abstände | 100 | 10 × 1000 | 20,493 | 22,328 | 8,18 % |

## Online-Outcomes: gleicher Mini-Test mit 200 Fällen und 30 Ziehungen

Ein Seed (0), identische Bank-Falldauern zwischen allen sechs Varianten, Bankmittel 20,374 Tage.

| Variante | Ziehungen/Option | Seeds × Fälle | Tage ↓ | Bank (Tage) | Verbesserung ggü. Bank |
|---|---|---|---|---|---|
| (c), 1 ms | 30 | 1 × 200 | 20,198 | 20,374 | 0,86 % |
| (c), ein Abstand je Option | 30 | 1 × 200 | 21,212 | 20,374 | -4,11 % |
| (d), Abstand je Suffix | 30 | 1 × 200 | 20,087 | 20,374 | 1,41 % |
| OE, Abstand je Suffix | 30 | 1 × 200 | 21,195 | 20,374 | -4,03 % |
| OE, ein Abstand je Option | 30 | 1 × 200 | 20,828 | 20,374 | -2,23 % |
| OE + Übergangssperren | 30 | 1 × 200 | 21,737 | 20,374 | -6,69 % |

## Online-Outcomes: 1.000 Fälle und 300 Ziehungen, Seed 0

Beide Varianten haben exakt dieselben Bank-Falldauern. Kein Zehn-Seed-Ergebnis.

| Variante | Ziehungen/Option | Seeds × Fälle | Tage ↓ | Bank (Tage) | Verbesserung ggü. Bank |
|---|---|---|---|---|---|
| (d), Abstand je Suffix | 300 | 1 × 1000 | 20,918 | 21,709 | 3,65 % |
| OE + Übergangssperren | 300 | 1 × 1000 | 22,132 | 21,709 | -1,95 % |

## Weitere historische Zielgrößen und Zwischenstände

- (c) mit 300 Suffixen: fünf fertig gewordene Seeds 0–4 des später gestoppten Laufs, ungefähr 19,88 Tage gegenüber 22,33 Banktagen. Nur gerundete Log-Zwischenstände.
- OE ohne Übergangssperre mit 300 Suffixen: sieben fertig gewordene Seeds 0,1,3,4,5,6,8, ungefähr 21,30 Tage gegenüber 22,42 Banktagen. Andere Seedmenge als (c), daher nicht direkt dieselbe Stichprobe.
- PSP-ERM auf (c), γ=0,1/Tag: 22,437 Tage gegenüber 22,600 Banktagen; empirisches entropisches Risiko 63,487 gegenüber 62,364 Tagen. Zehn Seeds, je 1.000 Fälle; learned-Mitarbeiterzuteilung und anderes Optimierungsziel. Quelle: ../neue_punkte_online_psp_erm_gamma0.1_suffixe100_bank/auswertung.md.
- (d) mit Eventkosten als Ziel: 209,99 €/Fall gegenüber 174,03 €/Fall der Bank, 200 Fälle, 30 Ziehungen, Seed 0, konfigurierte Beispieltarife. Outcome ist hier negative Euro, nicht negative Tage. Dies ist eine andere Bewertung derselben Suffixgewichte, kein neu trainiertes PSP.

## (e) und fehlende eigenständige Messungen

(e): Status training; letzte im Log ausgewiesene Epoche 8/15. Es liegen noch keine fertigen Suffixgüte- oder Online-Ergebnisse vor.

Für (c) mit geteiltem Optionsabstand, OE mit geteiltem Optionsabstand sowie OE mit Übergangssperren gibt es Online-Ergebnisse, aber keine neu gemessene vollständige Suffixgüte in dieser Zusammenstellung. Bei reinen Bewertungsvarianten bleiben die Basisgewichte gleich; das ersetzt keine gemessene Diagnose des jeweiligen Entscheidungsverfahrens. Pilot-TF-0,8 und einige *_od/*_oe-Kopien haben ebenfalls keine eigenständige vollständige Suffixgüte-Datei.

## Pilotmodelle: 3.000 Trainingsfälle, sechs Epochen

Gleiche 1.000 Testprefixe und je 100 Suffixe, aber erheblich kleineres Training als die Vollmodelle. k = getrennte Klassen, t = gemeinsames Event-/Zeitklassen-Token, n = normale Abstände, TF = unterste Teacher-Forcing-Quote, kx = Encoderkontext in jedem Decoderschritt. Für diese Piloten liegt keine entsprechende vollständige Online-Outcome-Tabelle vor.

| Modell | Längen-MAE ↓ | DLS ↑ | S1 ↑ | S2 ↑ | S3 ↑ | S5 ↑ | S10 ↑ | S20 ↑ |
|---|---|---|---|---|---|---|---|---|
| Pilot Klassen, TF 0 | 9,90 | 0,185 | 61,6 % | 49,5 % | 40,6 % | 29,3 % | 12,1 % | 2,6 % |
| Pilot Klassen, TF 1 | 10,62 | 0,255 | 66,1 % | 52,9 % | 42,7 % | 31,1 % | 9,5 % | 0,0 % |
| Pilot Token, TF 1 | 11,05 | 0,246 | 66,2 % | 54,7 % | 44,3 % | 31,3 % | 9,7 % | 0,5 % |
| Pilot normal, TF 1 | 11,60 | 0,272 | 66,4 % | 52,8 % | 43,3 % | 32,8 % | 12,1 % | 0,0 % |
| Pilot Token, TF 0,5 | 9,91 | 0,218 | 62,8 % | 49,3 % | 40,1 % | 23,1 % | 8,8 % | 0,3 % |
| Pilot Token, TF 0 | 10,81 | 0,181 | 53,3 % | 39,8 % | 34,7 % | 20,5 % | 9,5 % | 2,8 % |
| Pilot Token, TF 1 + Encoderkontext | 10,97 | 0,227 | 60,5 % | 49,8 % | 39,1 % | 26,9 % | 9,0 % | 0,0 % |
| Pilot Token, TF 0,5 + Encoderkontext | 9,78 | 0,215 | 56,7 % | 47,5 % | 38,1 % | 21,2 % | 9,1 % | 1,3 % |

| Modell | Erster Abstand MAE (h) ↓ | Restzeit-MAE (d) ↓ | Bias (d) | CRPS (d) ↓ | 80-%-Abdeckung |
|---|---|---|---|---|---|
| Pilot Klassen, TF 0 | 12,86 | 8,09 | -2,79 | 5,70 | 72,0 % |
| Pilot Klassen, TF 1 | 13,53 | 8,50 | -1,22 | 5,62 | 76,3 % |
| Pilot Token, TF 1 | 12,60 | 8,42 | -1,33 | 5,66 | 73,5 % |
| Pilot normal, TF 1 | 15,66 | 8,96 | -0,23 | 5,79 | 84,0 % |
| Pilot Token, TF 0,5 | 12,92 | 8,21 | -2,19 | 5,67 | 73,3 % |
| Pilot Token, TF 0 | 13,08 | 8,16 | -2,54 | 5,68 | 71,7 % |
| Pilot Token, TF 1 + Encoderkontext | 13,34 | 8,58 | -1,32 | 5,74 | 73,5 % |
| Pilot Token, TF 0,5 + Encoderkontext | 12,82 | 8,11 | -2,94 | 5,74 | 70,2 % |

## Dateien

Alle Tabellen stehen zusätzlich als CSV in diesem Ordner. manifest.json enthält die Quellenprüfsummen. Die ältere übergreifende Inventarliste einschließlich BPIC19/Repair steht unter ../psp_online_uebersicht_20261113/online_alle_berichte.csv; diese anderen Datensätze sind nicht mit BPIC17 zu vermischen.
