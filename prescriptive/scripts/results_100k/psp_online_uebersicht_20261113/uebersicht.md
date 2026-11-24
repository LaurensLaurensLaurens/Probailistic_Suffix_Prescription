**PSP: Online-Outcome, Suffixvorhersage und Zeitgüte**

Erstellt: 2026-11-13T13:16:28+01:00.

Die Hauptübersicht betrifft BPIC17 und die vier zuletzt untersuchten Control-Flow-Punkte: angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads. Ein vollständiges Inventar aller gefundenen PSP-Online-Berichte einschließlich älterer BPIC17-, BPIC19- und Repair-Versuche steht am Ende und in online_alle_berichte.csv. Mehrere Berichtsordner können denselben Lauf enthalten; ihre Anzahl ist keine Zahl unabhängiger Versuche.

Outcome ist bei den BPIC17-Hauptvergleichen die negative Durchlaufzeit: größerer Outcome = kürzere Zeit. Die Tabellen zeigen positive Tage; kleiner ist besser. Online-Dauern werden aus dem Simulator-Log gemessen, nicht aus den vorhergesagten Suffixen. Die künstlichen 1 ms beeinflussen die Optionsbewertung. Ein daraus folgender Policy-Vorteil ist deshalb kein Rechenfehler der Online-Dauermessung.

**Vollständig abgeschlossene Hauptvergleiche: zehn Seeds, je 1.000 gewertete Fälle**

Je Lauf 3.000 Ankünfte, 700 Warmup- und 1.300 Cooldown-Fälle, Seed 0–9, zufällige Mitarbeiterzuteilung. Die gespeicherten Bank-Mittelwerte je Seed stimmen in diesen zehn Versuchen überein; die Bank liegt insgesamt bei 22,328 Tagen. Positive Verbesserung bedeutet kürzer als die Bank. Die Prozentwerte sind Mittel der gepaarten Seed-Uplifts; sie können vom Quotienten der Gesamtmittel geringfügig abweichen.

| Variante | Bewertung je Option | Tage ↓ | Verbesserung ggü. Bank | 95-%-KI der Verbesserung | Quelle |
|---|---|---:|---:|---|---|
| Ursprüngliches Echtlog-PSP | 30 Suffixe | 22,573 | -1,15 % | [-3,64; 1,33] % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/neue_punkte_online/prescription/PSP/bpic17_online.csv) |
| Auf neue Wahlpunkte neu trainiert | 30 Suffixe | 22,420 | -0,44 % | [-1,90; 1,01] % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/neue_punkte_online_psp_neu/prescription/PSP/bpic17_online.csv) |
| s40, Zeit aus vorhergesagtem Fallalter | 30 Suffixe | 21,806 | 2,28 % | [-0,11; 4,68] % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/neue_punkte_online_real_np_s40_ohne_zukunft/prescription/PSP/bpic17_online.csv) |
| s40, Summe der Zeitabstände | 30 Suffixe | 21,482 | 3,75 % | [1,70; 5,80] % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/neue_punkte_online_variante_abstaende/prescription/PSP/bpic17_online.csv) |
| s40, Summe der Zeitabstände | 100 Suffixe | 20,938 | 6,19 % | [3,98; 8,41] % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/neue_punkte_online_variante_100ziehungen/prescription/PSP/bpic17_online.csv) |
| s40, Angebot nur beim ersten Besuch wählen | 100 Suffixe | 21,024 | 5,81 % | [3,87; 7,75] % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/neue_punkte_online_real_np_s40_ohne_zukunft_abstaende_erste_entscheidung/prescription/PSP/bpic17_online.csv) |
| (a), logarithmierte Abstände | 100 Suffixe | 22,819 | -2,25 % | [-4,82; 0,32] % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/neue_punkte_online_stufe2_ohne_kopf_suffixe100/prescription/PSP/bpic17_online.csv) |
| (b), mit Restzeitkopf, Suffixbewertung | 100 Suffixe | 22,626 | -1,36 % | [-2,52; -0,19] % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/neue_punkte_online_stufe2_kopf_suffixe100/prescription/PSP/bpic17_online.csv) |
| (b), direkte Restzeitbewertung | 30 Dropout-Auswertungen des Kopfs | 19,830 | 11,15 % | [9,15; 13,14] % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/neue_punkte_online_stufe2_kopf/prescription/PSP/bpic17_online.csv) |
| (c), normale Abstände | 100 Suffixe | 20,493 | 8,18 % | [6,42; 9,93] % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/neue_punkte_online_stufe2c_suffixe100/prescription/PSP/bpic17_online.csv) |

Als Vergleich auf diesen zehn Seeds: Causal Forest 19,477 Tage, X-Learner 19,621, S-Learner 20,044, feste Regel 19,356. Der direkte Restzeitkopf ist mit 19,830 Tagen der beste PSP unter diesen vollständig abgeschlossenen Hauptvergleichen. Das ist eine Aussage über diesen Simulator und diese Konfiguration, keine Garantie für den echten Prozess.

Auch die langen Fälle unterscheiden sich: Bank P95 47,415 / CVaR90 51,265 Tage; direkter Restzeitkopf (b) 44,195 / 48,964; (c) mit 100 Suffixen 45,918 / 50,953; (a) 49,002 / 53,763. CVaR90 ist der Mittelwert der längsten 10 % je Seed, danach über die Seeds gemittelt. Alle Tail-Werte der zehn Varianten stehen zusätzlich in online_bpic17_vier_punkte.csv.

Die ursprünglichen PSPs verwendeten sehr kurzes Decodertraining und teilweise statische Merkmale aus dem vollständigen Fall. Bei s40 und später sind die statischen Merkmale auf LoanGoal, ApplicationType und RequestedAmount beschränkt. Die s60-Modelle verwenden alle 25.208 Fälle außerhalb der 6.301 zurückgehaltenen Testfälle, Decoder 60, dreifach gewichtete Wahlpunkte und regulär 15 Epochen. Die alten und neuen Online-Resultate bleiben historische Auswertungen ihrer jeweiligen Modell- und Codeversion.

**300 Suffixe: fertig gewordene Seeds der später gestoppten Läufe**

| Variante | Fertige Seeds | Tage, ungefähr | Bank auf diesen Seeds | CF auf diesen Seeds | Verbesserung zur Bank, ungefähr |
|---|---|---:|---:|---:|---:|
| (c), 300 Suffixe | 0, 1, 2, 3, 4 | 19,88 | 22,33 | 19,64 | 10,95 % |
| OE, 300 Suffixe | 0, 1, 3, 4, 5, 6, 8 | 21,30 | 22,42 | 19,47 | 4,92 % |

Die 300er-Logs enthalten die Outcomes nur auf 0,1 Tage gerundet. Es gibt keine vollständige Zehn-Seed-Auswertung. Die Tabellenzeilen verwenden unterschiedliche Seed-Mengen und sind daher keine unmittelbar gleiche Stichprobe. Auf den gemeinsamen Seeds 0, 1, 3, 4 erreicht (c) ungefähr 19,650 und OE 21,275 Tage. Auf den fünf fertig gewordenen (c)-Seeds lagen 100 Suffixe bei 20,601 gegenüber ungefähr 19,88 Tagen mit 300 Suffixen. Das ist ein gepaarter Zwischenstand, keine vollständige Wiederholung.

**Gepaarte Mini-Tests: ein Seed, 30 Ziehungen, 200 gewertete Fälle**

Alle vier Varianten: Seed 0, Fallfenster 700–899, 2.200 Ankünfte insgesamt, zufällige Mitarbeiterzuteilung. Die einzelnen Bank-Dauern aller 200 Fälle sind zwischen den vier Varianten exakt gleich; Bankmittel 20,374 Tage.

| Variante | Tage ↓ | Verbesserung ggü. Bank | Quelle |
|---|---:|---:|---|
| (c), 1 ms | 20,198 | 0,86 % | [JSON](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/c_shared_gap_suffixe30_paired200_seed0/original_1ms/psp_seed0_row.json) |
| (c), ein Normal-Abstand je Option | 21,212 | -4,11 % | [JSON](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/c_shared_gap_suffixe30_paired200_seed0/normal_einmal_pro_option/psp_seed0_row.json) |
| OE, ohne Sperre | 21,195 | -4,03 % | [JSON](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/token_oe_suffixe30_vergleich_c_seed0_20261113/oe_ohne_sperre/psp_seed0_row.json) |
| OE, beobachtete Übergänge | 21,737 | -6,69 % | [JSON](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/token_oe_suffixe30_paired200_seed0_v2/mit_sperre/psp_seed0_row.json) |

Beim neuen (c) bekommt jede Option einmal je Entscheidung einen eigenen gezogenen Zeitabstand; alle 30 Suffixe dieser Option teilen ihn. Die Normalparameter stammen aus den beobachteten Trainingsabständen je Wahlpunkt und Option, nicht aus einem zusätzlichen prefixabhängigen Zeitmodell. Die Verteilungen sind bei null trunkiert: Bei den stark schiefen Daten kann dies die erwarteten Abstände erhöhen. Die Unsicherheit des ersten Abstands wird bei dieser Variante nicht über 30 Zeitziehungen gemittelt. OE zieht dagegen je Suffix einen Optionsabstand über die vorhergesagte Zeitklasse und deren Zeitverteilung. Die Sperre in der vierten Zeile erlaubt nur die im äußeren Trainingssplit beobachteten Aktivität|Lifecycle-Paare und Fallenden.

Diese konkrete Zeitvariante verschlechtert (c) um 1,013 Tage; die Übergangssperre verschlechtert OE um 0,542 Tage. Das sind zwei einzelne Mini-Tests. Sie belegen weder, dass eine korrekte Zeitmodellierung schadet, noch, dass der ursprüngliche Unterschied zwischen (c) und OE überwiegend auf dem Optionszeitpunkt beruht. Der erneute OE-Lauf reproduziert alle Entscheidungen und Falldauern des früheren OE-Laufs desselben Seeds exakt; er ist kein zusätzlicher unabhängiger Seed.

**Suffixvorhersage und Zeitschätzung auf identischen Testprefixen**

Die folgenden Werte stammen aus 1.000 identischen (Testfall, Prefixlänge)-Paaren aus den zurückgehaltenen echten Fällen, jeweils 100 Suffixe pro Prefix. Die Übereinstimmung der Prefixmengen wurde bei dieser Zusammenstellung überprüft. Das sind freie Fortsetzungen eines echten Prefix, keine Messung bekannter individueller Gegenfakten an einem Wahlpunkt.

Treffer Schritt k: Das häufigste gezogene Token an Position k entspricht dem beobachteten Token; nur Prefixe mit einem echten Event an dieser Position werden gewertet. DLS ist die mittlere normierte Sequenzähnlichkeit der gezogenen Suffixe (1 = identisch). MAE und CRPS sind Fehlermaße: kleiner ist besser. Zeit-Bias = mittlere vorhergesagte Restzeit minus beobachtete Restzeit, also positiv = zu lang. Die Abdeckung eines 80-%-Intervalls sollte ungefähr 80 % erreichen.

| Modell | Schritt 1 ↑ | Schritt 5 ↑ | Schritt 10 ↑ | DLS ↑ | Längen-MAE ↓ | Erster Abstand: MAE (h) ↓ | Restzeit-MAE (Tage) ↓ | Zeit-Bias (Tage) | CRPS (Tage) ↓ | 80-%-Abdeckung |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Semi-Markov | 76,4 % | 30,4 % | 9,4 % | 0,264 | 13,28 | 15,38 | 9,51 | 2,36 | 5,81 | 88,3 % |
| s40, Gaps | 83,8 % | 40,7 % | 8,5 % | 0,234 | 13,12 | 13,83 | 9,69 | 6,41 | 6,08 | 87,6 % |
| (a), Log-Gaps | 85,5 % | 42,3 % | 12,7 % | 0,278 | 8,99 | 17,44 | 15,57 | 13,64 | 9,21 | 95,4 % |
| (c), normal | 83,4 % | 38,5 % | 9,4 % | 0,249 | 10,57 | 13,13 | 8,13 | -0,05 | 5,40 | 87,8 % |
| (d), Zeitklassen; nur offline | 85,4 % | 49,2 % | 21,1 % | 0,281 | 12,12 | 11,71 | 7,81 | -0,16 | 5,23 | 75,3 % |
| OE | 84,1 % | 49,8 % | 21,2 % | 0,274 | 12,12 | 11,89 | 7,98 | 0,36 | 5,31 | 77,4 % |
| (c), Teacher Forcing 1; nur offline | 85,8 % | 50,3 % | 25,2 % | 0,404 | 10,65 | 12,35 | 8,62 | 0,96 | 5,49 | 83,6 % |

(c) hat fast keinen mittleren Restzeit-Bias, aber trotzdem rund 8,13 Tage mittleren absoluten Fehler. Sich ausgleichende Über- und Unterschätzungen ergeben keine genaue Vorhersage einzelner Fälle. OE ist bei längeren Eventfolgen und beim ersten Zeitabstand besser als (c), bei der Restzeit-MAE nur geringfügig besser. Das Modell mit Teacher Forcing 1 hat die beste DLS dieser Tabelle, aber schwächere Optionssignale als (c). Für (d) und Teacher Forcing 1 wurde kein abgeschlossener Online-Lauf an den vier aktuellen Punkten gefunden. Für das neue (c) mit geteiltem Optionsabstand und OE mit Übergangssperre gibt es noch keine eigenständige Suffix-/Zeitgüte-Auswertung dieser Art; Werte des unveränderten Basismodells dürfen nicht als neu gemessene Variante ausgegeben werden.

Für (b) gibt es nur die ältere Eventauswertung mit 200 Prefixen aus dem gesamten echten Log: nächstes Event 84,5 %, DLS 0,266. Diese Stichprobe kann Trainingsfälle enthalten; sie ist keine saubere Auswertung auf ausschließlich zurückgehaltenen Fällen und nicht die obige 1.000-Prefix-Stichprobe. Eine separate Diagnose an je 100 Wahlpunkt-Prefixen zurückgehaltener Fälle ergibt für den direkten Restzeitkopf faktische Gesamtzeit-MAE von 9,25–10,73 Tagen, gegenüber 19,91–26,00 Tagen bei Suffixbewertung desselben Modells. Das sind andere Stichproben und Zielgrößen als die Restzeit-MAE in der obigen Tabelle.

Die ältere 200-Prefix-Tabelle ist hier vollständig enthalten, damit auch die ursprünglichen Online-Modelle dokumentiert sind: [Original-CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/suffix_vorhersage_stufe2.csv). Ihre Werte sind wegen der möglichen Überschneidung mit Trainingsfällen vorsichtig zu interpretieren.

| Ältere Eventauswertung, 200 Prefixe aus dem gesamten Log | Schritt 1 | Schritt 5 | DLS | Längen-MAE |
|---|---:|---:|---:|---:|
| Markov-Kette | 74,5 % | 26,0 % | 0,240 | 18,53 |
| PSP Nachtlauf (5.000 Fälle, 1e-4) | 79,0 % | 29,6 % | 0,146 | 50,87 |
| PSP echter Log (10.000 Fälle, 1e-5, neue Punkte 3-fach) | 79,0 % | 37,3 % | 0,246 | 18,89 |
| PSP-Variante (Decoder 40, ohne Zukunftswissen, 1e-4) | 83,0 % | 39,1 % | 0,239 | 13,20 |
| Stufe 2 (a): alle Fälle, Decoder 60, log-normale Abstände | 83,5 % | 42,0 % | 0,277 | 8,69 |
| Stufe 2 (b): wie (a) mit Restzeit-Kopf | 84,5 % | 40,2 % | 0,266 | 8,84 |

**Optionssignal und Rauschen**

Je Wahlpunkt 200 identische erste Entscheidungen aus echten Testfällen, hälftig beobachtete Aktion bzw. Referenz, 100 Suffixe je Option. Positiver geschätzter Aktionsvorteil bedeutet kürzere Durchlaufzeit. Die Vorzeichenquote vergleicht mit der Richtung des AIPW-Effekts aus dem Screening; sie ist keine bekannte individuelle Entscheidungsgenauigkeit.

| Aktion | (c): Vorteil (Tage) | OE: Vorteil (Tage) | (c): Vorzeichen wie AIPW | OE: Vorzeichen wie AIPW | (c): SE-Median (Tage) | OE: SE-Median (Tage) |
|---|---:|---:|---:|---:|---:|---:|
| annehmen_bei_konzept | 1,46 | 1,02 | 84,5 % | 73,0 % | 1,27 | 1,76 |
| annehmen_nach_vervollstaendigung | 2,09 | 1,11 | 96,5 % | 72,0 % | 1,29 | 1,86 |
| angebot_nachlegen | -1,21 | -0,67 | 79,5 % | 65,5 % | 1,36 | 1,93 |
| handle_leads | -1,23 | 0,21 | 80,5 % | 47,0 % | 1,25 | 1,89 |

OE hat hier rund 40–50 % höhere Standardfehler als (c). Beim Handle-leads-Punkt weicht außerdem die mittlere Richtung vom Screening ab. In einer separaten Wiederholungsprobe mit 48 Prefixen und jeweils 300 Suffixen wechselte OE bei 11/48 Prefixen die Wahl zwischen zwei Zufallsstarts. Diese kleinen Proben erklären keine feste Anzahl verlorener Online-Tage.

Die OE-Pfadprüfung fand bei ungefähr 99,3 % der gezogenen Suffixe mindestens einen im äußeren Trainingssplit unbeobachteten Übergang; das ist kein Anteil ungültiger Einzelschritte. Ungefähr 35,6 % der Suffixe enthielten keines der Events A_Pending, A_Denied, A_Cancelled. Auch das ist keine allgemeine Quote aller PSP-Modelle. Die Mini-Auswertung mit Übergangssperre zeigt bisher keine Verbesserung des Online-Outcomes. Details: [OE-Diagnose](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/token_oe_300_diagnose_20261113/auswertung.md).

**Abweichendes Ziel: PSP-ERM**

Zusätzlich abgeschlossen: (c) mit 100 Suffixen und entropischer Risikobewertung, γ = 0,1/Tag, zehn Seeds mit je 1.000 Fällen. Hier ist die Mitarbeiterzuteilung learned; der Lauf gehört daher nicht in die Rangliste der obigen Random-Zuteilung. Er optimiert zudem ein anderes Risiko-Ziel.

PSP-ERM: 22,437 Tage; zugehörige Bank 22,600 Tage. Empirisches ERM: 63,487 gegenüber 62,364 Tagen bei der Bank; kleiner wäre besser. Die gepaarte ERM-Differenz +1,124 Tage hat ein 95-%-KI von [−16,038; +18,285]; kein gesicherter ERM-Vorteil. Das Netz ist dasselbe (c), die Auswahlregel ist anders. Quelle: [ERM-Auswertung](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/neue_punkte_online_psp_erm_gamma0.1_suffixe100_bank/auswertung.md).

**Noch ohne Ergebnis**

OE mit Übergangssperre und 300 Ziehungen, Seed 0: gespeicherter Status `running`, 1275 Entscheidungen, zuletzt aktualisiert 2026-11-13T13:14:22+01:00. Noch kein fertiger Online-Outcome. Der vorherige abgelöste Lauf und der fehlgeschlagene erste Sperren-Kurzlauf werden nicht als zusätzliche Ergebnisse gezählt. Die Annahmepunkt-Trainingsvariante hat ebenfalls keinen fertigen Online-Nachweis in den gefundenen Berichten.

**Archiv: andere BPIC17-Wahlpunkte und ältere Datensätze**

Die folgenden vollständigen Berichtszeilen sind zur Vollständigkeit aufgeführt. Wahlpunkte, Daten, Fallzahlen und teilweise Outcome unterscheiden sich. Kosten-Outcomes im Repair-Datensatz sind keine negativen Tage. Gleiche PSP-Ergebnisse in mehreren Vergleichsordnern können dieselben Läufe wiedergeben.

| Bericht / Policy | Datensatz | Fälle/Seed | Seeds | Zeit (Tage) | Outcome | Verbesserung zum jeweiligen Bank-Outcome | Quelle |
|---|---|---:|---:|---:|---:|---:|---|
| results / PSP/Mitarbeiter_W_Call_incomplete_files (psp) | bpic17 | 2000 | 5 | 22,347 | -22,347 | -1,14 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results/prescription/PSP/Mitarbeiter_W_Call_incomplete_files/bpic17_online.csv) |
| results / PSP_Vorausschau/Mitarbeiter_W_Call_incomplete_files (psp_lookahead) | bpic17 | 2000 | 5 | 22,330 | -22,330 | -1,08 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results/prescription/PSP_Vorausschau/Mitarbeiter_W_Call_incomplete_files/bpic17_online.csv) |
| results_100k/PSP_lr1e4_online / PSP (psp) | bpic17 | 1000 | 10 | 21,655 | -21,655 | 2,99 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/PSP_lr1e4_online/prescription/PSP/bpic17_online.csv) |
| results_100k/PSP_real_w3_10k_online / PSP (psp) | bpic17 | 1000 | 10 | 21,732 | -21,732 | 2,62 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/PSP_real_w3_10k_online/prescription/PSP/bpic17_online.csv) |
| results_100k/PSP_real_w3_10k_online_channel / PSP (psp) | bpic17 | 1000 | 10 | 20,738 | -20,738 | 7,07 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/PSP_real_w3_10k_online_channel/prescription/PSP/bpic17_online.csv) |
| results_100k/PSP_real_w3_10k_online_nur_angebot / PSP/channel_angebot_nachlegen (psp) | bpic17 | 1000 | 10 | 22,715 | -22,715 | -1,78 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/PSP_real_w3_10k_online_nur_angebot/prescription/PSP/channel_angebot_nachlegen/bpic17_online.csv) |
| results_100k/PSP_real_w3_10k_online_nur_controlflow / PSP/channel_angebot_nachlegen_handle_leads_annehmen_nach_vervollstaendigung_neues_angebot_beim_nachfassen (psp) | bpic17 | 1000 | 10 | 21,798 | -21,798 | 2,35 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/PSP_real_w3_10k_online_nur_controlflow/prescription/PSP/channel_angebot_nachlegen_handle_leads_annehmen_nach_vervollstaendigung_neues_angebot_beim_nachfassen/bpic17_online.csv) |
| results_100k/token_oe_suffixe30_paired200_seed0/ohne_sperre / PSP (psp) | bpic17 | 200 | 1 | 21,195 | -21,195 | -4,03 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/token_oe_suffixe30_paired200_seed0/ohne_sperre/prescription/PSP/bpic17_online.csv) |
| results_bpic19_durchlaufzeit / PSP (psp) | bpic19 | 5000 | 10 | 82,720 | -82,720 | -2,19 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_bpic19_durchlaufzeit/prescription/PSP/bpic19_online.csv) |
| results_bpic19_psp_w5 / PSP (psp) | bpic19 | 5000 | 10 | 84,921 | -84,921 | -4,91 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_bpic19_psp_w5/prescription/PSP/bpic19_online.csv) |
| results_repair_erm/gamma_0.3 / PSP_ERM (psp_erm) | repair | 1000 | 10 | 12,854 | -12,854 | 3,53 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_repair_erm/gamma_0.3/prescription/PSP_ERM/repair_online.csv) |
| results_repair_erm/gamma_1 / PSP_ERM (psp_erm) | repair | 1000 | 10 | 12,960 | -12,960 | 2,73 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_repair_erm/gamma_1/prescription/PSP_ERM/repair_online.csv) |
| results_repair_erm/gamma_3 / PSP_ERM (psp_erm) | repair | 1000 | 10 | 12,903 | -12,903 | 3,16 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_repair_erm/gamma_3/prescription/PSP_ERM/repair_online.csv) |
| results_repair_erm_w5/gamma_0.3 / PSP_ERM (psp_erm) | repair | 1000 | 10 | 12,455 | -12,455 | 6,52 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_repair_erm_w5/gamma_0.3/prescription/PSP_ERM/repair_online.csv) |
| results_repair_erm_w5/gamma_1 / PSP_ERM (psp_erm) | repair | 1000 | 10 | 12,476 | -12,476 | 6,36 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_repair_erm_w5/gamma_1/prescription/PSP_ERM/repair_online.csv) |
| results_repair_erm_w5/gamma_3 / PSP_ERM (psp_erm) | repair | 1000 | 10 | 12,522 | -12,522 | 6,02 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_repair_erm_w5/gamma_3/prescription/PSP_ERM/repair_online.csv) |
| results_repair_kosten / PSP (psp) | repair | 1000 | 10 | 12,591 | -11533,593 | 17,05 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_repair_kosten/prescription/PSP/repair_online.csv) |
| results_repair_kosten_b / PSP (psp) | repair | 1000 | 10 | 12,655 | -11456,509 | 16,98 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_repair_kosten_b/prescription/PSP/repair_online.csv) |
| results_repair_pipeline / PSP (psp) | repair | 1000 | 10 | 13,253 | -13,253 | 0,53 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_repair_pipeline/prescription/PSP/repair_online.csv) |
| results_repair_psp100 / PSP (psp) | repair | 1000 | 10 | 12,816 | -12,816 | 3,81 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_repair_psp100/prescription/PSP/repair_online.csv) |
| results_repair_psp_w5 / PSP (psp) | repair | 1000 | 10 | 12,446 | -12,446 | 6,59 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_repair_psp_w5/prescription/PSP/repair_online.csv) |
| results_repair_vorausschau_dlz / PSP (psp) | repair | 1000 | 10 | 12,446 | -12,446 | 6,59 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_repair_vorausschau_dlz/prescription/PSP/repair_online.csv) |
| results_repair_vorausschau_dlz / PSP_Vorausschau (psp_lookahead) | repair | 1000 | 10 | 12,438 | -12,438 | 6,66 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_repair_vorausschau_dlz/prescription/PSP_Vorausschau/repair_online.csv) |
| results_repair_vorausschau_kosten / PSP (psp) | repair | 1000 | 10 | 12,591 | -11533,593 | 17,05 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_repair_vorausschau_kosten/prescription/PSP/repair_online.csv) |
| results_repair_vorausschau_kosten / PSP_Vorausschau (psp_lookahead) | repair | 1000 | 10 | 15,142 | -13357,886 | 3,94 % | [CSV](/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_repair_vorausschau_kosten/prescription/PSP_Vorausschau/repair_online.csv) |

**Einordnung**

Bei den vollständigen Hauptvergleichen ist der direkte Restzeitkopf der stärkste PSP; unter den dort vollständig ausgewerteten Suffix-Policies ist (c) stärker als die Log-Gap-Varianten. OE verbessert Teile der freien Suffixvorhersage, liefert aber schwächere und noisigere Optionsvergleiche. Gute Prognosen beobachteter Fortsetzungen, gute Zeitkalibrierung und gute Entscheidungen sind hier drei getrennt zu prüfende Eigenschaften. Weder CF-Effekte noch AIPW-Vorzeichen liefern eine bekannte individuelle Gegenfakten-Wahrheit.

Alle Zahlen sind aus vorhandenen CSV-, JSON- und Logdateien zusammengeführt. Es wurden keine neuen Trainings oder Simulationen gestartet. Die Originaldateien wurden nicht geändert.
