# Ergebnisregister BPIC17-PSP (erzeugt 2026-11-23 11:57)

Erzeugt von `register_erstellen.py` aus den Ergebnisordnern; Deutungen in `befunde.md`; maschinenlesbar in `register.csv`, Verluste je Epoche in `verlustkurven.csv`. Erste Zeile jeder Tabelle = Wahrheit/Referenz, sofern vorhanden.

**Messungen** (alle mit Durchlaufzeit als Ziel, Katalog `catalog_neue_punkte_zuteilung_random.yaml`):
- Optionssignal (`psp_rauschen.py`): 200 echte Testentscheidungen je Wahlpunkt, 100 Ziehungen je Option; Anteil mit Vorzeichen wie AIPW auf dem echten Log (AIPW: konzept +5,2, nach V. +3,0, nachlegen −1,4, leads −2,2 Tage; Vorzeichen-Konvention der Datei), gepaarter SE je Entscheidung, Korrelation der Optionen (gemeinsame Zufallszahlen), gezogene vs. echte Durchlaufzeit.
- Simulator-Wahrheit (`punkte_alle.py`): 60 Simulatorzustände je Gruppe, Vorteil der Regel-Option in Tagen und Anteil Zustände mit richtigem Vorzeichen, MC-SE je Entscheidung.
- Gedächtnis: `nachlegen_erklaerung.py` (1. Besuch Nachlegen, extra minus senden: Restzeit, zusätzlich versendete Angebote, Nachfassen bis Validierung), `angebote_einfuegen.py` (zweites Angebot in echte Prefixe eingefügt: Restzeit B−A; richtige Richtung: positiv), `leads_gedaechtnis.py` (Restzeit ab erstem O_Sent, Lead bearbeitet minus zurückgezogen; echt +2,83 ± 1,11).
- Suffixgüte (`psp_suffixguete.py`): 1.000 Testprefixe, 100 Ziehungen.

## Konfigurationen

| label | tag | konfig |
|---|---|---|
| Wahrheit/Referenz | – | Simulator-RCT (Vorteil Regel-Option), Simulator mit Bank danach (Nachlegen), echter Log (Lead-Gedächtnis, Durchlaufzeit) |
| (a) | real_alle_s60_loggaps | alle Trainingsfälle (20.482/4.726), Decoder 60, 15 Ep., LR 1e-4, Wahlpunkte ×3, TF-Plan max(0, 1−ep/7,5); Log-Abstände; Option +1 ms |
| (b) | real_alle_s60_loggaps_rest | wie (a) + Restzeitkopf (Kopfbewertung verboten, hier Suffixe) |
| (c) | real_alle_s60_normal | alle Trainingsfälle (20.482/4.726), Decoder 60, 15 Ep., LR 1e-4, Wahlpunkte ×3, TF-Plan max(0, 1−ep/7,5); normale Abstände, Durchlaufzeit = Summe der Abstände; statische Merkmale ohne Zukunft; Option +1 ms |
| (d) 1 ms | real_alle_s60_klassen | alle Trainingsfälle (20.482/4.726), Decoder 60, 15 Ep., LR 1e-4, Wahlpunkte ×3, TF-Plan max(0, 1−ep/7,5); Zeitklassen (Klassenkopf); Option +1 ms |
| D | real_alle_s60_klassen_option_encoder | (d)-Netz, Option als letztes Prefix-Event mit eigenem gezogenem Abstand (option_as encoder) |
| OE | real_alle_s60_token_oe | alle Trainingsfälle (20.482/4.726), Decoder 60, 15 Ep., LR 1e-4, Wahlpunkte ×3, TF-Plan max(0, 1−ep/7,5); gemeinsames Token Aktivität/Lifecycle/Zeitklasse; Option im Encoder |
| (e) | real_alle_s60_klassen_event_oe | Zeitklasse bedingt auf das Event desselben Schritts, Option im Encoder |
| (c)-TF1 | real_alle_s60_normal_tf1 | wie (c), Teacher Forcing immer 1 |
| D-TF1 | real_alle_s60_klassen_tf1_option_encoder | wie D, Teacher Forcing immer 1 |
| OE-TF1 Ep. 26 | (Checkpoint) oe_tf1_100_training_20261115 | wie OE, TF immer 1, 100 Ep. geplant, bei Ep. ~27 gestoppt; Stand Ep. 26 |
| OE-TF1 Ep. 1 | (Checkpoint) oe_tf1_100_training_20261115 | wie OE-TF1, Stand Ep. 1 (bester Val-Verlust) |
| s40 | real_np_s40_ohne_zukunft | 10k Fälle, Decoder 40, 20 Ep., LR 1e-4, Abstände, statische ohne Zukunft |
| 10k neue Punkte | real_w3_10k_np | 10k Fälle, Decoder 5, 100 Ep., LR 1e-5 (ursprüngliches Training) |
| ZZ ep3 | real_alle_s60_normal_zaehler_zeit (Checkpoint ep3) | (c) + Zähler je Aktivität/Lifecycle in Encoder/Decoder + Zeitköpfe sehen das Event des Schritts; Option im Encoder nach dem Training; Stand Epoche 3 |
| ZZ ep4 | real_alle_s60_normal_zaehler_zeit (Checkpoint ep4) | (c) + Zähler je Aktivität/Lifecycle in Encoder/Decoder + Zeitköpfe sehen das Event des Schritts; Option im Encoder nach dem Training; Stand Epoche 4 |
| ZZ ep5 | real_alle_s60_normal_zaehler_zeit (Checkpoint ep5) | (c) + Zähler je Aktivität/Lifecycle in Encoder/Decoder + Zeitköpfe sehen das Event des Schritts; Option im Encoder nach dem Training; Stand Epoche 5 |
| ZZ ep6 | real_alle_s60_normal_zaehler_zeit (Checkpoint ep6) | (c) + Zähler je Aktivität/Lifecycle in Encoder/Decoder + Zeitköpfe sehen das Event des Schritts; Option im Encoder nach dem Training; Stand Epoche 6 |
| ZZ ep7 | real_alle_s60_normal_zaehler_zeit (Checkpoint ep7) | (c) + Zähler je Aktivität/Lifecycle in Encoder/Decoder + Zeitköpfe sehen das Event des Schritts; Option im Encoder nach dem Training; Stand Epoche 7 |
| ZZ ep9 | real_alle_s60_normal_zaehler_zeit (Checkpoint ep9) | (c) + Zähler je Aktivität/Lifecycle in Encoder/Decoder + Zeitköpfe sehen das Event des Schritts; Option im Encoder nach dem Training; Stand Epoche 9 |
| ZZ ep11 | real_alle_s60_normal_zaehler_zeit (Checkpoint ep11) | (c) + Zähler je Aktivität/Lifecycle in Encoder/Decoder + Zeitköpfe sehen das Event des Schritts; Option im Encoder nach dem Training; Stand Epoche 11 |
| ZZ ep13 | real_alle_s60_normal_zaehler_zeit (Checkpoint ep13) | (c) + Zähler je Aktivität/Lifecycle in Encoder/Decoder + Zeitköpfe sehen das Event des Schritts; Option im Encoder nach dem Training; Stand Epoche 13 |
| ZZ final | real_alle_s60_normal_zaehler_zeit | (c) + Zähler je Aktivität/Lifecycle in Encoder/Decoder + Zeitköpfe sehen das Event des Schritts; Option im Encoder nach dem Training; Stand Epoche 15 |
| ZZ-TF1 ep3 | real_alle_s60_normal_zaehler_zeit_tf1 (Checkpoint ep3) | wie ZZ, aber durchgehend Teacher Forcing 1 (konsistentes Trainingsziel); Stand Epoche 3 |
| ZZ-TF1 ep6 | real_alle_s60_normal_zaehler_zeit_tf1 (Checkpoint ep6) | wie ZZ, aber durchgehend Teacher Forcing 1 (konsistentes Trainingsziel); Stand Epoche 6 |
| ZZ-TF1 ep9 | real_alle_s60_normal_zaehler_zeit_tf1 (Checkpoint ep9) | wie ZZ, aber durchgehend Teacher Forcing 1 (konsistentes Trainingsziel); Stand Epoche 9 |
| ZZ-TF1 ep12 | real_alle_s60_normal_zaehler_zeit_tf1 (Checkpoint ep12) | wie ZZ, aber durchgehend Teacher Forcing 1 (konsistentes Trainingsziel); Stand Epoche 12 |
| ZZ-TF1 final | real_alle_s60_normal_zaehler_zeit_tf1 | wie ZZ, aber durchgehend Teacher Forcing 1 (konsistentes Trainingsziel); Stand Epoche 15 |
| ZZU-TF1 ep3 | real_alle_s60_normal_zaehler_uhren_tf1 (Checkpoint ep3) | wie ZZ-TF1, dazu Uhren: Tage seit dem letzten Auftreten je Aktivität/Lifecycle-Token als Eingabe; Stand Epoche 3 |
| ZZU-TF1 ep6 | real_alle_s60_normal_zaehler_uhren_tf1 (Checkpoint ep6) | wie ZZ-TF1, dazu Uhren: Tage seit dem letzten Auftreten je Aktivität/Lifecycle-Token als Eingabe; Stand Epoche 6 |
| ZZU-TF1 ep9 | real_alle_s60_normal_zaehler_uhren_tf1 (Checkpoint ep9) | wie ZZ-TF1, dazu Uhren: Tage seit dem letzten Auftreten je Aktivität/Lifecycle-Token als Eingabe; Stand Epoche 9 |
| ZZU-TF1 ep12 | real_alle_s60_normal_zaehler_uhren_tf1 (Checkpoint ep12) | wie ZZ-TF1, dazu Uhren: Tage seit dem letzten Auftreten je Aktivität/Lifecycle-Token als Eingabe; Stand Epoche 12 |
| ZZU-TF1 final | real_alle_s60_normal_zaehler_uhren_tf1 | wie ZZ-TF1, dazu Uhren: Tage seit dem letzten Auftreten je Aktivität/Lifecycle-Token als Eingabe; Stand Epoche 15 |
| ZZ-TF1-WP ep5 | real_alle_s60_normal_zaehler_zeit_tf1_wp (Checkpoint ep5) | wie ZZ-TF1, aber nur die Trainingsproben am Options-Event eines Wahlpunkts (only_points, ~75.000 statt ~933.000), 30 Epochen geplant, bei Ep. 18 abgebrochen; Stand Epoche 5 |
| ZZ-TF1-WP ep10 | real_alle_s60_normal_zaehler_zeit_tf1_wp (Checkpoint ep10) | wie ZZ-TF1, aber nur die Trainingsproben am Options-Event eines Wahlpunkts (only_points, ~75.000 statt ~933.000), 30 Epochen geplant, bei Ep. 18 abgebrochen; Stand Epoche 10 |
| ZZ-TF1-WP150 ep15 | real_alle_s60_normal_zaehler_zeit_tf1_wp150 (Checkpoint ep15) | wie ZZ-TF1-WP, Training und Validierung nur am Options-Event (only_points + only_points_validation), 150 Epochen (≈ 10 ZZ-TF1-Epochen an Schritten); Stand Epoche 15 |
| ZZ-TF1-WP150 ep45 | real_alle_s60_normal_zaehler_zeit_tf1_wp150 (Checkpoint ep45) | wie ZZ-TF1-WP, Training und Validierung nur am Options-Event (only_points + only_points_validation), 150 Epochen (≈ 10 ZZ-TF1-Epochen an Schritten); Stand Epoche 45 |
| ZZ-TF1-WP150 ep90 | real_alle_s60_normal_zaehler_zeit_tf1_wp150 (Checkpoint ep90) | wie ZZ-TF1-WP, Training und Validierung nur am Options-Event (only_points + only_points_validation), 150 Epochen (≈ 10 ZZ-TF1-Epochen an Schritten); Stand Epoche 90 |
| (c) Masken | (beschränkt) c_masken | Kopie mit beschränktem Ziehen (psp_beschraenkt.py): Übergänge zwischen Events und Lifecycle-Folgen innerhalb von Aktivitäten wie in den Trainingsfällen, Fallende erst nach A_Pending/Denied/Cancelled |
| ZZ ep9 Masken+Gumbel | (beschränkt) zz9_masken_gumbel | Kopie mit beschränktem Ziehen (psp_beschraenkt.py): Übergänge zwischen Events und Lifecycle-Folgen innerhalb von Aktivitäten wie in den Trainingsfällen, Fallende erst nach A_Pending/Denied/Cancelled; Gumbel-Kopplung der Eventwahl |
| (c) Masken+Gumbel | (beschränkt) c_masken_gumbel | Kopie mit beschränktem Ziehen (psp_beschraenkt.py): Übergänge zwischen Events und Lifecycle-Folgen innerhalb von Aktivitäten wie in den Trainingsfällen, Fallende erst nach A_Pending/Denied/Cancelled; Gumbel-Kopplung der Eventwahl |
| ZZ-TF1 ep12 Gumbel | (beschränkt) zztf1_12_gumbel | Kopie mit beschränktem Ziehen (psp_beschraenkt.py): Übergänge zwischen Events und Lifecycle-Folgen innerhalb von Aktivitäten wie in den Trainingsfällen, Fallende erst nach A_Pending/Denied/Cancelled; Gumbel-Kopplung der Eventwahl |
| T-Learner (ZZ-TF1) | (t_learner_20261117) t_zztf1 | T-Learner: zwei PSPs (Aktion / Referenz), warm vom fertigen ZZ-TF1, je nachtrainiert nur auf den Wahlpunkt-Proben ihrer Gruppe (LR 3e-5, Aktion 40 / Referenz 20 Epochen); jede Option zieht aus ihrem Modell; Gedächtnistests ohne Option messen nur das Referenz-Modell |
| ZZ-TF1 (RCT) Ep. 5 | (rct_tlearner_20261118) zzrct_ep5 | trainiert auf Simulator-RCT-Daten des korrigierten Simulators (8 × 4.000 Fälle, Optionen 50/50, Nachlegen nur 1. Besuch randomisiert), sonst wie ZZ-TF1 bzw. T-Learner; Testprefixe aus dem echten Log |
| ZZ-TF1 (RCT) | (rct_tlearner_20261118) zzrct | trainiert auf Simulator-RCT-Daten des korrigierten Simulators (8 × 4.000 Fälle, Optionen 50/50, Nachlegen nur 1. Besuch randomisiert), sonst wie ZZ-TF1 bzw. T-Learner; Testprefixe aus dem echten Log |
| T-Learner (RCT) | (rct_tlearner_20261118) tlrct | trainiert auf Simulator-RCT-Daten des korrigierten Simulators (8 × 4.000 Fälle, Optionen 50/50, Nachlegen nur 1. Besuch randomisiert), sonst wie ZZ-TF1 bzw. T-Learner; Testprefixe aus dem echten Log |
| ZZ-TF1 + Restzeit-Hilfsziel | (restzeit_hilfsziel_20261118) zz_restzeit | ZZ-TF1 fertig + Restzeit-Kopf als reines Hilfsziel im Training (5 Epochen nachtrainiert, LR 3e-5, warm vom ZZ-TF1); Bewertung weiter nur über gezogene Suffixe |
| T-Learner Suffix-Verlust (Zufallszahlen je Event) | (suffix_verlust_20261119) t_sv_crn | T-Learner (ZZ-TF1), beide Arme warm nachtrainiert mit zusätzlichem Suffix-Verlust (sequence_loss: CRPS der frei gezogenen Restzeit gegen die echte Restzeit, je Batch 32 Prefixe × 16 Ziehungen, Gewicht 1; Aktion 10 / Referenz 5 Epochen, LR 3e-5, Teacher Forcing 1); Bewertung mit Zufallszahlen je Event (keyed_noise); Stand „+ Kalibrierung“: warm von diesen Armen weiter mit Kalibrierung je (Wahlpunkt, Option) im Suffix-Verlust (Gewicht 3, Prefixe je Gruppe gleich häufig; Aktion 10 / Referenz 5 Epochen); Stand „+ gepaarte Kalibrierung“ (20.11.): wie „+ Kalibrierung“, aber Zielwert = echtes Mittel derselben Prefixe im Batch und Quadrat der Abweichung aus zwei unabhängigen Ziehungshälften (kein Schrumpfdruck gegen fallabhängige Effekte); Stand „+ Kalibrierung Hälften“ (20.11.): Prefixe einer Gruppe im Batch in zwei disjunkte Hälften, Strafe (Mittel 1 − Gruppenmittel) · (Mittel 2 − Gruppenmittel), sonst wie „+ Kalibrierung“ |
| T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event) | (suffix_verlust_20261119) t_svk_crn | T-Learner (ZZ-TF1), beide Arme warm nachtrainiert mit zusätzlichem Suffix-Verlust (sequence_loss: CRPS der frei gezogenen Restzeit gegen die echte Restzeit, je Batch 32 Prefixe × 16 Ziehungen, Gewicht 1; Aktion 10 / Referenz 5 Epochen, LR 3e-5, Teacher Forcing 1); Bewertung mit Zufallszahlen je Event (keyed_noise); Stand „+ Kalibrierung“: warm von diesen Armen weiter mit Kalibrierung je (Wahlpunkt, Option) im Suffix-Verlust (Gewicht 3, Prefixe je Gruppe gleich häufig; Aktion 10 / Referenz 5 Epochen); Stand „+ gepaarte Kalibrierung“ (20.11.): wie „+ Kalibrierung“, aber Zielwert = echtes Mittel derselben Prefixe im Batch und Quadrat der Abweichung aus zwei unabhängigen Ziehungshälften (kein Schrumpfdruck gegen fallabhängige Effekte); Stand „+ Kalibrierung Hälften“ (20.11.): Prefixe einer Gruppe im Batch in zwei disjunkte Hälften, Strafe (Mittel 1 − Gruppenmittel) · (Mittel 2 − Gruppenmittel), sonst wie „+ Kalibrierung“ |
| T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event) | (suffix_verlust_20261119) t_svp_crn | T-Learner (ZZ-TF1), beide Arme warm nachtrainiert mit zusätzlichem Suffix-Verlust (sequence_loss: CRPS der frei gezogenen Restzeit gegen die echte Restzeit, je Batch 32 Prefixe × 16 Ziehungen, Gewicht 1; Aktion 10 / Referenz 5 Epochen, LR 3e-5, Teacher Forcing 1); Bewertung mit Zufallszahlen je Event (keyed_noise); Stand „+ Kalibrierung“: warm von diesen Armen weiter mit Kalibrierung je (Wahlpunkt, Option) im Suffix-Verlust (Gewicht 3, Prefixe je Gruppe gleich häufig; Aktion 10 / Referenz 5 Epochen); Stand „+ gepaarte Kalibrierung“ (20.11.): wie „+ Kalibrierung“, aber Zielwert = echtes Mittel derselben Prefixe im Batch und Quadrat der Abweichung aus zwei unabhängigen Ziehungshälften (kein Schrumpfdruck gegen fallabhängige Effekte); Stand „+ Kalibrierung Hälften“ (20.11.): Prefixe einer Gruppe im Batch in zwei disjunkte Hälften, Strafe (Mittel 1 − Gruppenmittel) · (Mittel 2 − Gruppenmittel), sonst wie „+ Kalibrierung“ |
| T-Learner Suffix-Verlust + Kalibrierung Hälften (Zufallszahlen je Event) | (suffix_verlust_20261119) t_svh_crn | T-Learner (ZZ-TF1), beide Arme warm nachtrainiert mit zusätzlichem Suffix-Verlust (sequence_loss: CRPS der frei gezogenen Restzeit gegen die echte Restzeit, je Batch 32 Prefixe × 16 Ziehungen, Gewicht 1; Aktion 10 / Referenz 5 Epochen, LR 3e-5, Teacher Forcing 1); Bewertung mit Zufallszahlen je Event (keyed_noise); Stand „+ Kalibrierung“: warm von diesen Armen weiter mit Kalibrierung je (Wahlpunkt, Option) im Suffix-Verlust (Gewicht 3, Prefixe je Gruppe gleich häufig; Aktion 10 / Referenz 5 Epochen); Stand „+ gepaarte Kalibrierung“ (20.11.): wie „+ Kalibrierung“, aber Zielwert = echtes Mittel derselben Prefixe im Batch und Quadrat der Abweichung aus zwei unabhängigen Ziehungshälften (kein Schrumpfdruck gegen fallabhängige Effekte); Stand „+ Kalibrierung Hälften“ (20.11.): Prefixe einer Gruppe im Batch in zwei disjunkte Hälften, Strafe (Mittel 1 − Gruppenmittel) · (Mittel 2 − Gruppenmittel), sonst wie „+ Kalibrierung“ |
| T-Learner Kopf je Event (Steigung) | (zeitkopf_je_event_20261118) t_kopf_steigung | T-Learner (ZZ-TF1), Zeitköpfe beider Arme um eigene Parameter je Event erweitert (Versatz je Event je Ausgabe für Mittel und log-Varianz getrennt; Variante Steigung zusätzlich Steigung je Event auf h) und nur die Zeitköpfe nachtrainiert (Rest eingefroren, Wahlpunkt-Proben des Arms, Teacher Forcing 1, LR 1e-3 / Versätze 3e-2, Aktion 30 / Referenz 12 Epochen) |
| T-Learner Kopf je Event (Versatz) | (zeitkopf_je_event_20261118) t_kopf_versatz | T-Learner (ZZ-TF1), Zeitköpfe beider Arme um eigene Parameter je Event erweitert (Versatz je Event je Ausgabe für Mittel und log-Varianz getrennt; Variante Steigung zusätzlich Steigung je Event auf h) und nur die Zeitköpfe nachtrainiert (Rest eingefroren, Wahlpunkt-Proben des Arms, Teacher Forcing 1, LR 1e-3 / Versätze 3e-2, Aktion 30 / Referenz 12 Epochen) |
| T-Learner Kopf je Event (Steigung, β-NLL 1) | (zeitkopf_je_event_20261118) t_kopf_steigung_bnll1 | T-Learner (ZZ-TF1), Zeitköpfe beider Arme um eigene Parameter je Event erweitert (Versatz je Event je Ausgabe für Mittel und log-Varianz getrennt; Variante Steigung zusätzlich Steigung je Event auf h) und nur die Zeitköpfe nachtrainiert (Rest eingefroren, Wahlpunkt-Proben des Arms, Teacher Forcing 1, LR 1e-3 / Versätze 3e-2, Aktion 30 / Referenz 12 Epochen) |

## Trainingseinstellungen (aus den gespeicherten Modellen ausgelesen)

Teacher-Forcing-Plan des Trainers je Epoche: max(tf_untergrenze, 1 − Epoche/(0,5 · epochen_geplant)); tf_untergrenze 1 = immer Teacher Forcing. epoche_stand = ausgewerteter Stand (bei Checkpoints die Epoche, sonst das fertige Modell). weitere_params = alle übrigen gesetzten Parameter (Zeitdarstellung, Option, Zähler usw.); leer = Standard (normale Abstände, Option +1 ms nach dem Entscheidungspunkt). Checkpoints haben keine Trainingszusammenfassung (Fallzahlen wie beim fertigen Lauf).

| label | epoche_stand | epochen_geplant | epochen_trainiert | lernrate | batch | regularisierung | decoderlaenge | wahlpunkt_gewicht | tf_untergrenze | zeit_aus | faelle_training | faelle_validierung | hidden | schichten | dropout | seed | statische_merkmale | val_bester | val_letzte |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Wahrheit/Referenz | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| (a) | 15 | 15 | 15 | 1e-04 | 256 | 1e-04 | 60 | 3 | 0.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 7.73 | 8.47 |
| (b) | 15 | 15 | 15 | 1e-04 | 256 | 1e-04 | 60 | 3 | 0.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 7.75 | 8.16 |
| (c) | 15 | 15 | 15 | 1e-04 | 256 | 1e-04 | 60 | 3 | 0.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 8.85 | 9.20 |
| (d) 1 ms | 15 | 15 | 15 | 1e-04 | 256 | 1e-04 | 60 | 3 | 0.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 10.22 | 11.61 |
| D | 15 | 15 | 15 | 1e-04 | 256 | 1e-04 | 60 | 3 | 0.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 10.22 | 11.61 |
| OE | 15 | 15 | 15 | 1e-04 | 256 | 1e-04 | 60 | 3 | 0.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 9.27 | 9.60 |
| (e) | 15 | 15 | 15 | 1e-04 | 256 | 1e-04 | 60 | 3 | 0.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 9.37 | 9.64 |
| (c)-TF1 | 15 | 15 | 15 | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 9.59 | 13.13 |
| D-TF1 | 15 | 15 | 15 | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 12.47 | 17.33 |
| OE-TF1 Ep. 26 | 26 | 100 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| OE-TF1 Ep. 1 | 1 | 100 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| s40 | 20 | 20 | 20 | 1e-04 | 256 | 1e-04 | 40 | 3 | 0.00 | (Original: Fallalter) | 10001 | 2307 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 7.84 | 7.86 |
| 10k neue Punkte | 100 | 100 | 100 | 1e-05 | 256 | 1e-04 | 5 | 3 | 0.00 | (Original: Fallalter) | 10001 | 2307 | 128 | 4 | 0.10 | 42 | alle (mit Zukunftswissen) | 2.81 | 2.82 |
| ZZ ep3 | 3 | 15 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 0.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZ ep4 | 4 | 15 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 0.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZ ep5 | 5 | 15 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 0.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZ ep6 | 6 | 15 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 0.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZ ep7 | 7 | 15 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 0.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZ ep9 | 9 | 15 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 0.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZ ep11 | 11 | 15 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 0.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZ ep13 | 13 | 15 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 0.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZ final | 15 | 15 | 15 | 1e-04 | 256 | 1e-04 | 60 | 3 | 0.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 8.99 | 9.14 |
| ZZ-TF1 ep3 | 3 | 15 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZ-TF1 ep6 | 6 | 15 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZ-TF1 ep9 | 9 | 15 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZ-TF1 ep12 | 12 | 15 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZ-TF1 final | 15 | 15 | 15 | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 9.48 | 12.87 |
| ZZU-TF1 ep3 | 3 | 15 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZU-TF1 ep6 | 6 | 15 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZU-TF1 ep9 | 9 | 15 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZU-TF1 ep12 | 12 | 15 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZU-TF1 final | 15 | 15 | 15 | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 9.61 | 12.96 |
| ZZ-TF1-WP ep5 | 5 | 30 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZ-TF1-WP ep10 | 10 | 30 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZ-TF1-WP150 ep15 | 15 | 150 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZ-TF1-WP150 ep45 | 45 | 150 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZ-TF1-WP150 ep90 | 90 | 150 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| (c) Masken | 15 | 15 | 15 | 1e-04 | 256 | 1e-04 | 60 | 3 | 0.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 8.85 | 9.20 |
| ZZ ep9 Masken+Gumbel | – | 15 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 0.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| (c) Masken+Gumbel | 15 | 15 | 15 | 1e-04 | 256 | 1e-04 | 60 | 3 | 0.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 8.85 | 9.20 |
| ZZ-TF1 ep12 Gumbel | – | 15 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| T-Learner (ZZ-TF1) | 20 | 20 | 20 | 3e-05 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 14.84 | 15.21 |
| ZZ-TF1 (RCT) Ep. 5 | – | 15 | – | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | – | – | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | – | – |
| ZZ-TF1 (RCT) | 15 | 15 | 15 | 1e-04 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | 20801 | 4799 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 9.52 | 12.58 |
| T-Learner (RCT) | 20 | 20 | 20 | 3e-05 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | 20801 | 4799 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 13.10 | 13.25 |
| ZZ-TF1 + Restzeit-Hilfsziel | 5 | 5 | 5 | 3e-05 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 13.00 | 13.26 |
| T-Learner Suffix-Verlust (Zufallszahlen je Event) | 5 | 5 | 5 | 3e-05 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 15.18 | 15.29 |
| T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event) | 5 | 5 | 5 | 3e-05 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 15.34 | 15.51 |
| T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event) | 5 | 5 | 5 | 3e-05 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 15.13 | 15.40 |
| T-Learner Suffix-Verlust + Kalibrierung Hälften (Zufallszahlen je Event) | 5 | 5 | 5 | 3e-05 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 15.27 | 15.33 |
| T-Learner Kopf je Event (Steigung) | 12 | 12 | 12 | 1e-03 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 14.20 | 14.20 |
| T-Learner Kopf je Event (Versatz) | 12 | 12 | 12 | 1e-03 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 15.18 | 15.18 |
| T-Learner Kopf je Event (Steigung, β-NLL 1) | 12 | 12 | 12 | 1e-03 | 256 | 1e-04 | 60 | 3 | 1.00 | gaps | 20482 | 4726 | 128 | 4 | 0.10 | 42 | case:LoanGoal+case:ApplicationType+case:RequestedAmount | 14.80 | 15.09 |

| label | weitere_params | modelldatei | modell_datum |
|---|---|---|---|
| Wahrheit/Referenz | – | – | – |
| (a) | log_gaps=True | prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_loggaps.pkl | 2026-11-10 06:16 |
| (b) | log_gaps=True; remaining_head=True | prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_loggaps_rest.pkl | 2026-11-10 10:09 |
| (c) |  | prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal.pkl | 2026-11-11 05:27 |
| (d) 1 ms | gap_classes=True | prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_klassen.pkl | 2026-11-11 10:49 |
| D | gap_classes=True; option_as=encoder | prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_klassen_option_encoder.pkl | 2026-11-13 14:09 |
| OE | gap_tokens=True; option_as=encoder | prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_token_oe.pkl | 2026-11-12 18:54 |
| (e) | gap_classes=True; gap_condition_on_event=True; option_as=encoder | prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_klassen_event_oe.pkl | 2026-11-14 10:02 |
| (c)-TF1 |  | prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal_tf1.pkl | 2026-11-12 14:32 |
| D-TF1 | gap_classes=True; option_as=encoder | prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_klassen_tf1_option_encoder.pkl | 2026-11-15 05:03 |
| OE-TF1 Ep. 26 | gap_tokens=True; option_as=encoder | prescriptive/scripts/results_100k/modellvergleich_20261116/oe_tf1_schnappschuss/psp_oe_tf1_letzte.pkl | 2026-11-16 10:42 |
| OE-TF1 Ep. 1 | gap_tokens=True; option_as=encoder | prescriptive/scripts/results_100k/modellvergleich_20261116/oe_tf1_schnappschuss/psp_oe_tf1_beste_val.pkl | 2026-11-16 10:42 |
| s40 |  | prescriptive/models/psp_model_durchlaufzeit_real_np_s40_ohne_zukunft.pkl | 2026-11-08 21:08 |
| 10k neue Punkte |  | prescriptive/models/psp_model_durchlaufzeit_real_w3_10k_np.pkl | 2026-11-07 16:56 |
| ZZ ep3 | counters=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/zaehler_zeit_20261116/modelle/psp_zz_ep3.pkl | 2026-11-16 16:44 |
| ZZ ep4 | counters=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/zaehler_zeit_20261116/modelle/psp_zz_ep4.pkl | 2026-11-17 11:07 |
| ZZ ep5 | counters=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/zaehler_zeit_20261116/modelle/psp_zz_ep5.pkl | 2026-11-17 11:18 |
| ZZ ep6 | counters=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/zaehler_zeit_20261116/modelle/psp_zz_ep6.pkl | 2026-11-17 11:28 |
| ZZ ep7 | counters=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/zaehler_zeit_20261116/modelle/psp_zz_ep7.pkl | 2026-11-16 21:24 |
| ZZ ep9 | counters=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/zaehler_zeit_20261116/modelle/psp_zz_ep9.pkl | 2026-11-16 23:35 |
| ZZ ep11 | counters=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/zaehler_zeit_20261116/modelle/psp_zz_ep11.pkl | 2026-11-17 02:07 |
| ZZ ep13 | counters=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/zaehler_zeit_20261116/modelle/psp_zz_ep13.pkl | 2026-11-17 04:39 |
| ZZ final | counters=True; option_as=encoder; time_on_event=True | prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit.pkl | 2026-11-17 07:11 |
| ZZ-TF1 ep3 | counters=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/zaehler_zeit_tf1_20261117/modelle/psp_zztf1_ep3.pkl | 2026-11-17 05:02 |
| ZZ-TF1 ep6 | counters=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/zaehler_zeit_tf1_20261117/modelle/psp_zztf1_ep6.pkl | 2026-11-17 08:36 |
| ZZ-TF1 ep9 | counters=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/zaehler_zeit_tf1_20261117/modelle/psp_zztf1_ep9.pkl | 2026-11-17 12:10 |
| ZZ-TF1 ep12 | counters=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/zaehler_zeit_tf1_20261117/modelle/psp_zztf1_ep12.pkl | 2026-11-17 15:56 |
| ZZ-TF1 final | counters=True; option_as=encoder; time_on_event=True | prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1.pkl | 2026-11-17 19:53 |
| ZZU-TF1 ep3 | clocks=True; counters=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/zaehler_uhren_tf1_20261117/modelle/psp_zzu_ep3.pkl | 2026-11-17 14:24 |
| ZZU-TF1 ep6 | clocks=True; counters=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/zaehler_uhren_tf1_20261117/modelle/psp_zzu_ep6.pkl | 2026-11-17 18:17 |
| ZZU-TF1 ep9 | clocks=True; counters=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/zaehler_uhren_tf1_20261117/modelle/psp_zzu_ep9.pkl | 2026-11-17 22:23 |
| ZZU-TF1 ep12 | clocks=True; counters=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/zaehler_uhren_tf1_20261117/modelle/psp_zzu_ep12.pkl | 2026-11-18 03:17 |
| ZZU-TF1 final | clocks=True; counters=True; option_as=encoder; time_on_event=True | prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_uhren_tf1.pkl | 2026-11-18 07:24 |
| ZZ-TF1-WP ep5 | counters=True; only_points=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/nur_wahlpunkte_20261117/modelle/psp_wp_ep5.pkl | 2026-11-17 20:56 |
| ZZ-TF1-WP ep10 | counters=True; only_points=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/nur_wahlpunkte_20261117/modelle/psp_wp_ep10.pkl | 2026-11-17 21:40 |
| ZZ-TF1-WP150 ep15 | counters=True; only_points=True; only_points_validation=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/nur_wahlpunkte150_20261117/modelle/psp_wp150_ep15.pkl | 2026-11-18 00:58 |
| ZZ-TF1-WP150 ep45 | counters=True; only_points=True; only_points_validation=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/nur_wahlpunkte150_20261117/modelle/psp_wp150_ep45.pkl | 2026-11-18 04:50 |
| ZZ-TF1-WP150 ep90 | counters=True; only_points=True; only_points_validation=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/nur_wahlpunkte150_20261117/modelle/psp_wp150_ep90.pkl | 2026-11-18 10:13 |
| (c) Masken |  | prescriptive/scripts/results_100k/beschraenkt_20261117/modelle/c_masken.pkl | 2026-11-17 00:31 |
| ZZ ep9 Masken+Gumbel | counters=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/beschraenkt_20261117/modelle/zz9_masken_gumbel.pkl | 2026-11-17 00:31 |
| (c) Masken+Gumbel |  | prescriptive/scripts/results_100k/beschraenkt_20261117/modelle/c_masken_gumbel.pkl | 2026-11-17 00:31 |
| ZZ-TF1 ep12 Gumbel | counters=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/beschraenkt_20261117/modelle/zztf1_12_gumbel.pkl | 2026-11-17 17:25 |
| T-Learner (ZZ-TF1) | counters=True; init_from=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1.pkl; only_options=['O_Sent (mail and online)/complete', 'W_Complete application/resume', 'W_Complete application/start', 'W_Handle leads/withdraw']; only_points=True; only_points_validation=True; option_as=encoder; time_on_event=True | prescriptive/models/psp_model_durchlaufzeit_t_learner_zztf1.pkl | 2026-11-18 03:07 |
| ZZ-TF1 (RCT) Ep. 5 | counters=True; option_as=encoder; time_on_event=True | prescriptive/scripts/results_100k/rct_tlearner_20261118/modelle/psp_zzrct_ep5.pkl | 2026-11-18 19:44 |
| ZZ-TF1 (RCT) | counters=True; option_as=encoder; time_on_event=True | prescriptive/models/psp_model_durchlaufzeit_rct_zaehler_zeit_tf1.pkl | 2026-11-19 10:47 |
| T-Learner (RCT) | counters=True; init_from=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/models/psp_model_durchlaufzeit_rct_zaehler_zeit_tf1.pkl; only_options=['O_Sent (mail and online)/complete', 'W_Complete application/resume', 'W_Complete application/start', 'W_Handle leads/withdraw']; only_points=True; only_points_validation=True; option_as=encoder; time_on_event=True | prescriptive/models/psp_model_durchlaufzeit_t_learner_rct.pkl | 2026-11-19 14:45 |
| ZZ-TF1 + Restzeit-Hilfsziel | counters=True; init_from=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1.pkl; option_as=encoder; remaining_head=True; time_on_event=True | prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_restzeit.pkl | 2026-11-18 19:01 |
| T-Learner Suffix-Verlust (Zufallszahlen je Event) | counters=True; init_from=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_referenz.pkl; keyed_noise=True; only_options=['O_Sent (mail and online)/complete', 'W_Complete application/resume', 'W_Complete application/start', 'W_Handle leads/withdraw']; only_points=True; only_points_validation=True; option_as=encoder; sequence_loss=True; time_on_event=True | prescriptive/models/psp_model_durchlaufzeit_t_learner_sv_crn.pkl | 2026-11-19 18:01 |
| T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event) | counters=True; init_from=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_referenz_sv.pkl; keyed_noise=True; only_options=['O_Sent (mail and online)/complete', 'W_Complete application/resume', 'W_Complete application/start', 'W_Handle leads/withdraw']; only_points=True; only_points_validation=True; option_as=encoder; sequence_loss={'calibration': 3.0}; time_on_event=True | prescriptive/models/psp_model_durchlaufzeit_t_learner_svk_crn.pkl | 2026-11-20 00:04 |
| T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event) | counters=True; init_from=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_referenz_sv.pkl; keyed_noise=True; only_options=['O_Sent (mail and online)/complete', 'W_Complete application/resume', 'W_Complete application/start', 'W_Handle leads/withdraw']; only_points=True; only_points_validation=True; option_as=encoder; sequence_loss={'calibration': 3.0, 'calibration_mode': 'paired'}; time_on_event=True | prescriptive/models/psp_model_durchlaufzeit_t_learner_svp_crn.pkl | 2026-11-20 03:22 |
| T-Learner Suffix-Verlust + Kalibrierung Hälften (Zufallszahlen je Event) | counters=True; init_from=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_referenz_sv.pkl; keyed_noise=True; only_options=['O_Sent (mail and online)/complete', 'W_Complete application/resume', 'W_Complete application/start', 'W_Handle leads/withdraw']; only_points=True; only_points_validation=True; option_as=encoder; sequence_loss={'calibration': 3.0, 'calibration_mode': 'split'}; time_on_event=True | prescriptive/models/psp_model_durchlaufzeit_t_learner_svh_crn.pkl | 2026-11-20 10:48 |
| T-Learner Kopf je Event (Steigung) | counters=True; head_only=True; init_from=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_referenz.pkl; offset_learning_rate=0.03; only_options=['O_Sent (mail and online)/complete', 'W_Complete application/resume', 'W_Complete application/start', 'W_Handle leads/withdraw']; only_points=True; only_points_validation=True; option_as=encoder; time_event_head=slope; time_on_event=True | prescriptive/models/psp_model_durchlaufzeit_t_learner_kopf_steigung.pkl | 2026-11-18 23:22 |
| T-Learner Kopf je Event (Versatz) | counters=True; head_only=True; init_from=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_referenz.pkl; offset_learning_rate=0.03; only_options=['O_Sent (mail and online)/complete', 'W_Complete application/resume', 'W_Complete application/start', 'W_Handle leads/withdraw']; only_points=True; only_points_validation=True; option_as=encoder; time_event_head=offset; time_on_event=True | prescriptive/models/psp_model_durchlaufzeit_t_learner_kopf_versatz.pkl | 2026-11-19 00:19 |
| T-Learner Kopf je Event (Steigung, β-NLL 1) | beta_nll=1.0; counters=True; head_only=True; init_from=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_referenz.pkl; offset_learning_rate=0.03; only_options=['O_Sent (mail and online)/complete', 'W_Complete application/resume', 'W_Complete application/start', 'W_Handle leads/withdraw']; only_points=True; only_points_validation=True; option_as=encoder; time_event_head=slope; time_on_event=True | prescriptive/models/psp_model_durchlaufzeit_t_learner_kopf_steigung_bnll1.pkl | 2026-11-19 02:30 |

### Netzaufbau und Daten (aus dem gespeicherten Netz)

Merkmale mit Vokabulargröße in Klammern (concept:name = Aktivität|Lifecycle-Token, bei OE mit Zeitklasse); fensterbreite = Prefixfenster; proben_wahlpunkt = Proben an den Wahlpunkten (mit wahlpunkt_gewicht gewichtet); ziehungen_standard = Ziehungen je Option, wenn die Auswertung nichts anderes setzt.

| label | encoder_merkmale | decoder_merkmale | statische_eingaben | ausgaben | eingabe_encoder | eingabe_decoder | parameter_anzahl |
|---|---|---|---|---|---|---|---|
| Wahrheit/Referenz | – | – | – | – | – | – | – |
| (a) | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, day_in_week, seconds_in_day, event_elapsed_time | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1092761 |
| (b) | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, day_in_week, seconds_in_day, event_elapsed_time | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1109339 |
| (c) | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1092761 |
| (d) 1 ms | concept:name(68), lifecycle:transition(9), org:resource(151), gap_class(11), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name, gap_class + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), gap_class(11), case_elapsed_time(1), event_elapsed_time(1) | 59 | 25 | 1097267 |
| D | concept:name(68), lifecycle:transition(9), org:resource(151), gap_class(11), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name, gap_class + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), gap_class(11), case_elapsed_time(1), event_elapsed_time(1) | 59 | 25 | 1097267 |
| OE | concept:name(284), lifecycle:transition(9), org:resource(151), gap_class(11), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(284), case_elapsed_time(1), event_elapsed_time(1) | 80 | 40 | 1173971 |
| (e) | concept:name(68), lifecycle:transition(9), org:resource(151), gap_class(11), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name, gap_class + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), gap_class(11), case_elapsed_time(1), event_elapsed_time(1) | 59 | 25 | 1098797 |
| (c)-TF1 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1092761 |
| D-TF1 | concept:name(68), lifecycle:transition(9), org:resource(151), gap_class(11), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name, gap_class + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), gap_class(11), case_elapsed_time(1), event_elapsed_time(1) | 59 | 25 | 1097267 |
| OE-TF1 Ep. 26 | concept:name(284), lifecycle:transition(9), org:resource(151), gap_class(11), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(284), case_elapsed_time(1), event_elapsed_time(1) | 80 | 40 | 1173971 |
| OE-TF1 Ep. 1 | concept:name(284), lifecycle:transition(9), org:resource(151), gap_class(11), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(284), case_elapsed_time(1), event_elapsed_time(1) | 80 | 40 | 1173971 |
| s40 | concept:name(66), lifecycle:transition(9), org:resource(149), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(15), case:ApplicationType(3), case:RequestedAmount | concept:name(66), case_elapsed_time(1), event_elapsed_time(1) | 52 | 19 | 1091853 |
| 10k neue Punkte | concept:name(66), lifecycle:transition(9), org:resource(149), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(15), case:ApplicationType(3), Accepted(3), Selected(3), case:RequestedAmount, FirstWithdrawalAmount, NumberOfTerms, MonthlyCost, CreditScore | concept:name(66), case_elapsed_time(1), event_elapsed_time(1) | 52 | 19 | 1093151 |
| ZZ ep3 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| ZZ ep4 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| ZZ ep5 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| ZZ ep6 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| ZZ ep7 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| ZZ ep9 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| ZZ ep11 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| ZZ ep13 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| ZZ final | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| ZZ-TF1 ep3 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| ZZ-TF1 ep6 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| ZZ-TF1 ep9 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| ZZ-TF1 ep12 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| ZZ-TF1 final | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| ZZU-TF1 ep3 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1128805 |
| ZZU-TF1 ep6 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1128805 |
| ZZU-TF1 ep9 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1128805 |
| ZZU-TF1 ep12 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1128805 |
| ZZU-TF1 final | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1128805 |
| ZZ-TF1-WP ep5 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| ZZ-TF1-WP ep10 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| ZZ-TF1-WP150 ep15 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| ZZ-TF1-WP150 ep45 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| ZZ-TF1-WP150 ep90 | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| (c) Masken | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1092761 |
| ZZ ep9 Masken+Gumbel | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| (c) Masken+Gumbel | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1092761 |
| ZZ-TF1 ep12 Gumbel | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| T-Learner (ZZ-TF1) | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| ZZ-TF1 (RCT) Ep. 5 | concept:name(67), lifecycle:transition(9), org:resource(108), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(67), case_elapsed_time(1), event_elapsed_time(1) | 48 | 19 | 1108491 |
| ZZ-TF1 (RCT) | concept:name(67), lifecycle:transition(9), org:resource(108), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(67), case_elapsed_time(1), event_elapsed_time(1) | 48 | 19 | 1108491 |
| T-Learner (RCT) | concept:name(67), lifecycle:transition(9), org:resource(108), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(67), case_elapsed_time(1), event_elapsed_time(1) | 48 | 19 | 1108491 |
| ZZ-TF1 + Restzeit-Hilfsziel | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1127975 |
| T-Learner Suffix-Verlust (Zufallszahlen je Event) | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event) | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event) | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| T-Learner Suffix-Verlust + Kalibrierung Hälften (Zufallszahlen je Event) | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111397 |
| T-Learner Kopf je Event (Steigung) | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1146485 |
| T-Learner Kopf je Event (Versatz) | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1111669 |
| T-Learner Kopf je Event (Steigung, β-NLL 1) | concept:name(68), lifecycle:transition(9), org:resource(151), case_elapsed_time, event_elapsed_time, day_in_week, seconds_in_day | concept:name + case_elapsed_time, event_elapsed_time | case:LoanGoal(14), case:ApplicationType(3), case:RequestedAmount | concept:name(68), case_elapsed_time(1), event_elapsed_time(1) | 53 | 19 | 1146485 |

| label | fensterbreite | proben | proben_validierung | proben_wahlpunkt | test_faelle | wahlpunkte | ziehungen_standard |
|---|---|---|---|---|---|---|---|
| Wahrheit/Referenz | – | – | – | – | – | – | – |
| (a) | 207 | 933392 | 179856 | 75381 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| (b) | 207 | 933392 | 179856 | 75381 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| (c) | 207 | 933392 | 179856 | 75381 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| (d) 1 ms | 207 | 933392 | 179856 | 75381 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| D | 207 | 933392 | 179856 | 75381 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| OE | 207 | 1084154 | 179856 | 150762 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| (e) | 207 | 933392 | 179856 | 75381 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| (c)-TF1 | 207 | 933392 | 179856 | 75381 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| D-TF1 | 207 | 933392 | 179856 | 75381 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| OE-TF1 Ep. 26 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| OE-TF1 Ep. 1 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| s40 | 167 | 453364 | 87441 | 36653 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| 10k neue Punkte | 97 | 453364 | 87441 | 36653 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ ep3 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ ep4 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ ep5 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ ep6 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ ep7 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ ep9 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ ep11 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ ep13 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ final | 207 | 933392 | 179856 | 75381 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ-TF1 ep3 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ-TF1 ep6 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ-TF1 ep9 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ-TF1 ep12 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ-TF1 final | 207 | 933392 | 179856 | 75381 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZU-TF1 ep3 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZU-TF1 ep6 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZU-TF1 ep9 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZU-TF1 ep12 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZU-TF1 final | 207 | 933392 | 179856 | 75381 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ-TF1-WP ep5 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ-TF1-WP ep10 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ-TF1-WP150 ep15 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ-TF1-WP150 ep45 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ-TF1-WP150 ep90 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| (c) Masken | 207 | 933392 | 179856 | 75381 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ ep9 Masken+Gumbel | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| (c) Masken+Gumbel | 207 | 933392 | 179856 | 75381 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ-TF1 ep12 Gumbel | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| T-Learner (ZZ-TF1) | 207 | 58371 | 13643 | 58371 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ-TF1 (RCT) Ep. 5 | – | – | – | – | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ-TF1 (RCT) | 231 | 1072580 | 210114 | 77837 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| T-Learner (RCT) | 231 | 43398 | 10138 | 43398 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| ZZ-TF1 + Restzeit-Hilfsziel | 207 | 933392 | 179856 | 75381 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| T-Learner Suffix-Verlust (Zufallszahlen je Event) | 207 | 58371 | 13643 | 58371 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event) | 207 | 58371 | 13643 | 58371 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event) | 207 | 58371 | 13643 | 58371 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| T-Learner Suffix-Verlust + Kalibrierung Hälften (Zufallszahlen je Event) | 207 | 58371 | 13643 | 58371 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| T-Learner Kopf je Event (Steigung) | 207 | 58371 | 13643 | 58371 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| T-Learner Kopf je Event (Versatz) | 207 | 58371 | 13643 | 58371 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |
| T-Learner Kopf je Event (Steigung, β-NLL 1) | 207 | 58371 | 13643 | 58371 | 6301 | angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads | 50 |

### Exakter Trainingsaufruf (Parameter laut Trainingslog, Startskripte in prescriptive/data)

| label | trainingsparameter_log | startskripte | trainingslog |
|---|---|---|---|
| Wahrheit/Referenz | – | – | – |
| (a) | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'log_gaps': True, 'time_from': 'gaps', 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (12:47:31) | bpic17_psp_real_stufe2.sh, bpic17_psp_stufe2_online.sh, bpic17_psp_stufe2_online_lauf3.sh | prescriptive/data/PSP_real_alle_s60_loggaps.log |
| (b) | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'log_gaps': True, 'time_from': 'gaps', 'remaining_head': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (16:59:57) | bpic17_psp_stufe2_online.sh | prescriptive/data/PSP_real_alle_s60_loggaps_rest.log |
| (c) | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (11:38:41) | bpic17_psp_annahme_check.sh, bpic17_psp_real_stufe2c.sh, bpic17_psp_stufe2c_online.sh | prescriptive/data/PSP_real_alle_s60_normal.log |
| (d) 1 ms | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'gap_classes': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (16:38:18) | bpic17_psp_real_stufe2d.sh | prescriptive/data/PSP_real_alle_s60_klassen.log |
| D | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'gap_classes': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (16:38:18) | kein eigenes Training: Kopie von (d) 1 ms, params option_as = encoder nachträglich gesetzt | prescriptive/data/PSP_real_alle_s60_klassen.log |
| OE | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'gap_tokens': True, 'option_as': 'encoder', 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (20:40:13) | bpic17_psp_annahme_check.sh, bpic17_psp_stufe3_voll.sh | prescriptive/data/PSP_real_alle_s60_token_oe.log |
| (e) | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'gap_classes': True, 'gap_condition_on_event': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (16:49:58) | bpic17_psp_real_stufe2e.sh | prescriptive/scripts/results_100k/e_suffixe30_cases200_seed0_20261113/pipeline.log |
| (c)-TF1 | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'teacher_forcing': 1.0, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (20:50:14) | bpic17_psp_annahme_check.sh, bpic17_psp_stufe3_voll.sh | prescriptive/data/PSP_real_alle_s60_normal_tf1.log |
| D-TF1 | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'gap_classes': True, 'teacher_forcing': 1.0, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (10:56:47) | – | prescriptive/scripts/results_100k/d_tf1_training_20261114/pipeline.log |
| OE-TF1 Ep. 26 | {'train_cases': 0, 'epochs': 100, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'teacher_forcing': 1.0, 'gap_tokens': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (05:34:04) | – | prescriptive/scripts/results_100k/oe_tf1_100_training_20261115/pipeline.log |
| OE-TF1 Ep. 1 | {'train_cases': 0, 'epochs': 100, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'teacher_forcing': 1.0, 'gap_tokens': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (05:34:04) | – | prescriptive/scripts/results_100k/oe_tf1_100_training_20261115/pipeline.log |
| s40 | {'train_cases': 10000, 'epochs': 20, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 40, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (12:54:41) | bpic17_psp_real_variante.sh | prescriptive/data/PSP_real_np_s40_ohne_zukunft.log |
| 10k neue Punkte | {'train_cases': 10000, 'epochs': 100, 'learning_rate': 1e-05, 'point_weight': 3, 'checkpoints': '/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/models/psp_training_real_w3_10k_np'} (15:14:48) | bpic17_psp_real_neue_punkte.sh | prescriptive/data/PSP_real_w3_10k_np.log |
| ZZ ep3 | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'counters': True, 'time_on_event': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (13:12:20) | bpic17_psp_real_zaehler_zeit.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_zeit.log |
| ZZ ep4 | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'counters': True, 'time_on_event': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (13:12:20) | bpic17_psp_real_zaehler_zeit.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_zeit.log |
| ZZ ep5 | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'counters': True, 'time_on_event': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (13:12:20) | bpic17_psp_real_zaehler_zeit.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_zeit.log |
| ZZ ep6 | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'counters': True, 'time_on_event': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (13:12:20) | bpic17_psp_real_zaehler_zeit.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_zeit.log |
| ZZ ep7 | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'counters': True, 'time_on_event': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (13:12:20) | bpic17_psp_real_zaehler_zeit.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_zeit.log |
| ZZ ep9 | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'counters': True, 'time_on_event': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (13:12:20) | bpic17_psp_real_zaehler_zeit.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_zeit.log |
| ZZ ep11 | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'counters': True, 'time_on_event': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (13:12:20) | bpic17_psp_real_zaehler_zeit.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_zeit.log |
| ZZ ep13 | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'counters': True, 'time_on_event': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (13:12:20) | bpic17_psp_real_zaehler_zeit.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_zeit.log |
| ZZ final | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'counters': True, 'time_on_event': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (13:12:20) | bpic17_psp_real_zaehler_zeit.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_zeit.log |
| ZZ-TF1 ep3 | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'teacher_forcing': 1.0, 'counters': True, 'time_on_event': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (01:07:49) | bpic17_psp_real_zaehler_zeit_tf1.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_zeit_tf1.log |
| ZZ-TF1 ep6 | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'teacher_forcing': 1.0, 'counters': True, 'time_on_event': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (01:07:49) | bpic17_psp_real_zaehler_zeit_tf1.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_zeit_tf1.log |
| ZZ-TF1 ep9 | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'teacher_forcing': 1.0, 'counters': True, 'time_on_event': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (01:07:49) | bpic17_psp_real_zaehler_zeit_tf1.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_zeit_tf1.log |
| ZZ-TF1 ep12 | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'teacher_forcing': 1.0, 'counters': True, 'time_on_event': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (01:07:49) | bpic17_psp_real_zaehler_zeit_tf1.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_zeit_tf1.log |
| ZZ-TF1 final | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'teacher_forcing': 1.0, 'counters': True, 'time_on_event': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (01:07:49) | bpic17_psp_real_zaehler_zeit_tf1.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_zeit_tf1.log |
| ZZU-TF1 ep3 | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'teacher_forcing': 1.0, 'counters': True, 'time_on_event': True, 'clocks': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (10:19:38) | bpic17_psp_real_zaehler_uhren_tf1.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_uhren_tf1.log |
| ZZU-TF1 ep6 | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'teacher_forcing': 1.0, 'counters': True, 'time_on_event': True, 'clocks': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (10:19:38) | bpic17_psp_real_zaehler_uhren_tf1.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_uhren_tf1.log |
| ZZU-TF1 ep9 | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'teacher_forcing': 1.0, 'counters': True, 'time_on_event': True, 'clocks': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (10:19:38) | bpic17_psp_real_zaehler_uhren_tf1.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_uhren_tf1.log |
| ZZU-TF1 ep12 | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'teacher_forcing': 1.0, 'counters': True, 'time_on_event': True, 'clocks': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (10:19:38) | bpic17_psp_real_zaehler_uhren_tf1.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_uhren_tf1.log |
| ZZU-TF1 final | {'train_cases': 0, 'epochs': 15, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'teacher_forcing': 1.0, 'counters': True, 'time_on_event': True, 'clocks': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (10:19:38) | bpic17_psp_real_zaehler_uhren_tf1.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_uhren_tf1.log |
| ZZ-TF1-WP ep5 | {'train_cases': 0, 'epochs': 30, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'teacher_forcing': 1.0, 'counters': True, 'time_on_event': True, 'only_points': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (19:56:50) | bpic17_psp_real_zaehler_zeit_tf1_wp.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_zeit_tf1_wp.log |
| ZZ-TF1-WP ep10 | {'train_cases': 0, 'epochs': 30, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'teacher_forcing': 1.0, 'counters': True, 'time_on_event': True, 'only_points': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (19:56:50) | bpic17_psp_real_zaehler_zeit_tf1_wp.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_zeit_tf1_wp.log |
| ZZ-TF1-WP150 ep15 | {'train_cases': 0, 'epochs': 150, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'teacher_forcing': 1.0, 'counters': True, 'time_on_event': True, 'only_points': True, 'only_points_validation': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (22:55:44) | bpic17_psp_real_zaehler_zeit_tf1_wp150.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_zeit_tf1_wp150.log |
| ZZ-TF1-WP150 ep45 | {'train_cases': 0, 'epochs': 150, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'teacher_forcing': 1.0, 'counters': True, 'time_on_event': True, 'only_points': True, 'only_points_validation': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (22:55:44) | bpic17_psp_real_zaehler_zeit_tf1_wp150.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_zeit_tf1_wp150.log |
| ZZ-TF1-WP150 ep90 | {'train_cases': 0, 'epochs': 150, 'learning_rate': 0.0001, 'point_weight': 3, 'suffix_size': 60, 'time_from': 'gaps', 'teacher_forcing': 1.0, 'counters': True, 'time_on_event': True, 'only_points': True, 'only_points_validation': True, 'statics': (['case:LoanGoal', 'case:ApplicationType'], ['case:RequestedAmount'])} (22:55:44) | bpic17_psp_real_zaehler_zeit_tf1_wp150.sh | prescriptive/data/PSP_real_alle_s60_normal_zaehler_zeit_tf1_wp150.log |
| (c) Masken | – | – | – |
| ZZ ep9 Masken+Gumbel | – | – | – |
| (c) Masken+Gumbel | – | – | – |
| ZZ-TF1 ep12 Gumbel | – | – | – |
| T-Learner (ZZ-TF1) | – | – | – |
| ZZ-TF1 (RCT) Ep. 5 | – | – | – |
| ZZ-TF1 (RCT) | – | – | – |
| T-Learner (RCT) | – | – | – |
| ZZ-TF1 + Restzeit-Hilfsziel | – | – | – |
| T-Learner Suffix-Verlust (Zufallszahlen je Event) | – | – | – |
| T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event) | – | – | – |
| T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event) | – | – | – |
| T-Learner Suffix-Verlust + Kalibrierung Hälften (Zufallszahlen je Event) | – | – | – |
| T-Learner Kopf je Event (Steigung) | – | – | – |
| T-Learner Kopf je Event (Versatz) | – | – | – |
| T-Learner Kopf je Event (Steigung, β-NLL 1) | – | – | – |

### Für alle mit PSPModel.fit trainierten Modelle gleich (Code-Stand, prescriptive/estimators/psp_model.py `fit`)

- Optimierer AdamW (weight_decay 0), Lernrate wie oben; ReduceLROnPlateau (factor 0,1, patience 15, min_lr 1e-8) auf dem Validierungsverlust; Batchgröße, Regularisierung, Epochen wie oben; Mischen der Proben an.
- Verlust: Loss Attenuation (gelernte Varianz je Ausgabe), GradNorm (alpha 1,5, eigene Lernrate 1e-4, eine Aufgabe je Ausgabe).
- Teacher Forcing je Epoche von 1,0 linear abnehmend bis tf_untergrenze (Trainer des U-ED-LSTM); Masken für Auffüllung und EOS an.
- Netz: U-ED-LSTM mit MC-Dropout (Variational Dropout, gleiche Maske über die Zeitschritte), Gewichte je Epoche in last.pt.
- Ältere Modelle (10k, s40) können mit einem früheren Code-Stand trainiert sein; ihre Parameter oben stammen aus dem Modell selbst.

## Optionssignal offline

| label | aipw_vorzeichen_konzept | aipw_vorzeichen_nach_v | aipw_vorzeichen_nachlegen | aipw_vorzeichen_leads | se_gepaart | korrelation_optionen | dlz_gezogen | dlz_sd_gezogen | dlz_echt |
|---|---|---|---|---|---|---|---|---|---|
| Wahrheit/Referenz | – | – | – | – | – | – | – | – | 22.82 |
| (a) | 46 % | 58 % | 52 % | 44 % | 8.13 | 0.36 | 40.18 | 71.58 | 22.82 |
| (b) | 57 % | 54 % | 28 % | 60 % | 7.80 | 0.42 | 42.52 | 72.55 | 22.82 |
| (c) | 84 % | 96 % | 80 % | 80 % | 1.29 | 0.64 | 21.95 | 15.42 | 22.82 |
| (d) 1 ms | 62 % | 84 % | 49 % | 78 % | 1.63 | 0.31 | 22.32 | 13.95 | 22.82 |
| D | 68 % | 76 % | 50 % | 86 % | 1.64 | 0.30 | 22.46 | 13.96 | 22.82 |
| OE | 73 % | 72 % | 66 % | 47 % | 1.86 | 0.34 | 23.02 | 16.33 | 22.82 |
| (e) | 60 % | 76 % | 53 % | 43 % | 1.73 | 0.35 | 23.12 | 15.30 | 22.82 |
| (c)-TF1 | 66 % | 76 % | 50 % | 56 % | 1.97 | 0.15 | 21.87 | 15.15 | 22.82 |
| D-TF1 | 67 % | 72 % | 50 % | 54 % | 1.78 | 0.13 | 20.67 | 13.56 | 22.82 |
| OE-TF1 Ep. 26 | 70 % | 80 % | 50 % | 48 % | 1.54 | 0.12 | 21.04 | 11.89 | 22.82 |
| OE-TF1 Ep. 1 | 63 % | 55 % | 46 % | 50 % | 0.97 | 0.57 | 19.12 | 10.79 | 22.82 |
| s40 | 90 % | 88 % | 78 % | 65 % | 1.95 | 0.68 | 31.49 | 24.64 | 22.82 |
| 10k neue Punkte | 54 % | 58 % | 44 % | 40 % | 0.62 | 0.43 | 13.96 | 5.96 | 22.82 |
| ZZ ep3 | 69 % | 96 % | 55 % | 40 % | 1.53 | 0.39 | 16.40 | 14.08 | 22.82 |
| ZZ ep4 | 58 % | 96 % | 60 % | 42 % | 2.15 | 0.39 | 17.34 | 19.82 | 22.82 |
| ZZ ep5 | 57 % | 75 % | 58 % | 56 % | 2.69 | 0.39 | 18.52 | 24.67 | 22.82 |
| ZZ ep6 | 59 % | 73 % | 53 % | 57 % | 2.20 | 0.40 | 16.36 | 20.95 | 22.82 |
| ZZ ep7 | 55 % | 70 % | 56 % | 76 % | 2.61 | 0.45 | 18.88 | 25.88 | 22.82 |
| ZZ ep9 | 82 % | 96 % | 87 % | 77 % | 4.05 | 0.44 | 30.14 | 39.41 | 22.82 |
| ZZ ep11 | 74 % | 96 % | 82 % | 78 % | 4.33 | 0.41 | 29.34 | 41.21 | 22.82 |
| ZZ ep13 | 78 % | 92 % | 68 % | 67 % | 4.05 | 0.40 | 27.70 | 38.41 | 22.82 |
| ZZ final | 76 % | 86 % | 78 % | 67 % | 4.91 | 0.43 | 35.06 | 46.15 | 22.82 |
| ZZ-TF1 ep3 | 82 % | 96 % | 50 % | 29 % | 1.79 | 0.25 | 21.88 | 15.00 | 22.82 |
| ZZ-TF1 ep6 | 88 % | 96 % | 46 % | 32 % | 1.67 | 0.19 | 20.34 | 13.15 | 22.82 |
| ZZ-TF1 ep9 | 80 % | 97 % | 46 % | 41 % | 1.81 | 0.17 | 21.08 | 14.08 | 22.82 |
| ZZ-TF1 ep12 | 76 % | 97 % | 50 % | 60 % | 1.81 | 0.15 | 20.99 | 13.92 | 22.82 |
| ZZ-TF1 final | 74 % | 96 % | 49 % | 50 % | 1.85 | 0.16 | 21.48 | 14.26 | 22.82 |
| ZZU-TF1 ep3 | 85 % | 98 % | 42 % | 34 % | 1.72 | 0.25 | 21.11 | 14.30 | 22.82 |
| ZZU-TF1 ep6 | 78 % | 97 % | 46 % | 30 % | 1.69 | 0.19 | 20.13 | 13.32 | 22.82 |
| ZZU-TF1 ep9 | 77 % | 93 % | 49 % | 41 % | 1.72 | 0.15 | 20.09 | 13.45 | 22.82 |
| ZZU-TF1 ep12 | 78 % | 92 % | 50 % | 58 % | 1.84 | 0.14 | 20.60 | 14.10 | 22.82 |
| ZZU-TF1 final | 77 % | 92 % | 44 % | 54 % | 1.81 | 0.14 | 20.64 | 13.94 | 22.82 |
| ZZ-TF1-WP ep5 | 50 % | 58 % | 36 % | 62 % | 0.93 | 0.76 | 22.20 | 13.79 | 22.82 |
| ZZ-TF1-WP ep10 | 74 % | 80 % | 48 % | 43 % | 1.20 | 0.51 | 20.03 | 12.59 | 22.82 |
| ZZ-TF1-WP150 ep15 | 80 % | 93 % | 56 % | 34 % | 1.36 | 0.43 | 20.48 | 13.06 | 22.82 |
| ZZ-TF1-WP150 ep45 | 86 % | 95 % | 46 % | 32 % | 1.53 | 0.31 | 20.74 | 13.14 | 22.82 |
| ZZ-TF1-WP150 ep90 | 85 % | 95 % | 46 % | 30 % | 1.54 | 0.31 | 20.73 | 13.14 | 22.82 |
| (c) Masken | 40 % | 86 % | 62 % | 67 % | 2.78 | 0.36 | 28.38 | 24.63 | 22.82 |
| ZZ ep9 Masken+Gumbel | 75 % | 82 % | 70 % | 68 % | 3.88 | 0.31 | 29.82 | 33.92 | 22.82 |
| (c) Masken+Gumbel | 37 % | 85 % | 68 % | 66 % | 2.75 | 0.36 | 28.41 | 24.63 | 22.82 |
| ZZ-TF1 ep12 Gumbel | 74 % | 93 % | 45 % | 59 % | 1.81 | 0.15 | 21.11 | 13.97 | 22.82 |
| T-Learner (ZZ-TF1) | 70 % | 85 % | 56 % | 68 % | 1.76 | 0.12 | 20.60 | 13.33 | 22.82 |
| ZZ-TF1 (RCT) Ep. 5 | 68 % | 80 % | 56 % | 56 % | 1.88 | 0.21 | 21.08 | 15.31 | 22.82 |
| ZZ-TF1 (RCT) | 82 % | 85 % | 66 % | 78 % | 1.74 | 0.18 | 20.36 | 13.67 | 22.82 |
| T-Learner (RCT) | 78 % | 86 % | 66 % | 78 % | 1.91 | 0.17 | 21.67 | 14.98 | 22.82 |
| ZZ-TF1 + Restzeit-Hilfsziel | 80 % | 95 % | 48 % | 62 % | 1.74 | 0.13 | 20.59 | 13.38 | 22.82 |
| T-Learner Suffix-Verlust (Zufallszahlen je Event) | 66 % | 90 % | 75 % | 94 % | 1.10 | 0.72 | 23.11 | 14.78 | 22.82 |
| T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event) | 97 % | 98 % | 88 % | 95 % | 1.19 | 0.57 | 23.38 | 12.82 | 22.82 |
| T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event) | 98 % | 98 % | 28 % | 24 % | 1.43 | 0.67 | 23.56 | 17.92 | 22.82 |
| T-Learner Suffix-Verlust + Kalibrierung Hälften (Zufallszahlen je Event) | 85 % | 86 % | 80 % | 86 % | 1.48 | 0.63 | 23.27 | 17.73 | 22.82 |
| T-Learner Kopf je Event (Steigung) | 84 % | 92 % | 48 % | 80 % | 1.59 | 0.12 | 24.09 | 12.16 | 22.82 |
| T-Learner Kopf je Event (Versatz) | 77 % | 90 % | 48 % | 75 % | 1.80 | 0.12 | 20.43 | 13.56 | 22.82 |
| T-Learner Kopf je Event (Steigung, β-NLL 1) | 82 % | 88 % | 44 % | 72 % | 1.73 | 0.12 | 25.61 | 13.32 | 22.82 |

## Effektgrößen offline (gleiche 200 Testentscheidungen je Punkt, 100 Ziehungen)

optionsdiff = mittlerer Optionsunterschied des Modells (Aktion minus Referenz, Tage, Vorzeichen wie AIPW); effekt_anteil = optionsdiff / AIPW-Effekt (1 = volle Effektgröße, 0 = kein Signal, negativ = falsches Vorzeichen); diff_betrag_median = typischer Betrag des Unterschieds je Entscheidung. Erste Zeile: AIPW auf dem echten Log (Screening). Simulator-Wahrheit (RCT) im nächsten Abschnitt: konzept 2,79, nach V. 2,40, nachlegen 2,11 / 0,42, leads 1,32 Tage Vorteil der Regel-Option.

| label | optionsdiff_konzept | effekt_anteil_konzept | diff_betrag_median_konzept | optionsdiff_nach_v | effekt_anteil_nach_v | diff_betrag_median_nach_v | optionsdiff_nachlegen | effekt_anteil_nachlegen | diff_betrag_median_nachlegen | optionsdiff_leads | effekt_anteil_leads | diff_betrag_median_leads |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Wahrheit/Referenz | 5.37 | 1.00 | – | 2.94 | 1.00 | – | -1.46 | 1.00 | – | -2.22 | 1.00 | – |
| (a) | 0.08 | 0.02 | 5.30 | 2.11 | 0.72 | 5.91 | -0.25 | 0.17 | 5.51 | 1.36 | -0.61 | 4.55 |
| (b) | 0.93 | 0.17 | 5.45 | 1.32 | 0.45 | 5.56 | 4.54 | -3.10 | 5.51 | -1.71 | 0.77 | 6.45 |
| (c) | 1.46 | 0.27 | 1.50 | 2.09 | 0.71 | 1.99 | -1.21 | 0.83 | 1.33 | -1.23 | 0.56 | 1.32 |
| (d) 1 ms | 0.45 | 0.08 | 1.00 | 1.82 | 0.62 | 1.91 | 0.01 | -0.01 | 1.29 | -1.41 | 0.64 | 1.40 |
| D | 0.81 | 0.15 | 1.20 | 1.14 | 0.39 | 1.42 | -0.14 | 0.10 | 1.12 | -1.78 | 0.80 | 1.89 |
| OE | 1.02 | 0.19 | 1.39 | 1.11 | 0.38 | 1.42 | -0.67 | 0.46 | 1.18 | 0.21 | -0.09 | 1.24 |
| (e) | 0.48 | 0.09 | 1.20 | 1.12 | 0.38 | 1.37 | -0.17 | 0.11 | 1.40 | 0.30 | -0.13 | 1.24 |
| (c)-TF1 | 0.93 | 0.17 | 1.58 | 1.58 | 0.54 | 1.91 | 0.14 | -0.10 | 1.34 | -0.44 | 0.20 | 1.43 |
| D-TF1 | 0.72 | 0.13 | 1.21 | 1.07 | 0.36 | 1.22 | 0.03 | -0.02 | 1.34 | -0.02 | 0.01 | 1.12 |
| OE-TF1 Ep. 26 | 0.97 | 0.18 | 1.31 | 1.49 | 0.51 | 1.35 | 0.06 | -0.04 | 1.04 | 0.07 | -0.03 | 1.03 |
| OE-TF1 Ep. 1 | 0.35 | 0.07 | 0.78 | 0.13 | 0.05 | 0.83 | 0.22 | -0.15 | 0.66 | 0.01 | -0.01 | 0.49 |
| s40 | 2.39 | 0.45 | 2.54 | 2.63 | 0.89 | 2.70 | -1.49 | 1.02 | 1.47 | -0.80 | 0.36 | 1.24 |
| 10k neue Punkte | 0.09 | 0.02 | 0.38 | 0.21 | 0.07 | 0.47 | 0.11 | -0.07 | 0.47 | 0.17 | -0.08 | 0.44 |
| ZZ ep3 | 0.90 | 0.17 | 1.32 | 2.76 | 0.94 | 2.62 | -0.38 | 0.26 | 1.03 | 0.35 | -0.16 | 0.95 |
| ZZ ep4 | 0.38 | 0.07 | 1.40 | 3.59 | 1.22 | 3.71 | -0.48 | 0.33 | 1.40 | 0.35 | -0.16 | 1.32 |
| ZZ ep5 | 0.55 | 0.10 | 1.95 | 1.70 | 0.58 | 2.44 | -0.40 | 0.28 | 1.97 | -0.41 | 0.18 | 1.82 |
| ZZ ep6 | 0.50 | 0.09 | 1.55 | 1.21 | 0.41 | 1.78 | -0.27 | 0.18 | 2.01 | -0.64 | 0.29 | 1.43 |
| ZZ ep7 | 0.25 | 0.05 | 1.65 | 1.55 | 0.53 | 1.95 | -0.36 | 0.24 | 2.00 | -2.09 | 0.94 | 2.22 |
| ZZ ep9 | 3.70 | 0.69 | 4.09 | 7.12 | 2.43 | 6.87 | -5.18 | 3.54 | 5.05 | -3.56 | 1.61 | 3.83 |
| ZZ ep11 | 3.26 | 0.61 | 4.16 | 7.81 | 2.66 | 8.05 | -3.90 | 2.66 | 4.37 | -3.21 | 1.45 | 3.56 |
| ZZ ep13 | 3.45 | 0.64 | 4.10 | 5.52 | 1.88 | 5.27 | -2.15 | 1.47 | 3.18 | -1.96 | 0.88 | 3.55 |
| ZZ final | 4.25 | 0.79 | 4.77 | 5.81 | 1.98 | 6.14 | -3.66 | 2.50 | 4.66 | -2.47 | 1.11 | 3.51 |
| ZZ-TF1 ep3 | 1.61 | 0.30 | 1.70 | 3.44 | 1.17 | 3.46 | 0.16 | -0.11 | 1.22 | 0.79 | -0.36 | 1.27 |
| ZZ-TF1 ep6 | 1.76 | 0.33 | 1.61 | 3.45 | 1.18 | 3.32 | 0.15 | -0.10 | 1.15 | 0.64 | -0.29 | 1.21 |
| ZZ-TF1 ep9 | 1.66 | 0.31 | 1.93 | 3.61 | 1.23 | 3.68 | 0.23 | -0.16 | 1.32 | 0.27 | -0.12 | 1.14 |
| ZZ-TF1 ep12 | 1.45 | 0.27 | 1.52 | 3.51 | 1.19 | 3.53 | 0.18 | -0.12 | 1.13 | -0.31 | 0.14 | 1.31 |
| ZZ-TF1 final | 1.35 | 0.25 | 1.81 | 3.31 | 1.13 | 3.13 | 0.15 | -0.10 | 1.48 | -0.06 | 0.03 | 1.21 |
| ZZU-TF1 ep3 | 1.61 | 0.30 | 1.70 | 3.70 | 1.26 | 3.74 | 0.20 | -0.13 | 1.03 | 0.72 | -0.32 | 1.18 |
| ZZU-TF1 ep6 | 1.28 | 0.24 | 1.60 | 3.17 | 1.08 | 3.17 | 0.14 | -0.09 | 1.07 | 0.86 | -0.39 | 1.30 |
| ZZU-TF1 ep9 | 1.16 | 0.22 | 1.26 | 2.42 | 0.83 | 2.38 | 0.07 | -0.05 | 1.12 | 0.40 | -0.18 | 1.13 |
| ZZU-TF1 ep12 | 1.53 | 0.28 | 1.97 | 2.84 | 0.97 | 2.85 | 0.04 | -0.03 | 1.28 | -0.35 | 0.16 | 1.13 |
| ZZU-TF1 final | 1.35 | 0.25 | 1.64 | 2.66 | 0.91 | 2.37 | 0.23 | -0.16 | 1.18 | -0.21 | 0.09 | 1.28 |
| ZZ-TF1-WP ep5 | 0.01 | 0.00 | 0.58 | 0.20 | 0.07 | 0.67 | 0.36 | -0.25 | 0.90 | -0.27 | 0.12 | 0.53 |
| ZZ-TF1-WP ep10 | 0.79 | 0.15 | 1.10 | 1.12 | 0.38 | 1.13 | 0.01 | -0.00 | 0.85 | 0.14 | -0.06 | 0.53 |
| ZZ-TF1-WP150 ep15 | 1.32 | 0.25 | 1.47 | 2.32 | 0.79 | 2.31 | -0.06 | 0.04 | 1.03 | 0.31 | -0.14 | 0.65 |
| ZZ-TF1-WP150 ep45 | 1.73 | 0.32 | 1.77 | 3.02 | 1.03 | 3.10 | -0.01 | 0.01 | 1.04 | 0.77 | -0.35 | 1.14 |
| ZZ-TF1-WP150 ep90 | 1.71 | 0.32 | 1.72 | 3.01 | 1.02 | 3.03 | 0.02 | -0.02 | 1.05 | 0.78 | -0.35 | 1.16 |
| (c) Masken | -1.65 | -0.31 | 2.80 | 2.63 | 0.89 | 2.56 | -1.07 | 0.73 | 1.80 | -1.07 | 0.48 | 2.04 |
| ZZ ep9 Masken+Gumbel | 2.56 | 0.48 | 3.24 | 3.87 | 1.32 | 4.20 | -2.15 | 1.47 | 3.32 | -1.56 | 0.70 | 2.80 |
| (c) Masken+Gumbel | -2.03 | -0.38 | 3.02 | 2.93 | 1.00 | 2.85 | -1.05 | 0.72 | 2.18 | -1.01 | 0.46 | 2.02 |
| ZZ-TF1 ep12 Gumbel | 1.13 | 0.21 | 1.55 | 3.21 | 1.09 | 3.30 | 0.09 | -0.06 | 1.16 | -0.35 | 0.16 | 1.20 |
| T-Learner (ZZ-TF1) | 1.16 | 0.22 | 1.57 | 2.20 | 0.75 | 2.23 | -0.28 | 0.19 | 1.21 | -1.02 | 0.46 | 1.61 |
| ZZ-TF1 (RCT) Ep. 5 | 0.93 | 0.17 | 1.50 | 1.55 | 0.53 | 1.73 | -0.42 | 0.29 | 1.45 | -0.36 | 0.16 | 1.24 |
| ZZ-TF1 (RCT) | 1.36 | 0.25 | 1.52 | 1.88 | 0.64 | 1.95 | -0.78 | 0.54 | 1.32 | -1.34 | 0.61 | 1.70 |
| T-Learner (RCT) | 1.45 | 0.27 | 1.72 | 2.14 | 0.73 | 2.10 | -0.75 | 0.51 | 1.28 | -1.45 | 0.65 | 1.75 |
| ZZ-TF1 + Restzeit-Hilfsziel | 1.33 | 0.25 | 1.42 | 2.95 | 1.01 | 2.78 | 0.18 | -0.12 | 1.15 | -0.66 | 0.30 | 1.30 |
| T-Learner Suffix-Verlust (Zufallszahlen je Event) | 0.58 | 0.11 | 0.91 | 1.60 | 0.54 | 1.65 | -0.68 | 0.46 | 0.87 | -1.66 | 0.75 | 1.72 |
| T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event) | 2.35 | 0.44 | 2.34 | 3.12 | 1.06 | 3.14 | -1.91 | 1.31 | 2.00 | -1.55 | 0.70 | 1.66 |
| T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event) | 3.15 | 0.59 | 3.18 | 3.61 | 1.23 | 3.62 | 0.83 | -0.57 | 1.20 | 0.97 | -0.44 | 1.22 |
| T-Learner Suffix-Verlust + Kalibrierung Hälften (Zufallszahlen je Event) | 1.88 | 0.35 | 1.86 | 2.01 | 0.68 | 2.15 | -1.27 | 0.87 | 1.46 | -1.72 | 0.78 | 1.79 |
| T-Learner Kopf je Event (Steigung) | 1.72 | 0.32 | 1.81 | 2.89 | 0.99 | 2.99 | 0.03 | -0.02 | 1.11 | -1.52 | 0.69 | 1.71 |
| T-Learner Kopf je Event (Versatz) | 1.51 | 0.28 | 1.82 | 2.53 | 0.86 | 2.48 | 0.07 | -0.05 | 1.10 | -1.23 | 0.56 | 1.53 |
| T-Learner Kopf je Event (Steigung, β-NLL 1) | 1.77 | 0.33 | 1.88 | 2.39 | 0.81 | 2.48 | -0.03 | 0.02 | 1.11 | -1.03 | 0.47 | 1.62 |

AIPW-Details (echter Log): Effekt mit 95-%-KI, Policy-Gewinn einer Einzelfall-Policy (DR-Learner) gegenüber der besten festen Regel (negativ = Einzelfallentscheidung lohnt nicht), BLP-p (Heterogenität), E-Wert, Überlappung.

| label | aipw_effekt_konzept | aipw_ki_unten_konzept | aipw_ki_oben_konzept | aipw_policy_gewinn_konzept | aipw_blp_p_konzept | aipw_e_wert_konzept | aipw_effekt_nach_v | aipw_ki_unten_nach_v | aipw_ki_oben_nach_v | aipw_policy_gewinn_nach_v | aipw_blp_p_nach_v | aipw_e_wert_nach_v | aipw_effekt_nachlegen | aipw_ki_unten_nachlegen | aipw_ki_oben_nachlegen | aipw_policy_gewinn_nachlegen | aipw_blp_p_nachlegen | aipw_e_wert_nachlegen | aipw_effekt_leads | aipw_ki_unten_leads | aipw_ki_oben_leads | aipw_policy_gewinn_leads | aipw_blp_p_leads | aipw_e_wert_leads |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Wahrheit/Referenz | 5.37 | 4.81 | 5.92 | -0.60 | 0.09 | 2.25 | 2.94 | 2.45 | 3.43 | -0.66 | 0.39 | 1.75 | -1.46 | -2.09 | -0.83 | -0.51 | 0.05 | 1.45 | -2.22 | -2.78 | -1.65 | -0.59 | 0.50 | 1.60 |

## Entscheidungssicherheit offline (z = Optionsunterschied / gepaarter SE je Entscheidung)

anteil_z_gt2 = Anteil Entscheidungen mit |z| > 2 (sicher); vorzeichen_z_gt2 = davon Vorzeichen wie AIPW; anteil_aktion = Anteil Entscheidungen für die Aktion nach dem Mittel; noetige_ziehungen = Median der Ziehungen, die für |z| = 2 nötig wären.

| label | anteil_z_gt2_konzept | vorzeichen_z_gt2_konzept | anteil_aktion_konzept | noetige_ziehungen_konzept | anteil_z_gt2_nach_v | vorzeichen_z_gt2_nach_v | anteil_aktion_nach_v | noetige_ziehungen_nach_v | anteil_z_gt2_nachlegen | vorzeichen_z_gt2_nachlegen | anteil_aktion_nachlegen | noetige_ziehungen_nachlegen | anteil_z_gt2_leads | vorzeichen_z_gt2_leads | anteil_aktion_leads | noetige_ziehungen_leads |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Wahrheit/Referenz | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| (a) | 6 % | 75 % | 46 % | 944 | 6 % | 67 % | 58 % | 767 | 1 % | 100 % | 48 % | 770 | 3 % | 33 % | 56 % | 1168 |
| (b) | 6 % | 82 % | 57 % | 779 | 4 % | 75 % | 54 % | 724 | 11 % | 0 % | 72 % | 802 | 6 % | 73 % | 40 % | 612 |
| (c) | 17 % | 100 % | 84 % | 267 | 32 % | 100 % | 96 % | 159 | 15 % | 97 % | 20 % | 392 | 15 % | 97 % | 20 % | 354 |
| (d) 1 ms | 6 % | 92 % | 62 % | 841 | 20 % | 100 % | 84 % | 298 | 2 % | 60 % | 51 % | 805 | 10 % | 100 % | 22 % | 538 |
| D | 10 % | 100 % | 68 % | 595 | 10 % | 100 % | 76 % | 532 | 6 % | 58 % | 50 % | 898 | 16 % | 100 % | 14 % | 367 |
| OE | 10 % | 95 % | 73 % | 629 | 10 % | 100 % | 72 % | 671 | 6 % | 100 % | 34 % | 1052 | 6 % | 45 % | 53 % | 876 |
| (e) | 7 % | 64 % | 60 % | 662 | 8 % | 100 % | 76 % | 682 | 8 % | 62 % | 47 % | 736 | 6 % | 33 % | 57 % | 695 |
| (c)-TF1 | 11 % | 91 % | 66 % | 622 | 12 % | 100 % | 76 % | 444 | 7 % | 43 % | 50 % | 775 | 6 % | 69 % | 44 % | 747 |
| D-TF1 | 4 % | 100 % | 67 % | 843 | 10 % | 95 % | 72 % | 823 | 4 % | 43 % | 50 % | 777 | 4 % | 22 % | 46 % | 932 |
| OE-TF1 Ep. 26 | 8 % | 100 % | 70 % | 527 | 15 % | 100 % | 80 % | 551 | 8 % | 44 % | 50 % | 884 | 8 % | 53 % | 52 % | 844 |
| OE-TF1 Ep. 1 | 8 % | 67 % | 63 % | 638 | 5 % | 80 % | 55 % | 634 | 4 % | 25 % | 54 % | 1057 | 6 % | 42 % | 50 % | 814 |
| s40 | 15 % | 100 % | 90 % | 280 | 24 % | 100 % | 88 % | 226 | 12 % | 100 % | 22 % | 497 | 5 % | 100 % | 35 % | 852 |
| 10k neue Punkte | 2 % | 60 % | 54 % | 945 | 7 % | 79 % | 58 % | 661 | 4 % | 57 % | 56 % | 820 | 5 % | 10 % | 60 % | 690 |
| ZZ ep3 | 9 % | 100 % | 69 % | 508 | 36 % | 100 % | 96 % | 169 | 4 % | 78 % | 46 % | 946 | 3 % | 0 % | 60 % | 726 |
| ZZ ep4 | 6 % | 58 % | 58 % | 800 | 42 % | 100 % | 96 % | 146 | 4 % | 67 % | 40 % | 1056 | 4 % | 38 % | 57 % | 806 |
| ZZ ep5 | 8 % | 81 % | 57 % | 559 | 10 % | 100 % | 75 % | 488 | 6 % | 55 % | 42 % | 804 | 6 % | 54 % | 44 % | 829 |
| ZZ ep6 | 6 % | 77 % | 59 % | 627 | 7 % | 86 % | 73 % | 598 | 4 % | 56 % | 47 % | 544 | 6 % | 91 % | 43 % | 861 |
| ZZ ep7 | 4 % | 67 % | 55 % | 654 | 12 % | 100 % | 70 % | 652 | 5 % | 80 % | 44 % | 769 | 10 % | 100 % | 24 % | 626 |
| ZZ ep9 | 14 % | 100 % | 82 % | 371 | 46 % | 100 % | 96 % | 110 | 26 % | 100 % | 13 % | 274 | 17 % | 100 % | 23 % | 397 |
| ZZ ep11 | 14 % | 100 % | 74 % | 409 | 43 % | 100 % | 96 % | 116 | 16 % | 100 % | 18 % | 396 | 10 % | 100 % | 22 % | 575 |
| ZZ ep13 | 16 % | 97 % | 78 % | 321 | 24 % | 100 % | 92 % | 223 | 9 % | 83 % | 32 % | 556 | 5 % | 100 % | 33 % | 565 |
| ZZ final | 14 % | 100 % | 76 % | 395 | 21 % | 100 % | 86 % | 252 | 11 % | 91 % | 22 % | 422 | 8 % | 94 % | 33 % | 645 |
| ZZ-TF1 ep3 | 12 % | 100 % | 82 % | 431 | 44 % | 100 % | 96 % | 117 | 3 % | 50 % | 50 % | 877 | 6 % | 8 % | 71 % | 654 |
| ZZ-TF1 ep6 | 18 % | 100 % | 88 % | 412 | 50 % | 100 % | 96 % | 100 | 4 % | 25 % | 54 % | 835 | 7 % | 14 % | 68 % | 685 |
| ZZ-TF1 ep9 | 14 % | 100 % | 80 % | 326 | 47 % | 100 % | 97 % | 104 | 5 % | 30 % | 55 % | 733 | 7 % | 29 % | 59 % | 961 |
| ZZ-TF1 ep12 | 13 % | 100 % | 76 % | 540 | 48 % | 100 % | 97 % | 106 | 6 % | 36 % | 50 % | 932 | 7 % | 71 % | 40 % | 707 |
| ZZ-TF1 final | 12 % | 100 % | 74 % | 447 | 41 % | 100 % | 96 % | 143 | 7 % | 21 % | 51 % | 575 | 5 % | 70 % | 50 % | 878 |
| ZZU-TF1 ep3 | 13 % | 96 % | 85 % | 453 | 50 % | 100 % | 98 % | 99 | 2 % | 40 % | 58 % | 1093 | 10 % | 14 % | 66 % | 651 |
| ZZU-TF1 ep6 | 10 % | 100 % | 78 % | 446 | 42 % | 100 % | 97 % | 117 | 4 % | 50 % | 54 % | 1084 | 12 % | 0 % | 70 % | 582 |
| ZZU-TF1 ep9 | 8 % | 100 % | 77 % | 787 | 25 % | 100 % | 93 % | 254 | 6 % | 50 % | 51 % | 853 | 4 % | 0 % | 59 % | 839 |
| ZZU-TF1 ep12 | 14 % | 100 % | 78 % | 412 | 28 % | 100 % | 92 % | 173 | 5 % | 50 % | 50 % | 789 | 4 % | 33 % | 42 % | 1091 |
| ZZU-TF1 final | 10 % | 100 % | 77 % | 500 | 30 % | 100 % | 92 % | 232 | 5 % | 40 % | 56 % | 876 | 3 % | 67 % | 46 % | 786 |
| ZZ-TF1-WP ep5 | 6 % | 50 % | 50 % | 768 | 6 % | 83 % | 58 % | 919 | 6 % | 8 % | 64 % | 605 | 5 % | 100 % | 38 % | 811 |
| ZZ-TF1-WP ep10 | 10 % | 89 % | 74 % | 519 | 14 % | 100 % | 80 % | 602 | 2 % | 67 % | 52 % | 1076 | 4 % | 33 % | 57 % | 711 |
| ZZ-TF1-WP150 ep15 | 16 % | 100 % | 80 % | 363 | 36 % | 100 % | 93 % | 153 | 6 % | 42 % | 44 % | 963 | 4 % | 11 % | 66 % | 807 |
| ZZ-TF1-WP150 ep45 | 20 % | 100 % | 86 % | 330 | 48 % | 100 % | 95 % | 112 | 4 % | 57 % | 54 % | 927 | 8 % | 6 % | 68 % | 466 |
| ZZ-TF1-WP150 ep90 | 22 % | 100 % | 85 % | 343 | 46 % | 100 % | 95 % | 110 | 4 % | 57 % | 54 % | 976 | 7 % | 7 % | 70 % | 495 |
| (c) Masken | 12 % | 24 % | 40 % | 437 | 16 % | 100 % | 86 % | 349 | 6 % | 92 % | 38 % | 846 | 4 % | 89 % | 33 % | 590 |
| ZZ ep9 Masken+Gumbel | 12 % | 92 % | 75 % | 529 | 16 % | 100 % | 82 % | 350 | 6 % | 92 % | 30 % | 571 | 9 % | 78 % | 32 % | 652 |
| (c) Masken+Gumbel | 19 % | 21 % | 37 % | 364 | 20 % | 98 % | 85 % | 310 | 5 % | 70 % | 32 % | 604 | 7 % | 100 % | 34 % | 592 |
| ZZ-TF1 ep12 Gumbel | 8 % | 94 % | 74 % | 536 | 42 % | 100 % | 93 % | 127 | 4 % | 43 % | 55 % | 956 | 4 % | 88 % | 41 % | 828 |
| T-Learner (ZZ-TF1) | 9 % | 94 % | 70 % | 498 | 24 % | 100 % | 85 % | 260 | 3 % | 67 % | 44 % | 814 | 8 % | 100 % | 32 % | 472 |
| ZZ-TF1 (RCT) Ep. 5 | 8 % | 100 % | 68 % | 621 | 12 % | 100 % | 80 % | 474 | 8 % | 67 % | 44 % | 696 | 8 % | 62 % | 44 % | 836 |
| ZZ-TF1 (RCT) | 14 % | 100 % | 82 % | 508 | 18 % | 100 % | 85 % | 302 | 6 % | 92 % | 34 % | 666 | 12 % | 100 % | 22 % | 447 |
| T-Learner (RCT) | 12 % | 96 % | 78 % | 455 | 15 % | 100 % | 86 % | 362 | 8 % | 71 % | 34 % | 893 | 10 % | 95 % | 22 % | 475 |
| ZZ-TF1 + Restzeit-Hilfsziel | 13 % | 100 % | 80 % | 553 | 32 % | 100 % | 95 % | 162 | 6 % | 31 % | 52 % | 929 | 6 % | 92 % | 38 % | 670 |
| T-Learner Suffix-Verlust (Zufallszahlen je Event) | 13 % | 88 % | 66 % | 523 | 26 % | 98 % | 90 % | 201 | 9 % | 100 % | 25 % | 536 | 28 % | 100 % | 6 % | 170 |
| T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event) | 60 % | 100 % | 97 % | 76 | 79 % | 99 % | 98 % | 47 | 27 % | 96 % | 12 % | 260 | 28 % | 98 % | 5 % | 168 |
| T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event) | 58 % | 100 % | 98 % | 81 | 64 % | 99 % | 98 % | 62 | 11 % | 5 % | 72 % | 623 | 10 % | 5 % | 76 % | 607 |
| T-Learner Suffix-Verlust + Kalibrierung Hälften (Zufallszahlen je Event) | 26 % | 100 % | 85 % | 223 | 31 % | 98 % | 86 % | 204 | 12 % | 96 % | 20 % | 452 | 18 % | 97 % | 14 % | 301 |
| T-Learner Kopf je Event (Steigung) | 20 % | 95 % | 84 % | 327 | 42 % | 100 % | 92 % | 120 | 8 % | 40 % | 52 % | 795 | 16 % | 97 % | 20 % | 401 |
| T-Learner Kopf je Event (Versatz) | 13 % | 92 % | 77 % | 389 | 30 % | 100 % | 90 % | 218 | 4 % | 12 % | 52 % | 1023 | 11 % | 91 % | 25 % | 549 |
| T-Learner Kopf je Event (Steigung, β-NLL 1) | 18 % | 97 % | 82 % | 350 | 28 % | 100 % | 88 % | 220 | 6 % | 46 % | 56 % | 950 | 9 % | 89 % | 28 % | 522 |

## Gegen die Simulator-Wahrheit (Vorteil Regel-Option, Tage / Anteil richtig)

| label | sim_vorteil_konzept | sim_richtig_konzept | sim_vorteil_nach_v | sim_richtig_nach_v | sim_vorteil_nachlegen1 | sim_richtig_nachlegen1 | sim_vorteil_nachlegen2 | sim_richtig_nachlegen2 | sim_vorteil_leads | sim_richtig_leads | sim_mcse |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Wahrheit/Referenz | 2.79 | – | 2.40 | – | 2.11 | – | 0.42 | – | 1.32 | – | – |
| (a) | – | – | – | – | – | – | – | – | – | – | – |
| (b) | – | – | – | – | – | – | – | – | – | – | – |
| (c) | 1.61 | 82 % | 2.17 | 92 % | 1.18 | 78 % | -1.91 | 25 % | 1.13 | 85 % | 1.31 |
| (d) 1 ms | – | – | – | – | – | – | – | – | – | – | – |
| D | 1.10 | 78 % | 1.97 | 87 % | -0.29 | 45 % | -2.03 | 15 % | 1.60 | 80 % | 1.61 |
| OE | 1.54 | 78 % | 1.66 | 82 % | 0.48 | 58 % | -0.22 | 47 % | -0.14 | 47 % | 1.88 |
| (e) | – | – | – | – | – | – | – | – | – | – | – |
| (c)-TF1 | – | – | – | – | – | – | – | – | – | – | – |
| D-TF1 | – | – | – | – | – | – | – | – | – | – | – |
| OE-TF1 Ep. 26 | 1.17 | 77 % | 1.80 | 87 % | 0.19 | 58 % | -0.32 | 47 % | -0.44 | 43 % | 1.59 |
| OE-TF1 Ep. 1 | 0.57 | 68 % | 0.20 | 57 % | -0.15 | 42 % | -2.62 | 12 % | -0.10 | 45 % | 1.01 |
| s40 | – | – | – | – | – | – | – | – | – | – | – |
| 10k neue Punkte | – | – | – | – | – | – | – | – | – | – | – |
| ZZ ep3 | 1.30 | 78 % | 2.45 | 90 % | 0.30 | 58 % | -2.44 | 20 % | -0.37 | 43 % | 1.61 |
| ZZ ep4 | 0.61 | 60 % | 3.13 | 93 % | 0.69 | 58 % | -1.27 | 42 % | -0.38 | 45 % | 2.21 |
| ZZ ep5 | 0.79 | 70 % | 1.84 | 75 % | 1.01 | 67 % | -0.06 | 50 % | 0.44 | 57 % | 2.57 |
| ZZ ep6 | 0.06 | 50 % | 0.95 | 65 % | -0.22 | 43 % | -1.92 | 37 % | 1.48 | 78 % | 2.36 |
| ZZ ep7 | 0.05 | 52 % | 1.21 | 67 % | 0.15 | 58 % | -3.10 | 28 % | 1.22 | 67 % | 2.75 |
| ZZ ep9 | 3.61 | 80 % | 6.71 | 95 % | 3.76 | 75 % | -3.01 | 28 % | 3.52 | 80 % | 4.17 |
| ZZ ep11 | 3.54 | 80 % | 8.67 | 92 % | 3.81 | 72 % | -7.60 | 15 % | 2.75 | 77 % | 4.53 |
| ZZ ep13 | 3.79 | 78 % | 6.71 | 93 % | 1.43 | 70 % | -5.19 | 22 % | 2.17 | 70 % | 4.14 |
| ZZ final | 4.24 | 78 % | 6.07 | 88 % | 2.74 | 72 % | -3.18 | 22 % | 3.40 | 87 % | 4.95 |
| ZZ-TF1 ep3 | 1.77 | 82 % | 2.92 | 98 % | -0.36 | 48 % | -1.64 | 25 % | -1.02 | 27 % | 1.78 |
| ZZ-TF1 ep6 | 1.74 | 87 % | 3.43 | 97 % | -0.27 | 43 % | -0.10 | 50 % | -0.96 | 27 % | 1.66 |
| ZZ-TF1 ep9 | 2.02 | 87 % | 2.86 | 92 % | -0.30 | 47 % | 0.35 | 57 % | -0.27 | 53 % | 1.80 |
| ZZ-TF1 ep12 | 1.61 | 83 % | 3.04 | 93 % | -0.13 | 47 % | 1.44 | 77 % | 0.32 | 62 % | 1.79 |
| ZZ-TF1 final | 1.36 | 80 % | 2.29 | 77 % | 0.12 | 52 % | 0.54 | 68 % | -0.03 | 52 % | 1.84 |
| ZZU-TF1 ep3 | 1.53 | 85 % | 3.91 | 100 % | -0.33 | 43 % | -0.88 | 35 % | -0.97 | 32 % | 1.69 |
| ZZU-TF1 ep6 | 1.36 | 82 % | 2.71 | 95 % | -0.15 | 47 % | -1.66 | 27 % | -0.86 | 35 % | 1.65 |
| ZZU-TF1 ep9 | 1.24 | 75 % | 2.04 | 87 % | 0.01 | 48 % | -1.64 | 28 % | -0.12 | 50 % | 1.74 |
| ZZU-TF1 ep12 | 1.81 | 90 % | 2.12 | 87 % | -0.09 | 50 % | -0.81 | 38 % | 0.29 | 58 % | 1.81 |
| ZZU-TF1 final | 2.06 | 87 % | 2.06 | 83 % | -0.11 | 43 % | -1.19 | 32 % | 0.41 | 57 % | 1.78 |
| ZZ-TF1-WP ep5 | 0.19 | 58 % | 0.46 | 73 % | -0.49 | 30 % | -3.58 | 8 % | 0.10 | 57 % | 0.97 |
| ZZ-TF1-WP ep10 | 0.98 | 70 % | 1.18 | 83 % | -0.08 | 47 % | -3.95 | 10 % | -0.17 | 45 % | 1.26 |
| ZZ-TF1-WP150 ep15 | 1.12 | 70 % | 1.81 | 90 % | 0.01 | 53 % | -1.76 | 27 % | -0.24 | 40 % | 1.42 |
| ZZ-TF1-WP150 ep45 | 1.52 | 87 % | 2.63 | 92 % | 0.02 | 47 % | -0.31 | 55 % | -0.87 | 27 % | 1.56 |
| ZZ-TF1-WP150 ep90 | 1.45 | 82 % | 2.56 | 93 % | 0.04 | 48 % | -0.25 | 52 % | -0.82 | 32 % | 1.55 |
| (c) Masken | -4.30 | 13 % | 1.76 | 78 % | 2.10 | 72 % | -0.06 | 47 % | 0.50 | 57 % | 2.79 |
| ZZ ep9 Masken+Gumbel | 3.79 | 83 % | 3.86 | 85 % | 1.47 | 75 % | -1.61 | 45 % | 2.13 | 67 % | 4.16 |
| (c) Masken+Gumbel | -3.93 | 10 % | 2.81 | 88 % | 1.23 | 58 % | -0.63 | 37 % | 0.24 | 50 % | 2.79 |
| ZZ-TF1 ep12 Gumbel | 1.11 | 75 % | 3.60 | 95 % | -0.05 | 43 % | 1.70 | 73 % | 0.35 | 53 % | 1.80 |
| T-Learner (ZZ-TF1) | 1.47 | 75 % | 2.47 | 93 % | 0.28 | 68 % | 0.77 | 53 % | 0.80 | 72 % | 1.74 |
| ZZ-TF1 (RCT) Ep. 5 | 1.01 | 73 % | 1.69 | 78 % | 0.38 | 60 % | 1.84 | 77 % | 0.22 | 58 % | 1.96 |
| ZZ-TF1 (RCT) | 1.35 | 77 % | 2.13 | 90 % | 0.58 | 65 % | 2.07 | 67 % | 1.15 | 75 % | 1.81 |
| T-Learner (RCT) | 1.64 | 77 % | 1.63 | 78 % | 0.94 | 72 % | 1.85 | 68 % | 1.23 | 72 % | 1.95 |
| ZZ-TF1 + Restzeit-Hilfsziel | 1.61 | 85 % | 2.83 | 87 % | -0.10 | 48 % | 0.63 | 58 % | 0.62 | 70 % | 1.75 |
| T-Learner Suffix-Verlust (Zufallszahlen je Event) | 1.04 | 87 % | 1.79 | 90 % | 0.98 | 80 % | 0.93 | 72 % | 1.86 | 95 % | 1.19 |
| T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event) | 2.84 | 100 % | 3.42 | 100 % | 2.11 | 90 % | 4.15 | 95 % | 1.58 | 98 % | 1.34 |
| T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event) | 3.64 | 98 % | 3.48 | 97 % | -0.88 | 33 % | 5.40 | 88 % | -0.76 | 30 % | 1.74 |
| T-Learner Suffix-Verlust + Kalibrierung Hälften (Zufallszahlen je Event) | 2.05 | 92 % | 2.16 | 85 % | 0.85 | 73 % | 8.49 | 92 % | 2.08 | 95 % | 1.79 |
| T-Learner Kopf je Event (Steigung) | 2.47 | 95 % | 2.88 | 93 % | 0.37 | 58 % | 0.82 | 70 % | 1.27 | 78 % | 1.58 |
| T-Learner Kopf je Event (Versatz) | 1.82 | 78 % | 2.89 | 93 % | -0.14 | 53 % | 1.04 | 60 % | 0.90 | 77 % | 1.75 |
| T-Learner Kopf je Event (Steigung, β-NLL 1) | 2.22 | 95 % | 3.09 | 90 % | 0.57 | 65 % | 0.47 | 63 % | 1.07 | 72 % | 1.76 |

## Gedächtnis (Angebote, Leads)

| label | nachlegen_restzeit | nachlegen_versendet | nachlegen_nachfassen | nachlegen_ohne_ende | angebot_eingefuegt_rest | lead_gedaechtnis |
|---|---|---|---|---|---|---|
| Wahrheit/Referenz | 1.55 | 1.07 | 1.92 | 0 % | – | 2.83 |
| (a) | – | – | – | – | – | – |
| (b) | – | – | – | – | – | – |
| (c) | 1.06 | 0.34 | 0.07 | 47 % | 0.51 | 0.15 |
| (d) 1 ms | – | – | – | – | – | – |
| D | -0.28 | 0.77 | -0.25 | 39 % | 0.48 | 0.36 |
| OE | 0.56 | 0.29 | 0.32 | 38 % | 1.18 | -0.16 |
| (e) | – | – | – | – | – | – |
| (c)-TF1 | – | – | – | – | – | – |
| D-TF1 | – | – | – | – | – | – |
| OE-TF1 Ep. 26 | 0.46 | 0.76 | 0.33 | 2 % | -0.17 | 0.31 |
| OE-TF1 Ep. 1 | -0.29 | -0.59 | -0.38 | 25 % | -0.36 | -0.22 |
| s40 | – | – | – | – | – | – |
| 10k neue Punkte | – | – | – | – | – | – |
| ZZ ep3 | 0.11 | 0.15 | 0.24 | 25 % | -1.12 | 0.51 |
| ZZ ep4 | 0.72 | 0.47 | 0.18 | 36 % | -0.21 | 0.25 |
| ZZ ep5 | 0.78 | 0.48 | -0.16 | 36 % | -0.07 | 0.53 |
| ZZ ep6 | 0.15 | 0.39 | -0.29 | 39 % | -0.33 | 0.98 |
| ZZ ep7 | -0.10 | 0.68 | -1.18 | 48 % | -0.55 | 1.49 |
| ZZ ep9 | 4.55 | 0.82 | 2.37 | 47 % | -0.51 | 3.70 |
| ZZ ep11 | 1.84 | 0.35 | 1.27 | 47 % | 0.29 | 2.77 |
| ZZ ep13 | 1.31 | 0.46 | 1.25 | 46 % | 0.71 | 2.78 |
| ZZ final | 2.31 | 0.58 | 2.63 | 42 % | 1.12 | 2.19 |
| ZZ-TF1 ep3 | 0.09 | 0.26 | 0.37 | 2 % | -0.55 | 0.44 |
| ZZ-TF1 ep6 | 0.30 | 0.25 | 0.29 | 1 % | -0.17 | 0.65 |
| ZZ-TF1 ep9 | 0.39 | 0.55 | 0.50 | 1 % | -0.13 | 0.53 |
| ZZ-TF1 ep12 | 0.42 | 0.68 | 0.53 | 1 % | -0.06 | 0.45 |
| ZZ-TF1 final | 0.34 | 0.55 | 0.49 | 1 % | -0.01 | 0.21 |
| ZZU-TF1 ep3 | -0.14 | 0.19 | 0.26 | 2 % | -0.20 | 0.51 |
| ZZU-TF1 ep6 | 0.59 | 0.28 | 0.63 | 2 % | -0.17 | 0.59 |
| ZZU-TF1 ep9 | 0.35 | 0.32 | 0.57 | 2 % | -0.00 | 0.37 |
| ZZU-TF1 ep12 | 0.69 | 0.34 | 0.60 | 1 % | 0.24 | 0.39 |
| ZZU-TF1 final | 0.29 | 0.45 | 0.28 | 2 % | -0.14 | 0.39 |
| ZZ-TF1-WP ep5 | -0.31 | -0.78 | -1.50 | 35 % | -0.25 | -0.23 |
| ZZ-TF1-WP ep10 | -0.04 | -0.22 | -0.39 | 28 % | -0.92 | -0.24 |
| ZZ-TF1-WP150 ep15 | 0.33 | 0.01 | 0.21 | 16 % | -0.27 | 0.15 |
| ZZ-TF1-WP150 ep45 | -0.07 | 0.17 | 0.39 | 5 % | -0.22 | 0.17 |
| ZZ-TF1-WP150 ep90 | -0.09 | 0.17 | 0.33 | 5 % | -0.23 | 0.22 |
| (c) Masken | 2.24 | 1.15 | 0.95 | 0 % | 0.32 | 0.41 |
| ZZ ep9 Masken+Gumbel | 1.97 | 2.47 | 1.69 | 0 % | -0.48 | 1.62 |
| (c) Masken+Gumbel | 1.08 | 1.10 | 0.75 | 0 % | 0.23 | -0.57 |
| ZZ-TF1 ep12 Gumbel | 0.13 | 0.66 | 0.07 | 1 % | -0.10 | 0.45 |
| T-Learner (ZZ-TF1) | -0.07 | 0.94 | -0.28 | 1 % | -0.00 | 0.56 |
| ZZ-TF1 (RCT) Ep. 5 | 0.95 | 0.77 | 0.43 | 1 % | 0.63 | 0.23 |
| ZZ-TF1 (RCT) | 1.07 | 0.87 | 0.49 | 1 % | 0.86 | 0.85 |
| T-Learner (RCT) | 1.32 | 0.93 | 0.81 | 1 % | 0.76 | 0.84 |
| ZZ-TF1 + Restzeit-Hilfsziel | 0.38 | 0.85 | 0.30 | 1 % | 0.03 | 0.47 |
| T-Learner Suffix-Verlust (Zufallszahlen je Event) | 1.11 | 0.99 | 0.41 | 1 % | -0.07 | 0.06 |
| T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event) | 2.44 | 0.57 | -1.74 | 2 % | 0.18 | 0.07 |
| T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event) | -0.99 | 1.05 | -0.01 | 2 % | 0.16 | 0.22 |
| T-Learner Suffix-Verlust + Kalibrierung Hälften (Zufallszahlen je Event) | 0.58 | 0.94 | 0.90 | 2 % | 0.51 | 0.07 |
| T-Learner Kopf je Event (Steigung) | -0.38 | 0.92 | -0.26 | 1 % | 0.17 | 0.33 |
| T-Learner Kopf je Event (Versatz) | -0.44 | 0.95 | -0.44 | 1 % | 0.04 | 0.52 |
| T-Learner Kopf je Event (Steigung, β-NLL 1) | -0.30 | 0.91 | -0.28 | 1 % | 0.53 | 0.54 |

## Suffixgüte (1.000 Testprefixe, 100 Ziehungen)

suffix_sK = Anteil Prefixe, bei denen das häufigste gezogene Token an Position K stimmt; DLS = mittlere normierte Damerau-Levenshtein-Ähnlichkeit der Einzelziehungen; Restzeit ab dem letzten Prefix-Event: MAE, Bias (gezogen − echt), CRPS, Anteil echter Restzeiten im 80-%-Intervall.

| label | suffix_s1 | suffix_s2 | suffix_s3 | suffix_s5 | suffix_s10 | suffix_s20 | suffix_dls | suffix_laenge_mae |
|---|---|---|---|---|---|---|---|---|
| Wahrheit/Referenz | – | – | – | – | – | – | – | – |
| (a) | 86 % | 74 % | 62 % | 42 % | 13 % | 1 % | 0.28 | 8.99 |
| (b) | – | – | – | – | – | – | – | – |
| (c) | 83 % | 71 % | 61 % | 39 % | 9 % | 1 % | 0.25 | 10.57 |
| (d) 1 ms | 85 % | 75 % | 65 % | 49 % | 21 % | 4 % | 0.28 | 12.12 |
| D | 85 % | 75 % | 65 % | 49 % | 21 % | 4 % | 0.28 | 12.12 |
| OE | 84 % | 72 % | 64 % | 50 % | 21 % | 4 % | 0.27 | 12.12 |
| (e) | 84 % | 73 % | 64 % | 49 % | 20 % | 5 % | 0.27 | 12.44 |
| (c)-TF1 | 86 % | 76 % | 65 % | 50 % | 25 % | 4 % | 0.40 | 10.65 |
| D-TF1 | 86 % | 74 % | 63 % | 52 % | 26 % | 2 % | 0.39 | 10.37 |
| OE-TF1 Ep. 26 | 85 % | 74 % | 64 % | 49 % | 25 % | 3 % | 0.40 | 10.37 |
| OE-TF1 Ep. 1 | 64 % | 51 % | 41 % | 27 % | 10 % | 0 % | 0.23 | 10.69 |
| s40 | 84 % | 71 % | 58 % | 41 % | 9 % | 2 % | 0.23 | 13.12 |
| 10k neue Punkte | – | – | – | – | – | – | – | – |
| ZZ ep3 | 82 % | 70 % | 59 % | 46 % | 17 % | 0 % | 0.34 | 10.08 |
| ZZ ep4 | – | – | – | – | – | – | – | – |
| ZZ ep5 | – | – | – | – | – | – | – | – |
| ZZ ep6 | – | – | – | – | – | – | – | – |
| ZZ ep7 | 85 % | 72 % | 59 % | 42 % | 13 % | 2 % | 0.31 | 9.61 |
| ZZ ep9 | 85 % | 73 % | 61 % | 41 % | 10 % | 2 % | 0.27 | 11.30 |
| ZZ ep11 | 85 % | 72 % | 60 % | 45 % | 11 % | 3 % | 0.27 | 10.93 |
| ZZ ep13 | 86 % | 73 % | 60 % | 44 % | 14 % | 1 % | 0.27 | 11.47 |
| ZZ final | 85 % | 74 % | 63 % | 45 % | 13 % | 4 % | 0.27 | 12.95 |
| ZZ-TF1 ep3 | 83 % | 72 % | 59 % | 46 % | 24 % | 2 % | 0.36 | 11.64 |
| ZZ-TF1 ep6 | 85 % | 74 % | 63 % | 49 % | 26 % | 3 % | 0.40 | 11.04 |
| ZZ-TF1 ep9 | 86 % | 75 % | 64 % | 50 % | 28 % | 3 % | 0.40 | 11.24 |
| ZZ-TF1 ep12 | 86 % | 76 % | 65 % | 52 % | 27 % | 3 % | 0.41 | 10.85 |
| ZZ-TF1 final | 88 % | 77 % | 65 % | 51 % | 29 % | 2 % | 0.41 | 11.39 |
| ZZU-TF1 ep3 | 82 % | 70 % | 59 % | 47 % | 24 % | 3 % | 0.36 | 11.43 |
| ZZU-TF1 ep6 | 84 % | 73 % | 63 % | 48 % | 25 % | 2 % | 0.39 | 10.79 |
| ZZU-TF1 ep9 | 85 % | 75 % | 63 % | 49 % | 26 % | 2 % | 0.40 | 10.88 |
| ZZU-TF1 ep12 | 86 % | 75 % | 65 % | 50 % | 26 % | 3 % | 0.41 | 10.70 |
| ZZU-TF1 final | 86 % | 76 % | 65 % | 50 % | 26 % | 4 % | 0.41 | 10.64 |
| ZZ-TF1-WP ep5 | 38 % | 25 % | 17 % | 9 % | 6 % | 0 % | 0.14 | 16.32 |
| ZZ-TF1-WP ep10 | 40 % | 31 % | 24 % | 21 % | 10 % | 1 % | 0.21 | 12.54 |
| ZZ-TF1-WP150 ep15 | 40 % | 35 % | 29 % | 24 % | 12 % | 1 % | 0.24 | 12.81 |
| ZZ-TF1-WP150 ep45 | 40 % | 36 % | 29 % | 23 % | 13 % | 1 % | 0.26 | 12.46 |
| ZZ-TF1-WP150 ep90 | 41 % | 36 % | 29 % | 24 % | 12 % | 2 % | 0.26 | 12.37 |
| (c) Masken | – | – | – | – | – | – | – | – |
| ZZ ep9 Masken+Gumbel | – | – | – | – | – | – | – | – |
| (c) Masken+Gumbel | – | – | – | – | – | – | – | – |
| ZZ-TF1 ep12 Gumbel | – | – | – | – | – | – | – | – |
| T-Learner (ZZ-TF1) | – | – | – | – | – | – | – | – |
| ZZ-TF1 (RCT) Ep. 5 | – | – | – | – | – | – | – | – |
| ZZ-TF1 (RCT) | – | – | – | – | – | – | – | – |
| T-Learner (RCT) | – | – | – | – | – | – | – | – |
| ZZ-TF1 + Restzeit-Hilfsziel | – | – | – | – | – | – | – | – |
| T-Learner Suffix-Verlust (Zufallszahlen je Event) | – | – | – | – | – | – | – | – |
| T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event) | – | – | – | – | – | – | – | – |
| T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event) | – | – | – | – | – | – | – | – |
| T-Learner Suffix-Verlust + Kalibrierung Hälften (Zufallszahlen je Event) | – | – | – | – | – | – | – | – |
| T-Learner Kopf je Event (Steigung) | – | – | – | – | – | – | – | – |
| T-Learner Kopf je Event (Versatz) | – | – | – | – | – | – | – | – |
| T-Learner Kopf je Event (Steigung, β-NLL 1) | – | – | – | – | – | – | – | – |

| label | suffix_erster_abstand_mae_h | restzeit_mae | restzeit_bias | restzeit_crps | abdeckung80 | restzeit_gezogen | restzeit_echt | suffix_quelle |
|---|---|---|---|---|---|---|---|---|
| Wahrheit/Referenz | – | – | – | – | – | – | – | – |
| (a) | 17.44 | 15.57 | 13.64 | 9.21 | 95 % | 27.97 | 14.33 | psp_gesamtuebersicht_20261114/suffix_vollmodelle_1000_testprefixe.csv |
| (b) | – | – | – | – | – | – | – | – |
| (c) | 13.13 | 8.13 | -0.05 | 5.40 | 88 % | 14.29 | 14.33 | modellvergleich_20261116/suffixguete/suffixguete.csv |
| (d) 1 ms | 11.71 | 7.81 | -0.16 | 5.23 | 75 % | 14.17 | 14.33 | psp_gesamtuebersicht_20261114/suffix_vollmodelle_1000_testprefixe.csv |
| D | 11.71 | 7.81 | -0.16 | 5.23 | 75 % | 14.17 | 14.33 | psp_gesamtuebersicht_20261114/suffix_vollmodelle_1000_testprefixe.csv (wie (d) 1 ms: gleiches Netz, option_as wirkt nur bei der Optionsbewertung) |
| OE | 11.89 | 7.98 | 0.36 | 5.31 | 77 % | 14.70 | 14.33 | psp_gesamtuebersicht_20261114/suffix_vollmodelle_1000_testprefixe.csv |
| (e) | 11.55 | 7.93 | 0.20 | 5.24 | 78 % | 14.54 | 14.33 | modellvergleich_20261116/suffixguete/suffixguete.csv |
| (c)-TF1 | 12.35 | 8.62 | 0.96 | 5.49 | 84 % | 15.30 | 14.33 | psp_gesamtuebersicht_20261114/suffix_vollmodelle_1000_testprefixe.csv |
| D-TF1 | 11.51 | 7.93 | -0.95 | 5.33 | 75 % | 13.39 | 14.33 | modellvergleich_20261116/suffixguete/suffixguete.csv |
| OE-TF1 Ep. 26 | 11.50 | 7.96 | -0.64 | 5.33 | 67 % | 13.69 | 14.33 | modellvergleich_20261116/suffixguete_oe_tf1/suffixguete.csv |
| OE-TF1 Ep. 1 | 13.58 | 8.39 | -1.49 | 5.63 | 72 % | 12.85 | 14.33 | modellvergleich_20261116/suffixguete_oe_tf1/suffixguete.csv |
| s40 | 13.83 | 9.69 | 6.41 | 6.08 | 88 % | 20.74 | 14.33 | psp_gesamtuebersicht_20261114/suffix_vollmodelle_1000_testprefixe.csv |
| 10k neue Punkte | – | – | – | – | – | – | – | – |
| ZZ ep3 | 12.96 | 8.23 | -3.67 | 6.14 | 74 % | 10.67 | 14.33 | zaehler_zeit_20261116/suffixguete_ep3/suffixguete.csv |
| ZZ ep4 | – | – | – | – | – | – | – | – |
| ZZ ep5 | – | – | – | – | – | – | – | – |
| ZZ ep6 | – | – | – | – | – | – | – | – |
| ZZ ep7 | 13.40 | 7.87 | -2.03 | 6.34 | 84 % | 12.31 | 14.33 | zaehler_zeit_20261116/suffixguete_ep7/suffixguete.csv |
| ZZ ep9 | 14.45 | 8.92 | 3.61 | 6.05 | 91 % | 17.95 | 14.33 | zaehler_zeit_20261116/suffixguete_ep9/suffixguete.csv |
| ZZ ep11 | 15.42 | 9.10 | 4.24 | 6.05 | 92 % | 18.58 | 14.33 | zaehler_zeit_20261116/suffixguete_ep11/suffixguete.csv |
| ZZ ep13 | 15.31 | 8.75 | 3.65 | 5.84 | 91 % | 17.99 | 14.33 | zaehler_zeit_20261116/suffixguete_ep13/suffixguete.csv |
| ZZ final | 15.68 | 10.59 | 7.89 | 6.04 | 91 % | 22.22 | 14.33 | zaehler_zeit_20261116/suffixguete_final/suffixguete.csv |
| ZZ-TF1 ep3 | 12.87 | 8.56 | 0.56 | 5.53 | 84 % | 14.89 | 14.33 | zaehler_zeit_tf1_20261117/suffixguete_ep3/suffixguete.csv |
| ZZ-TF1 ep6 | 11.66 | 8.25 | -0.74 | 5.54 | 81 % | 13.60 | 14.33 | zaehler_zeit_tf1_20261117/suffixguete_ep6/suffixguete.csv |
| ZZ-TF1 ep9 | 11.88 | 8.48 | -0.00 | 5.55 | 82 % | 14.33 | 14.33 | zaehler_zeit_tf1_20261117/suffixguete_ep9/suffixguete.csv |
| ZZ-TF1 ep12 | 12.14 | 8.28 | -0.20 | 5.49 | 82 % | 14.13 | 14.33 | zaehler_zeit_tf1_20261117/suffixguete_ep12/suffixguete.csv |
| ZZ-TF1 final | 11.97 | 8.50 | 0.35 | 5.53 | 80 % | 14.69 | 14.33 | zaehler_zeit_tf1_20261117/suffixguete_final/suffixguete.csv |
| ZZU-TF1 ep3 | 12.48 | 8.54 | 0.03 | 5.60 | 83 % | 14.37 | 14.33 | zaehler_uhren_tf1_20261117/suffixguete_ep3/suffixguete.csv |
| ZZU-TF1 ep6 | 11.87 | 8.29 | -0.78 | 5.55 | 80 % | 13.55 | 14.33 | zaehler_uhren_tf1_20261117/suffixguete_ep6/suffixguete.csv |
| ZZU-TF1 ep9 | 11.61 | 8.23 | -0.78 | 5.53 | 82 % | 13.56 | 14.33 | zaehler_uhren_tf1_20261117/suffixguete_ep9/suffixguete.csv |
| ZZU-TF1 ep12 | 11.87 | 8.44 | -0.39 | 5.57 | 81 % | 13.94 | 14.33 | zaehler_uhren_tf1_20261117/suffixguete_ep12/suffixguete.csv |
| ZZU-TF1 final | 11.66 | 8.43 | -0.37 | 5.58 | 83 % | 13.97 | 14.33 | zaehler_uhren_tf1_20261117/suffixguete_final/suffixguete.csv |
| ZZ-TF1-WP ep5 | 17.24 | 10.07 | 2.53 | 6.26 | 84 % | 16.87 | 14.33 | nur_wahlpunkte_20261117/suffixguete_ep5/suffixguete.csv |
| ZZ-TF1-WP ep10 | 19.94 | 9.27 | 0.38 | 5.94 | 84 % | 14.71 | 14.33 | nur_wahlpunkte_20261117/suffixguete_ep10/suffixguete.csv |
| ZZ-TF1-WP150 ep15 | 16.90 | 9.27 | 0.77 | 5.91 | 84 % | 15.11 | 14.33 | nur_wahlpunkte150_20261117/suffixguete_ep15/suffixguete.csv |
| ZZ-TF1-WP150 ep45 | 16.51 | 9.24 | 0.83 | 5.94 | 83 % | 15.17 | 14.33 | nur_wahlpunkte150_20261117/suffixguete_ep45/suffixguete.csv |
| ZZ-TF1-WP150 ep90 | 16.73 | 9.33 | 0.78 | 5.97 | 81 % | 15.11 | 14.33 | nur_wahlpunkte150_20261117/suffixguete_ep90/suffixguete.csv |
| (c) Masken | – | – | – | – | – | – | – | – |
| ZZ ep9 Masken+Gumbel | – | – | – | – | – | – | – | – |
| (c) Masken+Gumbel | – | – | – | – | – | – | – | – |
| ZZ-TF1 ep12 Gumbel | – | – | – | – | – | – | – | – |
| T-Learner (ZZ-TF1) | – | – | – | – | – | – | – | – |
| ZZ-TF1 (RCT) Ep. 5 | – | – | – | – | – | – | – | – |
| ZZ-TF1 (RCT) | – | – | – | – | – | – | – | – |
| T-Learner (RCT) | – | – | – | – | – | – | – | – |
| ZZ-TF1 + Restzeit-Hilfsziel | – | – | – | – | – | – | – | – |
| T-Learner Suffix-Verlust (Zufallszahlen je Event) | – | – | – | – | – | – | – | – |
| T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event) | – | – | – | – | – | – | – | – |
| T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event) | – | – | – | – | – | – | – | – |
| T-Learner Suffix-Verlust + Kalibrierung Hälften (Zufallszahlen je Event) | – | – | – | – | – | – | – | – |
| T-Learner Kopf je Event (Steigung) | – | – | – | – | – | – | – | – |
| T-Learner Kopf je Event (Versatz) | – | – | – | – | – | – | – | – |
| T-Learner Kopf je Event (Steigung, β-NLL 1) | – | – | – | – | – | – | – | – |

## Gültigkeit der gezogenen Abläufe (uebergaenge_pruefen.py; Verläufe ab dem 1. Besuch Nachlegen)

Übergangsmodelle aus 80 % der echten Fälle: zwischen Events (Token → nächstes Token, wie die Maske des Simulators) und innerhalb einer Aktivität (letzter Lifecycle derselben Aktivität → nächster, z. B. suspend → resume; A_Pending nur einmal). Die übrigen 20 % echten Fälle zeigen die Fehlalarm-Rate (ungesehen, aber echt).

| Quelle | Suffixe | Events je Suffix | mit Verstoß zwischen Events | mit Verstoß innerhalb Aktivität | mit Verstoß Optionsprüfung | Verstöße zwischen je 100 Events | Verstöße innerhalb je 100 Events |
|---|---|---|---|---|---|---|---|
| echter Log (20 % Prüffälle, Fehlalarm-Rate) | 6302 | 27.06 | 0 % | 0 % | 2 % | 0.00 | 0.00 |
| (c) | 12000 | 29.52 | 99 % | 94 % | 79 % | 47.68 | 26.15 |
| D-Netz | 12000 | 33.35 | 98 % | 94 % | 77 % | 45.80 | 24.36 |
| OE | 12000 | 32.61 | 98 % | 94 % | 74 % | 39.88 | 22.98 |
| Simulator (Bank danach) | 7647 | 33.46 | 0 % | 0 % | 4 % | 0.00 | 0.02 |
| (c) Masken | 12000 | 33.02 | 6 % | 3 % | 34 % | 0.20 | 0.09 |
| (c) Masken+Gumbel | 12000 | 32.82 | 7 % | 3 % | 33 % | 0.21 | 0.09 |
| OE-TF1 Ep. 1 | 12000 | 29.10 | 97 % | 89 % | 51 % | 24.97 | 13.93 |
| OE-TF1 Ep. 26 | 12000 | 28.75 | 12 % | 12 % | 5 % | 0.62 | 0.59 |
| T-Learner Kopf je Event (Steigung) | 12000 | 29.78 | 6 % | 5 % | 3 % | 0.31 | 0.25 |
| T-Learner Kopf je Event (Steigung, β-NLL 1) | 12000 | 29.74 | 6 % | 5 % | 3 % | 0.29 | 0.25 |
| T-Learner Kopf je Event (Versatz) | 12000 | 31.80 | 16 % | 9 % | 4 % | 0.63 | 0.41 |
| T-Learner Suffix-Verlust (Zufallszahlen je Event) | 12000 | 32.15 | 15 % | 9 % | 4 % | 0.61 | 0.40 |
| T-Learner Suffix-Verlust + Kalibrierung Hälften (Zufallszahlen je Event) | 12000 | 31.74 | 18 % | 11 % | 4 % | 0.73 | 0.49 |
| T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event) | 12000 | 28.46 | 18 % | 11 % | 5 % | 0.85 | 0.48 |
| T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event) | 12000 | 32.42 | 19 % | 12 % | 5 % | 0.81 | 0.62 |
| T-Learner (ZZ-TF1) | 12000 | 31.58 | 15 % | 9 % | 4 % | 0.63 | 0.39 |
| T-Learner (RCT) | 12000 | 34.56 | 8 % | 6 % | 5 % | 0.27 | 0.21 |
| ZZ-TF1-WP150 ep15 | 12000 | 33.50 | 82 % | 74 % | 30 % | 8.34 | 5.93 |
| ZZ-TF1-WP150 ep45 | 12000 | 31.20 | 46 % | 45 % | 11 % | 3.09 | 2.51 |
| ZZ-TF1-WP150 ep90 | 12000 | 31.16 | 46 % | 45 % | 11 % | 3.05 | 2.47 |
| ZZ-TF1-WP ep10 | 12000 | 34.03 | 97 % | 91 % | 55 % | 19.78 | 12.94 |
| ZZ-TF1-WP ep5 | 12000 | 41.30 | 99 % | 96 % | 82 % | 49.24 | 24.96 |
| ZZ ep9 Masken+Gumbel | 12000 | 36.70 | 7 % | 0 % | 42 % | 0.19 | 0.01 |
| ZZ ep11 | 12000 | 31.82 | 99 % | 95 % | 83 % | 44.46 | 24.63 |
| ZZ ep13 | 12000 | 32.78 | 99 % | 95 % | 83 % | 45.03 | 24.70 |
| ZZ ep3 | 12000 | 28.04 | 74 % | 67 % | 27 % | 7.01 | 5.31 |
| ZZ ep4 | 12000 | 26.33 | 85 % | 79 % | 39 % | 10.47 | 7.87 |
| ZZ ep5 | 12000 | 27.72 | 90 % | 84 % | 48 % | 12.56 | 9.43 |
| ZZ ep6 | 12000 | 30.16 | 94 % | 88 % | 58 % | 16.95 | 11.78 |
| ZZ ep7 | 12000 | 28.31 | 96 % | 91 % | 67 % | 23.11 | 15.04 |
| ZZ ep9 | 12000 | 32.49 | 98 % | 95 % | 83 % | 41.82 | 24.09 |
| ZZ final | 12000 | 36.02 | 99 % | 96 % | 86 % | 46.89 | 26.09 |
| ZZ-TF1 + Restzeit-Hilfsziel | 12000 | 30.20 | 11 % | 7 % | 3 % | 0.46 | 0.31 |
| ZZ-TF1 (RCT) | 12000 | 33.33 | 10 % | 7 % | 4 % | 0.36 | 0.25 |
| ZZ-TF1 (RCT) Ep. 5 | 12000 | 34.06 | 10 % | 8 % | 5 % | 0.38 | 0.30 |
| ZZ-TF1 ep12 Gumbel | 12000 | 30.81 | 16 % | 9 % | 4 % | 0.66 | 0.40 |
| ZZ-TF1 ep12 | 12000 | 30.34 | 16 % | 9 % | 4 % | 0.64 | 0.42 |
| ZZ-TF1 ep3 | 12000 | 31.16 | 25 % | 18 % | 6 % | 1.18 | 0.83 |
| ZZ-TF1 ep6 | 12000 | 30.61 | 17 % | 10 % | 4 % | 0.70 | 0.44 |
| ZZ-TF1 ep9 | 12000 | 30.95 | 16 % | 9 % | 4 % | 0.63 | 0.41 |
| ZZ-TF1 final | 12000 | 31.54 | 17 % | 11 % | 5 % | 0.68 | 0.47 |
| ZZU-TF1 ep12 | 12000 | 29.70 | 16 % | 13 % | 4 % | 0.77 | 0.60 |
| ZZU-TF1 ep3 | 12000 | 30.72 | 24 % | 18 % | 5 % | 1.17 | 0.87 |
| ZZU-TF1 ep6 | 12000 | 29.60 | 17 % | 12 % | 4 % | 0.73 | 0.54 |
| ZZU-TF1 ep9 | 12000 | 30.02 | 15 % | 11 % | 4 % | 0.66 | 0.52 |
| ZZU-TF1 final | 12000 | 29.55 | 14 % | 11 % | 4 % | 0.63 | 0.49 |

## Diagnose an echten Entscheidungen (psp_diagnose, 100 je Punkt, Mittel über die vier Punkte)

diag_mae_faktisch / diag_korrelation_faktisch: gezogene Durchlaufzeit der tatsächlich gewählten Option gegen die echte; diag_diff_nach_5_schritten: Optionsunterschied nach 5 Schritten; diag_suffixlaenge: mittlere gezogene Suffixlänge.

| label | diag_mae_faktisch | diag_korrelation_faktisch | diag_diff_nach_5_schritten | diag_suffixlaenge |
|---|---|---|---|---|
| Wahrheit/Referenz | – | – | – | – |
| (a) | 20.68 | 0.09 | -0.13 | 26.19 |
| (b) | 23.05 | -0.03 | -0.14 | 28.10 |
| (c) | 10.18 | 0.11 | 0.43 | 32.76 |
| (d) 1 ms | 10.11 | 0.09 | 0.40 | 38.10 |
| D | 10.11 | 0.09 | 0.40 | 38.10 |
| OE | – | – | – | – |
| (e) | – | – | – | – |
| (c)-TF1 | – | – | – | – |
| D-TF1 | – | – | – | – |
| OE-TF1 Ep. 26 | – | – | – | – |
| OE-TF1 Ep. 1 | – | – | – | – |
| s40 | 10.25 | 0.02 | 0.51 | 39.63 |
| 10k neue Punkte | 10.93 | 0.02 | 0.60 | 48.81 |
| ZZ ep3 | – | – | – | – |
| ZZ ep4 | – | – | – | – |
| ZZ ep5 | – | – | – | – |
| ZZ ep6 | – | – | – | – |
| ZZ ep7 | – | – | – | – |
| ZZ ep9 | – | – | – | – |
| ZZ ep11 | – | – | – | – |
| ZZ ep13 | – | – | – | – |
| ZZ final | 15.66 | 0.17 | 0.36 | 41.73 |
| ZZ-TF1 ep3 | – | – | – | – |
| ZZ-TF1 ep6 | – | – | – | – |
| ZZ-TF1 ep9 | – | – | – | – |
| ZZ-TF1 ep12 | – | – | – | – |
| ZZ-TF1 final | 10.08 | 0.08 | 0.76 | 35.38 |
| ZZU-TF1 ep3 | – | – | – | – |
| ZZU-TF1 ep6 | – | – | – | – |
| ZZU-TF1 ep9 | – | – | – | – |
| ZZU-TF1 ep12 | – | – | – | – |
| ZZU-TF1 final | 10.18 | 0.04 | 0.76 | 33.34 |
| ZZ-TF1-WP ep5 | – | – | – | – |
| ZZ-TF1-WP ep10 | – | – | – | – |
| ZZ-TF1-WP150 ep15 | – | – | – | – |
| ZZ-TF1-WP150 ep45 | – | – | – | – |
| ZZ-TF1-WP150 ep90 | – | – | – | – |
| (c) Masken | – | – | – | – |
| ZZ ep9 Masken+Gumbel | – | – | – | – |
| (c) Masken+Gumbel | – | – | – | – |
| ZZ-TF1 ep12 Gumbel | – | – | – | – |
| T-Learner (ZZ-TF1) | – | – | – | – |
| ZZ-TF1 (RCT) Ep. 5 | – | – | – | – |
| ZZ-TF1 (RCT) | – | – | – | – |
| T-Learner (RCT) | – | – | – | – |
| ZZ-TF1 + Restzeit-Hilfsziel | – | – | – | – |
| T-Learner Suffix-Verlust (Zufallszahlen je Event) | – | – | – | – |
| T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event) | – | – | – | – |
| T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event) | – | – | – | – |
| T-Learner Suffix-Verlust + Kalibrierung Hälften (Zufallszahlen je Event) | – | – | – | – |
| T-Learner Kopf je Event (Steigung) | – | – | – | – |
| T-Learner Kopf je Event (Versatz) | – | – | – | – |
| T-Learner Kopf je Event (Steigung, β-NLL 1) | – | – | – | – |

## Zerlegung des Optionsvorteils in Pfad und Zeit (psp_zerlegung, echte Testentscheidungen)

zerl_vorteil = Vorteil der Aktion (Tage), davon zerl_pfad (andere Abfolge) und zerl_zeit (andere Abstände); erste Zeile = beobachtete Gruppenunterschiede im echten Log (nicht kausal). zerl_ohne_ende = Anteil Ziehungen ohne Endzustand.

| label | zerl_vorteil_konzept | zerl_pfad_konzept | zerl_zeit_konzept | zerl_vorteil_nach_v | zerl_pfad_nach_v | zerl_zeit_nach_v | zerl_vorteil_nachlegen | zerl_pfad_nachlegen | zerl_zeit_nachlegen | zerl_vorteil_leads | zerl_pfad_leads | zerl_zeit_leads | zerl_ohne_ende |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Wahrheit/Referenz | 2.41 | 1.41 | 1.00 | 3.46 | 0.87 | 2.59 | -4.56 | -0.97 | -3.59 | -2.82 | -0.19 | -2.63 | – |
| (a) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| (b) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| (c) | 1.46 | -0.02 | 1.48 | 2.09 | 0.10 | 1.99 | -1.21 | -0.19 | -1.03 | -1.23 | -0.00 | -1.23 | 47 % |
| (d) 1 ms | 0.45 | 0.03 | 0.42 | 1.82 | 0.05 | 1.77 | 0.01 | -0.02 | 0.03 | -1.41 | -0.06 | -1.36 | 39 % |
| D | – | – | – | – | – | – | – | – | – | – | – | – | – |
| OE | 1.02 | 0.18 | 0.84 | 1.11 | 0.26 | 0.85 | -0.67 | -0.51 | -0.16 | 0.21 | 0.14 | 0.06 | 36 % |
| (e) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| (c)-TF1 | 0.93 | -0.09 | 1.02 | 1.58 | -0.11 | 1.69 | 0.14 | -0.02 | 0.16 | -0.44 | -0.00 | -0.44 | 1 % |
| D-TF1 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| OE-TF1 Ep. 26 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| OE-TF1 Ep. 1 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| s40 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| 10k neue Punkte | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ ep3 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ ep4 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ ep5 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ ep6 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ ep7 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ ep9 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ ep11 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ ep13 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ final | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ-TF1 ep3 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ-TF1 ep6 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ-TF1 ep9 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ-TF1 ep12 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ-TF1 final | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZU-TF1 ep3 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZU-TF1 ep6 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZU-TF1 ep9 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZU-TF1 ep12 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZU-TF1 final | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ-TF1-WP ep5 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ-TF1-WP ep10 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ-TF1-WP150 ep15 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ-TF1-WP150 ep45 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ-TF1-WP150 ep90 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| (c) Masken | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ ep9 Masken+Gumbel | – | – | – | – | – | – | – | – | – | – | – | – | – |
| (c) Masken+Gumbel | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ-TF1 ep12 Gumbel | – | – | – | – | – | – | – | – | – | – | – | – | – |
| T-Learner (ZZ-TF1) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ-TF1 (RCT) Ep. 5 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ-TF1 (RCT) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| T-Learner (RCT) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ZZ-TF1 + Restzeit-Hilfsziel | – | – | – | – | – | – | – | – | – | – | – | – | – |
| T-Learner Suffix-Verlust (Zufallszahlen je Event) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| T-Learner Suffix-Verlust + Kalibrierung Hälften (Zufallszahlen je Event) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| T-Learner Kopf je Event (Steigung) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| T-Learner Kopf je Event (Versatz) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| T-Learner Kopf je Event (Steigung, β-NLL 1) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| Pilot 2_t_tf0 | 0.52 | 0.06 | 0.46 | 0.05 | -0.01 | 0.06 | -0.13 | -0.02 | -0.11 | -0.01 | 0.02 | -0.03 | 40 % |
| Pilot 2_t_tf05 | 0.81 | 0.04 | 0.77 | 0.20 | 0.01 | 0.20 | -0.07 | 0.10 | -0.16 | -0.05 | -0.02 | -0.03 | 39 % |
| Pilot 3_t_tf05_kx | 0.44 | -0.04 | 0.48 | 0.18 | 0.00 | 0.17 | -0.10 | -0.00 | -0.09 | -0.06 | -0.03 | -0.03 | 38 % |
| Pilot 3_t_tf1_kx | 0.14 | -0.06 | 0.20 | -0.05 | 0.02 | -0.07 | 0.04 | 0.03 | 0.01 | 0.12 | 0.04 | 0.08 | 27 % |
| Pilot k_tf0 | 0.88 | 0.01 | 0.87 | 1.66 | 0.03 | 1.64 | -0.02 | -0.01 | -0.01 | -0.47 | -0.01 | -0.45 | 45 % |
| Pilot k_tf0_od | – | – | – | – | – | – | – | – | – | – | – | – | – |
| Pilot k_tf1 | -0.06 | 0.00 | -0.06 | 0.03 | -0.00 | 0.04 | 0.27 | 0.00 | 0.27 | -0.04 | -0.00 | -0.04 | 24 % |
| Pilot k_tf1_od | 0.01 | 0.00 | 0.00 | -0.05 | -0.01 | -0.04 | 0.14 | 0.00 | 0.14 | -0.10 | -0.00 | -0.10 | 24 % |
| Pilot klassen_tf0_oe | 0.72 | 0.00 | 0.72 | 1.52 | 0.03 | 1.49 | 0.03 | -0.02 | 0.05 | -0.25 | -0.01 | -0.24 | 45 % |
| Pilot klassen_tf1_oe | -0.01 | -0.00 | -0.00 | -0.21 | -0.02 | -0.19 | 0.05 | 0.01 | 0.04 | 0.04 | -0.00 | 0.04 | 24 % |
| Pilot n_tf1 | 0.53 | -0.03 | 0.56 | 0.76 | -0.01 | 0.77 | -0.01 | 0.02 | -0.03 | 0.05 | -0.01 | 0.06 | 15 % |
| Pilot n_tf1_od | 0.68 | -0.01 | 0.70 | 0.73 | -0.04 | 0.77 | -0.02 | 0.00 | -0.03 | -0.00 | -0.00 | 0.00 | 15 % |
| Pilot normal_tf1_oe | 0.59 | 0.01 | 0.58 | 0.58 | 0.00 | 0.58 | 0.02 | 0.00 | 0.02 | 0.09 | -0.01 | 0.09 | 15 % |
| Pilot t_tf1_od | 0.17 | -0.09 | 0.26 | -0.05 | -0.12 | 0.07 | 0.09 | 0.10 | -0.01 | 0.04 | 0.05 | -0.01 | 24 % |
| Pilot token_tf05_oe | 0.62 | 0.00 | 0.62 | 0.23 | 0.03 | 0.20 | -0.00 | 0.03 | -0.04 | -0.10 | -0.05 | -0.05 | 40 % |
| Pilot token_tf08_oe | 0.39 | -0.01 | 0.40 | 0.08 | 0.02 | 0.06 | 0.01 | 0.02 | -0.01 | -0.11 | -0.04 | -0.08 | 34 % |
| Pilot token_tf0_oe | 0.88 | 0.03 | 0.86 | 0.39 | 0.08 | 0.31 | -0.21 | 0.00 | -0.21 | -0.29 | 0.06 | -0.35 | 40 % |
| Pilot token_tf1_oe | 0.26 | -0.05 | 0.31 | 0.03 | -0.00 | 0.03 | -0.20 | -0.06 | -0.14 | -0.09 | -0.01 | -0.08 | 24 % |

## Verluste der Prüfstände (Epoche des Stands)

| label | train_att | val_std | val_att |
|---|---|---|---|
| Wahrheit/Referenz | – | – | – |
| (a) | – | – | – |
| (b) | – | – | – |
| (c) | – | – | – |
| (d) 1 ms | – | – | – |
| D | – | – | – |
| OE | – | – | – |
| (e) | – | – | – |
| (c)-TF1 | – | – | – |
| D-TF1 | – | – | – |
| OE-TF1 Ep. 26 | -6.19 | 13.83 | 392.36 |
| OE-TF1 Ep. 1 | 0.28 | 10.26 | 352.40 |
| s40 | – | – | – |
| 10k neue Punkte | – | – | – |
| ZZ ep3 | 1.52 | 10.74 | 326.68 |
| ZZ ep4 | -0.14 | 10.86 | 285.19 |
| ZZ ep5 | -2.36 | 10.30 | 241.91 |
| ZZ ep6 | -2.45 | 9.95 | 223.71 |
| ZZ ep7 | -2.32 | 9.32 | 191.74 |
| ZZ ep9 | -2.29 | 8.99 | 276.52 |
| ZZ ep11 | -2.69 | 9.13 | 292.98 |
| ZZ ep13 | -3.02 | 9.08 | 286.77 |
| ZZ final | -3.25 | 9.14 | 296.02 |
| ZZ-TF1 ep3 | 1.33 | 11.39 | 349.73 |
| ZZ-TF1 ep6 | 1.59 | 12.11 | 392.99 |
| ZZ-TF1 ep9 | 1.47 | 12.54 | 395.50 |
| ZZ-TF1 ep12 | 1.41 | 12.83 | 382.86 |
| ZZ-TF1 final | 1.47 | 12.87 | 401.55 |
| ZZU-TF1 ep3 | 1.14 | 11.34 | 358.32 |
| ZZU-TF1 ep6 | 1.56 | 12.19 | 400.74 |
| ZZU-TF1 ep9 | 1.42 | 12.53 | 431.26 |
| ZZU-TF1 ep12 | 1.36 | 12.72 | 434.31 |
| ZZU-TF1 final | 1.41 | 12.96 | 423.55 |
| ZZ-TF1-WP ep5 | -1.04 | 8.67 | 332.20 |
| ZZ-TF1-WP ep10 | -2.10 | 9.60 | 368.41 |
| ZZ-TF1-WP150 ep15 | -2.19 | 10.46 | 432.14 |
| ZZ-TF1-WP150 ep45 | -1.98 | 11.27 | 432.56 |
| ZZ-TF1-WP150 ep90 | -2.52 | 11.27 | 429.49 |
| (c) Masken | – | – | – |
| ZZ ep9 Masken+Gumbel | – | – | – |
| (c) Masken+Gumbel | – | – | – |
| ZZ-TF1 ep12 Gumbel | – | – | – |
| T-Learner (ZZ-TF1) | – | – | – |
| ZZ-TF1 (RCT) Ep. 5 | – | – | – |
| ZZ-TF1 (RCT) | – | – | – |
| T-Learner (RCT) | – | – | – |
| ZZ-TF1 + Restzeit-Hilfsziel | – | – | – |
| T-Learner Suffix-Verlust (Zufallszahlen je Event) | – | – | – |
| T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event) | – | – | – |
| T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event) | – | – | – |
| T-Learner Suffix-Verlust + Kalibrierung Hälften (Zufallszahlen je Event) | – | – | – |
| T-Learner Kopf je Event (Steigung) | – | – | – |
| T-Learner Kopf je Event (Versatz) | – | – | – |
| T-Learner Kopf je Event (Steigung, β-NLL 1) | – | – | – |

## Piloten (3.000 Fälle, 6 Epochen): Optionssignal

| label | aipw_vorzeichen_konzept | aipw_vorzeichen_nach_v | aipw_vorzeichen_nachlegen | aipw_vorzeichen_leads | se_gepaart | korrelation_optionen |
|---|---|---|---|---|---|---|
| Pilot 2_t_tf0 | 74 % | 52 % | 54 % | 48 % | 0.65 | 0.83 |
| Pilot 2_t_tf05 | 78 % | 58 % | 51 % | 54 % | 0.89 | 0.69 |
| Pilot 3_t_tf05_kx | 68 % | 54 % | 52 % | 56 % | 0.93 | 0.64 |
| Pilot 3_t_tf1_kx | 52 % | 48 % | 52 % | 43 % | 1.04 | 0.54 |
| Pilot k_tf0 | 86 % | 90 % | 50 % | 64 % | 1.05 | 0.57 |
| Pilot k_tf0_od | 54 % | 64 % | 56 % | 40 % | 0.56 | 0.87 |
| Pilot k_tf1 | 46 % | 52 % | 44 % | 50 % | 1.09 | 0.58 |
| Pilot k_tf1_od | 50 % | 49 % | 42 % | 56 % | 1.07 | 0.58 |
| Pilot klassen_tf0_oe | 78 % | 91 % | 46 % | 56 % | 1.06 | 0.58 |
| Pilot klassen_tf1_oe | 52 % | 37 % | 48 % | 48 % | 1.07 | 0.58 |
| Pilot n_tf1 | 68 % | 69 % | 48 % | 48 % | 1.31 | 0.52 |
| Pilot n_tf1_od | 68 % | 66 % | 48 % | 46 % | 1.32 | 0.53 |
| Pilot normal_tf1_oe | 64 % | 68 % | 50 % | 48 % | 1.29 | 0.53 |
| Pilot t_tf1_od | 55 % | 46 % | 46 % | 49 % | 1.00 | 0.56 |
| Pilot token_tf05_oe | 70 % | 56 % | 52 % | 60 % | 0.99 | 0.63 |
| Pilot token_tf08_oe | 62 % | 53 % | 49 % | 57 % | 0.93 | 0.62 |
| Pilot token_tf0_oe | 84 % | 60 % | 60 % | 57 % | 1.02 | 0.59 |
| Pilot token_tf1_oe | 58 % | 52 % | 56 % | 58 % | 1.04 | 0.54 |

Einstellungen der Piloten:

| label | epochen_trainiert | lernrate | decoderlaenge | tf_untergrenze | faelle_training | zeit_aus | weitere_params | modelldatei |
|---|---|---|---|---|---|---|---|---|
| Pilot 2_t_tf0 | 6 | 1e-04 | 60 | 0.00 | 3000 | gaps | gap_tokens=True; option_as=decoder | prescriptive/models/psp_model_durchlaufzeit_pilot_token_tf0.pkl |
| Pilot 2_t_tf05 | 6 | 1e-04 | 60 | 0.50 | 3000 | gaps | gap_tokens=True; option_as=decoder | prescriptive/models/psp_model_durchlaufzeit_pilot_token_tf05.pkl |
| Pilot 3_t_tf05_kx | 6 | 1e-04 | 60 | 0.50 | 3000 | gaps | decoder_context=True; gap_tokens=True; option_as=encoder | prescriptive/models/psp_model_durchlaufzeit_pilot_token_tf05_kontext.pkl |
| Pilot 3_t_tf1_kx | 6 | 1e-04 | 60 | 1.00 | 3000 | gaps | decoder_context=True; gap_tokens=True; option_as=encoder | prescriptive/models/psp_model_durchlaufzeit_pilot_token_tf1_kontext.pkl |
| Pilot k_tf0 | 6 | 1e-04 | 60 | 0.00 | 3000 | gaps | gap_classes=True | prescriptive/models/psp_model_durchlaufzeit_pilot_klassen_tf0.pkl |
| Pilot k_tf0_od | 6 | 1e-04 | 60 | 0.00 | 3000 | gaps | gap_classes=True; option_as=decoder | prescriptive/models/psp_model_durchlaufzeit_pilot_klassen_tf0_od.pkl |
| Pilot k_tf1 | 6 | 1e-04 | 60 | 1.00 | 3000 | gaps | gap_classes=True | prescriptive/models/psp_model_durchlaufzeit_pilot_klassen_tf1.pkl |
| Pilot k_tf1_od | 6 | 1e-04 | 60 | 1.00 | 3000 | gaps | gap_classes=True; option_as=decoder | prescriptive/models/psp_model_durchlaufzeit_pilot_klassen_tf1_od.pkl |
| Pilot klassen_tf0_oe | 6 | 1e-04 | 60 | 0.00 | 3000 | gaps | gap_classes=True; option_as=encoder | prescriptive/models/psp_model_durchlaufzeit_pilot_klassen_tf0_oe.pkl |
| Pilot klassen_tf1_oe | 6 | 1e-04 | 60 | 1.00 | 3000 | gaps | gap_classes=True; option_as=encoder | prescriptive/models/psp_model_durchlaufzeit_pilot_klassen_tf1_oe.pkl |
| Pilot n_tf1 | 6 | 1e-04 | 60 | 1.00 | 3000 | gaps |  | prescriptive/models/psp_model_durchlaufzeit_pilot_normal_tf1.pkl |
| Pilot n_tf1_od | 6 | 1e-04 | 60 | 1.00 | 3000 | gaps | option_as=decoder | prescriptive/models/psp_model_durchlaufzeit_pilot_normal_tf1_od.pkl |
| Pilot normal_tf1_oe | 6 | 1e-04 | 60 | 1.00 | 3000 | gaps | option_as=encoder | prescriptive/models/psp_model_durchlaufzeit_pilot_normal_tf1_oe.pkl |
| Pilot t_tf1_od | 6 | 1e-04 | 60 | 1.00 | 3000 | gaps | gap_tokens=True; option_as=decoder | prescriptive/models/psp_model_durchlaufzeit_pilot_token_tf1.pkl |
| Pilot token_tf05_oe | 6 | 1e-04 | 60 | 0.50 | 3000 | gaps | gap_tokens=True; option_as=encoder | prescriptive/models/psp_model_durchlaufzeit_pilot_token_tf05_oe.pkl |
| Pilot token_tf08_oe | 6 | 1e-04 | 60 | 0.80 | 3000 | gaps | gap_tokens=True; option_as=encoder | prescriptive/models/psp_model_durchlaufzeit_pilot_token_tf08_oe.pkl |
| Pilot token_tf0_oe | 6 | 1e-04 | 60 | 0.00 | 3000 | gaps | gap_tokens=True; option_as=encoder | prescriptive/models/psp_model_durchlaufzeit_pilot_token_tf0_oe.pkl |
| Pilot token_tf1_oe | 6 | 1e-04 | 60 | 1.00 | 3000 | gaps | gap_tokens=True; option_as=encoder | prescriptive/models/psp_model_durchlaufzeit_pilot_token_tf1_oe.pkl |

## Online-Ergebnisse (alle Läufe, Langformat in online.csv mit Quelle je Zeile)

gewinn_tage = Durchlaufzeit Bank − Policy im selben Aufbau (positiv = schneller). Nur innerhalb eines Aufbaus vergleichen: Simulatorstand, Zuteilung, Fallzahl und Ziel unterscheiden sich. In register.csv steht je Modell und Aufbau die beste Variante (online_<Buchstabe>_gewinn / _variante). (b) „direkte Restzeitbewertung“ nutzt den Restzeitkopf (für PSP nicht mehr erlaubt).

Beste erlaubte Variante je Modell und Aufbau (Gewinn in Tagen; Aufbauten: A = A: älterer Simulator, 10 Seeds × 1.000 Fälle (Lauf 12.11. mit CATE); B = B: gelernte Zuteilung, 10 Seeds × 1.000 Fälle, ERM-Ziel γ=0,1; C = C: 1 Seed × 200 Fälle (Seed 0), 30 Ziehungen; D = D: 1 Seed × 1.000 Fälle (Seed 0), 300 Ziehungen; E = E: ältere Zuteilung, fertige Seeds × 1.000 Fälle, 300 Ziehungen (Werte gerundet); F = F: korrigierter Simulator, 3 Seeds × 200 Fälle, 100 Ziehungen; G = G: Eventkosten-Ziel, 1 Seed × 200 Fälle (Seed 0); L = L: korrigierter Simulator, 12 Seeds × 200 Fälle, first_only (17.11.)):

| label | online_A_gewinn | online_B_gewinn | online_C_gewinn | online_D_gewinn | online_E_gewinn | online_F_gewinn | online_G_gewinn | online_L_gewinn |
|---|---|---|---|---|---|---|---|---|
| (a) | -0.49 | – | – | – | – | – | – | – |
| (b) | -0.30 | – | – | – | – | – | – | – |
| (c) | 1.84 | 0.16 | 0.18 | – | 2.45 | 1.78 | – | 1.97 |
| D | – | – | 0.29 | 0.79 | – | 0.82 | 0.03 | – |
| OE | – | – | -0.45 | -0.42 | 1.12 | -0.05 | – | – |
| (e) | – | – | -1.67 | – | – | – | – | – |
| (c)-TF1 | – | – | -1.29 | – | – | – | – | – |
| s40 | 1.39 | – | – | – | – | – | – | – |
| 10k neue Punkte | -0.09 | – | – | – | – | – | – | – |
| T-Learner (ZZ-TF1) | – | – | – | – | – | – | – | 2.28 |
| ZZ-TF1 (RCT) | – | – | – | – | – | – | – | 0.99 |
| T-Learner (RCT) | – | – | – | – | – | – | – | 1.98 |
| T-Learner Suffix-Verlust (Zufallszahlen je Event) | – | – | – | – | – | – | – | 2.23 |
| T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event) | – | – | – | – | – | – | – | 2.61 |
| T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event) | – | – | – | – | – | – | – | -0.48 |
| T-Learner Suffix-Verlust + Kalibrierung Hälften (Zufallszahlen je Event) | – | – | – | – | – | – | – | 2.15 |
| T-Learner Kopf je Event (Steigung) | – | – | – | – | – | – | – | 1.41 |

### A: älterer Simulator, 10 Seeds × 1.000 Fälle

| label | variante | ziel | ziehungen | seeds | faelle_je_seed | dlz_tage | bank_tage | gewinn_tage | se_tage | causal_forest_gewinn_gleiche_seeds | angebote_je_fall | kosten_eur | bank_kosten_eur |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (b) | (b), direkte Restzeitbewertung | Durchlaufzeit | 30 | 10 | 1000 | 19.83 | 22.33 | 2.50 | 0.23 | – | – | – | – |
| (c) | (c), normale Abstände | Durchlaufzeit | 100 | 10 | 1000 | 20.49 | 22.33 | 1.84 | 0.20 | – | – | – | – |
| s40 | s40, Summe der Zeitabstände | Durchlaufzeit | 100 | 10 | 1000 | 20.94 | 22.33 | 1.39 | 0.25 | – | – | – | – |
| s40 | s40, Angebot nur beim ersten Besuch wählen | Durchlaufzeit | 100 | 10 | 1000 | 21.02 | 22.33 | 1.30 | 0.22 | – | – | – | – |
| s40 | s40, Summe der Zeitabstände | Durchlaufzeit | 30 | 10 | 1000 | 21.48 | 22.33 | 0.85 | 0.23 | – | – | – | – |
| s40 | s40, Zeit aus vorhergesagtem Fallalter | Durchlaufzeit | 30 | 10 | 1000 | 21.81 | 22.33 | 0.52 | 0.27 | – | – | – | – |
| 10k neue Punkte | Auf neue Wahlpunkte neu trainiert | Durchlaufzeit | 30 | 10 | 1000 | 22.42 | 22.33 | -0.09 | 0.17 | – | – | – | – |
| 10k Original | Ursprüngliches Echtlog-PSP | Durchlaufzeit | 30 | 10 | 1000 | 22.57 | 22.33 | -0.25 | 0.28 | – | – | – | – |
| (b) | (b), mit Restzeitkopf, Suffixbewertung | Durchlaufzeit | 100 | 10 | 1000 | 22.63 | 22.33 | -0.30 | 0.13 | – | – | – | – |
| (a) | (a), logarithmierte Abstände | Durchlaufzeit | 100 | 10 | 1000 | 22.82 | 22.33 | -0.49 | 0.29 | – | – | – | – |

### A: älterer Simulator, 10 Seeds × 1.000 Fälle (Lauf 12.11. mit CATE)

| label | variante | ziel | ziehungen | seeds | faelle_je_seed | dlz_tage | bank_tage | gewinn_tage | se_tage | causal_forest_gewinn_gleiche_seeds | angebote_je_fall | kosten_eur | bank_kosten_eur |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Regel (AIPW) | regel | Durchlaufzeit | – | 10 | 1000 | 19.36 | 22.33 | 2.97 | 0.19 | – | – | – | – |
| Causal Forest | causal_forest | Durchlaufzeit | – | 10 | 1000 | 19.48 | 22.33 | 2.85 | 0.23 | – | – | – | – |
| X-Learner | x_learner | Durchlaufzeit | – | 10 | 1000 | 19.62 | 22.33 | 2.71 | 0.26 | – | – | – | – |
| S-Learner | s_learner | Durchlaufzeit | – | 10 | 1000 | 20.04 | 22.33 | 2.29 | 0.08 | – | – | – | – |
| XGB | xgb | Durchlaufzeit | – | 10 | 1000 | 20.41 | 22.33 | 1.92 | 0.13 | – | – | – | – |
| (c) | psp (c) | Durchlaufzeit | 100 | 10 | 1000 | 20.49 | 22.33 | 1.84 | 0.18 | – | – | – | – |

### B: gelernte Zuteilung, 10 Seeds × 1.000 Fälle, ERM-Ziel γ=0,1

| label | variante | ziel | ziehungen | seeds | faelle_je_seed | dlz_tage | bank_tage | gewinn_tage | se_tage | causal_forest_gewinn_gleiche_seeds | angebote_je_fall | kosten_eur | bank_kosten_eur |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Causal Forest | causal_forest_erm | ERM γ=0,1 (Durchlaufzeit) | – | 10 | 1000 | 20.81 | 22.60 | 1.79 | – | – | – | – | – |
| X-Learner | x_learner_erm | ERM γ=0,1 (Durchlaufzeit) | – | 10 | 1000 | 22.19 | 22.60 | 0.41 | – | – | – | – | – |
| (c) | psp_erm | ERM γ=0,1 (Durchlaufzeit) | 100 | 10 | 1000 | 22.44 | 22.60 | 0.16 | – | – | – | – | – |
| S-Learner | s_learner_erm | ERM γ=0,1 (Durchlaufzeit) | – | 10 | 1000 | 22.65 | 22.60 | -0.05 | – | – | – | – | – |

### C: 1 Seed × 200 Fälle (Seed 0), 30 Ziehungen

| label | variante | ziel | ziehungen | seeds | faelle_je_seed | dlz_tage | bank_tage | gewinn_tage | se_tage | causal_forest_gewinn_gleiche_seeds | angebote_je_fall | kosten_eur | bank_kosten_eur |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D | (d), Abstand je Suffix | Durchlaufzeit | 30 | 1 | 200 | 20.09 | 20.37 | 0.29 | – | – | – | – | – |
| (c) | (c), 1 ms | Durchlaufzeit | 30 | 1 | 200 | 20.20 | 20.37 | 0.18 | – | – | – | – | – |
| OE | OE, ein Abstand je Option | Durchlaufzeit | 30 | 1 | 200 | 20.83 | 20.37 | -0.45 | – | – | – | – | – |
| OE | OE, Abstand je Suffix | Durchlaufzeit | 30 | 1 | 200 | 21.20 | 20.37 | -0.82 | – | – | – | – | – |
| (c) | (c), ein Abstand je Option | Durchlaufzeit | 30 | 1 | 200 | 21.21 | 20.37 | -0.84 | – | – | – | – | – |
| (c)-TF1 | c_tf1 | Durchlaufzeit | 30 | 1 | 200 | 21.66 | 20.37 | -1.29 | – | – | – | – | – |
| OE | OE + Übergangssperren | Durchlaufzeit | 30 | 1 | 200 | 21.74 | 20.37 | -1.36 | – | – | – | – | – |
| (e) | e | Durchlaufzeit | 30 | 1 | 200 | 22.05 | 20.37 | -1.67 | – | – | – | – | – |

### D: 1 Seed × 1.000 Fälle (Seed 0), 300 Ziehungen

| label | variante | ziel | ziehungen | seeds | faelle_je_seed | dlz_tage | bank_tage | gewinn_tage | se_tage | causal_forest_gewinn_gleiche_seeds | angebote_je_fall | kosten_eur | bank_kosten_eur |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D | (d), Abstand je Suffix | Durchlaufzeit | 300 | 1 | 1000 | 20.92 | 21.71 | 0.79 | – | – | – | – | – |
| OE | OE mit Unsicherheitsrückfall (token_oe_ue) | Durchlaufzeit | 300 | 1 | 1000 | 22.13 | 21.71 | -0.42 | – | – | – | – | – |
| OE | OE + Übergangssperren | Durchlaufzeit | 300 | 1 | 1000 | 22.13 | 21.71 | -0.42 | – | – | – | – | – |

### E: ältere Zuteilung, fertige Seeds × 1.000 Fälle, 300 Ziehungen (Werte gerundet)

| label | variante | ziel | ziehungen | seeds | faelle_je_seed | dlz_tage | bank_tage | gewinn_tage | se_tage | causal_forest_gewinn_gleiche_seeds | angebote_je_fall | kosten_eur | bank_kosten_eur |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (c) | (c), 300 Suffixe | Durchlaufzeit | 300 | 5 | 1000 | 19.88 | 22.33 | 2.45 | 0.29 | 2.69 | – | – | – |
| OE | OE, 300 Suffixe | Durchlaufzeit | 300 | 7 | 1000 | 21.30 | 22.42 | 1.12 | 0.25 | 2.94 | – | – | – |

### F: korrigierter Simulator, 3 Seeds × 200 Fälle, 100 Ziehungen

| label | variante | ziel | ziehungen | seeds | faelle_je_seed | dlz_tage | bank_tage | gewinn_tage | se_tage | causal_forest_gewinn_gleiche_seeds | angebote_je_fall | kosten_eur | bank_kosten_eur |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Causal Forest | Ziel Gesamtkosten | Durchlaufzeit | – | 3 | 200 | 18.53 | 21.14 | 2.62 | 0.61 | – | 1.27 | – | – |
| Causal Forest | Ziel Durchlaufzeit | Durchlaufzeit | – | 3 | 200 | 18.53 | 21.14 | 2.62 | 0.61 | – | 1.27 | – | – |
| Regel (AIPW) | feste AIPW-Regel | Durchlaufzeit | – | 3 | 200 | 18.53 | 21.14 | 2.62 | 0.61 | – | 1.27 | – | – |
| X-Learner | Ziel Durchlaufzeit | Durchlaufzeit | – | 3 | 200 | 18.85 | 21.14 | 2.29 | 0.59 | – | 1.44 | – | – |
| (c) | nur 1. Besuch Nachlegen (first_only), 300 Ziehungen fest | Durchlaufzeit | 300 | 3 | 200 | 19.36 | 21.14 | 1.78 | 1.00 | – | 1.40 | – | – |
| (c) | nur 1. Besuch Nachlegen (first_only), fest 100 | Durchlaufzeit | 100 | 3 | 200 | 19.41 | 21.14 | 1.73 | 1.04 | – | 1.53 | – | – |
| (c) | nur 1. Besuch Nachlegen (first_only), adaptiv ohne | Durchlaufzeit | 100 | 3 | 200 | 19.85 | 21.14 | 1.30 | 0.72 | – | 1.50 | – | – |
| XGB | Zeit | Durchlaufzeit | – | 3 | 200 | 19.96 | 21.14 | 1.18 | 0.58 | – | 2.23 | – | – |
| (c) | adaptiv ohne (Original-Netz, Decoder-Zeiten) | Durchlaufzeit | 100 | 3 | 200 | 20.26 | 21.14 | 0.88 | 0.53 | – | 2.31 | – | – |
| D | Zeit ohne (zustandsbewusste Kopie: Arbeitszustand + empirische Abstandstabelle) | Durchlaufzeit | 100 | 3 | 200 | 20.32 | 21.14 | 0.82 | 0.57 | – | 1.81 | – | – |
| D | Zeit mit (zustandsbewusste Kopie: Arbeitszustand + empirische Abstandstabelle) | Durchlaufzeit | 100 | 3 | 200 | 21.08 | 21.14 | 0.07 | 0.90 | – | 1.40 | – | – |
| (c) | fest 100 (Original-Netz, Decoder-Zeiten) | Durchlaufzeit | 100 | 3 | 200 | 21.12 | 21.14 | 0.02 | 0.74 | – | 2.44 | – | – |
| OE | Zeit ohne (zustandsbewusste Kopie: Arbeitszustand + empirische Abstandstabelle) | Durchlaufzeit | 100 | 3 | 200 | 21.20 | 21.14 | -0.05 | 0.64 | – | 1.93 | – | – |
| (c) | nur 1. Besuch Nachlegen (first_only), adaptiv mit | Durchlaufzeit | 100 | 3 | 200 | 21.42 | 21.14 | -0.28 | 0.76 | – | 1.34 | – | – |
| (c) | adaptiv mit (Original-Netz, Decoder-Zeiten) | Durchlaufzeit | 100 | 3 | 200 | 21.47 | 21.14 | -0.33 | 0.42 | – | 1.86 | – | – |
| OE | Zeit mit (zustandsbewusste Kopie: Arbeitszustand + empirische Abstandstabelle) | Durchlaufzeit | 100 | 3 | 200 | 21.63 | 21.14 | -0.48 | 0.31 | – | 1.41 | – | – |

### G: Eventkosten-Ziel, 1 Seed × 200 Fälle (Seed 0)

| label | variante | ziel | ziehungen | seeds | faelle_je_seed | dlz_tage | bank_tage | gewinn_tage | se_tage | causal_forest_gewinn_gleiche_seeds | angebote_je_fall | kosten_eur | bank_kosten_eur |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Causal Forest | causal_forest | Eventkosten (€) | – | 1 | 200 | 17.76 | 19.41 | 1.65 | – | – | – | 165.48 | 174.03 |
| X-Learner | x_learner | Eventkosten (€) | – | 1 | 200 | 18.74 | 19.41 | 0.67 | – | – | – | 192.79 | 174.03 |
| D | psp, 500 Ziehungen, Unsicherheitsrückfall | Eventkosten (€) | 500 | 1 | 200 | 19.38 | 19.41 | 0.03 | – | – | – | 176.98 | 174.03 |
| D | psp, 300 Ziehungen | Eventkosten (€) | 300 | 1 | 200 | 19.93 | 19.41 | -0.52 | – | – | – | 198.18 | 174.03 |
| D | psp, 30 Ziehungen | Eventkosten (€) | 30 | 1 | 200 | 19.95 | 19.41 | -0.54 | – | – | – | 209.99 | 174.03 |
| D | psp, 300 Ziehungen, Unsicherheitsrückfall | Eventkosten (€) | 300 | 1 | 200 | 20.02 | 19.41 | -0.61 | – | – | – | 179.27 | 174.03 |
| S-Learner | s_learner | Eventkosten (€) | – | 1 | 200 | 20.30 | 19.41 | -0.89 | – | – | – | 189.10 | 174.03 |

### H: 12 Interventionspunkte (Kontrollfluss + 8 Mitarbeiterpunkte), älterer Simulator, 10 Seeds × 1.000 Fälle (Anfang Nov.)

| label | variante | ziel | ziehungen | seeds | faelle_je_seed | dlz_tage | bank_tage | gewinn_tage | se_tage | causal_forest_gewinn_gleiche_seeds | angebote_je_fall | kosten_eur | bank_kosten_eur |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| XGB | XGB, 5.000 Fälle | Durchlaufzeit | – | 10 | 1000 | 16.48 | 22.33 | 5.85 | 0.21 | – | – | – | – |
| 10k Original | PSP 10k, Kanal statt Kanal+Nachlegen | Durchlaufzeit | 30 | 10 | 1000 | 20.74 | 22.33 | 1.59 | 0.23 | – | – | – | – |
| Zufall | zufällige Wahl | Durchlaufzeit | – | 10 | 1000 | 21.53 | 22.33 | 0.80 | 0.17 | – | – | – | – |
| PSP Nachtlauf | PSP 5.000 Fälle, 25 Ep., 1e-4, Decoder 5 | Durchlaufzeit | 30 | 10 | 1000 | 21.65 | 22.33 | 0.67 | 0.14 | – | – | – | – |
| 10k Original | PSP 10.000 Fälle, 1e-5, Decoder 5 | Durchlaufzeit | 30 | 10 | 1000 | 21.73 | 22.33 | 0.60 | 0.23 | – | – | – | – |
| 10k Original | PSP 10k, nur die 4 Kontrollfluss-Punkte | Durchlaufzeit | 30 | 10 | 1000 | 21.80 | 22.33 | 0.53 | 0.17 | – | – | – | – |
| 10k Original | PSP 10k, nur Kanal+Nachlegen | Durchlaufzeit | 30 | 10 | 1000 | 22.72 | 22.33 | -0.39 | 0.21 | – | – | – | – |

### I: Kanal + 2 Mitarbeiterpunkte, 5 Seeds × 2.000 Fälle (28.10.)

| label | variante | ziel | ziehungen | seeds | faelle_je_seed | dlz_tage | bank_tage | gewinn_tage | se_tage | causal_forest_gewinn_gleiche_seeds | angebote_je_fall | kosten_eur | bank_kosten_eur |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LSTM | LSTM | Durchlaufzeit | – | 5 | 2000 | 20.22 | 22.10 | 1.88 | 0.21 | – | – | – | – |
| XGB | XGB | Durchlaufzeit | – | 5 | 2000 | 20.33 | 22.10 | 1.76 | 0.10 | – | – | – | – |
| Zufall | zufällige Wahl | Durchlaufzeit | – | 5 | 2000 | 21.44 | 22.10 | 0.65 | 0.22 | – | – | – | – |

### J: nur Mitarbeiterwahl W_Call incomplete files, 5 Seeds × 2.000 Fälle

| label | variante | ziel | ziehungen | seeds | faelle_je_seed | dlz_tage | bank_tage | gewinn_tage | se_tage | causal_forest_gewinn_gleiche_seeds | angebote_je_fall | kosten_eur | bank_kosten_eur |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PSP | PSP mit Vorausschau | Durchlaufzeit | – | 5 | 2000 | 22.33 | 22.10 | -0.23 | 0.19 | – | – | – | – |
| PSP | PSP | Durchlaufzeit | – | 5 | 2000 | 22.35 | 22.10 | -0.25 | 0.09 | – | – | – | – |
| Zufall | zufällige Wahl | Durchlaufzeit | – | 5 | 2000 | 22.35 | 22.10 | -0.26 | 0.13 | – | – | – | – |

### K: Mitarbeiterwahl, Simulator-RCT-Modell, 5 Seeds × 2.000 Fälle (27.10.)

| label | variante | ziel | ziehungen | seeds | faelle_je_seed | dlz_tage | bank_tage | gewinn_tage | se_tage | causal_forest_gewinn_gleiche_seeds | angebote_je_fall | kosten_eur | bank_kosten_eur |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Zufall | zufälliger Mitarbeiter | Durchlaufzeit | – | 5 | 2000 | 21.37 | 21.38 | 0.01 | 0.07 | – | – | – | – |
| XGB | XGB auf Simulator-RCT trainiert | Durchlaufzeit | – | 5 | 2000 | 21.40 | 21.38 | -0.02 | 0.12 | – | – | – | – |

### L: korrigierter Simulator, 12 Seeds × 200 Fälle, first_only (17.11.)

| label | variante | ziel | ziehungen | seeds | faelle_je_seed | dlz_tage | bank_tage | gewinn_tage | se_tage | causal_forest_gewinn_gleiche_seeds | angebote_je_fall | kosten_eur | bank_kosten_eur |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Netz je Wahlpunkt und Option + Suffix-Verlust, Causal Forest bei Unsicherheit | keyed_noise, first_only, 300 Ziehungen fest; /Δ/ < 1,96·SE → Causal Forest | Durchlaufzeit | 300 | 12 | 200 | 18.61 | 21.25 | 2.64 | 0.35 | – | 1.26 | – | – |
| Regel (AIPW) | feste AIPW-Regel | Durchlaufzeit | – | 12 | 200 | 18.62 | 21.25 | 2.63 | 0.37 | – | 1.26 | – | – |
| T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event) | Zufallszahlen je Ereignis (keyed_noise), first_only, 300 Ziehungen fest | Durchlaufzeit | 300 | 12 | 200 | 18.65 | 21.25 | 2.61 | 0.46 | – | 1.30 | – | – |
| Netz je Wahlpunkt und Option + Suffix-Verlust (Zufallszahlen je Event) | Zufallszahlen je Ereignis (keyed_noise), first_only, 300 Ziehungen fest | Durchlaufzeit | 300 | 12 | 200 | 18.79 | 21.25 | 2.46 | 0.33 | – | 1.27 | – | – |
| T-Learner (ZZ-TF1) | Zufallszahlen je Ereignis (keyed_noise), first_only, 300 Ziehungen fest | Durchlaufzeit | 300 | 12 | 200 | 18.97 | 21.25 | 2.28 | 0.44 | – | 1.68 | – | – |
| T-Learner Suffix-Verlust (Zufallszahlen je Event) | Zufallszahlen je Ereignis (keyed_noise), first_only, 300 Ziehungen fest | Durchlaufzeit | 300 | 12 | 200 | 19.02 | 21.25 | 2.23 | 0.50 | – | 1.41 | – | – |
| T-Learner Suffix-Verlust + Kalibrierung Hälften (Zufallszahlen je Event) | Zufallszahlen je Ereignis (keyed_noise), first_only, 300 Ziehungen fest | Durchlaufzeit | 300 | 12 | 200 | 19.10 | 21.25 | 2.15 | 0.42 | – | 1.51 | – | – |
| T-Learner (RCT) | auf Simulator-RCT-Daten trainiert, first_only, 300 Ziehungen fest | Durchlaufzeit | 300 | 12 | 200 | 19.27 | 21.25 | 1.98 | 0.42 | – | 1.51 | – | – |
| (c) | first_only, 300 Ziehungen fest | Durchlaufzeit | 300 | 12 | 200 | 19.28 | 21.25 | 1.97 | 0.51 | – | 1.39 | – | – |
| (c) | first_only, fest 100 | Durchlaufzeit | 100 | 12 | 200 | 19.68 | 21.25 | 1.57 | 0.43 | – | 1.55 | – | – |
| T-Learner (ZZ-TF1) | first_only, 300 Ziehungen fest | Durchlaufzeit | 300 | 12 | 200 | 19.73 | 21.25 | 1.52 | 0.43 | – | 1.76 | – | – |
| S-Learner je Wahlpunkt + Suffix-Verlust (Zufallszahlen je Event) | Zufallszahlen je Ereignis (keyed_noise), first_only, 300 Ziehungen fest | Durchlaufzeit | 300 | 12 | 200 | 19.76 | 21.25 | 1.49 | 0.46 | – | 1.82 | – | – |
| T-Learner Kopf je Event (Steigung) | Zeitkopf je Event (Versatz + Steigung), nur Kopf nachtrainiert, first_only, 300 Ziehungen fest | Durchlaufzeit | 300 | 12 | 200 | 19.84 | 21.25 | 1.41 | 0.38 | – | 1.79 | – | – |
| T-Learner (ZZ-TF1) | first_only, 500 Ziehungen fest | Durchlaufzeit | 500 | 12 | 200 | 19.86 | 21.25 | 1.39 | 0.44 | – | 1.72 | – | – |
| ZZ-TF1 (RCT) | auf Simulator-RCT-Daten trainiert, first_only, fest 100 | Durchlaufzeit | 100 | 12 | 200 | 20.26 | 21.25 | 0.99 | 0.50 | – | 1.68 | – | – |
| ZZ-TF1 | first_only, fest 100 | Durchlaufzeit | 100 | 12 | 200 | 21.33 | 21.25 | -0.07 | 0.48 | – | 1.92 | – | – |
| T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event) | Zufallszahlen je Ereignis (keyed_noise), first_only, 300 Ziehungen fest | Durchlaufzeit | 300 | 12 | 200 | 21.73 | 21.25 | -0.48 | 0.47 | – | 2.31 | – | – |

## Simulator: Treue gegenüber dem echten Log (Langformat in simulator.csv)

Wichtig für die Deutung aller Online-Ergebnisse. Die Messungen vom 06.11. stammen **vor** der Umstellung auf gelernte Zuteilung (08.11.) und der korrigierten Optionsprüfung (15.11.); die RCT-Wahrheit ist vom 14.11. Nach Simulatoränderungen neu messen (main.py --steps kpis; scripts/validate_screening.py).

### Allgemein (10.000 simulierte Fälle gegen 31.509 echte)

| bereich | kennzahl | echt | simulator | abweichung | abweichung_als | stand |
|---|---|---|---|---|---|---|
| Umfang | Fälle | 31509.00 | 10000.00 | -68.26 | % von echt | 2026-11-06 |
| Umfang | Events | 1202267.00 | 422056.00 | -64.89 | % von echt | 2026-11-06 |
| Umfang | verschiedene Aktivitäten | 26.00 | 25.00 | -3.85 | % von echt | 2026-11-06 |
| Events pro Fall | Mittel | 38.16 | 42.21 | 10.61 | % von echt | 2026-11-06 |
| Events pro Fall | Median | 35.00 | 37.00 | 5.71 | % von echt | 2026-11-06 |
| Events pro Fall | P90 | 60.00 | 70.00 | 16.67 | % von echt | 2026-11-06 |
| Events pro Fall | Maximum | 180.00 | 199.00 | 10.56 | % von echt | 2026-11-06 |
| Durchlaufzeit pro Fall | Mittel (h) | 525.59 | 529.12 | 0.67 | % von echt | 2026-11-06 |
| Durchlaufzeit pro Fall | Median (h) | 458.11 | 450.37 | -1.69 | % von echt | 2026-11-06 |
| Durchlaufzeit pro Fall | P75 (h) | 755.90 | 756.78 | 0.12 | % von echt | 2026-11-06 |
| Durchlaufzeit pro Fall | P90 (h) | 841.32 | 859.55 | 2.17 | % von echt | 2026-11-06 |
| Durchlaufzeit pro Fall | P95 (h) | 1020.50 | 1062.95 | 4.16 | % von echt | 2026-11-06 |
| Durchlaufzeit pro Fall | Maximum (h) | 6865.74 | 3417.75 | -50.22 | % von echt | 2026-11-06 |
| Ankünfte pro Tag | Mittel | 86.09 | 75.19 | -12.66 | % von echt | 2026-11-06 |
| Ankünfte pro Tag | Median | 88.00 | 80.00 | -9.09 | % von echt | 2026-11-06 |
| Ankünfte pro Tag | P90 | 126.00 | 105.00 | -16.67 | % von echt | 2026-11-06 |
| Ankünfte pro Tag | Maximum | 178.00 | 122.00 | -31.46 | % von echt | 2026-11-06 |
| Abschlüsse pro Tag | Mittel | 81.00 | 48.54 | -40.07 | % von echt | 2026-11-06 |
| Abschlüsse pro Tag | Median | 79.00 | 35.00 | -55.70 | % von echt | 2026-11-06 |
| Abschlüsse pro Tag | P90 | 128.00 | 102.50 | -19.92 | % von echt | 2026-11-06 |
| Abschlüsse pro Tag | Maximum | 527.00 | 162.00 | -69.26 | % von echt | 2026-11-06 |
| Kontrollfluss (Kanten) | Kanten Anzahl | 178.00 | 138.00 | -22.47 | % von echt | 2026-11-06 |
| Kontrollfluss (Kanten) | Kanten Überlappung (%) | 100.00 | 77.53 | -22.47 | % von echt | 2026-11-06 |
| Kontrollfluss (Kanten) | Kanten Jaccard (%) | 100.00 | 77.53 | -22.47 | % von echt | 2026-11-06 |
| Kontrollfluss (Kanten) | Kanten Top 20 getroffen | 20.00 | 20.00 | 0.00 | % von echt | 2026-11-06 |
| Kontrollfluss (Kanten) | Kanten Abstand der Anteile (%) | 0.00 | 7.16 | – | % von echt | 2026-11-06 |
| Trace-Varianten | Varianten Anzahl | 15930.00 | 6391.00 | -59.88 | % von echt | 2026-11-06 |
| Trace-Varianten | Varianten Überlappung (%) | 100.00 | 7.48 | -92.52 | % von echt | 2026-11-06 |
| Trace-Varianten | Varianten Jaccard (%) | 100.00 | 5.64 | -94.36 | % von echt | 2026-11-06 |
| Trace-Varianten | Varianten Top 10 getroffen | 10.00 | 9.00 | -10.00 | % von echt | 2026-11-06 |
| Trace-Varianten | Fälle mit echter Variante (%) | 100.00 | 46.81 | -53.19 | % von echt | 2026-11-06 |
| Trace-Varianten | Varianten nur complete: Anzahl | 5623.00 | 2487.00 | -55.77 | % von echt | 2026-11-06 |
| Trace-Varianten | Varianten nur complete: Jaccard (%) | 100.00 | 13.60 | -86.40 | % von echt | 2026-11-06 |
| Trace-Varianten | Fälle mit echter Variante, nur complete (%) | 100.00 | 83.58 | -16.42 | % von echt | 2026-11-06 |
| Aktivitätsverteilung | Aktivitäten Jaccard (%) | 100.00 | 96.15 | -3.85 | % von echt | 2026-11-06 |
| Aktivitätsverteilung | Aktivitäten Abstand der Anteile (%) | 0.00 | 5.53 | – | % von echt | 2026-11-06 |
| Startaktivitäten | verschiedene Aktivitäten | 1.00 | 1.00 | 0.00 | % von echt | 2026-11-06 |
| Startaktivitäten | Abstand der Anteile (%) | 0.00 | 0.00 | – | % von echt | 2026-11-06 |
| Endaktivitäten | verschiedene Aktivitäten | 12.00 | 8.00 | -33.33 | % von echt | 2026-11-06 |
| Endaktivitäten | Abstand der Anteile (%) | 0.00 | 7.18 | – | % von echt | 2026-11-06 |
| Aktivitätsdauern | Aktivitäten mit mittlerer Dauer innerhalb ±25 % oder 1 Minute | 26.00 | 19.00 | -26.92 | % von echt | 2026-11-06 |
| Aktivitätsdauern | mittlere Zeit bis zum nächsten Event (h) | 14.15 | 12.84 | -9.26 | % von echt | 2026-11-06 |
| Ressourcen | Anzahl | 149.00 | 109.00 | -26.85 | % von echt | 2026-11-06 |
| Ressourcen | Events pro Ressource, Mittel | 8068.91 | 3872.07 | -52.01 | % von echt | 2026-11-06 |
| Ressourcen | Events pro Ressource, Median | 6037.00 | 3149.00 | -47.84 | % von echt | 2026-11-06 |
| Ressourcen | Events pro Ressource, P90 | 16576.60 | 7383.80 | -55.46 | % von echt | 2026-11-06 |
| Ressourcen | Events pro Ressource, Maximum | 148404.00 | 45512.00 | -69.33 | % von echt | 2026-11-06 |
| Ressourcen | Übergaben pro Fall | 6.91 | 8.29 | 19.97 | % von echt | 2026-11-06 |
| Ressourcen | Gini der Arbeitslast (0 = gleich verteilt) | 0.49 | 0.48 | -2.04 | % von echt | 2026-11-06 |
| Mitarbeiter-Effekte | Arbeitsabschnitt: Streuung durch den Mitarbeiter (%) | 19.21 | 11.46 | -40.34 | % von echt | 2026-11-06 |
| Mitarbeiter-Effekte | Tempo je Mitarbeiter und Aktivität: Korrelation mit echt | 1.00 | 0.68 | -32.00 | % von echt | 2026-11-06 |
| Mitarbeiter-Effekte | Bewilligt je erstem Anrufer: Streuung (Std) | 0.10 | 0.03 | -70.00 | % von echt | 2026-11-06 |
| Mitarbeiter-Effekte | Bewilligt je erstem Anrufer: Korrelation mit echt | 1.00 | -0.33 | -133.00 | % von echt | 2026-11-06 |
| Zeit pro Fall nach Zustand | in Arbeit: Mittel (h) | 0.78 | 1.01 | 29.49 | % von echt | 2026-11-06 |
| Zeit pro Fall nach Zustand | in Arbeit: Median (h) | 0.42 | 0.50 | 19.05 | % von echt | 2026-11-06 |
| Zeit pro Fall nach Zustand | in Arbeit: P90 (h) | 1.43 | 1.57 | 9.79 | % von echt | 2026-11-06 |
| Zeit pro Fall nach Zustand | pausiert: Mittel (h) | 477.67 | 513.90 | 7.58 | % von echt | 2026-11-06 |
| Zeit pro Fall nach Zustand | pausiert: Median (h) | 397.65 | 431.04 | 8.40 | % von echt | 2026-11-06 |
| Zeit pro Fall nach Zustand | pausiert: P90 (h) | 810.95 | 837.53 | 3.28 | % von echt | 2026-11-06 |
| Zeit pro Fall nach Zustand | wartet auf Start: Mittel (h) | 47.04 | 14.14 | -69.94 | % von echt | 2026-11-06 |
| Zeit pro Fall nach Zustand | wartet auf Start: Median (h) | 12.86 | 6.73 | -47.67 | % von echt | 2026-11-06 |
| Zeit pro Fall nach Zustand | wartet auf Start: P90 (h) | 76.66 | 39.46 | -48.53 | % von echt | 2026-11-06 |
| Zeit pro Fall nach Zustand | gesamt: Mittel (h) | 525.59 | 529.12 | 0.67 | % von echt | 2026-11-06 |
| Zeit pro Fall nach Zustand | gesamt: Median (h) | 458.11 | 450.37 | -1.69 | % von echt | 2026-11-06 |
| Zeit pro Fall nach Zustand | gesamt: P90 (h) | 841.32 | 859.55 | 2.17 | % von echt | 2026-11-06 |
| Entscheidungspunkt | erreicht Entscheidungspunkt (%) | 97.09 | 97.23 | 0.14 | % von echt | 2026-11-06 |
| Entscheidungspunkt | Aktion am ersten Entscheidungspunkt (%) | 10.61 | 8.41 | -20.74 | % von echt | 2026-11-06 |
| Entscheidungspunkt | Outcome je Fall: Mittel | -21.90 | -22.05 | 0.68 | % von echt | 2026-11-06 |
| Entscheidungspunkt | Outcome ohne Aktion: Mittel | -21.82 | -22.02 | 0.92 | % von echt | 2026-11-06 |
| Entscheidungspunkt | Outcome mit Aktion: Mittel | -24.24 | -24.46 | 0.91 | % von echt | 2026-11-06 |
| Entscheidungspunkt | bewilligt, alle Fälle (%) | 54.68 | 58.70 | 7.35 | % von echt | 2026-11-06 |
| Entscheidungspunkt | bewilligt ohne Call (%) | 66.72 | 72.95 | 9.34 | % von echt | 2026-11-06 |
| Entscheidungspunkt | bewilligt mit Call (%) | 84.30 | 87.42 | 3.70 | % von echt | 2026-11-06 |
| Fallattribute | LoanGoal: Abstand der Anteile (%) | 0.00 | 4.57 | – | % von echt | 2026-11-06 |
| Fallattribute | ApplicationType: Abstand der Anteile (%) | 0.00 | 1.80 | – | % von echt | 2026-11-06 |
| Fallattribute | LoanGoal × ApplicationType: Abstand der Anteile (%) | 0.00 | 5.43 | – | % von echt | 2026-11-06 |
| Fallattribute | RequestedAmount: Mittel | 16233.74 | 14431.06 | -11.10 | % von echt | 2026-11-06 |
| Fallattribute | RequestedAmount: Median | 12500.00 | 10000.00 | -20.00 | % von echt | 2026-11-06 |
| Fallattribute | RequestedAmount: P90 | 35000.00 | 30000.00 | -14.29 | % von echt | 2026-11-06 |
| Fallattribute | RequestedAmount: Anteil 0 (%) | 9.77 | 9.26 | -5.22 | % von echt | 2026-11-06 |
| Fallattribute | RequestedAmount: KS-Abstand | 0.00 | 0.05 | – | % von echt | 2026-11-06 |
| Fallattribute | RequestedAmount je LoanGoal: mittl. Abweichung des Medians (%) | 0.00 | 8.64 | – | % von echt | 2026-11-06 |
| Fallattribute | LoanGoal: Abstand der Anteile je Startmonat, Mittel (%) | 0.00 | 3.09 | – | % von echt | 2026-11-06 |
| Fallattribute | ApplicationType: Anteil Limit raise (%) | 10.76 | 12.56 | 16.73 | % von echt | 2026-11-06 |
| Angebotsattribute | Angebote | 42995.00 | 13349.00 | -68.95 | % von echt | 2026-11-06 |
| Angebotsattribute | Angebote pro Fall mit Angebot | 1.36 | 1.33 | -2.21 | % von echt | 2026-11-06 |
| Angebotsattribute | Fälle mit mehreren Angeboten: Angebote verschieden (%) | 41.84 | 42.69 | 2.03 | % von echt | 2026-11-06 |
| Angebotsattribute | CreditScore: Anteil 0 (%) | 64.51 | 61.61 | -4.50 | % von echt | 2026-11-06 |
| Angebotsattribute | CreditScore > 0: Median | 904.00 | 904.00 | 0.00 | % von echt | 2026-11-06 |
| Angebotsattribute | FirstWithdrawalAmount: Anteil 0 (%) | 29.74 | 32.07 | 7.83 | % von echt | 2026-11-06 |
| Angebotsattribute | CreditScore: Median | 0.00 | 0.00 | – | % von echt | 2026-11-06 |
| Angebotsattribute | CreditScore: KS-Abstand | 0.00 | 0.03 | – | % von echt | 2026-11-06 |
| Angebotsattribute | OfferedAmount: Median | 15000.00 | 13585.71 | -9.43 | % von echt | 2026-11-06 |
| Angebotsattribute | OfferedAmount: KS-Abstand | 0.00 | 0.07 | – | % von echt | 2026-11-06 |
| Angebotsattribute | FirstWithdrawalAmount: Median | 5000.00 | 5000.00 | 0.00 | % von echt | 2026-11-06 |
| Angebotsattribute | FirstWithdrawalAmount: KS-Abstand | 0.00 | 0.05 | – | % von echt | 2026-11-06 |
| Angebotsattribute | NumberOfTerms: Median | 77.00 | 72.00 | -6.49 | % von echt | 2026-11-06 |
| Angebotsattribute | NumberOfTerms: KS-Abstand | 0.00 | 0.04 | – | % von echt | 2026-11-06 |
| Angebotsattribute | MonthlyCost: Median | 244.52 | 220.01 | -10.02 | % von echt | 2026-11-06 |
| Angebotsattribute | MonthlyCost: KS-Abstand | 0.00 | 0.07 | – | % von echt | 2026-11-06 |
| Angebotsattribute | OfferedAmount / RequestedAmount: Median | 1.00 | 1.00 | 0.00 | % von echt | 2026-11-06 |
| Angebotsattribute | OfferedAmount = RequestedAmount (%) | 67.20 | 67.01 | -0.28 | % von echt | 2026-11-06 |
| Angebotsattribute | MonthlyCost × NumberOfTerms / OfferedAmount: Median | 1.17 | 1.17 | 0.00 | % von echt | 2026-11-06 |
| Angebotsattribute | Selected: Anteil (%) | 50.63 | 51.11 | 0.95 | % von echt | 2026-11-06 |
| Angebotsattribute | Accepted: Anteil (%) | 70.09 | 71.11 | 1.46 | % von echt | 2026-11-06 |
| Angebotsattribute | Fall bewilligt, wenn Accepted (%) | 62.10 | 65.17 | 4.94 | % von echt | 2026-11-06 |
| Angebotsattribute | Fall bewilligt, wenn nicht Accepted (%) | 42.27 | 46.78 | 10.67 | % von echt | 2026-11-06 |
| Endaktivität (Anteil der Fälle, %) | W_Validate application | 40.19 | 47.36 | 7.17 | Prozentpunkte | 2026-11-06 |
| Endaktivität (Anteil der Fälle, %) | W_Call after offers | 30.01 | 28.31 | -1.70 | Prozentpunkte | 2026-11-06 |
| Endaktivität (Anteil der Fälle, %) | W_Call incomplete files | 14.84 | 9.97 | -4.87 | Prozentpunkte | 2026-11-06 |
| Endaktivität (Anteil der Fälle, %) | O_Cancelled | 14.08 | 13.94 | -0.14 | Prozentpunkte | 2026-11-06 |
| Endaktivität (Anteil der Fälle, %) | W_Complete application | 0.46 | 0.27 | -0.19 | Prozentpunkte | 2026-11-06 |
| Endaktivität (Anteil der Fälle, %) | W_Assess potential fraud | 0.32 | 0.13 | -0.19 | Prozentpunkte | 2026-11-06 |

### Effekte an den Wahlpunkten: echter Log gegen Simulator

echt = AIPW auf dem echten Log; simulator = AIPW auf beobachtet simulierten Daten (Bank-Politik); simulator_zufall = wahrer Effekt bei zufälliger Wahl; abweichung = Anteil der echten Effektgröße, den der Simulator hat; z_echt_gegen_wahr = Unterschied echt − Simulator in Standardfehlern.

| bereich | echt | echt_se | simulator | simulator_zufall | simulator_zufall_se | abweichung | z_echt_gegen_wahr | ueberlappung_echt | ueberlappung_simulator | stand |
|---|---|---|---|---|---|---|---|---|---|---|
| handle_leads | -2.16 | 0.30 | -1.53 | -1.38 | 0.25 | 0.64 | -2.00 | 0.97 | 1.00 | 2026-11-06 |
| annehmen_nach_vervollstaendigung | 2.96 | 0.27 | 2.69 | 1.91 | 0.26 | 0.65 | 2.78 | 0.92 | 0.91 | 2026-11-06 |
| angebot_nachlegen | -1.42 | 0.32 | -1.62 | -1.64 | 0.20 | 1.16 | 0.59 | 0.80 | 0.75 | 2026-11-06 |
| annehmen_bei_konzept | 5.23 | 0.29 | 3.16 | 3.36 | 0.23 | 0.64 | 5.08 | 0.56 | 0.56 | 2026-11-06 |

RCT im Simulator (14.11.): Vorteil der Regel-Option und Zerlegung in Endzustand (die Option ändert, wie der Fall endet: Pending/Cancelled/Denied) und Dauer (bei gleichem Ende); Formel wie psp_zerlegung.py.

| bereich | simulator_zufall | simulator_zufall_se | endzustand_anteil | dauer_anteil | ende_regel | ende_andere |
|---|---|---|---|---|---|---|
| annehmen_bei_konzept | 2.79 | 0.32 | 0.42 | 2.36 | A_Pending 56%, A_Cancelled 32%, A_Denied 11%, ohne 1% | A_Pending 53%, A_Cancelled 36%, A_Denied 11%, ohne 1% |
| annehmen_nach_vervollstaendigung | 2.40 | 0.30 | 0.29 | 2.11 | A_Pending 57%, A_Cancelled 32%, A_Denied 11%, ohne 1% | A_Pending 54%, A_Cancelled 35%, A_Denied 11%, ohne 1% |
| handle_leads | 1.32 | 0.29 | -0.02 | 1.35 | A_Pending 54%, A_Cancelled 35%, A_Denied 11%, ohne 0% | A_Pending 53%, A_Cancelled 35%, A_Denied 12%, ohne 0% |
| nachlegen 1. Besuch | 2.11 | 0.23 | 0.70 | 1.41 | A_Pending 58%, A_Cancelled 30%, A_Denied 12%, ohne 0% | A_Pending 54%, A_Cancelled 35%, A_Denied 10%, ohne 1% |
| nachlegen ab 2. Besuch | 0.42 | 0.22 | 0.19 | 0.24 | A_Pending 55%, A_Cancelled 34%, A_Denied 10%, ohne 1% | A_Pending 52%, A_Cancelled 36%, A_Denied 10%, ohne 1% |

### Je Wahlpunkt (echt gegen Simulator mit Bank-Politik und mit Zufallswahl)

„Lage unterscheidbar“: AUC eines Klassifikators echt gegen simuliert auf den Zuständen am Wahlpunkt (0,5 = nicht unterscheidbar); genannt ist das wichtigste Merkmal. „Wirkung … im/außerhalb des Trägers“: Effekt dort, wo die Wahl auch im echten Log vorkommt, bzw. außerhalb.

| bereich | kennzahl | echt | simulator | simulator_zufall | abweichung | abweichung_als | bewertung |
|---|---|---|---|---|---|---|---|
| Outcome | Outcome je Fall: Mittel | -21.90 | -22.05 | -22.20 | -0.67 | % von echt (Bank) | ok |
| Outcome | Outcome je Fall: Median | -19.09 | -18.77 | -18.86 | 1.69 | % von echt (Bank) | ok |
| Outcome | Outcome: schlechteste 10 % (CVaR) | -46.56 | -49.17 | -50.14 | -5.60 | % von echt (Bank) | ok |
| Outcome | Outcome: Abstand der Verteilung (Wasserstein / Std echt) | 0.00 | 0.04 | 0.05 | 0.04 | Std echt (Bank) | ok |
| Outcome | Durchlaufzeit: Mittel (Tage) | 21.90 | 22.05 | 22.20 | 0.67 | % von echt (Bank) | ok |
| Outcome | Durchlaufzeit: P90 (Tage) | 35.05 | 35.81 | 36.98 | 2.17 | % von echt (Bank) | ok |
| Outcome | Durchlaufzeit: P95 (Tage) | 42.52 | 44.29 | 46.38 | 4.16 | % von echt (Bank) | ok |
| Control Flow: angebot_nachlegen | Fälle mit Wahl (%) | 97.09 | 97.23 | 97.57 | 0.14 | Prozentpunkte Bank − echt | ok |
| Control Flow: angebot_nachlegen | Wahlen je Fall mit Wahl | 1.29 | 1.26 | 2.21 | -2.23 | % von echt (Bank) | ok |
| Control Flow: angebot_nachlegen | Zeitpunkt der Wahl: Median (Tage seit Fallbeginn) | 1.07 | 0.98 | 0.87 | -8.38 | % von echt (Bank) | ok |
| Control Flow: angebot_nachlegen | Lage unterscheidbar: AUC echt gegen Bank (0,5 = gleich) | 0.50 | 0.76 | – | 0.26 | AUC − 0,5; v. a. now:MonthlyCost (25 %) | kritisch |
| Control Flow: angebot_nachlegen | Anteil O_Sent (mail and online) (%) | 90.16 | 92.21 | 50.20 | 2.05 | Prozentpunkte Bank − echt | ok |
| Control Flow: angebot_nachlegen | Anteil O_Create Offer (%) | 9.84 | 7.79 | 49.80 | -2.05 | Prozentpunkte Bank − echt | ok |
| Control Flow: angebot_nachlegen | Wahlen außerhalb des Trägers des echten Logs (%) | 0.59 | 0.51 | 9.47 | -0.09 | Prozentpunkte Bank − echt | ok |
| Control Flow: angebot_nachlegen | Kontrast O_Create Offer − O_Sent (mail and online): beobachtet (Zufall: randomisiert) | -1.69 | -1.04 | -0.76 | 0.65 | Differenz Bank − echt | ok |
| Control Flow: angebot_nachlegen | Zufallswahlen O_Create Offer außerhalb des Trägers (%) | – | – | 19.02 | – | – | – |
| Control Flow: angebot_nachlegen | Wirkung O_Create Offer − O_Sent (mail and online) im Träger (echt: IPW, Zufall: randomisiert) | -1.40 | – | -0.83 | 0.57 | Differenz Zufall − echt | ok |
| Control Flow: angebot_nachlegen | Wirkung O_Create Offer − O_Sent (mail and online) außerhalb des Trägers (Zufall) | – | – | -0.38 | 0.45 | Differenz außerhalb − im Träger | ok |
| Control Flow: annehmen_bei_konzept | Fälle mit Wahl (%) | 77.11 | 72.77 | 54.12 | -4.34 | Prozentpunkte Bank − echt | ok |
| Control Flow: annehmen_bei_konzept | Wahlen je Fall mit Wahl | 1.00 | 1.00 | 1.00 | 0.14 | % von echt (Bank) | ok |
| Control Flow: annehmen_bei_konzept | Zeitpunkt der Wahl: Median (Tage seit Fallbeginn) | 0.00 | 0.00 | 0.00 | 0.01 | % von echt (Bank) | ok |
| Control Flow: annehmen_bei_konzept | Lage unterscheidbar: AUC echt gegen Bank (0,5 = gleich) | 0.50 | 0.80 | – | 0.30 | AUC − 0,5; v. a. elapsed_hours (48 %) | kritisch |
| Control Flow: annehmen_bei_konzept | Anteil W_Complete application/start (%) | 62.01 | 69.05 | 49.93 | 7.05 | Prozentpunkte Bank − echt | prüfen |
| Control Flow: annehmen_bei_konzept | Anteil A_Accepted (%) | 37.99 | 30.95 | 50.07 | -7.05 | Prozentpunkte Bank − echt | prüfen |
| Control Flow: annehmen_bei_konzept | Wahlen außerhalb des Trägers des echten Logs (%) | 0.01 | 0.00 | 18.94 | -0.01 | Prozentpunkte Bank − echt | ok |
| Control Flow: annehmen_bei_konzept | Kontrast A_Accepted − W_Complete application/start: beobachtet (Zufall: randomisiert) | 3.36 | 3.02 | 2.76 | -0.34 | Differenz Bank − echt | ok |
| Control Flow: annehmen_bei_konzept | Zufallswahlen A_Accepted außerhalb des Trägers (%) | – | – | 0.18 | – | – | – |
| Control Flow: annehmen_bei_konzept | Wirkung A_Accepted − W_Complete application/start im Träger (echt: IPW, Zufall: randomisiert) | 3.93 | – | 3.46 | -0.46 | Differenz Zufall − echt | ok |
| Control Flow: annehmen_bei_konzept | Wirkung A_Accepted − W_Complete application/start außerhalb des Trägers (Zufall) | – | – | 1.56 | -1.90 | Differenz außerhalb − im Träger | prüfen |
| Control Flow: annehmen_nach_vervollstaendigung | Fälle mit Wahl (%) | 52.12 | 58.16 | 59.16 | 6.04 | Prozentpunkte Bank − echt | prüfen |
| Control Flow: annehmen_nach_vervollstaendigung | Wahlen je Fall mit Wahl | 1.94 | 1.99 | 1.59 | 2.56 | % von echt (Bank) | ok |
| Control Flow: annehmen_nach_vervollstaendigung | Zeitpunkt der Wahl: Median (Tage seit Fallbeginn) | 0.99 | 0.87 | 0.50 | -12.08 | % von echt (Bank) | prüfen |
| Control Flow: annehmen_nach_vervollstaendigung | Lage unterscheidbar: AUC echt gegen Bank (0,5 = gleich) | 0.50 | 0.63 | – | 0.13 | AUC − 0,5; v. a. elapsed_hours (21 %) | prüfen |
| Control Flow: annehmen_nach_vervollstaendigung | Anteil W_Complete application/resume (%) | 70.27 | 72.56 | 50.76 | 2.30 | Prozentpunkte Bank − echt | ok |
| Control Flow: annehmen_nach_vervollstaendigung | Anteil A_Accepted (%) | 29.73 | 27.43 | 49.24 | -2.30 | Prozentpunkte Bank − echt | ok |
| Control Flow: annehmen_nach_vervollstaendigung | Wahlen außerhalb des Trägers des echten Logs (%) | 0.00 | 0.03 | 2.97 | 0.03 | Prozentpunkte Bank − echt | ok |
| Control Flow: annehmen_nach_vervollstaendigung | Kontrast A_Accepted − W_Complete application/resume: beobachtet (Zufall: randomisiert) | 1.21 | 1.90 | 2.18 | 0.68 | Differenz Bank − echt | ok |
| Control Flow: annehmen_nach_vervollstaendigung | Zufallswahlen A_Accepted außerhalb des Trägers (%) | – | – | 6.04 | – | – | – |
| Control Flow: annehmen_nach_vervollstaendigung | Wirkung A_Accepted − W_Complete application/resume im Träger (echt: IPW, Zufall: randomisiert) | 2.33 | – | 2.24 | -0.10 | Differenz Zufall − echt | ok |
| Control Flow: annehmen_nach_vervollstaendigung | Wirkung A_Accepted − W_Complete application/resume außerhalb des Trägers (Zufall) | – | – | 1.27 | -0.96 | Differenz außerhalb − im Träger | ok |
| Control Flow: handle_leads | Fälle mit Wahl (%) | 64.82 | 64.67 | 64.67 | -0.15 | Prozentpunkte Bank − echt | ok |
| Control Flow: handle_leads | Wahlen je Fall mit Wahl | 1.00 | 1.00 | 1.00 | -0.11 | % von echt (Bank) | ok |
| Control Flow: handle_leads | Zeitpunkt der Wahl: Median (Tage seit Fallbeginn) | 0.00 | 0.00 | 0.00 | 0.00 | % von echt (Bank) | ok |
| Control Flow: handle_leads | Lage unterscheidbar: AUC echt gegen Bank (0,5 = gleich) | 0.50 | 0.79 | – | 0.28 | AUC − 0,5; v. a. elapsed_hours (78 %) | kritisch |
| Control Flow: handle_leads | Anteil W_Handle leads/withdraw (%) | 82.18 | 80.78 | 49.27 | -1.40 | Prozentpunkte Bank − echt | ok |
| Control Flow: handle_leads | Anteil W_Handle leads/start (%) | 17.82 | 19.22 | 50.73 | 1.40 | Prozentpunkte Bank − echt | ok |
| Control Flow: handle_leads | Wahlen außerhalb des Trägers des echten Logs (%) | 0.00 | 0.08 | 0.06 | 0.08 | Prozentpunkte Bank − echt | ok |
| Control Flow: handle_leads | Kontrast W_Handle leads/start − W_Handle leads/withdraw: beobachtet (Zufall: randomisiert) | -2.26 | -1.78 | -1.03 | 0.48 | Differenz Bank − echt | ok |
| Control Flow: handle_leads | Zufallswahlen W_Handle leads/start außerhalb des Trägers (%) | – | – | 0.12 | – | – | – |
| Control Flow: handle_leads | Wirkung W_Handle leads/start − W_Handle leads/withdraw im Träger (echt: IPW, Zufall: randomisiert) | -2.21 | – | -1.02 | 1.19 | Differenz Zufall − echt | ok |
| Control Flow: handle_leads | Wirkung W_Handle leads/start − W_Handle leads/withdraw außerhalb des Trägers (Zufall) | – | – | 0.08 | 1.11 | Differenz außerhalb − im Träger | ok |

## Trainingsläufe (aus data/PSP_*.log; Einzelwerte je Epoche in verlustkurven.csv)

| log | lauf | epochen | val_std_letzte | val_std_min | val_att_letzte |
|---|---|---|---|---|---|
| k3_zaehler_zeit_tf1_t_aktion | 1 | 10 | 11.84 | 11.74 | 438.40 |
| k3_zaehler_zeit_tf1_t_referenz | 1 | 5 | 15.01 | 14.69 | 553.18 |
| k3x4_zaehler_zeit_tf1_t_aktion | 1 | 10 | 12.12 | 12.04 | 466.23 |
| k3x4_zaehler_zeit_tf1_t_referenz | 1 | 5 | 14.84 | 14.75 | 484.26 |
| k3x4ol_konzept_annehmen | 1 | 10 | 11.57 | 11.49 | 432.77 |
| k3x4ol_konzept_bearbeiten | 1 | 5 | 16.76 | 16.52 | 512.94 |
| k3x4ol_leads_rueckruf | 1 | 10 | 13.72 | 13.26 | 419.00 |
| k3x4ol_leads_zurueck | 1 | 5 | 15.48 | 15.29 | 470.84 |
| k3x4ol_nachlegen_senden | 1 | 5 | 13.03 | 12.91 | 522.07 |
| k3x4ol_nachlegen_weiteres | 1 | 10 | 11.80 | 11.66 | 574.98 |
| k3x4ol_nachv_annehmen | 1 | 10 | 12.08 | 12.00 | 501.89 |
| k3x4ol_nachv_weiter | 1 | 5 | 15.93 | 15.93 | 499.78 |
| lr1e4 | 1 | 25 | 2.72 | 2.70 | 5.07 |
| oe_tf1_100 | 1 | 26 | 13.83 | 10.26 | 392.36 |
| ol_konzept_annehmen | 1 | 40 | 11.23 | 11.08 | 398.21 |
| ol_konzept_annehmen_sv | 1 | 10 | 11.20 | 11.10 | 399.05 |
| ol_konzept_bearbeiten | 1 | 20 | 17.19 | 16.65 | 519.40 |
| ol_konzept_bearbeiten_sv | 1 | 3 | 17.29 | 17.16 | 568.77 |
| ol_konzept_bearbeiten_sv | 2 | 5 | 17.35 | 17.16 | 523.44 |
| ol_leads_rueckruf | 1 | 40 | 15.96 | 14.73 | 651.34 |
| ol_leads_rueckruf_sv | 1 | 10 | 15.86 | 14.86 | 571.34 |
| ol_leads_zurueck | 1 | 20 | 15.94 | 15.32 | 468.28 |
| ol_leads_zurueck_sv | 1 | 5 | 16.21 | 15.97 | 474.81 |
| ol_nachlegen_senden | 1 | 20 | 12.21 | 12.18 | 471.34 |
| ol_nachlegen_senden_sv | 1 | 5 | 12.20 | 12.16 | 473.59 |
| ol_nachlegen_weiteres | 1 | 40 | 13.72 | 13.21 | 668.12 |
| ol_nachlegen_weiteres_sv | 1 | 10 | 13.30 | 13.18 | 644.10 |
| ol_nachv_annehmen | 1 | 40 | 11.33 | 11.20 | 491.60 |
| ol_nachv_annehmen_sv | 1 | 10 | 11.24 | 11.21 | 477.47 |
| ol_nachv_weiter | 1 | 20 | 16.46 | 16.26 | 526.35 |
| ol_nachv_weiter_sv | 1 | 5 | 16.44 | 16.36 | 544.99 |
| pilot_klassen_tf0 | 1 | 6 | 12.86 | 12.18 | 38.33 |
| pilot_klassen_tf1 | 1 | 6 | 13.60 | 12.18 | 405.28 |
| pilot_normal_tf1 | 1 | 6 | 10.70 | 10.44 | 379.71 |
| pilot_token_tf0 | 1 | 6 | 12.14 | 10.75 | 37.70 |
| pilot_token_tf05 | 1 | 6 | 11.87 | 10.75 | 62.53 |
| pilot_token_tf05_kontext | 1 | 6 | 11.62 | 10.52 | 66.65 |
| pilot_token_tf08 | 1 | 6 | 11.68 | 10.46 | 157.93 |
| pilot_token_tf1 | 1 | 6 | 11.87 | 11.03 | 425.93 |
| pilot_token_tf1_c6000 | 1 | 2 | 9.88 | 9.88 | 261.89 |
| pilot_token_tf1_kontext | 1 | 6 | 11.86 | 11.28 | 366.39 |
| pl_konzept | 1 | 20 | 14.87 | 14.55 | 463.58 |
| pl_konzept_sv | 1 | 5 | 14.95 | 14.80 | 479.65 |
| pl_leads | 1 | 20 | 15.79 | 15.38 | 497.19 |
| pl_leads_sv | 1 | 5 | 16.07 | 15.79 | 531.13 |
| pl_nachlegen | 1 | 20 | 12.57 | 12.25 | 538.04 |
| pl_nachlegen_sv | 1 | 5 | 12.15 | 12.15 | 478.20 |
| pl_nachv | 1 | 20 | 15.09 | 14.88 | 517.78 |
| pl_nachv_sv | 1 | 5 | 15.20 | 15.10 | 537.98 |
| rct_zaehler_zeit_tf1 | 1 | 15 | 12.58 | 9.52 | 404.96 |
| rct_zaehler_zeit_tf1_t_aktion | 1 | 40 | 12.77 | 12.63 | 537.71 |
| rct_zaehler_zeit_tf1_t_referenz | 1 | 20 | 13.25 | 13.10 | 532.36 |
| real_alle_s60_klassen | 1 | 15 | 11.61 | 10.22 | 25.50 |
| real_alle_s60_klassen_event_oe | 1 | 15 | 9.64 | 9.37 | 26.34 |
| real_alle_s60_klassen_tf1_option_encoder | 1 | 15 | 17.33 | 12.47 | 384.37 |
| real_alle_s60_loggaps | 1 | 15 | 8.47 | 7.73 | 45.38 |
| real_alle_s60_loggaps_rest | 1 | 15 | 8.16 | 7.75 | 48.46 |
| real_alle_s60_loggaps_rest_abgebrochen | 1 | 2 | 8.56 | 8.54 | 135.51 |
| real_alle_s60_normal | 1 | 15 | 9.20 | 8.85 | 24.95 |
| real_alle_s60_normal_accepted | 1 | 6 | 9.54 | 9.54 | 67.50 |
| real_alle_s60_normal_tf1 | 1 | 15 | 13.13 | 9.59 | 372.47 |
| real_alle_s60_normal_zaehler_uhren_tf1 | 1 | 15 | 12.96 | 9.61 | 423.55 |
| real_alle_s60_normal_zaehler_uhren_tf1_bnll05 | 1 | 3 | 14.10 | 13.62 | 30.76 |
| real_alle_s60_normal_zaehler_zeit | 1 | 15 | 9.14 | 8.99 | 296.02 |
| real_alle_s60_normal_zaehler_zeit_tf1 | 1 | 15 | 12.87 | 9.48 | 401.55 |
| real_alle_s60_normal_zaehler_zeit_tf1_restzeit | 1 | 5 | 13.26 | 13.00 | 400.02 |
| real_alle_s60_normal_zaehler_zeit_tf1_t_aktion | 1 | 40 | 12.25 | 12.14 | 499.52 |
| real_alle_s60_normal_zaehler_zeit_tf1_t_aktion_kopf_steigung | 1 | 30 | 11.24 | 11.18 | 360.48 |
| real_alle_s60_normal_zaehler_zeit_tf1_t_aktion_kopf_steigung_bnll1 | 1 | 30 | 11.60 | 11.41 | 10.22 |
| real_alle_s60_normal_zaehler_zeit_tf1_t_aktion_kopf_versatz | 1 | 30 | 12.30 | 12.19 | 518.32 |
| real_alle_s60_normal_zaehler_zeit_tf1_t_aktion_sv | 1 | 10 | 12.27 | 12.21 | 492.34 |
| real_alle_s60_normal_zaehler_zeit_tf1_t_aktion_svh | 1 | 10 | 12.25 | 12.13 | 484.86 |
| real_alle_s60_normal_zaehler_zeit_tf1_t_aktion_svk | 1 | 10 | 12.93 | 12.23 | 530.25 |
| real_alle_s60_normal_zaehler_zeit_tf1_t_aktion_svp | 1 | 10 | 12.15 | 12.15 | 491.45 |
| real_alle_s60_normal_zaehler_zeit_tf1_t_referenz | 1 | 20 | 15.21 | 14.84 | 501.44 |
| real_alle_s60_normal_zaehler_zeit_tf1_t_referenz_kopf_steigung | 1 | 12 | 14.20 | 14.20 | 303.32 |
| real_alle_s60_normal_zaehler_zeit_tf1_t_referenz_kopf_steigung_bnll1 | 1 | 12 | 15.09 | 14.80 | 13.12 |
| real_alle_s60_normal_zaehler_zeit_tf1_t_referenz_kopf_versatz | 1 | 12 | 15.18 | 15.18 | 513.50 |
| real_alle_s60_normal_zaehler_zeit_tf1_t_referenz_sv | 1 | 5 | 15.29 | 15.18 | 518.09 |
| real_alle_s60_normal_zaehler_zeit_tf1_t_referenz_svh | 1 | 5 | 15.33 | 15.27 | 515.94 |
| real_alle_s60_normal_zaehler_zeit_tf1_t_referenz_svk | 1 | 5 | 15.51 | 15.34 | 529.22 |
| real_alle_s60_normal_zaehler_zeit_tf1_t_referenz_svp | 1 | 5 | 15.40 | 15.13 | 494.86 |
| real_alle_s60_normal_zaehler_zeit_tf1_wp | 1 | 18 | 10.91 | 8.67 | 372.65 |
| real_alle_s60_normal_zaehler_zeit_tf1_wp150 | 1 | 93 | 11.31 | 9.02 | 432.59 |
| real_alle_s60_token_oe | 1 | 15 | 9.60 | 9.27 | 24.19 |
| real_np_s40_ohne_zukunft | 1 | 20 | 7.86 | 7.84 | 21.81 |
| real_w3_10k | 1 | 100 | 2.80 | 2.79 | 3.30 |
| real_w3_10k_np | 1 | 100 | 2.82 | 2.81 | 4.26 |
| w3_10k | 1 | 100 | 2.73 | 2.72 | 7.10 |
