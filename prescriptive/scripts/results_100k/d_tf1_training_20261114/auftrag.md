# D mit durchgehendem Teacher Forcing 1

Neues Training der korrigierten D-Variante: getrennte Event- und Zeitklassen-Köpfe, Option im Encoder, Optionsabstand je Suffixziehung. Teacher Forcing bleibt in allen 15 Epochen bei 1,0.

Alle 25.208 Fälle außerhalb des bisherigen Testsplits werden für Training/Validierung verwendet (20.482 Training, 4.726 Validierung; 6.301 Testfälle ausgeschlossen). Seed 42, Decoderlänge 60, Lernrate 0,0001, Wahlpunktgewicht 3, Batchgröße 256. Statische Merkmale ohne Zukunftswissen; Zeit aus der Summe der Eventabstände.

Der Optionsmodus wird nach dem Training gesetzt, damit die Probengewichtung der bisherigen D-Variante entspricht. Danach werden Parameter, Fall-Split, Teacher-Forcing-Protokoll und Optionsziehung geprüft. Es folgt die übliche Diagnose auf echten Testentscheidungen: 100 je Wahlpunkt, 30 Ziehungen je Option.

Neues Modell: `/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_klassen_tf1_option_encoder.pkl`. Status: `status.json`; Fortschritt: `pipeline.log`; nach Abschluss: `model_validation.json`, `diagnose/diagnose.csv`, `auswertung.md`.
