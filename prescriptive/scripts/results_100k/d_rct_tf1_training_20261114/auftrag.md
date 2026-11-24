# D mit TF 1 auf RCT-Daten

Zusätzlicher Trainingslauf mit denselben D-TF1-Einstellungen wie beim echten Log: 15 Epochen, Teacher Forcing durchgehend 1,0, Lernrate 0,0001, Decoderlänge 60, Batchgröße 256, Wahlpunktgewicht 3, Seed 42, statische Merkmale ohne Zukunftswissen. Event- und Zeitklasse getrennt; Optionsmodus Encoder erst nach fit setzen; Abstand je Suffix.

Neue RCT-Daten werden mit dem vorhandenen Simulator an angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung und handle_leads erzeugt. Katalog und Ressourcenstrategie entsprechen der eingefrorenen Trainingskonfiguration des echten D-TF1. Wie in der früheren RCT-Pipeline bleibt ein Fall mit Wahrscheinlichkeit 0,5 bei der Bank; sonst werden seine Optionen an den vier Punkten randomisiert.

10 unabhängige Simulationsläufe zu je 3.151 Fällen ergeben 31.510 Fälle. Davon werden mit Seed 42 31.509 ausgewählt, damit der Umfang dem echten Log entspricht: 20.482 Training, 4.726 Validierung, 6.301 Test. Der Split erfolgt vollständig nach Fällen.

RCT-Erzeugung läuft auf der CPU; danach startet das GPU-Training bei ausreichendem Speicher automatisch. Vorhandene Standardmodelle, Kataloge und der laufende Echtlog-Trainingslauf werden nicht überschrieben.

Status: status.json; Protokoll: pipeline.log; Datenprüfung: data_validation.json; Modell: /home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/models/psp_model_durchlaufzeit_rct_s60_klassen_tf1_option_encoder_20261114.pkl; Abschlussprüfung: model_validation.json.
