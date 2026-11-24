# OE: 100 Epochen mit durchgehend TF 1

Neues Training von Anfang an auf dem echten BPIC17-Log. Einstellungen wie im gestoppten D-Lauf: 100 Epochen, Teacher Forcing 1, Start-Lernrate 0,0001, Batch 256, Decoderlänge 60, Wahlpunktgewicht 3, Seed 42, 20.482 Trainingsfälle, 4.726 Validierungsfälle und dieselben 6.301 ausgeschlossenen Testfälle. Statische Merkmale ohne Zukunftswissen, Zeit aus Eventabständen.

Architekturänderung: gemeinsames Token für Aktivität, Lifecycle und Zeitklasse (OE), statt getrennter D-Köpfe. Option im Encoder nach dem Training; individueller Optionsabstand je Suffix. Die Trainingsproben werden wie bei D gewichtet. Das ältere OE gewichtete zusätzliche Entscheidungs-Prefixe; diese zusätzliche Änderung wird hier nicht übernommen.

Start jetzt bei ausreichenden Ressourcen. D und RCT werden nicht neu gestartet. Eigenes Modell und eigene Checkpoints. Alle 100 Epochen laufen; beste Standard-Validierungsepoche und letzter Stand werden getrennt gespeichert. Anschließend Prüfung beider Modelle und die übliche Diagnose des besten Validierungsstands mit 100 Entscheidungen je Wahlpunkt und 30 Ziehungen je Option. Sitzungsregeln und Simulatorcode werden mit diesem Training nicht geändert.
