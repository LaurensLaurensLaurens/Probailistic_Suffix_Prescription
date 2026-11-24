# D mit TF 1: Anschlusslauf über 100 Epochen

Neues Training von Anfang an mit 100 Epochen auf dem echten BPIC17-Log. Gleiche D-Konfiguration wie im vorangehenden 15-Epochen-Lauf: TF durchgehend 1; 20.482 Trainingsfälle, 4.726 Validierungsfälle, 6.301 unveränderte Testfälle außerhalb des Trainings; Seed 42, Start-Lernrate 0,0001, Batchgröße 256, Decoderlänge 60, Wahlpunktgewicht 3. Option im Encoder nach dem Training, Optionsabstand je Suffix.

Start erst nach vollständigem und geprüftem Abschluss des bisherigen Echtlog-Laufs. Der RCT-Lauf wird weiterlaufen gelassen; zusätzlich mindestens 7.000 MiB freier GPU-Speicher und 18 GiB verfügbarer Arbeitsspeicher. Eigenes Modell und eigene Checkpoints.

100 Epochen werden vollständig trainiert. Zusätzlich zum letzten Stand wird die Epoche mit kleinstem Standard-Validierungsverlust gespeichert. Das ist die beste Epoche nach diesem Validierungskriterium, keine Zusage des besten Online-Outcomes. Der Trainingsalgorithmus und die bestehenden Quelldateien werden nicht geändert. Anschließend Parameter-/Ziehungsprüfung und die übliche Diagnose mit 100 Entscheidungen je Wahlpunkt und 30 Ziehungen je Option auf dem besten Validierungscheckpoint.
