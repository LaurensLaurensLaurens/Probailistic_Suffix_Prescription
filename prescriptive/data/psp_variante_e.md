# PSP-Variante (e): Event, bedingte Zeitklasse, Abstand

Variante (e) übernimmt (d) mit Optionsabstand je Suffixziehung. Die einzige Änderung
an der Vorhersage ist ein zusätzlicher Eingang des Zeitklassen-Kopfs:

1. Aktivität und Lifecycle als gemeinsames Event-Token ziehen.
2. Die Zeitklasse aus Decoderzustand **und diesem aktuellen Event-Token** vorhersagen
   und ziehen. Bei einer vorgegebenen Option wird auf deren Token bedingt.
3. Den Abstand wie bei (d) aus der Normalverteilung des stetigen Zeitkopfs ziehen,
   trunkiert auf die Grenzen der gezogenen Klasse. Die stetigen Köpfe erhalten
   keine zusätzlichen Eingänge.

Die getrennten Ausgabeköpfe, neun Zeitklassen, Rückgabe der gezogenen Events und
Zeiten ins Netz, MC-Dropout, gemeinsame Zufallszahlen der Optionen und Berechnung
der Durchlaufzeit bleiben bestehen. Je Suffix wird ein Optionsabstand gezogen;
Option samt Abstand und Zeitklasse werden im Encoder-Prefix eingesetzt, danach
folgt die Fortsetzung. Der Abstand zählt einmal. Übergangssperren und ein zwischen
Suffixen geteilter Optionsabstand sind im Standardaufruf nicht aktiviert.

Im Training und im Validierungsverlust erhält der Zeitklassen-Kopf das echte
aktuelle Ziel-Event. Das ist unabhängig vom Teacher Forcing der vorherigen Events.
Beim autoregressiven Feedback wird die Klasse hingegen zum vorhergesagten Event
berechnet; bei der Suffixziehung zum tatsächlich gezogenen Event. Der Event-Kopf
erhält keine Information über das aktuelle Ziel-Event oder dessen Klasse.

`gap_condition_on_event=True` aktiviert (e) und schließt `gap_classes=True` ein.
Das kann nicht nachträglich an einem trainierten (d)-Modell eingeschaltet werden:
Das zusätzliche Event-Embedding und die erweiterten Zeitklassen-Köpfe müssen
trainiert werden. Bestehende Modelle bleiben ohne diese Einstellung unverändert.

## Training und Diagnose

Aus dem Projektverzeichnis:

```bash
bash prescriptive/data/bpic17_psp_real_stufe2e.sh
```

Wie beim (d)-Training: alle Trainingsfälle, gleicher Fall-Split und Seed 42,
15 Epochen, Lernrate 1e-4, Decoderfenster 60, Wahlpunktgewicht 3, statische Merkmale
ohne Zukunftswissen und Durchlaufzeit aus der Summe der Abstände. Die Wahlpunkte
kommen aus `catalog_neue_punkte.yaml`, wie im (d)-Trainingsskript.

Der Optionsmodus `encoder` wird erst nach `fit` gesetzt, wie bei der späteren
(d)-Modellkopie. Damit ändert sich die Gewichtung der Trainingsproben nicht
zusätzlich. CLI: `--gap-condition-on-event --option-in-encoder-after-training`.

Standardmodell: `prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_klassen_event_oe.pkl`.
Eigene Checkpoints und Diagnose verwenden denselben Tag. Mit `TAG` und `EPOCHS`
können Modellname und Epochenzahl geändert werden; bestehende Modelle und
Checkpointordner werden nicht überschrieben. Das Skript trainiert und erstellt die
Offline-Diagnose. Einen Online-Vergleich mit 300 Ziehungen/1.000 Fällen startet es
nicht.
