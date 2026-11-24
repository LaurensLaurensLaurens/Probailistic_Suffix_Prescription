# Trainingsdaten und Lernraten der PSP-Modelle

Ausgelesen aus params und summary der gespeicherten Modelle. Fallzahlen trennen Training und Validierung. Lernrate bezeichnet den Startwert.

| Variante | Training | Validierung | Start-Lernrate | Epochen | Decoderlänge | TF-Untergrenze |
|---|---:|---:|---:|---:|---:|---:|
| Ursprüngliches Echtlog-PSP | 10.001 | 2.307 | 0,00001 | 100 | 5 | 0.0 |
| Echtlog, neue Wahlpunkte | 10.001 | 2.307 | 0,00001 | 100 | 5 | 0.0 |
| Früher Nachtlauf | 5.001 | 1.153 | 0,00010 | 25 | 5 | 0.0 |
| s40 | 10.001 | 2.307 | 0,00010 | 20 | 40 | 0.0 |
| a | 20.482 | 4.726 | 0,00010 | 15 | 60 | 0.0 |
| b | 20.482 | 4.726 | 0,00010 | 15 | 60 | 0.0 |
| c | 20.482 | 4.726 | 0,00010 | 15 | 60 | 0.0 |
| d | 20.482 | 4.726 | 0,00010 | 15 | 60 | 0.0 |
| OE | 20.482 | 4.726 | 0,00010 | 15 | 60 | 0.0 |
| c TF1 | 20.482 | 4.726 | 0,00010 | 15 | 60 | 1.0 |
| e | 20.482 | 4.726 | 0,00010 | 15 | 60 | 0.0 |
| Pilot Klassen TF1 | 3.000 | 692 | 0,00010 | 6 | 60 | 1.0 |
| Pilot Token TF1 | 3.000 | 692 | 0,00010 | 6 | 60 | 1.0 |
| Pilot Normal TF1 | 3.000 | 692 | 0,00010 | 6 | 60 | 1.0 |

Die Vollmodelle (a), (b), (c), (d), OE, (c) TF1 und (e) verwenden denselben Pool aus 25.208 Fällen: 20.482 Training und 4.726 Validierung; weitere 6.301 Fälle bleiben im äußeren Testsplit.

Die alten Bezeichnungen 5.000 bzw. 10.000 Fälle sind konfigurierte Sollzahlen; die gespeicherte Trainingszusammenfassung weist 5.001 bzw. 10.001 Trainingsfälle aus.

Der neue D-TF1-Lauf ist für denselben Vollmodell-Split, 15 Epochen, Start-Lernrate 0,0001 und durchgehend Teacher Forcing 1 konfiguriert. Sein fertiges Modell lag beim Erstellen dieser Tabelle noch nicht vor.

Bewertungsvarianten mit anderem Optionsabstand, Übergangssperren, Eventkosten-Ziel oder anderer Ziehungszahl verwenden dieselben Basisgewichte; dafür wurde kein neues Suffixmodell trainiert.

Einzelne Modellpfade und Rohmetadaten stehen in trainingsdaten.json.
