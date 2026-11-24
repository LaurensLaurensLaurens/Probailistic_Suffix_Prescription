# token_oe mit Übergangssperre, Seed 0

300 Ziehungen je Option, normale PSP-Erwartungswert-Policy, 3.000 Ankünfte, 1.000 gewertete Fälle. Gleiche Seeds, Fenster, vier Wahlpunkte und zufällige Zuteilung wie beim bisherigen 300er-Lauf.

Die Modellkopie verwendet die bisherigen trainierten Gewichte. Unbeobachtete direkte Aktivität-Lifecycle-Übergänge werden vor dem Ziehen auf Wahrscheinlichkeit 0 gesetzt. Zeitklassen bleiben frei. EOS ist nur nach Tokens erlaubt, nach denen ein Fall im Trainingssplit endet. Quelle: echte Fälle außerhalb der 6.301 Testfälle; 414 erlaubte Paare, 22 Fallend-Tokens. Die bisherige Längengrenze bleibt bestehen.

Sieben gezielte Tests und 2.400 Suffixe am echten Modell wurden geprüft: keine unbeobachteten Übergänge oder unzulässigen EOS-Enden.

- `progress.json`: laufender Status, nach der ersten und jeweils 25 weiteren Entscheidungen aktualisiert.
- `choices.jsonl`: jede abgeschlossene Entscheidung, sofort gespeichert.
- `run_manifest.json`: Parameter, Prüfergebnisse und Dateifingerabdrücke.
- `prescription/PSP/`: endgültige Auswertung gegen die Bank nach Laufende.
- Log: `prescriptive/data/PSP_online300_real_alle_s60_token_oe_ue_seed0.log`.
- tmux-Sitzung: `psp300_oe_ue_s0`.
