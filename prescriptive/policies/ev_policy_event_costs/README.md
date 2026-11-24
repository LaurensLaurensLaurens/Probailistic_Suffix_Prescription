# EV-Policies mit Eventkosten

Diese Variante verwendet die bisherigen Interventionspunkte, Optionen und
EV-Verfahren. Ziel ist der kleinste Erwartungswert der gesamten Eventkosten
eines Falls. Sie verändert weder den globalen Katalog noch die Standard-EV-Policies.

## Kostenkonfiguration

[event_costs.yaml](event_costs.yaml) enthält **editierbare Beispieltarife in Euro**,
keine aus dem Log ermittelten Geschäftskosten. Alle 26 BPIC17-Aktivitäten sind
eingetragen. Jeder Eventeintrag kostet:

```text
Basiskosten der Aktivität
+ Zuschlag für den Lifecycle dieser Aktivität
+ aktive Arbeitsstunden × Stundensatz der Aktivität
```

Basiskosten fallen je Eventeintrag an, auch bei wiederholten Aktivitäten und bei
jedem Lifecycle. `defaults` liefert gemeinsame Werte; `events` überschreibt sie
je Aktivität. Lifecycle-Zuschläge werden zusammengeführt. Ein nicht genannter
Lifecycle verwendet `lifecycle_eur.default`.

```yaml
events:
  W_Validate application:
    base_eur: 2
    eur_per_work_hour: 60
    lifecycle_eur:
      complete: 3
  A_Accepted:
    base_eur: 5
```

Mit der mitgelieferten Konfiguration kostet ein Validierungs-`start` 3 €.
Ein `complete` nach 30 Minuten kostet 2 + 3 + 30 = 35 €.
`A_Accepted|complete` kostet 5 + 2 = 7 €, auch ohne Beginn oder Arbeitsdauer.
Mitarbeiter-Identitäten beeinflussen die Tarife nicht.

## Aktive Arbeitsabschnitte

- `start` und `resume` öffnen einen Abschnitt der betreffenden Aktivität im Fall.
- `suspend` und `complete` schließen ihn; ebenso die im BPIC17 vorkommenden
  Abbrüche `ate_abort` und `withdraw`.
- Zeitkosten werden beim Schließen genau einmal verbucht. Andere Events und
  parallel offene Aktivitäten ändern diesen Abschnitt nicht.
- Die Pause zwischen `suspend` und `resume` sowie Warten vor `start` kosten keine
  Arbeitszeit. Innerhalb eines offenen Abschnitts zählt die Zeitdifferenz der
  Lifecycle-Zeitstempel; es wird kein zusätzlicher Dienstplanabzug vorgenommen.
- Bei einem Ende ohne bekannten Beginn entstehen Basis- und Lifecycle-Kosten,
  aber keine erfundene Arbeitsdauer. `orphan_closures` zählt solche Einträge bei
  Aktivitäten mit positivem Stundensatz.
- Ein erneutes Öffnen ohne vorheriges Ende rechnet den bisherigen Abschnitt bis
  zu diesem Zeitpunkt ab und beginnt einen neuen. `reopened_segments` zählt dies.
- Ein am beobachteten Ende noch offener Abschnitt wird bis zum letzten Event
  abgerechnet. Die Korrektur steht beim zugehörigen letzten `start`/`resume` dieser
  Aktivität, nicht bei einem fremden Abschluss-Event. Ein künstliches `complete`
  oder dessen Gebühr wird nicht ergänzt. `unclosed_segments` zeigt dies.

Unbekannte Aktivitäten verwenden standardmäßig den ausdrücklich konfigurierten
Standardtarif und werden als `unconfigured_events` gezählt. Mit
`unknown_activity: error` werden sie abgewiesen. Negative oder nicht endliche
Kostensätze und unbekannte Konfigurationsfelder werden abgewiesen.

## Modelle und Auswahl

`PSPEventCostPolicy` verwendet die vorhandene EV-Auswahl aus `ev_policy`:
Suffixe je Option ziehen, negativen Kostenmittelwert maximieren, Gleichstände
mit dem Seed auflösen. Der Adapter berechnet negative **verbleibende** Kosten.
Bereits angefallene Eventkosten und aktive Arbeitszeit bis zur Entscheidung
werden als gemeinsamer konstanter Anteil abgezogen. Offene Abschnitte werden
über die Prefix-/Suffix-Grenze fortgeführt.

Die Options-Events werden genau einmal berechnet, sowohl bei den ursprünglichen
PSP-Modellen mit Option im Prefix als auch bei `option_as=encoder/decoder` und
den Varianten mit gezogenem Optionsabstand. Zeiten werden konsistent aus den
gezogenen nichtnegativen Eventabständen aufgebaut. Das PSP muss Lifecycle-Tokens
vorhersagen. Seine Gewichte und seine Ziehungslogik bleiben unverändert; ein
Restzeit-Kopf wird für diese Kostenbewertung nicht verwendet.

`EventCostCATEModel` verwendet dieselben S-Learner, X-Learner und Causal Forests,
Merkmale und fallweisen Splits wie die Standard-EV-Modelle. Es berechnet das
Trainingsziel neu: **negative gesamte Fallkosten**. Positive CATE-Effekte bedeuten
erwartete Einsparungen in Euro gegenüber der Referenz. `threshold` ist ebenfalls
in Euro. Mitarbeiterpunkte bleiben wie bei den bisherigen CATE-Verfahren ohne
Modell. Untrainierte Aktionen erhalten keinen künstlich günstigen Nulleffekt.

CATE-Modelle speichern die vollständige Kostenkonfiguration mit Fingerabdruck.
Eine Policy akzeptiert nur Modelle mit diesem Kostenziel und passenden Tarifen.
Nach einer Tarifänderung müssen CATE-Modelle neu trainiert werden. PSP-Suffixe
können unmittelbar mit anderen Tarifen bewertet werden.

## Training

Alle Befehle aus dem Projektverzeichnis ausführen. Der folgende Befehl trainiert
alle drei CATE-Verfahren mit dem bisherigen äußeren Testsplit und allen übrigen
Fällen. Vorhandene Ausgabeordner werden nicht überschrieben:

```bash
process-simulation-engine/.venv/bin/python -m prescriptive.policies.ev_policy_event_costs.train \
  --tag real_alle --learner s x cf --cases 0 \
  --catalog prescriptive/data/catalog_neue_punkte.yaml
```

Modelle, Tarif- und Katalogkopie, Status und Testeffekte in Euro liegen unter
`models/real_alle/` in diesem Ordner. Eigene Tarife: `--cost-config PFAD`.

## Online-Auswertung

Zum Beispiel PSP mit dem vorhandenen (d)-Modell: 30 Ziehungen pro Option,
200 gewertete Fälle, 700 Warmup und 1.300 Cooldown, Seed 0:

```bash
CUDA_VISIBLE_DEVICES="" process-simulation-engine/.venv/bin/python \
  -m prescriptive.policies.ev_policy_event_costs.evaluate \
  --policies bank psp --samples 30 --scored-cases 200 \
  --psp-model prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_klassen_option_encoder.pkl \
  --catalog prescriptive/data/catalog_neue_punkte.yaml \
  --out prescriptive/policies/ev_policy_event_costs/results/psp_d_30_seed0
```

Für den gemeinsamen Vergleich zusätzlich `s_learner x_learner causal_forest` in
`--policies` und die trainierten Modelle angeben:

```text
--cate-model s_learner=prescriptive/policies/ev_policy_event_costs/models/real_alle/cate_model_s_event_costs.pkl
--cate-model x_learner=prescriptive/policies/ev_policy_event_costs/models/real_alle/cate_model_x_event_costs.pkl
--cate-model causal_forest=prescriptive/policies/ev_policy_event_costs/models/real_alle/cate_model_cf_event_costs.pkl
```

`--seeds` und `--workers` steuern Wiederholungen und parallele Prozesse. Es wird
immer gegen die Bank mit denselben Seeds und demselben Bewertungsfenster
verglichen. Der vorhandene Simulator muss trainiert sein; die Auswertung startet
kein Simulatortraining.

Ergebnisse: `comparison.csv`, `runs.csv`, Tarif- und Katalogkopie sowie `run.json`.
Je Policy/Seed gibt es außerdem `case_costs.csv`, `choices.csv` und
`events_with_costs.csv` mit echten simulierten Eventkosten. Hauptkennzahl ist
`Eventkosten je Fall (€)`. Ein positiver Uplift bedeutet eine Kosteneinsparung.
Durchlaufzeit und Arbeitsstunden werden zusätzlich ausgewiesen. Die
95-%-Intervalle werden über Seeds berechnet; ein einzelner Seed liefert dafür
keine belastbare Streuung.

### Gesamter Vergleich: CATE neu trainieren und mit PSP vergleichen

`run_comparison.py` hält Tarife und Katalog fest, trainiert alle drei CATE-Modelle
auf den Fällen außerhalb des gemeinsamen Testsplits und wertet anschließend Bank,
PSP, S-Learner, X-Learner und Causal Forest mit Seed 0 aus. Fest eingestellt sind
200 bewertete Fälle, 700 Warmup, 1.300 Cooldown und 30 PSP-Ziehungen je Option.
Standard-PSP ist das vorhandene (d)-Modell mit Optionsabstand je Suffixziehung.

```bash
process-simulation-engine/.venv/bin/python -m prescriptive.policies.ev_policy_event_costs.run_comparison \
  --prepare --out prescriptive/policies/ev_policy_event_costs/results/vergleich_neu

process-simulation-engine/.venv/bin/python -m prescriptive.policies.ev_policy_event_costs.run_comparison \
  --run --out prescriptive/policies/ev_policy_event_costs/results/vergleich_neu
```

Ein eigenes PSP lässt sich bei `--prepare` mit `--psp-model PFAD` festlegen.
Der Ausgabeordner enthält Status und Logs, die trainierten Modelle, vollständige
Online-Ergebnisse, eine Kostenübersicht als CSV und Markdown sowie das Diagramm
`event_cost_comparison.png`. `paired_case_costs.csv` zeigt die Kosten derselben
Fall-IDs je Verfahren. Quellcode-, Modell- und Konfigurationsprüfsummen verhindern
einen unbemerkten Wechsel der Grundlagen während dieses Vergleichs.

## Kostenfunktion direkt verwenden und testen

```python
from prescriptive.policies.ev_policy_event_costs.costs import EventCosts

costs = EventCosts()
ledger = costs.event_costs(log)       # Basis-, Lifecycle- und Zeitkosten je Event
bill = costs.case_costs(log)         # Summen je Fall
labelled = costs.label_log(log)      # Kopie mit event_cost_eur, case_event_cost_eur, outcome
```

```bash
process-simulation-engine/.venv/bin/python -m unittest discover \
  -s prescriptive/policies/ev_policy_event_costs/tests -v
```
