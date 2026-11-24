# CATE-Policies für den erwarteten Outcome

Die drei Ansätze `s_learner`, `x_learner` und `causal_forest` schätzen, welche
Auswirkung eine Aktion gegenüber einer Referenz für Fälle mit einem bestimmten
Prefix hat. Die gemeinsame Entscheidungslogik steht in [cate_policy.py](cate_policy.py),
das Training und die Schätzer in [cate_model.py](../../estimators/cate_model.py).
Die Verfahren unterscheiden sich darin, wie sie den Effekt schätzen.

| Policy | Schätzer | Grundidee |
| --- | --- | --- |
| `s_learner` | `_SLearner` | Ein Outcome-Modell zweimal abfragen: mit Aktion und mit Referenz. |
| `x_learner` | `_XLearner` | Zwei Outcome-Modelle, daraus imputierte Effekte und zwei Effektmodelle; gewichtet mit der Propensity zusammenführen. |
| `causal_forest` | `_CausalForest` | Outcome und Treatment mittels Cross-Fitting bereinigen; ein kausaler Wald schätzt daraus unterschiedliche Effekte nach Prefix. |

## Was wird geschätzt?

Für einen binären Vergleich bezeichnet:

- `X`: Merkmale des Falls bis zum aktuellen Entscheidungspunkt;
- `T = 0`: Referenz, die erste Option des Wahlpunkts;
- `T = 1`: die verglichene Aktion;
- `Y`: Outcome des abgeschlossenen Falls.

Das Ziel ist der **Conditional Average Treatment Effect (CATE)**:

$$
\tau(x) = E[Y(1)-Y(0)\mid X=x].
$$

Gemeint ist der mittlere Behandlungseffekt für Fälle mit diesem Merkmalsmuster.
Die beiden möglichen Outcomes eines einzelnen realen Falls sind nicht gleichzeitig
beobachtbar. Die Modelle müssen den fehlenden kontrafaktischen Outcome aus anderen
Fällen schätzen.

Im aktuellen BPIC17-Vergleich ist `Y` die **negative gesamte Durchlaufzeit in Tagen**.
Damit bedeutet ein positiver Effekt eine kürzere erwartete Durchlaufzeit durch die
Aktion. Beispiel: Erwarteter Outcome der Referenz `−18`, der Aktion `−14`:
`τ = −14 − (−18) = +4` Tage. Die Aktion verspricht vier Tage Verbesserung.

## Gemeinsame Trainingsdaten und Merkmale

Die drei Verfahren verwenden dieselben Entscheidungsproben und dieselbe
Merkmalsdarstellung. Beides stammt aus
[XGBModel.samples und Prefix](../../estimators/xgb_model.py).

Eine Control-Flow-Probe entsteht, wenn nach dem Entscheidungs-Event unmittelbar
eine der bekannten Optionen folgt. Das Prefix reicht bis einschließlich des
Entscheidungs-Events. Die folgende Option ist das Treatment, der finale Outcome
des Falls das Trainingsziel. Jedes erneute Erreichen des Punkts kann eine weitere
Probe erzeugen; das Training ist nicht auf die erste Entscheidung je Fall begrenzt.

Die Merkmale umfassen Fallattribute, Häufigkeiten der bisherigen Aktivität-Lifecycle-Tokens,
letztes Token, Eventanzahl, vergangene Zeit, Zeit seit dem letzten Event,
Wochentag und Uhrzeit sowie die bisherigen Werte der konfigurierten numerischen
Eventattribute, etwa Angebotsdaten. Es wird eine Merkmalsmatrix verwendet; ein
rekurrentes Netz über die vollständige Sequenz wird hier nicht trainiert.

Kategoriale Werte werden als One-Hot-Spalten kodiert. Fehlende numerische Werte
werden mit Trainingsmedianen ersetzt und gegebenenfalls durch eine zusätzliche
Fehlwertspalte markiert. Konstante Spalten entfallen. Die Kodierung wird im Modell
gespeichert und bei der Abfrage wiederverwendet.

Je Wahlpunkt wird ein eigener Schätzer trainiert. Bei mehr als zwei Optionen wird
jede weitere Aktion separat gegen dieselbe erste Option verglichen. Die Basis-S-Learner
dieser Umsetzung sind damit ebenfalls paarweise binäre Modelle.
Für Mitarbeiter-Wahlpunkte trainiert `CATEModel` keine Modelle.

## 1. S-Learner

Der S-Learner trainiert einen `XGBRegressor` auf Prefix-Merkmalen und Treatment:

$$
\hat\mu(x,t) \approx E[Y\mid X=x,T=t].
$$

Bei der Abfrage bleibt der Prefix unverändert und nur das Treatment wird umgestellt:

$$
\hat\tau(x) = \hat\mu(x,1)-\hat\mu(x,0).
$$

Beispiel: Derselbe bisherige Fall wird einmal unter der Annahme „weiter bearbeiten“
und einmal unter „annehmen“ bewertet. Die Differenz der vorhergesagten Outcomes
ist der geschätzte Effekt von „annehmen“.

Beide Gruppen teilen sich ein Modell. Das kann bei ähnlichen Zusammenhängen
hilfreich sein. Das Modell kann das Treatment aber auch ignorieren, wenn es daraus
keinen geeigneten Split lernt; dann werden die beiden Vorhersagen identisch und
der geschätzte Effekt ist null.

## 2. X-Learner

Der X-Learner verwendet in dieser Umsetzung fünf XGBoost-Modelle je binärem Vergleich.

Zuerst werden zwei Outcome-Modelle auf den jeweils beobachteten Gruppen trainiert:

$$
\hat\mu_0(x) \approx E[Y\mid X=x,T=0],\qquad
\hat\mu_1(x) \approx E[Y\mid X=x,T=1].
$$

Dann werden pro Trainingsprobe Effekte imputiert. Bei Fällen mit Aktion ist deren
Outcome beobachtet; das Referenzmodell liefert den fehlenden Referenz-Outcome:

$$
D_1 = Y-\hat\mu_0(X)\quad\text{auf Proben mit }T=1.
$$

Bei Referenzfällen liefert umgekehrt das Aktionsmodell den fehlenden Aktions-Outcome:

$$
D_0 = \hat\mu_1(X)-Y\quad\text{auf Proben mit }T=0.
$$

Auf diesen beiden künstlichen Effektzielen werden zwei weitere Regressoren trainiert:
`τ₁(X)` auf den Aktionsproben und `τ₀(X)` auf den Referenzproben. Ein
`XGBClassifier` schätzt zusätzlich die **Propensity**:

$$
\hat g(x) \approx P(T=1\mid X=x).
$$

Der endgültige Effekt kombiniert die beiden Effektmodelle genau wie im Code:

$$
\hat\tau(x)=\hat g(x)\hat\tau_0(x)+(1-\hat g(x))\hat\tau_1(x).
$$

Die Propensity wird dafür auf `[0,01; 0,99]` begrenzt. Sie beschreibt die beobachtete
Wahlwahrscheinlichkeit der Aktion, nicht deren Erfolgswahrscheinlichkeit.
Das Begrenzen der Gewichte erzeugt keine fehlende Überdeckung in den Daten.

Der wesentliche Unterschied zum S-Learner: Es werden eigene Modelle auf den
imputierten **Effekten** trainiert, statt ausschließlich zwei Abfragen eines
gemeinsamen Outcome-Modells voneinander abzuziehen.

## 3. Causal Forest mit Double Machine Learning

Verwendet wird `econml.dml.CausalForestDML`. XGBoost liefert zwei Hilfsmodelle:

$$
\hat m(x)\approx E[Y\mid X=x],\qquad
\hat g(x)\approx P(T=1\mid X=x).
$$

Diese Modelle sollen die Unterschiede erklären, die bereits aus den
Prefix-Merkmalen folgen. Anschließend werden die entsprechenden Anteile aus
Outcome und Treatment herausgerechnet:

$$
\widetilde Y=Y-\hat m(X),\qquad
\widetilde T=T-\hat g(X).
$$

Der kausale Wald lernt lokal nach `X` den Zusammenhang zwischen den verbleibenden
Outcome- und Treatment-Unterschieden. Vereinfacht lautet das Arbeitsmodell:

$$
\widetilde Y\approx\tau(X)\widetilde T+\text{Rest}.
$$

Das ist ein Effektwald mit DML-Bereinigung, dessen Ergebnis direkt `τ(x)` ist.
Er berechnet nicht einfach die Differenz zweier gewöhnlicher Outcome-Wälder.

Die konkrete Konfiguration:

- fünf Cross-Fitting-Folds für die Hilfsmodelle; Fall-IDs werden als Gruppen übergeben;
- 500 Bäume im kausalen Wald;
- `honest=True`: innerhalb der Bäume getrennte Teilproben für Struktur und Effektschätzung;
- `max_samples=0.45` und anfänglich `min_samples_leaf=5`;
- EconML wählt mit `tune` Baumtiefe, minimale Blattgewichtsfraktion und minimale
  Treatment-Varianzfraktion;
- `inference=False`: diese Konfiguration liefert keine CATE-Konfidenzintervalle.

Cross-Fitting bedeutet hier, dass die Hilfsmodell-Vorhersagen für die
Residualisierung auf jeweils zurückgehaltenen Folds entstehen. Das reduziert
die direkte Überanpassung dieser Bereinigung an dieselben Proben.

## Training und Validierung in diesem Projekt

[train_cate_real.py](../../scripts/train_cate_real.py) hält zunächst die gemeinsamen
Testfälle vollständig aus dem Training heraus. Im bisherigen BPIC17-Lauf waren
das 6.301 Testfälle und 25.208 Trainingsfälle.

Innerhalb der Trainingsfälle werden ungefähr 10 % der Fälle als Validierung für
das frühe Stoppen der XGBoost-Modelle markiert. Alle Entscheidungsproben desselben
Falls erhalten dieselbe Markierung. Die Voreinstellungen sind bis zu 2.000 Bäume,
Tiefe 6, Lernrate 0,03 und frühes Stoppen nach 50 Runden ohne Verbesserung.
Bei zu wenigen Proben für das frühe Stoppen werden 200 Bäume auf den verfügbaren
Proben trainiert.

Beim Causal Forest dient dieser Vorlauf zur Auswahl der Baumzahlen der XGBoost-Hilfsmodelle.
EconML trainiert diese anschließend mit fester Baumzahl und eigenem Cross-Fitting
auf den verfügbaren Trainingsdaten. Die im Modell ausgegebenen Effektdiagnosen
auf den inneren Validierungszeilen sind deshalb beim Causal Forest kein separater
unberührter Test. Die äußeren Testfälle bleiben bei allen drei Verfahren ausgeschlossen.

Ein Vergleich gegen die Referenz wird nur trainiert, wenn für beide Optionen
jeweils mindestens 30 Proben vorliegen. Diese Mindestzahl ist eine globale
Datenmengenprüfung; sie garantiert keine Überdeckung für jedes Merkmalsmuster.

Training aus `process-simulation-engine/`, mit dem Katalog des gewünschten Vergleichs:

```bash
PSP_CATALOG=../prescriptive/data/catalog_neue_punkte.yaml \
  .venv/bin/python ../prescriptive/scripts/train_cate_real.py \
  --tag real_alle --learner s x cf --cases 0 --seed 42
```

Die bisherigen EV-Modelle heißen
`cate_model_s_durchlaufzeit_real_alle.pkl`,
`cate_model_x_durchlaufzeit_real_alle.pkl` und
`cate_model_cf_durchlaufzeit_real_alle.pkl`. Vorhandene Dateien werden vom
Trainingsskript nicht überschrieben.

## Wie entscheidet die Policy im Simulator?

Die Engine ruft `CATEPolicy.choose(point, case, options, time, engine)` auf.
Die Policy berechnet die aktuellen Prefix-Merkmale und fragt für jede Aktion
den geschätzten Effekt ab. Dabei hat die Referenz den relativen Wert null:

$$
v(\text{Referenz})=0,\qquad
v(a)=\hat\tau_a(x)-\text{threshold}.
$$

Gewählt wird die verfügbare Option mit dem größten Wert. Standard ist
`threshold=0`. Bei zwei Optionen bedeutet das:

- `τ > threshold`: Aktion wählen;
- `τ < threshold`: Referenz wählen;
- `τ = threshold`: Gleichstand, zufällig und mit dem Seed des Laufs auflösen.

Die Schwelle ist in Einheiten des Outcomes, hier Tagen. Bei `threshold=2` muss
die Aktion mehr als zwei Tage Verbesserung versprechen, um eindeutig bevorzugt
zu werden. Die Modelle werden während der Simulation nicht nachtrainiert.

Ohne Modell für den Wahlpunkt gibt die Policy `None` zurück; die Engine verwendet
dann ihren normalen Verlauf. Bei der Mitarbeiterwahl ist dies für BPIC17 die
aus dem echten Log gelernte Bank-Zuteilung (`LearnedAllocation`). Sie ist Teil der
Simulationsumgebung: `policy=None`, ein nicht behandelter Wahlpunkt oder eine
Policy-Antwort `None` behalten deren Vorschlag bei. Kontinuität ist ein gelerntes
Merkmal statt einer erzwungenen Bevorzugung des letzten Bearbeiters. Es gibt
dafür keine zusätzliche preskriptive `choose`-Policy. Explizite Zuteilungsexperimente
können weiterhin andere Basisstrategien konfigurieren.

Mit `first_only` kann man einzelne Wahlpunkte auf
die erste Entscheidung je Fall begrenzen. Standardmäßig ist diese Einschränkung aus.

Bei teilweise modellierten Wahlpunkten mit mehr als zwei Optionen gibt die
aktuelle EV-Implementierung für eine nicht trainierte Aktion den Effekt null
zurück. Diese kann dadurch bei Gleichstand gewählt werden. Eine Option, die
das Modell überhaupt nicht kennt, erhält dagegen `−inf`.

## Wann ist die Interpretation kausal?

Die kausale Interpretation setzt voraus, dass die beobachteten Prefix-Merkmale
die relevanten gemeinsamen Ursachen von Aktionswahl und Outcome ausreichend
erfassen, vergleichbare Fälle beide Optionen erhalten können und die Aktionen
sowie Outcomes konsistent definiert sind. Ein komplexerer Schätzer allein stellt
diese Voraussetzungen nicht her. Insbesondere erfasst die aktuelle Merkmalsmatrix
keinen vollständigen Zustand aller gleichzeitig laufenden Fälle und belegten Ressourcen.

Das [AIPW-Screening](../../../screening/causal.py) bewertet zusätzlich Effekte,
Policy-Werte, Überlappung und Placebo-Diagnosen. Die hier beschriebene `CATEPolicy`
führt zur Laufzeit keine AIPW-Berechnung aus. Ein Vergleich des Vorzeichens ihrer
Testvorhersagen mit einem mittleren AIPW-Effekt ist eine Diagnose, kein Nachweis
korrekter individueller Effektschätzungen.

Die EV-Verfahren lernen Effekte auf den **erwarteten Outcome**. Die separat
dokumentierten [ERM-Varianten](../erm_policy/README.md) lernen stattdessen Effekte
auf den erwarteten exponentiellen Verlust und benötigen eigene Modelle.

Die bisherigen EV-Vergleichsdiagramme wurden noch mit expliziter `random`-Zuteilung
erstellt. Sie sind historische Ergebnisse dieser damaligen Umgebung; Vergleiche
mit der neuen Bank-Zuteilung benötigen eigene Simulatorläufe.

## T-Learner-PSP (`t_learner_policy.py`, 2026-11-17)

Die PSP-Policy (`psp_policy.py`) ist strukturell ein **S-Learner**: ein Modell, die Option ist nur das letzte
Event des Prefix. Ein solches Netz kann die Option weitgehend ignorieren; die geschätzten Effekte schrumpfen dann
gegen null (bei den Teacher-Forcing-1-Modellen an Nachlegen und Leads beobachtet). Der **T-Learner** nutzt zwei PSPs:

- **Aktions-Modell**: nur Trainingsproben, deren Prefix am Options-Event einer Aktion endet
  (annehmen bei Konzept, annehmen nach Vervollständigung, weiteres Angebot, Lead bearbeiten; `ARME["aktion"]`).
- **Referenz-Modell**: entsprechend für bearbeiten, fortsetzen, senden, zurückziehen (`ARME["referenz"]`).

Beide starten mit den Gewichten des fertigen ZZ-TF1 (`--init-from`) und werden nur auf ihrer Gruppe
nachtrainiert (`scripts/train_psp_real.py --only-points --only-points-validation --arm aktion|referenz`,
Startskript `data/bpic17_psp_t_learner.sh`). Bewertet wird wie beim S-Learner aus gezogenen Suffixen mit Event
und Zeit je Schritt; jede Option zieht aus ihrem eigenen Modell (gleiche Startsaat für beide).

`TLearnerPSP` ist eine Hülle mit der Schnittstelle von `PSPModel` (gespeichert mit `bauen(aktion, referenz, ziel)`,
ladbar mit `PSPModel.load`). Damit laufen `PSPPolicy`, die Prüfreihe (`psp_rauschen.py`, `punkte_alle.py`,
`nachlegen_erklaerung.py`) und die Online-Läufe unverändert. Ziehungen ohne Option (`draw_cases`, z. B.
Suffixgüte, Gedächtnistests) nutzen nur das Referenz-Modell. `TLearnerPolicy` ist `PSPPolicy` mit Hülle oder
zwei Modellpfaden. Zwei verschiedene Netze lassen sich kaum koppeln; deshalb sind mehr Ziehungen nötig (300 statt 100).
