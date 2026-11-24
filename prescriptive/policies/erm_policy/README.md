# ERM-Policies

Die BPIC17-Basiswahl der Mitarbeiter ist das gelernte Bankverhalten
(`LearnedAllocation` im Simulator). Ohne Policy-Überschreibung bleibt diese Wahl
erhalten; es ist keine zusätzliche preskriptive Policy nötig. Die bisherigen
ERM-Ergebnisse unter `results_100k/neue_punkte_online_cate_erm_gamma0.1` und dem
ursprünglichen PSP-ERM-Lauf verwendeten noch `random`. Sie bleiben als historische
Vergleiche erhalten und dürfen nicht als Ergebnisse der neuen Umgebung ausgegeben
oder mit deren Bank-Läufen kombiniert werden.

`psp_erm` bewertet die gezogenen Suffixe mit dem entropischen Risikomaß
`ρ_γ(L) = log(E[exp(γ L)]) / γ`, mit `L = -outcome` (Durchlaufzeit in Tagen).
Die Vergleichsmethoden `s_learner_erm`, `x_learner_erm` und `causal_forest_erm`
lernen stattdessen den Effekt der Aktion auf den erwarteten exponentiellen Verlust.
Das kleinste vorhergesagte `E[exp(γ L)]` minimiert auch das ERM.

Die Verfahren, Prefix-Merkmale und Fall-Splits entsprechen den EV-Varianten.
S-Learner: Outcome-Modell mit Treatment-Merkmal; X-Learner: zwei Outcome-Modelle,
imputierte Effekte und Propensity-Gewichtung; Causal Forest: DML auf dem
transformierten Outcome. Die kausale Interpretation setzt wie bei EV insbesondere
ausreichende Überdeckung und keine unbeobachteten Confounder voraus.

## Training

Aus `process-simulation-engine/`, mit demselben Katalog wie beim Vergleich:

```bash
.venv/bin/python ../prescriptive/scripts/train_cate_real.py \
    --tag real_alle --learner s x cf --cases 0 --erm-gamma 0.1
```

Erzeugt je Verfahren ein eigenes Modell, z. B.
`prescriptive/models/cate_erm_model_s_gamma0.1_durchlaufzeit_real_alle.pkl`.
Ohne `--erm-gamma` trainiert das Skript weiterhin die bisherigen EV-Modelle.
Die Testfälle bleiben draußen. Die ERM-Testtabelle enthält Effekte auf den
skalierten exponentiellen Verlust, keine Effekte in Tagen oder AIPW-Vergleiche
auf den Erwartungswert.

## Online-Vergleich

Aus `process-simulation-engine/` (das PSP-Modell muss ebenfalls vorhanden sein):

```bash
.venv/bin/python ../prescriptive/scripts/main.py --steps online \
    --online-policies psp_erm s_learner_erm x_learner_erm causal_forest_erm \
    --erm-gamma 0.1 \
    --cate-model \
      s_learner_erm=../prescriptive/models/cate_erm_model_s_gamma0.1_durchlaufzeit_real_alle.pkl \
      x_learner_erm=../prescriptive/models/cate_erm_model_x_gamma0.1_durchlaufzeit_real_alle.pkl \
      causal_forest_erm=../prescriptive/models/cate_erm_model_cf_gamma0.1_durchlaufzeit_real_alle.pkl
```

`--psp-model PFAD` und `--psp-samples N` gelten auch für PSP-ERM.
`--cate-first-only WAHLPUNKT ...` gilt für die EV- und ERM-CATE-Policies.
ERM verwendet Schwelle 0; `--cate-threshold` betrifft ausschließlich EV.
Der Referenzwert ist 0, negative Effekte sprechen für die Aktion. Unbekannte
oder nicht trainierte Aktionen werden ausgeschlossen; ohne Modell für einen
Wahlpunkt entscheidet die Simulation wie die Bank. Gleichstände werden mit
dem Seed des Laufs aufgelöst.

`CATEERMPolicy(model_path, gamma=0.1)` prüft das trainierte Gamma. Ohne Angabe
verwendet sie das Gamma aus dem Modell. Ein anderes Gamma erfordert neues
Training; EV-Modelle werden abgelehnt.

## Numerik und Interpretation

Je Wahlpunkt wird `Z = exp(γ L - b)` trainiert. Der feste Log-Skalierungsfaktor
`b` wird auf den inneren Trainingsfällen bestimmt, sodass deren Ziele Mittel 1
haben. Alle Aktionen einschließlich aller paarweisen Vergleiche verwenden
denselben Faktor. Er ändert die theoretische Rangfolge nicht und wird im Modell
als `log_scales` gespeichert. Die Schätzer liefern Effekte auf `E[Z]`, keine
absoluten ERM-Werte oder ERM-Differenzen in Tagen.

Sehr lange Fälle bestimmen das ERM bei großem Gamma stark. Verluste werden
nicht abgeschnitten; ist die Dynamik der transformierten Ziele für die
Float32-Modelle zu groß, bricht das Training mit einer Fehlermeldung ab.
