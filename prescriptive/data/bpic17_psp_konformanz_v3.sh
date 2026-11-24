#!/bin/bash
# PSP auf dem Simulator-Log mit künstlichen Effekten v3 + Conformance-Auswertung (2026-11-20, Nutzer: „okay mach das“).
# Log: results_100k/kuenstliche_effekte_20261120/log_effekte_v3/log_raw.pkl (log_erzeugen.py: Bank, 8 × 4.000 Fälle, Effekte v3).
# 1. T-Learner-Arme warm von den Armen des echten Logs (..._t_{aktion,referenz}.pkl, ZZ-TF1-Einstellungen) mit --vocab-from (Kategorien der Quelle; der Simulator-Log hat
#    weniger Mitarbeiter), nur Wahlpunkt-Proben ihrer Gruppe, LR 1e-4, Aktion 10 / Referenz 5 Epochen.
# 2. Hülle t_learner_k3.pkl. 3. Online-Auswertung Conformance (online_konformanz.py, sobald vorhanden).
# Aufruf (tmux): bash bpic17_psp_konformanz_v3.sh
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_100k/kuenstliche_effekte_20261120
LOGP=$OUT/log_effekte_v3/log_raw.pkl
HULL=$P/models/psp_model_durchlaufzeit_t_learner_k3.pkl
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
COMMON="--cases 0 --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps --counters --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training --log-pickle $LOGP"
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
note() { echo "$(date '+%Y-%m-%d %H:%M') k3: $1" >> $OUT/fortschritt.log; }
for ARM in aktion referenz; do
  BASE=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_${ARM}.pkl
  TAG=k3_zaehler_zeit_tf1_t_$ARM
  MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
  EPOCHS=$([ $ARM = aktion ] && echo 10 || echo 5)
  note "Training $ARM beginnt ($EPOCHS Epochen)"
  [ -e "$MODEL" ] || .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --epochs $EPOCHS --learning-rate 1e-4 $COMMON \
      --only-points --only-points-validation --arm $ARM --init-from "$BASE" --vocab-from "$BASE" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$P/data/PSP_$TAG.log"
  [ -e "$MODEL" ] || { note "Abbruch im Training $ARM"; exit 1; }
  note "Training $ARM fertig"
done
CUDA_VISIBLE_DEVICES="" .venv/bin/python ../prescriptive/policies/ev_policy/t_learner_policy.py \
    $P/models/psp_model_durchlaufzeit_k3_zaehler_zeit_tf1_t_aktion.pkl $P/models/psp_model_durchlaufzeit_k3_zaehler_zeit_tf1_t_referenz.pkl "$HULL" >> $OUT/fortschritt.log 2>&1 || { note "Abbruch: Hülle"; exit 1; }
note "Hülle gebaut"
until [ -e $OUT/online_konformanz.py ]; do sleep 60; done
note "Online-Auswertung Conformance beginnt"
.venv/bin/python -u $OUT/online_konformanz.py > $OUT/online_konformanz.log 2>&1
note "Online-Auswertung beendet (Code $?): $(grep -E '^FERTIG|Fehler' $OUT/online_konformanz.log | head -1)"
note "Alles fertig"
