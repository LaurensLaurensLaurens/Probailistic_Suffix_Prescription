#!/bin/bash
# Wie data/bpic17_psp_konformanz_v3.sh, aber mit 4-facher Datenmenge (2026-11-21, Nutzer: „mach das“ – 128.000 Fälle, Training auf der GPU):
# 1. Log log_effekte_v3_x4 (log_erzeugen.py 32 × 4.000, Saaten 2000–2031; die ersten 8 Läufe = bisheriger v3-Log).
# 2. T-Learner-Arme warm von den Armen des echten Logs (--vocab-from), nur Wahlpunkt-Proben, LR 1e-4, Aktion 10 / Referenz 5 Epochen.
# 3. Hülle t_learner_k3x4.pkl. 4. Online-Auswertung nur Arm konformanz_x4 (12 Seeds, gepaart mit den vorhandenen bank / regel / orakel).
# Aufruf (tmux): bash bpic17_psp_konformanz_v3x4.sh
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_100k/kuenstliche_effekte_20261120
LOGP=$OUT/log_effekte_v3_x4/log_raw.pkl
HULL=$P/models/psp_model_durchlaufzeit_t_learner_k3x4.pkl
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
COMMON="--cases 0 --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps --counters --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training --log-pickle $LOGP"
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
note() { echo "$(date '+%Y-%m-%d %H:%M') k3x4: $1" >> $OUT/fortschritt.log; }
if [ ! -e "$LOGP" ]; then
  note "Log 32 × 4.000 wird erzeugt"
  .venv/bin/python -u $OUT/log_erzeugen.py 32 4000 kuenstliche_effekte/effekte_v3.yaml _x4 > $OUT/log_erzeugen_x4.log 2>&1 || { note "Abbruch: Log"; exit 1; }
  note "Log fertig: $(grep -E '"cases"|"events"' $OUT/log_effekte_v3_x4/info.json | tr -s ' ' | tr '\n' ' ')"
fi
for ARM in aktion referenz; do
  BASE=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_${ARM}.pkl
  TAG=k3x4_zaehler_zeit_tf1_t_$ARM
  MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
  EPOCHS=$([ $ARM = aktion ] && echo 10 || echo 5)
  note "Training $ARM beginnt ($EPOCHS Epochen, GPU: $(.venv/bin/python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null))"
  [ -e "$MODEL" ] || .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --epochs $EPOCHS --learning-rate 1e-4 $COMMON \
      --only-points --only-points-validation --arm $ARM --init-from "$BASE" --vocab-from "$BASE" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$P/data/PSP_$TAG.log"
  [ -e "$MODEL" ] || { note "Abbruch im Training $ARM"; exit 1; }
  note "Training $ARM fertig"
done
CUDA_VISIBLE_DEVICES="" .venv/bin/python ../prescriptive/policies/ev_policy/t_learner_policy.py \
    $P/models/psp_model_durchlaufzeit_k3x4_zaehler_zeit_tf1_t_aktion.pkl $P/models/psp_model_durchlaufzeit_k3x4_zaehler_zeit_tf1_t_referenz.pkl "$HULL" >> $OUT/fortschritt.log 2>&1 || { note "Abbruch: Hülle"; exit 1; }
note "Hülle gebaut, Online-Auswertung konformanz_x4 beginnt"
.venv/bin/python -u $OUT/online_konformanz.py konformanz_x4 0-11 12 > $OUT/online_konformanz_x4.log 2>&1
note "Online-Auswertung beendet (Code $?): $(grep -E '^FERTIG' $OUT/online_konformanz_x4.log | head -1)"
note "Alles fertig"
