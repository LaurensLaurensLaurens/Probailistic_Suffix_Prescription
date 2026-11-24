#!/bin/bash
# ZZ-TF1 + T-Learner auf Simulator-RCT-Daten (2026-11-18, Wunsch des Nutzers): dieselbe Pipeline wie auf dem echten Log, nur mit dem
# RCT-Log des korrigierten Simulators (results_100k/rct_tlearner_20261118/rct_erzeugen.py: 8 × 4.000 Fälle, Optionen 50/50, Nachlegen nur
# beim 1. Besuch randomisiert). 1. Basis wie ZZ-TF1 (15 Epochen, LR 1e-4, TF 1, Zähler, Zeit je Event, Option im Encoder nach dem
# Training); 2. T-Learner-Arme warm von dieser Basis (Aktion 40 / Referenz 20 Epochen, LR 3e-5, nur Wahlpunkt-Proben ihrer Gruppe);
# 3. Hülle; 4. Prüfreihe für Basis und T-Learner (Testprefixe aus dem ECHTEN Log: test_cases von (c)); 5. Storno-Zeitpunkt;
# 6. online Aufbau F, 12 Seeds (Basis 100 Ziehungen, T-Learner 300, je first_only). Achtung: Training auf Daten der Bewertungsumgebung.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_100k/rct_tlearner_20261118
LOGP=$OUT/rct_log_raw.pkl
REF=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal.pkl
BASE_TAG=rct_zaehler_zeit_tf1
BASE=$P/models/psp_model_durchlaufzeit_$BASE_TAG.pkl
HULL=$P/models/psp_model_durchlaufzeit_t_learner_rct.pkl
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
COMMON="--cases 0 --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps --counters --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training --log-pickle $LOGP --test-cases-from $REF"
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
note() { echo "$(date '+%Y-%m-%d %H:%M') $1" >> $OUT/fortschritt.log; }
mkdir -p $OUT/logs
until [ -e "$LOGP" ]; do sleep 60; done
note "RCT-Daten vorhanden, Basis-Training beginnt (15 Epochen)"
LOG=$P/data/PSP_$BASE_TAG.log
[ -e "$BASE" ] || .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $BASE_TAG --epochs 15 --learning-rate 1e-4 $COMMON 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ -e "$BASE" ] || { note "Abbruch im Basis-Training"; exit 1; }
note "Basis fertig"
for ARM in aktion referenz; do
  TAG=${BASE_TAG}_t_$ARM
  MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
  EPOCHS=$([ $ARM = aktion ] && echo 40 || echo 20)
  note "Training $ARM beginnt ($EPOCHS Epochen)"
  [ -e "$MODEL" ] || .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --epochs $EPOCHS --learning-rate 3e-5 $COMMON \
      --only-points --only-points-validation --arm $ARM --init-from "$BASE" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$P/data/PSP_$TAG.log"
  [ -e "$MODEL" ] || { note "Abbruch im Training $ARM"; exit 1; }
  note "Training $ARM fertig"
done
CUDA_VISIBLE_DEVICES="" .venv/bin/python ../prescriptive/policies/ev_policy/t_learner_policy.py \
    $P/models/psp_model_durchlaufzeit_${BASE_TAG}_t_aktion.pkl $P/models/psp_model_durchlaufzeit_${BASE_TAG}_t_referenz.pkl "$HULL" >> $OUT/fortschritt.log 2>&1 || { note "Abbruch: Hülle"; exit 1; }
note "Hülle gebaut, Prüfreihe beginnt"
.venv/bin/python -u $OUT/pruefreihe.py "zzrct=ZZ-TF1 (RCT)=$BASE" "tlrct=T-Learner (RCT)=$HULL" > $OUT/pruefreihe.log 2>&1
note "Prüfreihe fertig, Storno-Messung"
CUDA_VISIBLE_DEVICES="" .venv/bin/python $P/scripts/results_100k/zaehler_uhren_tf1_20261117/storno_abstand.py 300 20 "ZZ-TF1 (RCT)=$BASE" \
    "T-Learner (RCT) Aktion=$P/models/psp_model_durchlaufzeit_${BASE_TAG}_t_aktion.pkl" "T-Learner (RCT) Referenz=$P/models/psp_model_durchlaufzeit_${BASE_TAG}_t_referenz.pkl" > $OUT/storno.log 2>&1
note "Storno-Messung fertig, Online-Test 12 Seeds beginnt"
.venv/bin/python -u $P/scripts/results_100k/modellvergleich_20261116/f_zwoelf_seeds.py --arms tlrct_fest300,zzrct_fest100 --seeds 0-11 --workers 8 > $OUT/online_12_seeds.log 2>&1
note "Online-Test fertig"
