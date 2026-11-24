#!/bin/bash
# T-Learner-PSP (2026-11-17, Wunsch des Nutzers): zwei PSPs, warm gestartet vom fertigen ZZ-TF1, je nachtrainiert nur auf den
# Wahlpunkt-Proben ihrer Gruppe (Aktion: annehmen, annehmen, weiteres Angebot, Lead bearbeiten; Referenz: bearbeiten, fortsetzen, senden,
# zurückziehen), Validierung ebenso gefiltert; Lernrate 3e-5 (Feinabstimmung), Aktion 40 Epochen (~15.000 Proben), Referenz 20 Epochen
# (~60.000 Proben). Nacheinander (je Training ~16 GB RAM). Danach: Hülle bauen (policies/ev_policy/t_learner_policy.py), Prüfreihe
# (results_100k/t_learner_20261117/pruefreihe.py), Online-Test Aufbau F auf 12 Seeds mit 300 Ziehungen + first_only (f_zwoelf_seeds.py).
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
BASE=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1.pkl
HULL=$P/models/psp_model_durchlaufzeit_t_learner_zztf1.pkl
OUT=$P/scripts/results_100k/t_learner_20261117
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
note() { echo "$(date '+%Y-%m-%d %H:%M') $1" >> $OUT/fortschritt.log; }
for ARM in aktion referenz; do
  TAG=real_alle_s60_normal_zaehler_zeit_tf1_t_$ARM
  LOG=$P/data/PSP_$TAG.log
  MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
  EPOCHS=$([ $ARM = aktion ] && echo 40 || echo 20)
  note "Training $ARM beginnt ($EPOCHS Epochen)"
  echo "######## Training $ARM ($(date +%H:%M:%S))" | tee -a "$LOG"
  [ -e "$MODEL" ] || .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --cases 0 --epochs $EPOCHS --learning-rate 3e-5 \
      --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps \
      --counters --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training \
      --only-points --only-points-validation --arm $ARM --init-from "$BASE" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
  [ -e "$MODEL" ] || { note "Abbruch im Training $ARM"; exit 1; }
  note "Training $ARM fertig"
done
CUDA_VISIBLE_DEVICES="" .venv/bin/python ../prescriptive/policies/ev_policy/t_learner_policy.py \
    $P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_aktion.pkl \
    $P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_referenz.pkl "$HULL" >> $OUT/fortschritt.log 2>&1 || { note "Abbruch: Hülle"; exit 1; }
note "Hülle gebaut, Prüfreihe beginnt"
.venv/bin/python -u $OUT/pruefreihe.py "t_zztf1=T-Learner (ZZ-TF1)=$HULL" > $OUT/pruefreihe.log 2>&1
note "Prüfreihe fertig, Online-Test 12 Seeds beginnt"
.venv/bin/python -u $P/scripts/results_100k/modellvergleich_20261116/f_zwoelf_seeds.py --arms tl_fest300 --seeds 0-11 --workers 6 > $OUT/online_12_seeds.log 2>&1
note "Online-Test fertig"
