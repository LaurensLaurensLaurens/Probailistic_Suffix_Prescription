#!/bin/bash
# PSP-Variante auf dem echten Log (Diagnose vom 2026-11-08: das originalgetreue PSP sagt an den Wahlpunkten keine
# fallbezogene Durchlaufzeit vorher; Decoder nur 5 Schritte trainiert, gezogen ~50; statische Merkmale mit Zukunftswissen):
# Decoder 40 Schritte im Training, statische Merkmale nur LoanGoal, ApplicationType, RequestedAmount (STATICS_WITHOUT_FUTURE),
# die neuen Wahlpunkte dreifach (Katalog-Kopie data/catalog_neue_punkte.yaml), Lernrate 1e-4, 20 Epochen, GPU.
# Danach: Diagnose (scripts/psp_diagnose.py) und online an den neuen Punkten (wie data/bpic17_neue_punkte.sh, nur psp).
# Es bleiben models/psp_model_durchlaufzeit_<TAG>.pkl, models/psp_training_<TAG>/, data/PSP_<TAG>.log,
# scripts/results_100k/psp_diagnose_<TAG>, scripts/results_100k/neue_punkte_online_<TAG>.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
TAG=${TAG:-real_np_s40_ohne_zukunft}
LOG=$P/data/PSP_$TAG.log
MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte.yaml
cd $ROOT/process-simulation-engine
echo "######## Training ($(date +%H:%M:%S))" | tee -a "$LOG"
[ -e "$MODEL" ] || .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --cases ${CASES:-10000} --epochs ${EPOCHS:-20} \
    --learning-rate ${RATE:-1e-4} --point-weight 3 --catalog-points --suffix-size ${SUFFIX:-40} --statics-without-future 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ -e "$MODEL" ] || { echo "######## Abbruch im Training ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Diagnose ($(date +%H:%M:%S))" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/psp_diagnose.py --model "$MODEL" --out "$P/scripts/results_100k/psp_diagnose_$TAG" 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
echo "######## Online ($(date +%H:%M:%S))" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/main.py --delta 0.5 --rct-cases 10000 --rct-runs 10 --steps online \
    --online-policies psp --psp-model "$MODEL" --online-seeds 10 --online-cases 3000 --online-warmup 700 --online-cooldown 1300 \
    --online-workers 12 --out "$P/scripts/results_100k/neue_punkte_online_$TAG" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch online ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
