#!/bin/bash
# PSP-Stufe 2 auf dem echten Log (Plan 2026-11-09): alle Trainingsfälle (--cases 0), Decoder 60, Abstände log-normal (log_gaps,
# gezogen höchstens 180 Tage), Durchlaufzeit aus der Summe der Abstände (time_from gaps), statische Merkmale ohne Zukunftswissen,
# die neuen Wahlpunkte dreifach (Katalog-Kopie data/catalog_neue_punkte.yaml), Lernrate 1e-4, 15 Epochen, GPU.
# EXTRA (z. B. --remaining-head für Variante b) und TAG aus der Umgebung. Danach die Diagnose (scripts/psp_diagnose.py);
# online erst nach Prüfung der Diagnose. Es bleiben models/psp_model_durchlaufzeit_<TAG>.pkl, models/psp_training_<TAG>/,
# data/PSP_<TAG>.log, scripts/results_100k/psp_diagnose_<TAG>.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
TAG=${TAG:-real_alle_s60_loggaps}
LOG=$P/data/PSP_$TAG.log
MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte.yaml
cd $ROOT/process-simulation-engine
echo "######## Training ($(date +%H:%M:%S))" | tee -a "$LOG"
[ -e "$MODEL" ] || .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --cases 0 --epochs ${EPOCHS:-15} --learning-rate 1e-4 \
    --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --log-gaps --time-from gaps $EXTRA 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ -e "$MODEL" ] || { echo "######## Abbruch im Training ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Diagnose ($(date +%H:%M:%S))" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/psp_diagnose.py --model "$MODEL" --out "$P/scripts/results_100k/psp_diagnose_$TAG" 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
