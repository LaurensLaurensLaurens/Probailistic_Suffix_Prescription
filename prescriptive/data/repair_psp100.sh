#!/bin/bash
# Werkstatt (repair): das PSP neu, 100 statt 25 Epochen, sonst wie repair_pipeline.sh (dieselben 5.000 Trainingsfälle der
# 6.250 gezogenen, Lernrate 1e-4). Das 25-Epochen-Modell bleibt als models/psp_model_repair_25ep.pkl (Checkpoints
# psp_training_repair_25ep). Danach scripts/psp_option_check.py für beide Modelle: unterscheidet das PSP die Mitarbeiter?
# Die Online-Auswertung erst danach und nur, wenn ja (nicht in diesem Skript).
# Es bleiben: models/psp_model_repair.pkl (100 Epochen), scripts/results_repair_psp100, data/repair_psp100.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_repair_psp100
LOG=$P/data/repair_psp100.log
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory|super\(\).__init__'
cd $ROOT/process-simulation-engine
[ -e $P/models/psp_model_repair_25ep.pkl ] || cp -p $P/models/psp_model_repair.pkl $P/models/psp_model_repair_25ep.pkl
[ -e $P/models/psp_training_repair_25ep ] || cp -rp $P/models/psp_training_repair $P/models/psp_training_repair_25ep
mkdir -p $OUT
echo "######## Training PSP, 100 Epochen ($(date +%H:%M:%S))" | tee -a "$LOG"
.venv/bin/python -u ../prescriptive/scripts/main.py --datasets repair --delta 0.5 --rct-cases 10000 --rct-runs 1 --workers 16 --estimators psp \
    --train-cases 6250 --out $OUT --steps rct train --psp-train-cases None --psp-epochs 100 --psp-learning-rate 1e-4 --retrain-estimators \
    2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch im Training ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
for MODEL in psp_model_repair_25ep psp_model_repair; do
    echo "######## Prüfung $MODEL ($(date +%H:%M:%S))" | tee -a "$LOG"
    .venv/bin/python -u ../prescriptive/scripts/psp_option_check.py --dataset repair --model $P/models/$MODEL.pkl --out $OUT/optionen_$MODEL.csv \
        2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch in der Prüfung ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
done
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
