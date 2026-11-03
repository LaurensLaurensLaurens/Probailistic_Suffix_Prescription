#!/bin/bash
# Werkstatt (repair): das PSP mit gewichteten Wahlpunkten (TRAIN point_weight 5: die Proben, deren Prefix an W_repair start endet,
# fünfmal im Training), sonst wie repair_psp100.sh (dieselben 5.000 Trainingsfälle, 100 Epochen, Lernrate 1e-4). Das PSP aus
# 100 Epochen ohne Gewicht bleibt als models/psp_model_repair_100ep.pkl (Checkpoints psp_training_repair_100ep). Danach
# psp_option_check.py (PSP gewichtet, XGBoost, LSTM) und die Suffix-Tabelle mit allen drei PSP-Modellen. Online erst danach.
# Es bleiben: models/psp_model_repair.pkl (gewichtet), scripts/results_repair_psp_w5, data/repair_psp_w5.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_repair_psp_w5
LOG=$P/data/repair_psp_w5.log
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory|super\(\).__init__'
cd $ROOT/process-simulation-engine
[ -e $P/models/psp_model_repair_100ep.pkl ] || cp -p $P/models/psp_model_repair.pkl $P/models/psp_model_repair_100ep.pkl
[ -e $P/models/psp_training_repair_100ep ] || cp -rp $P/models/psp_training_repair $P/models/psp_training_repair_100ep
mkdir -p $OUT
echo "######## Training PSP, Wahlpunkte fünffach ($(date +%H:%M:%S))" | tee -a "$LOG"
.venv/bin/python -u ../prescriptive/scripts/main.py --datasets repair --delta 0.5 --rct-cases 10000 --rct-runs 1 --workers 16 --estimators psp \
    --train-cases 6250 --out $OUT --steps rct train --psp-train-cases None --psp-epochs 100 --psp-learning-rate 1e-4 --psp-point-weight 5 \
    --retrain-estimators 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch im Training ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Prüfung ($(date +%H:%M:%S))" | tee -a "$LOG"
.venv/bin/python -u ../prescriptive/scripts/psp_option_check.py --dataset repair --models psp xgb lstm --out $OUT/optionen_psp_w5_xgb_lstm.csv \
    2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch in der Prüfung ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Suffix-Tabelle ($(date +%H:%M:%S))" | tee -a "$LOG"
.venv/bin/python -u ../prescriptive/scripts/zwischenergebnisse.py suffix --dataset repair --name "Werkstatt (repair)" --threads 8 \
    --psp "PSP (5.000 Fälle, 25 Epochen, 1e-4)=psp_model_repair_25ep.pkl" "PSP (5.000 Fälle, 100 Epochen, 1e-4)=psp_model_repair_100ep.pkl" \
    "PSP (100 Epochen, Wahlpunkte 5-fach)=psp_model_repair.pkl" --out ../EXTRAZWISCHENERGEBNISSE/repair/suffix_vorhersage_repair.png \
    2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch Suffix-Tabelle ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
