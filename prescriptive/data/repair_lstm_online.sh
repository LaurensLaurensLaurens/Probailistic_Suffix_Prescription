#!/bin/bash
# Werkstatt (repair): das LSTM (estimators/lstm_model.py, dieselben 5.000 Trainingsfälle wie XGBoost und PSP) online im gelernten
# Simulator (Bank, Zufall, LSTM; Seeds und Fenster wie repair_psp100_online.sh, also gepaart mit XGBoost und PSP dort) und in der
# echten Werkstatt (repair_truth.py). Ergebnisse in scripts/results_repair_lstm (wahrheit/), Log data/repair_lstm.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_repair_lstm
LOG=$P/data/repair_lstm.log
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory|super\(\).__init__'
cd $ROOT/process-simulation-engine
echo "######## online ($(date +%H:%M:%S))" | tee -a "$LOG"
.venv/bin/python -u ../prescriptive/scripts/main.py --datasets repair --delta 0.5 --rct-cases 10000 --rct-runs 1 --workers 16 --estimators lstm \
    --train-cases 6250 --out $OUT --steps online --online-policies bank random lstm --online-seeds 10 --online-workers 14 --online-cases 3000 \
    --online-warmup 700 --online-cooldown 1300 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch online ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Wahrheit: die echte Werkstatt ($(date +%H:%M:%S))" | tee -a "$LOG"
.venv/bin/python -u ../prescriptive/scripts/repair_truth.py --policies bank lstm --seeds 10 --workers 14 --out $OUT/wahrheit \
    2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch Wahrheit ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
