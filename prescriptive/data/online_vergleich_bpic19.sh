#!/bin/bash
# BPIC19, Outcome Durchlaufzeit: Bank, Zufall, XGBoost und PSP online. 1. PSP auf den vorhandenen RCT-Daten (5.000 Fälle, randomisiert nur an
# invoice_worker und block_remover), auf denselben 4.000 Trainingsfällen wie XGBoost (models/xgb_model_bpic19.pkl), 25 Epochen, Lernrate 1e-4
# (wie PSP_lr1e4 auf BPIC17). 2. Online Bank, Zufall, XGBoost, PSP: 10 Seeds, 5.000 Fälle, alle gewertet (wie bpic19_5k.sh). 3. Kennzahlen beider Schätzer.
# Es bleiben: models/psp_model_bpic19.pkl, models/psp_training_bpic19, scripts/results_bpic19_durchlaufzeit, data/online_vergleich_bpic19.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_bpic19_durchlaufzeit
LOG=$P/data/online_vergleich_bpic19.log
RUN=".venv/bin/python -u ../prescriptive/scripts/main.py --datasets bpic19 --delta 0.5 --rct-cases 5000 --rct-runs 1 --workers 12 --estimators xgb psp --out $OUT"
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory|super\(\).__init__'
cd $ROOT/process-simulation-engine
echo "######## PSP-Training ($(date +%H:%M:%S))" | tee -a "$LOG"
$RUN --steps train --psp-train-cases None --psp-epochs 25 --psp-learning-rate 1e-4 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch im Training ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## online ($(date +%H:%M:%S))" | tee -a "$LOG"
$RUN --steps online --online-policies bank random xgb psp --online-seeds 10 --online-workers 12 --online-cases 5000 --online-warmup 0 --online-cooldown 0 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch online ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Kennzahlen ($(date +%H:%M:%S))" | tee -a "$LOG"
$RUN --steps models 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch Kennzahlen ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
