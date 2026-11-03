#!/bin/bash
# Werkstatt (Datensatz repair), nach repair_simulator.sh: dieselbe Pipeline wie bei BPIC, dann die Wahrheit.
# 1. RCT im gelernten Simulator (10.000 Fälle, delta 0.5, Wahl nur bei W_repair start). 2. XGBoost und PSP auf denselben 6.250 zufällig gezogenen Fällen
# (5.000 Training, 1.250 Test), PSP 25 Epochen, Lernrate 1e-4 (wie PSP_lr1e4). 3. Online im gelernten Simulator: Bank, Zufall, XGBoost, PSP; 10 Seeds,
# 3.000 Ankünfte, 700 Warm- und 1.300 Auslauf. 4. Kennzahlen beider Schätzer. 5. Dieselben Policies und die Spezialisten-Regel in der echten Werkstatt
# (scripts/repair_truth.py, gleiches Fenster).
# Es bleiben: data/rct_repair_nacharbeit_delta50, models/xgb_model_repair.pkl, models/psp_model_repair.pkl, models/psp_training_repair,
# scripts/results_repair_pipeline (wahrheit/: die echte Werkstatt), data/repair_pipeline.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_repair_pipeline
LOG=$P/data/repair_pipeline.log
RUN=".venv/bin/python -u ../prescriptive/scripts/main.py --datasets repair --delta 0.5 --rct-cases 10000 --rct-runs 1 --workers 16 --estimators xgb psp --train-cases 6250 --out $OUT"
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory|super\(\).__init__'
cd $ROOT/process-simulation-engine
echo "######## RCT-Daten und Training ($(date +%H:%M:%S))" | tee -a "$LOG"
$RUN --steps rct train --psp-train-cases None --psp-epochs 25 --psp-learning-rate 1e-4 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch in RCT oder Training ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## online ($(date +%H:%M:%S))" | tee -a "$LOG"
$RUN --steps online --online-policies bank random xgb psp --online-seeds 10 --online-workers 14 --online-cases 3000 --online-warmup 700 --online-cooldown 1300 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch online ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Wahrheit: die echte Werkstatt ($(date +%H:%M:%S))" | tee -a "$LOG"
.venv/bin/python -u ../prescriptive/scripts/repair_truth.py --policies bank random xgb psp spezialist --seeds 10 --workers 14 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch Wahrheit ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Kennzahlen ($(date +%H:%M:%S))" | tee -a "$LOG"
$RUN --steps models 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch Kennzahlen ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
