#!/bin/bash
# Werkstatt (repair) mit dem PSP aus gewichteten Wahlpunkten (repair_psp_w5.sh, models/psp_model_repair.pkl): psp online im gelernten
# Simulator (die Bank kommt immer mit; Fenster und Seeds wie repair_psp100_online.sh, also gepaart mit Zufall, XGBoost und dem
# PSP aus 100 Epochen dort), danach Bank und psp in der echten Werkstatt (scripts/repair_truth.py, gleiches Fenster).
# Ergebnisse in scripts/results_repair_psp_w5 (prescription/PSP, wahrheit/), Log data/repair_psp_w5_online.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_repair_psp_w5
LOG=$P/data/repair_psp_w5_online.log
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory|super\(\).__init__'
cd $ROOT/process-simulation-engine
echo "######## online ($(date +%H:%M:%S))" | tee -a "$LOG"
.venv/bin/python -u ../prescriptive/scripts/main.py --datasets repair --delta 0.5 --rct-cases 10000 --rct-runs 1 --workers 16 --estimators xgb psp \
    --train-cases 6250 --out $OUT --steps online --online-policies psp --online-seeds 10 --online-workers 12 --online-cases 3000 \
    --online-warmup 700 --online-cooldown 1300 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch online ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Wahrheit: die echte Werkstatt ($(date +%H:%M:%S))" | tee -a "$LOG"
.venv/bin/python -u ../prescriptive/scripts/repair_truth.py --policies bank psp --seeds 10 --workers 12 --out $OUT/wahrheit \
    2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch Wahrheit ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
