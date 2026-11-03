#!/bin/bash
# PSP auf BPIC19: Training auf 20.000 Fällen des echten Logs, danach der Vergleich mit XGBoost (prescriptive/scripts/psp_risk_bpic19.py).
# Es bleiben: models/psp_model_bpic19_risk.pkl, models/psp_training_bpic19_risk, scripts/results_risk_bpic19, data/psp_risk_bpic19.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
LOG=$ROOT/prescriptive/data/psp_risk_bpic19.log
FILTER='Warning|warn\(|cuda|CUDA|^  [0-9]+ von'
cd $ROOT/process-simulation-engine
echo "######## Training ($(date +%H:%M:%S))" | tee -a "$LOG"
.venv/bin/python -u ../prescriptive/scripts/psp_risk_bpic19.py --train --cases 20000 --epochs 30 --lr 1e-4 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch im Training ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Vergleich ($(date +%H:%M:%S))" | tee -a "$LOG"
.venv/bin/python -u ../prescriptive/scripts/psp_risk_bpic19.py --cases 20000 --test 4000 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch im Vergleich ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
