#!/bin/bash
# Erkennt das PSP besser als XGBoost, welche laufenden Fälle ihre Frist reißen? Siehe prescriptive/scripts/psp_risk_check.py. Es bleiben: scripts/results_risk, data/psp_risk_check.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
LOG=$ROOT/prescriptive/data/psp_risk_check.log
cd $ROOT/process-simulation-engine
.venv/bin/python -u ../prescriptive/scripts/psp_risk_check.py --real 4000 --rct 2000 2>&1 | grep --line-buffered -v -E "Warning|warn\(|cuda|CUDA|^  [0-9]+ von" | tee "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] && echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG" || echo "######## Abbruch ($(date +%H:%M:%S))" | tee -a "$LOG"
