#!/bin/bash
# Wartet, bis die Online-Auswertung fertig ist (psp_lookahead_online.log), dann die Kennzahlen des PSP neu mit dem
# korrigierten Ziehen (psp_model.draw_cases ohne die Sperren von sample_suffix); dasselbe Modell, kein neues Training
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
until grep -q "######## Fertig\|Traceback" $ROOT/prescriptive/data/psp_lookahead_online.log; do sleep 60; done
cd $ROOT/process-simulation-engine
.venv/bin/python -u ../prescriptive/scripts/main.py --outcome bpic17=durchlaufzeit --steps models --estimators psp \
    --delta 0.5 --rct-cases 10000 \
    2>&1 | grep --line-buffered -v -E 'FutureWarning|applied = df'
