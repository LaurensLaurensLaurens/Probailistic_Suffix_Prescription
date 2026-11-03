#!/bin/bash
# BPIC19 mit dem PSP aus gewichteten Wahlpunkten (bpic19_psp_w5.sh, models/psp_model_bpic19.pkl): psp online im gelernten Simulator (die Bank kommt
# immer mit; Fenster und Seeds wie online_vergleich_bpic19.sh: 10 Seeds, 5.000 Fälle, alle gewertet, also gepaart mit Zufall, XGBoost und dem
# ungewichteten PSP in scripts/results_bpic19_durchlaufzeit). Ergebnisse in scripts/results_bpic19_psp_w5 (prescription/PSP), Log data/bpic19_psp_w5_online.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_bpic19_psp_w5
LOG=$P/data/bpic19_psp_w5_online.log
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory|super\(\).__init__'
cd $ROOT/process-simulation-engine
echo "######## online ($(date +%H:%M:%S))" | tee -a "$LOG"
.venv/bin/python -u ../prescriptive/scripts/main.py --datasets bpic19 --delta 0.5 --rct-cases 5000 --rct-runs 1 --workers 12 --estimators xgb psp --out $OUT \
    --steps online --online-policies psp --online-seeds 10 --online-workers 16 --online-cases 5000 --online-warmup 0 --online-cooldown 0 \
    2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch online ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
