#!/bin/bash
# Werkstatt (repair): psp_erm (policies/psp_erm_policy.py) mit dem PSP aus 100 Epochen für mehrere γ, je γ online im gelernten
# Simulator (nur psp_erm, die Bank kommt immer mit; Fenster und Seeds wie repair_psp100_online.sh, also gepaart mit den Läufen
# von Zufall, XGBoost und psp dort) und in der echten Werkstatt (repair_truth.py, Bank und psp_erm). Die γ laufen parallel.
# γ aus der Streuung der gezogenen Durchlaufzeit je Option (Median 1,65 Tage): 0,1 wählte wie psp, 0,3 / 1 / 3 anders in 11 / 19 / 29 %.
# Das PSP ist models/psp_model_repair.pkl (bis 02.11.2026 das aus 100 Epochen, seither das mit gewichteten Wahlpunkten, repair_psp_w5.sh).
# Ergebnisse in scripts/results_repair_erm/gamma_<γ>/ (prescription/PSP_ERM, wahrheit/), Log data/repair_erm.log; anderer Ordner mit
# NAME, z. B. NAME=repair_erm_w5: scripts/results_repair_erm_w5, data/repair_erm_w5.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
NAME=${NAME:-repair_erm}
LOG=$P/data/$NAME.log
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory|super\(\).__init__'
GAMMAS=${GAMMAS:-"0.3 1 3"}
cd $ROOT/process-simulation-engine
run() {
    G=$1
    OUT=$P/scripts/results_$NAME/gamma_$G
    echo "######## γ=$G online ($(date +%H:%M:%S))" | tee -a "$LOG"
    .venv/bin/python -u ../prescriptive/scripts/main.py --datasets repair --delta 0.5 --rct-cases 10000 --rct-runs 1 --workers 16 --estimators xgb psp \
        --train-cases 6250 --out $OUT --steps online --online-policies psp_erm --erm-gamma $G --online-seeds 10 --online-workers 10 --online-cases 3000 \
        --online-warmup 700 --online-cooldown 1300 2>&1 | grep --line-buffered -v -E "$FILTER" > $OUT.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch γ=$G online ($(date +%H:%M:%S))" | tee -a "$LOG"; return 1; }
    echo "######## γ=$G Wahrheit ($(date +%H:%M:%S))" | tee -a "$LOG"
    .venv/bin/python -u ../prescriptive/scripts/repair_truth.py --policies bank psp_erm --erm-gamma $G --seeds 10 --workers 10 --out $OUT/wahrheit \
        2>&1 | grep --line-buffered -v -E "$FILTER" >> $OUT.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch γ=$G Wahrheit ($(date +%H:%M:%S))" | tee -a "$LOG"; return 1; }
    echo "######## γ=$G fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
}
mkdir -p $P/scripts/results_$NAME
for G in $GAMMAS; do run $G & done
wait
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
