#!/bin/bash
# Werkstatt (repair): die neue psp_lookahead (policies/psp_lookahead_policy.py: Umverteilen und Warten mit Blick auf die anderen
# Fälle) gegen psp, mit demselben gewichteten PSP (models/psp_model_repair.pkl). Je Outcome (kosten, Durchlaufzeit) in der echten
# Werkstatt (repair_truth.py: Hausregel, psp, psp_lookahead, Spezialist) und im gelernten Simulator (Bank, psp, psp_lookahead),
# beide parallel; 10 Seeds, Fenster wie repair_kosten.sh. Erst kosten, dann Durchlaufzeit.
# Ergebnisse in scripts/results_repair_vorausschau_kosten und results_repair_vorausschau_dlz, je mit wahrheit/; Log data/repair_lookahead.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
LOG=$P/data/repair_lookahead.log
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory|super\(\).__init__'
RUN=".venv/bin/python -u ../prescriptive/scripts/main.py --datasets repair --delta 0.5 --rct-cases 10000 --rct-runs 1 --workers 16 --estimators psp --train-cases 6250"
WINDOW="--online-seeds 10 --online-workers 10 --online-cases 3000 --online-warmup 700 --online-cooldown 1300"
TRUTH=".venv/bin/python -u ../prescriptive/scripts/repair_truth.py --seeds 10 --workers 10 --psp-model $P/models/psp_model_repair.pkl"
cd $ROOT/process-simulation-engine
step() {  # Name, Befehl, Log des Schritts
    echo "######## $1 ($(date +%H:%M:%S))" | tee -a "$LOG"
    eval "$2" 2>&1 | grep --line-buffered -v -E "$FILTER" > "$3"
    if [ "${PIPESTATUS[0]}" -eq 0 ]; then echo "######## fertig: $1 ($(date +%H:%M:%S))" | tee -a "$LOG"; else echo "######## Abbruch: $1 ($(date +%H:%M:%S))" | tee -a "$LOG"; fi
}
for O in kosten dlz; do
    OUT=$P/scripts/results_repair_vorausschau_$O
    mkdir -p $OUT
    if [ $O = kosten ]; then OPT="--outcome repair=kosten"; TOPT="--outcome kosten"; else OPT=""; TOPT=""; fi
    step "$O Werkstatt" "$TRUTH $TOPT --policies bank psp psp_lookahead spezialist --out $OUT/wahrheit" $OUT/werkstatt.log &
    step "$O online" "$RUN $OPT --out $OUT --steps online --online-policies psp psp_lookahead $WINDOW" $OUT/online.log &
    wait
done
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
