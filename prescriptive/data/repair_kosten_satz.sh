#!/bin/bash
# Werkstatt (repair), Outcome kosten: XGBoost mit den Stundensätzen der Option als Eingabe (scripts/train_xgb_rates.py, nach Tarif A
# trainiert): satz = Satz zum Zeitpunkt der Wahl, satz48 = dazu das Mittel der nächsten 48 Stunden. Je Variante online und in der
# echten Werkstatt unter Tarif A und nach dem Wechsel auf B (die Sätze in der Eingabe dann nach B, das Modell nicht neu trainiert).
# Fenster und Seeds wie repair_kosten.sh, also gepaart mit Bank, Zufall, XGBoost ohne Sätze und PSP dort.
# Ergebnisse in scripts/results_repair_kosten_xgb<variante> und results_repair_kosten_b_xgb<variante>, je mit wahrheit/; Log data/repair_kosten_satz.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
M=$P/models
LOG=$P/data/repair_kosten_satz.log
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory|super\(\).__init__'
RUN=".venv/bin/python -u ../prescriptive/scripts/main.py --datasets repair --delta 0.5 --rct-cases 10000 --rct-runs 1 --workers 16 --estimators xgb --train-cases 6250"
WINDOW="--online-seeds 10 --online-workers 14 --online-cases 3000 --online-warmup 700 --online-cooldown 1300"
TRUTH=".venv/bin/python -u ../prescriptive/scripts/repair_truth.py --seeds 10 --workers 14"
cd $ROOT/process-simulation-engine
step() {  # Name, Befehl
    echo "######## $1 ($(date +%H:%M:%S))" | tee -a "$LOG"
    eval "$2" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch: $1 ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
}
for V in satz satz48; do
    MODEL=$M/xgb_model_repair_kosten_$V.pkl
    step "$V A online" "$RUN --outcome repair=kosten --xgb-model $MODEL --out $P/scripts/results_repair_kosten_xgb$V --steps online --online-policies xgb $WINDOW"
    step "$V A Werkstatt" "$TRUTH --outcome kosten --xgb-model $MODEL --policies bank xgb --out $P/scripts/results_repair_kosten_xgb$V/wahrheit"
    step "$V B online" "$RUN --outcome repair=kosten_b --xgb-model $MODEL --out $P/scripts/results_repair_kosten_b_xgb$V --steps online --online-policies xgb $WINDOW"
    step "$V B Werkstatt" "$TRUTH --outcome kosten_b --xgb-model $MODEL --policies bank xgb --out $P/scripts/results_repair_kosten_b_xgb$V/wahrheit"
done
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
