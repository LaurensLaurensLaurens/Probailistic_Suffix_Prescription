#!/bin/bash
# BPIC19 (Outcome Durchlaufzeit): das PSP mit gewichteten Wahlpunkten (TRAIN point_weight 5: die Proben, deren Prefix an invoice_worker oder
# block_remover endet, fünfmal im Training), sonst wie online_vergleich_bpic19.sh (dieselben 4.000 Fälle, 25 Epochen, Lernrate 1e-4). Das PSP
# ohne Gewicht bleibt als models/psp_model_bpic19_ungewichtet.pkl (Checkpoints psp_training_bpic19_ungewichtet). Danach psp_option_check.py
# (beide PSP gegen XGBoost; ohne Wahrheit: Übereinstimmung) und die Suffix-Tabelle mit beiden PSP. Online erst danach.
# Es bleiben: models/psp_model_bpic19.pkl (gewichtet), scripts/results_bpic19_psp_w5, data/bpic19_psp_w5.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_bpic19_psp_w5
LOG=$P/data/bpic19_psp_w5.log
RCT=rct_bpic19_remove_payment_block_delta50
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory|super\(\).__init__'
cd $ROOT/process-simulation-engine
[ -e $P/models/psp_model_bpic19_ungewichtet.pkl ] || cp -p $P/models/psp_model_bpic19.pkl $P/models/psp_model_bpic19_ungewichtet.pkl
[ -e $P/models/psp_training_bpic19_ungewichtet ] || cp -rp $P/models/psp_training_bpic19 $P/models/psp_training_bpic19_ungewichtet
mkdir -p $OUT
echo "######## Training PSP, Wahlpunkte fünffach ($(date +%H:%M:%S))" | tee -a "$LOG"
.venv/bin/python -u ../prescriptive/scripts/main.py --datasets bpic19 --delta 0.5 --rct-cases 5000 --rct-runs 1 --workers 12 --estimators psp --out $OUT \
    --steps train --psp-train-cases None --psp-epochs 25 --psp-learning-rate 1e-4 --psp-point-weight 5 --retrain-estimators \
    2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch im Training ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
for MODEL in psp_model_bpic19_ungewichtet psp_model_bpic19; do
    echo "######## Prüfung $MODEL ($(date +%H:%M:%S))" | tee -a "$LOG"
    .venv/bin/python -u ../prescriptive/scripts/psp_option_check.py --dataset bpic19 --rct $RCT --models psp xgb --model $P/models/$MODEL.pkl \
        --out $OUT/optionen_$MODEL.csv 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch in der Prüfung ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
done
echo "######## Suffix-Tabelle ($(date +%H:%M:%S))" | tee -a "$LOG"
.venv/bin/python -u ../prescriptive/scripts/zwischenergebnisse.py suffix --dataset bpic19 --name BPIC19 --threads 8 \
    --psp "PSP (4.000 Fälle, 25 Epochen, 1e-4)=psp_model_bpic19_ungewichtet.pkl" "PSP (25 Epochen, Wahlpunkte 5-fach)=psp_model_bpic19.pkl" \
    --out ../EXTRAZWISCHENERGEBNISSE/2019/suffix_vorhersage_bpic19.png 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch Suffix-Tabelle ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
