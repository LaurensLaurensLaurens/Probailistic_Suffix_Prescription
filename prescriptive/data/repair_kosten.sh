#!/bin/bash
# Werkstatt (repair) mit dem Outcome kosten (prescriptive/costs/repair: Lohnkosten nach Stundensatz + 400 € je Tag Durchlaufzeit).
# Online im gelernten Simulator und in der echten Werkstatt (10 Seeds, Fenster wie repair_psp100_online.sh), in drei Teilen:
#   A   Tarif A (wie beim Training): Bank, Zufall, XGBoost (auf Kosten nach A trainiert), PSP; in der Werkstatt dazu der Spezialist
#   B   Tarifwechsel auf B: dasselbe, XGBoost weiter das nach A trainierte (es kennt B nicht), das PSP rechnet mit B
#   B2  XGBoost neu auf Kosten nach B trainiert (derselbe RCT-Datensatz, neu bewertet), online und in der Werkstatt
# Das PSP ist in allen Teilen das gewichtete models/psp_model_repair.pkl, nicht neu trainiert: die Pipeline sucht es unter dem Namen
# mit dem Outcome (psp_model_repair_kosten.pkl, _kosten_b.pkl), das sind Verknüpfungen darauf. Ebenso ist xgb_model_repair_kosten_b.pkl
# in Teil B eine Verknüpfung auf das nach A trainierte XGBoost; Teil B2 ersetzt sie durch das neu auf B trainierte.
# Ergebnisse in scripts/results_repair_kosten (A), results_repair_kosten_b (B), results_repair_kosten_b_xgbneu (B2), je mit wahrheit/;
# Log data/repair_kosten.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
M=$P/models
LOG=$P/data/repair_kosten.log
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory|super\(\).__init__'
RUN=".venv/bin/python -u ../prescriptive/scripts/main.py --datasets repair --delta 0.5 --rct-cases 10000 --rct-runs 1 --workers 16 --estimators xgb psp --train-cases 6250"
WINDOW="--online-seeds 10 --online-workers 14 --online-cases 3000 --online-warmup 700 --online-cooldown 1300"
TRUTH=".venv/bin/python -u ../prescriptive/scripts/repair_truth.py --seeds 10 --workers 14 --psp-model $M/psp_model_repair.pkl"
cd $ROOT/process-simulation-engine
ln -sfn psp_model_repair.pkl $M/psp_model_repair_kosten.pkl
ln -sfn psp_model_repair.pkl $M/psp_model_repair_kosten_b.pkl
ln -sfn xgb_model_repair_kosten.pkl $M/xgb_model_repair_kosten_b.pkl
step() {  # Name, Befehl
    echo "######## $1 ($(date +%H:%M:%S))" | tee -a "$LOG"
    eval "$2" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch: $1 ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
}
step "A online" "$RUN --outcome repair=kosten --out $P/scripts/results_repair_kosten --steps online --online-policies bank random xgb psp $WINDOW"
step "A Werkstatt" "$TRUTH --outcome kosten --policies bank random xgb psp spezialist --out $P/scripts/results_repair_kosten/wahrheit"
step "B online" "$RUN --outcome repair=kosten_b --out $P/scripts/results_repair_kosten_b --steps online --online-policies bank random xgb psp $WINDOW"
step "B Werkstatt" "$TRUTH --outcome kosten_b --policies bank random xgb psp spezialist --out $P/scripts/results_repair_kosten_b/wahrheit"
rm -f $M/xgb_model_repair_kosten_b.pkl
step "B2 XGBoost neu auf B" "$RUN --estimators xgb --outcome repair=kosten_b --out $P/scripts/results_repair_kosten_b_xgbneu --steps rct train --retrain-estimators"
step "B2 online" "$RUN --estimators xgb --outcome repair=kosten_b --out $P/scripts/results_repair_kosten_b_xgbneu --steps online --online-policies xgb $WINDOW"
step "B2 Werkstatt" "$TRUTH --outcome kosten_b --policies bank xgb --out $P/scripts/results_repair_kosten_b_xgbneu/wahrheit"
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
