#!/bin/bash
# BPIC19 mit der Engine ohne Lifecycles: Simulator trainieren (fehlt er) und gegen den echten Log prüfen, dann 5.000 RCT-Fälle (delta 0.5), randomisiert nur an den
# zwei Mitarbeiterpunkten (invoice_worker, block_remover), XGBoost darauf, online Bank, Zufall, XGBoost: 10 Seeds, alle 5.000 Fälle gewertet (ohne Belegung kein Warm-
# und Auslauf), zuletzt die Kennzahlen. Outcome: negative Tage bis zum letzten Clear Invoice. Es bleiben: models/next_event/bpic19, data/rct_bpic19_remove_payment_block_delta50,
# models/xgb_model_bpic19.pkl, scripts/results_bpic19, data/bpic19_5k.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_bpic19
LOG=$P/data/bpic19_5k.log
RUN=".venv/bin/python -u ../prescriptive/scripts/main.py --datasets bpic19 --delta 0.5 --rct-cases 5000 --rct-runs 1 --workers 1 --estimators xgb --out $OUT"
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory'
cd $ROOT/process-simulation-engine
echo "######## Simulator, Check, RCT, Training ($(date +%H:%M:%S))" | tee -a "$LOG"
$RUN --steps check rct train --n-cases 10000 --sim-threads 16 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch in Simulator, Check, RCT oder Training ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## online ($(date +%H:%M:%S))" | tee -a "$LOG"
$RUN --steps online --online-policies bank random xgb --online-seeds 10 --online-workers 8 --online-cases 5000 --online-warmup 0 --online-cooldown 0 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch online ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Kennzahlen ($(date +%H:%M:%S))" | tee -a "$LOG"
$RUN --steps models 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch Kennzahlen ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
