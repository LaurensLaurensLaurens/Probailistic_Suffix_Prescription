#!/bin/bash
# PSP mit Online-Auswertung in einer Kette, auf den RCT-Daten AllePunkte (kein neues Simulieren). Outcome: Durchlaufzeit.
# 1. PSP trainieren (Token, 5 Decoder-Schritte), bei instabilem Validierungsverlust Abbruch. 2. Online Bank, Zufall, PSP an allen 12 Punkten.
# Daneben auf den freien Kernen: XGBoost auf gleich vielen Trainingsfällen mit derselben Online-Auswertung (fairer Vergleich), danach die Kennzahlen des PSP.
# Argumente: Trainingsfälle, Epochen, Ablage, Lernrate, Ankünfte, Warmlauf, Auslauf, Seeds, XGB-Fälle (4/5 Training), Grenze Verlust.
# Es bleiben: models/psp_model_durchlaufzeit_<Ablage>.pkl, models/xgb_model_durchlaufzeit_<Ablage>.pkl, models/psp_training_<Ablage>, scripts/results_100k/<Ablage>
# (Kennzahlen), <Ablage>_online, <Ablage>_xgb, data/<Ablage>.log, data/<Ablage>_neben.log. Danach stehen Katalog, Standardmodelle und models/psp_training wie vorher.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
CASES=${1:-5000}; EPOCHS=${2:-25}; TAG=${3:-PSP_lr1e4}; RATE=${4:-1e-4}; ONLINE=${5:-3000}; WARM=${6:-700}; COOL=${7:-1300}; SEEDS=${8:-10}; XGB_CASES=${9:-6250}; MAX_LOSS=${10:-4.5}
OUT=$P/scripts/results_100k/$TAG
LOG=$P/data/$TAG.log
SIDE=$P/data/${TAG}_neben.log
MODEL=$P/models/psp_model_durchlaufzeit.pkl
XGB=$P/models/xgb_model_durchlaufzeit.pkl
DATA=$P/data/rct_bpic17_channel_delta50
RUN=".venv/bin/python -u ../prescriptive/scripts/main.py --outcome bpic17=durchlaufzeit --delta 0.5 --rct-cases 10000 --rct-runs 10"
WINDOW="--online-seeds $SEEDS --online-cases $ONLINE --online-warmup $WARM --online-cooldown $COOL"
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory|super\(\).__init__'
set_list() {
    python3 - "$P/catalog.yaml" "$1" "$2" <<'PY'
import re, sys
path, key, value = sys.argv[1:4]
text = open(path, encoding="utf-8").read()
new, n = re.subn(rf"^    {key}: \[.*\]$", f"    {key}: {value}", text, flags=re.M)
assert n == 1
open(path, "w", encoding="utf-8").write(new)
PY
}
EMPLOYEES='["Mitarbeiter: W_Call incomplete files", "Mitarbeiter: W_Complete application", "Mitarbeiter: W_Handle leads", "Mitarbeiter: W_Validate application", "Mitarbeiter: W_Call after offers (resume)", "Mitarbeiter: W_Call incomplete files (resume)", "Mitarbeiter: W_Complete application (resume)", "Mitarbeiter: W_Validate application (resume)"]'
THREE='["Mitarbeiter: W_Call incomplete files", "Mitarbeiter: W_Complete application", "Mitarbeiter: W_Handle leads"]'
[ ! -e "$DATA" ] || { echo "######## $DATA ist belegt, Abbruch"; exit 1; }
[ ! -e "$P/models/psp_training_$TAG" ] || { echo "######## models/psp_training_$TAG gibt es schon, Abbruch"; exit 1; }
restore() {
    set_list mitarbeiterwahl "$THREE"; set_list control_flow '[]'
    rm -f "$DATA"
    mv "$P/models/psp_training" "$P/models/psp_training_$TAG"
    mv "$P/models/psp_training_vor_$TAG" "$P/models/psp_training"
    mv "$MODEL.vor_$TAG" "$MODEL"
    mv "$XGB.vor_$TAG" "$XGB"
    echo "######## Katalog und Standardmodelle wiederhergestellt ($(date +%H:%M:%S))" | tee -a "$LOG"
}
# Auf den freien Kernen neben der Online-Auswertung; ein Fehler hier stoppt die Online-Auswertung nicht
side() {
    echo "######## XGBoost auf $XGB_CASES Fällen ($(date +%H:%M:%S))"
    $RUN --steps rct train models --estimators xgb --retrain-estimators --train-cases $XGB_CASES --out "${OUT}_xgb" 2>&1 | grep --line-buffered -v -E "$FILTER"
    if [ "${PIPESTATUS[0]}" -eq 0 ]; then
        cp -p "$XGB" "$P/models/xgb_model_durchlaufzeit_$TAG.pkl"
        echo "######## XGBoost online ($(date +%H:%M:%S))"
        $RUN --steps online --online-policies bank random xgb $WINDOW --online-workers 5 --out "${OUT}_xgb" 2>&1 | grep --line-buffered -v -E "$FILTER"
        [ "${PIPESTATUS[0]}" -eq 0 ] || echo "######## Abbruch XGBoost online"
    else
        echo "######## Abbruch XGBoost-Training"
    fi
    echo "######## Kennzahlen PSP ($(date +%H:%M:%S))"
    OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 $RUN --steps models --estimators psp --out "$OUT" 2>&1 | grep --line-buffered -v -E "$FILTER"
    [ "${PIPESTATUS[0]}" -eq 0 ] || echo "######## Abbruch Kennzahlen PSP"
    echo "######## Nebenarbeiten fertig ($(date +%H:%M:%S))"
}
cp -p "$MODEL" "$MODEL.vor_$TAG"
cp -p "$XGB" "$XGB.vor_$TAG"
mv "$P/models/psp_training" "$P/models/psp_training_vor_$TAG"
trap restore EXIT
ln -s "$(basename "$DATA")_AllePunkte" "$DATA"
set_list mitarbeiterwahl "$EMPLOYEES"
set_list control_flow '["handle_leads", "annehmen_nach_vervollstaendigung", "angebot_nachlegen", "neues_angebot_beim_nachfassen"]'
cd $ROOT/process-simulation-engine
echo "######## Training: $CASES Fälle, $EPOCHS Epochen, Lernrate $RATE ($(date +%H:%M:%S))" | tee -a "$LOG"
$RUN --steps rct train --estimators psp --retrain-estimators --psp-train-cases $CASES --psp-epochs $EPOCHS --psp-learning-rate $RATE --out "$OUT" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch im Training" | tee -a "$LOG"; exit 1; }
cp -p "$MODEL" "$P/models/psp_model_durchlaufzeit_$TAG.pkl"
LOSS=$(grep "Avg Standard Validation Loss" "$LOG" | tail -1 | sed 's/.*: //')
python3 -c "import sys; sys.exit(0 if float('$LOSS') < $MAX_LOSS else 1)" || { echo "######## Validierungsverlust $LOSS über $MAX_LOSS: instabil, keine Online-Auswertung" | tee -a "$LOG"; exit 1; }
side 2>&1 | tee -a "$SIDE" > /dev/null &
echo "######## PSP online, Verlust $LOSS ($(date +%H:%M:%S))" | tee -a "$LOG"
$RUN --steps online --online-policies bank random psp $WINDOW --online-workers 12 --out "${OUT}_online" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || echo "######## Abbruch PSP online" | tee -a "$LOG"
wait
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
