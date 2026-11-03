#!/bin/bash
# BPIC17 (Outcome Durchlaufzeit): das PSP mit gewichteten Wahlpunkten (TRAIN point_weight 5: die Proben, deren Prefix am Options-Event eines der
# 12 Wahlpunkte endet, fünfmal im Training; etwa 28 % der Proben, also gut doppelt so viele Proben), sonst wie PSP_lr1e4 (psp_nacht_online.sh:
# RCT-Daten AllePunkte, dieselben 5.000 Fälle, 25 Epochen, Lernrate 1e-4, 8 Mitarbeiter- und 4 Control-Flow-Punkte im Katalog). Danach die
# Suffix-Tabelle (PSP_lr1e4 und das gewichtete PSP) und psp_option_check.py (beide PSP gegen XGBoost PSP_lr1e4, 50 Proben je Wahlpunkt).
# Online erst danach. Es bleiben: models/psp_model_durchlaufzeit_PSP_w5.pkl, models/psp_training_PSP_w5, scripts/results_100k/PSP_w5,
# data/PSP_w5.log. Danach stehen Katalog, Standardmodell und models/psp_training wie vorher.
# Die Durchlaufzeit ist inzwischen das Standard-Outcome von BPIC17 (kein --outcome mehr); das Training schreibt models/psp_model.pkl.
# Zum Probelauf: TAG, CASES, EPOCHS, SAMPLES aus der Umgebung (z. B. TAG=PSP_w5_test CASES=100 EPOCHS=1 SAMPLES=2).
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
TAG=${TAG:-PSP_w5}; CASES=${CASES:-5000}; EPOCHS=${EPOCHS:-25}; SAMPLES=${SAMPLES:-50}
OUT=$P/scripts/results_100k/$TAG
LOG=$P/data/$TAG.log
MODEL=$P/models/psp_model.pkl
DATA=$P/data/rct_bpic17_channel_delta50
RUN=".venv/bin/python -u ../prescriptive/scripts/main.py --datasets bpic17 --delta 0.5 --rct-cases 10000 --rct-runs 10"
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory|super\(\).__init__'
# Liste key im Abschnitt bpic17 des Katalogs auf value setzen; gibt den alten Wert aus
set_list() {
    python3 - "$P/catalog.yaml" "$1" "$2" <<'PY'
import re, sys
path, key, value = sys.argv[1:4]
text = open(path, encoding="utf-8").read()
start = text.index("\nbpic17:\n") + len("\nbpic17:\n")
end = re.compile(r"^\S", re.M).search(text, start).start()
pattern = re.compile(rf"^    {key}: (\[.*\])$", re.M)
found = pattern.findall(text[start:end])
assert len(found) == 1
part = pattern.sub(lambda m: f"    {key}: {value}", text[start:end])
open(path, "w", encoding="utf-8").write(text[:start] + part + text[end:])
print(found[0])
PY
}
EMPLOYEES='["Mitarbeiter: W_Call incomplete files", "Mitarbeiter: W_Complete application", "Mitarbeiter: W_Handle leads", "Mitarbeiter: W_Validate application", "Mitarbeiter: W_Call after offers (resume)", "Mitarbeiter: W_Call incomplete files (resume)", "Mitarbeiter: W_Complete application (resume)", "Mitarbeiter: W_Validate application (resume)"]'
CONTROL='["handle_leads", "annehmen_nach_vervollstaendigung", "angebot_nachlegen", "neues_angebot_beim_nachfassen"]'
[ ! -e "$DATA" ] || { echo "######## $DATA ist belegt, Abbruch" | tee -a "$LOG"; exit 1; }
[ ! -e "$P/models/psp_training_$TAG" ] || { echo "######## models/psp_training_$TAG gibt es schon, Abbruch" | tee -a "$LOG"; exit 1; }
OLD_EMPLOYEES=""; OLD_CONTROL=""
restore() {
    [ -z "$OLD_EMPLOYEES" ] || set_list mitarbeiterwahl "$OLD_EMPLOYEES" > /dev/null
    [ -z "$OLD_CONTROL" ] || set_list control_flow "$OLD_CONTROL" > /dev/null
    rm -f "$DATA"
    [ ! -e "$MODEL" ] || mv "$MODEL" "$P/models/psp_model_durchlaufzeit_$TAG.pkl"
    [ ! -e "$MODEL.vor_$TAG" ] || mv "$MODEL.vor_$TAG" "$MODEL"
    [ ! -e "$P/models/psp_training" ] || mv "$P/models/psp_training" "$P/models/psp_training_$TAG"
    mv "$P/models/psp_training_vor_$TAG" "$P/models/psp_training"
    echo "######## Katalog, Standardmodell und models/psp_training wiederhergestellt ($(date +%H:%M:%S))" | tee -a "$LOG"
}
[ ! -e "$MODEL" ] || mv "$MODEL" "$MODEL.vor_$TAG"
mv "$P/models/psp_training" "$P/models/psp_training_vor_$TAG"
trap restore EXIT
ln -s "$(basename "$DATA")_AllePunkte" "$DATA"
OLD_EMPLOYEES=$(set_list mitarbeiterwahl "$EMPLOYEES") || exit 1
OLD_CONTROL=$(set_list control_flow "$CONTROL") || exit 1
cd $ROOT/process-simulation-engine
echo "######## Training: $CASES Fälle, $EPOCHS Epochen, Lernrate 1e-4, Wahlpunkte fünffach ($(date +%H:%M:%S))" | tee -a "$LOG"
$RUN --steps rct train --estimators psp --retrain-estimators --psp-train-cases $CASES --psp-epochs $EPOCHS --psp-learning-rate 1e-4 --psp-point-weight 5 \
    --out "$OUT" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch im Training ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
cp -p "$MODEL" "$P/models/psp_model_durchlaufzeit_$TAG.pkl"
echo "######## Suffix-Tabelle ($(date +%H:%M:%S))" | tee -a "$LOG"
.venv/bin/python -u ../prescriptive/scripts/zwischenergebnisse.py suffix --dataset bpic17 --name BPIC17 --rct "$(basename "$DATA")_AllePunkte" --threads 8 \
    --psp "PSP (5.000 Fälle, 25 Epochen, 1e-4)=psp_model_durchlaufzeit_PSP_lr1e4.pkl" "PSP (25 Epochen, Wahlpunkte 5-fach)=psp_model_durchlaufzeit_$TAG.pkl" \
    --out "$OUT/suffix_vorhersage_bpic17.png" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || echo "######## Abbruch Suffix-Tabelle ($(date +%H:%M:%S))" | tee -a "$LOG"
for NAME in PSP_lr1e4 $TAG; do
    echo "######## Prüfung psp_model_durchlaufzeit_$NAME ($(date +%H:%M:%S))" | tee -a "$LOG"
    .venv/bin/python -u ../prescriptive/scripts/psp_option_check.py --dataset bpic17 --rct "$(basename "$DATA")_AllePunkte" --models psp xgb --samples $SAMPLES \
        --model "$P/models/psp_model_durchlaufzeit_$NAME.pkl" --paths "xgb=$P/models/xgb_model_durchlaufzeit_PSP_lr1e4.pkl" \
        --out "$OUT/optionen_psp_$NAME.csv" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || echo "######## Abbruch in der Prüfung ($(date +%H:%M:%S))" | tee -a "$LOG"
done
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
