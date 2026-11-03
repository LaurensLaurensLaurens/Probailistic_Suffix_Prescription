#!/bin/bash
# PSP über Nacht: Training auf den RCT-Daten AllePunkte (kein neues Simulieren), lifecycle-aware, danach die Kennzahlen (Schritt models). Outcome: Durchlaufzeit.
# Argumente: Trainingsfälle, Epochen, Ablage, Decoder-Schritte (Standard: 5000, 30, PSP_AllePunkte, 5). Es bleiben: models/psp_model_durchlaufzeit_<Ablage>.pkl,
# models/psp_training_<Ablage> (nach jeder Epoche, auch bei Abbruch), scripts/results_100k/<Ablage>. Danach stehen Katalog, Standardmodell und models/psp_training wie vorher.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
CASES=${1:-5000}; EPOCHS=${2:-30}; TAG=${3:-PSP_AllePunkte}; STEPS=${4:-5}
OUT=$P/scripts/results_100k/$TAG
MODEL=$P/models/psp_model_durchlaufzeit.pkl
DATA=$P/data/rct_bpic17_channel_delta50
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
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr'
[ ! -e "$DATA" ] || { echo "######## $DATA ist belegt, Abbruch"; exit 1; }
[ ! -e "$P/models/psp_training_$TAG" ] || { echo "######## models/psp_training_$TAG gibt es schon, Abbruch"; exit 1; }
restore() {
    set_list mitarbeiterwahl "$THREE"; set_list control_flow '[]'
    rm -f "$DATA"
    mv "$P/models/psp_training" "$P/models/psp_training_$TAG"
    mv "$P/models/psp_training_vor_$TAG" "$P/models/psp_training"
    mv "$MODEL.vor_$TAG" "$MODEL"
}
cp -p "$MODEL" "$MODEL.vor_$TAG"
mv "$P/models/psp_training" "$P/models/psp_training_vor_$TAG"
trap restore EXIT
ln -s "$(basename "$DATA")_AllePunkte" "$DATA"
set_list mitarbeiterwahl "$EMPLOYEES"
set_list control_flow '["handle_leads", "annehmen_nach_vervollstaendigung", "angebot_nachlegen", "neues_angebot_beim_nachfassen"]'
cd $ROOT/process-simulation-engine
echo "######## Training: $CASES Fälle, $EPOCHS Epochen ($(date +%H:%M:%S))"
.venv/bin/python -u ../prescriptive/scripts/main.py --outcome bpic17=durchlaufzeit --steps rct train --estimators psp --retrain-estimators --psp-train-cases $CASES --psp-epochs $EPOCHS --psp-decoder-steps $STEPS \
    --delta 0.5 --rct-cases 10000 --rct-runs 10 --out "$OUT" 2>&1 | grep --line-buffered -v -E "$FILTER"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch im Training"; exit 1; }
cp -p "$MODEL" "$P/models/psp_model_durchlaufzeit_$TAG.pkl"
echo "######## Kennzahlen ($(date +%H:%M:%S))"
.venv/bin/python -u ../prescriptive/scripts/main.py --outcome bpic17=durchlaufzeit --steps models --estimators psp \
    --delta 0.5 --rct-cases 10000 --rct-runs 10 --out "$OUT" 2>&1 | grep --line-buffered -v -E "$FILTER"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch in den Kennzahlen, das Modell ist gespeichert"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))"
