#!/bin/bash
# Ein gemeinsamer Lauf über alle Punkte: alle Control-Flow-Punkte (am Event O_Created eine Wahl aus channel und angebot_nachlegen) und alle acht Mitarbeiterpunkte.
# 100.000 RCT-Fälle (10 Läufe zu je 10.000, delta 0.5), ein XGBoost-Modell, online Bank, Zufall, XGBoost an allen Punkten, 10 Seeds. Outcome: Durchlaufzeit.
# Argumente: Fälle je Lauf, Läufe, Seeds, Ablage (Standard: wie oben); danach stehen Katalog und Modell wieder wie vorher.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
CASES=${1:-10000}; RUNS=${2:-10}; SEEDS=${3:-10}; TAG=${4:-AllePunkte}; DELTA=${5:-0.5}; ONLINE=${6:-4000}; WARM=${7:-700}; COOL=${8:-1300}
OUT=$P/scripts/results_100k/$TAG
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
restore() { set_list mitarbeiterwahl "$THREE"; set_list control_flow '[]'; cp -p "$P/models/xgb_model_durchlaufzeit_30k.pkl" "$P/models/xgb_model_durchlaufzeit.pkl"; }
trap restore EXIT
set_list mitarbeiterwahl "$EMPLOYEES"
set_list control_flow '["handle_leads", "annehmen_nach_vervollstaendigung", "angebot_nachlegen", "neues_angebot_beim_nachfassen"]'
cd $ROOT/process-simulation-engine
echo "######## Simulation und Training ($(date +%H:%M:%S))"
.venv/bin/python -u ../prescriptive/scripts/main.py --outcome bpic17=durchlaufzeit --steps rct train models --estimators xgb --retrain-estimators \
    --delta $DELTA --rct-cases $CASES --rct-runs $RUNS --workers 10 --out "$OUT" 2>&1 | grep --line-buffered -v -E "$FILTER"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch in Simulation oder Training"; exit 1; }
echo "######## online ($(date +%H:%M:%S))"
.venv/bin/python -u ../prescriptive/scripts/main.py --outcome bpic17=durchlaufzeit --steps online --delta $DELTA --rct-cases $CASES --rct-runs $RUNS \
    --online-policies bank random xgb --online-seeds $SEEDS --online-workers 14 --online-cases $ONLINE --online-warmup $WARM --online-cooldown $COOL --out "$OUT" 2>&1 | grep --line-buffered -v -E "$FILTER"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch online"; exit 1; }
SUF=$([ "$DELTA" = "0.5" ] && echo delta50 || echo delta25)
mv "$P/data/rct_bpic17_channel_$SUF" "$P/data/rct_bpic17_channel_${SUF}_$TAG"
cp -p "$P/models/xgb_model_durchlaufzeit.pkl" "$P/models/xgb_model_durchlaufzeit_$TAG.pkl"
echo "######## Alles fertig ($(date +%H:%M:%S))"
