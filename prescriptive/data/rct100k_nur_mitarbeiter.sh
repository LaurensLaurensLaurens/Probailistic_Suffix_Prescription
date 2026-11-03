#!/bin/bash
# Ein Lauf nur mit den acht Mitarbeiterpunkten: 100.000 RCT-Fälle (10 Läufe zu je 10.000, delta 0.5), randomisiert nur dort, channel läuft wie die Bank. XGBoost darauf trainieren,
# online Bank, Zufall, XGBoost, 10 Seeds, je 2.000 gewertete Fälle. Outcome: Durchlaufzeit. Es bleiben: data/rct_bpic17_channel_delta50_NurMitarbeiter,
# models/xgb_model_durchlaufzeit_NurMitarbeiter.pkl, scripts/results_100k/NurMitarbeiter, data/rct100k_nur_mitarbeiter.log. Danach stehen Katalog und Standardmodell wie vorher.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_100k/NurMitarbeiter
LOG=$P/data/rct100k_nur_mitarbeiter.log
XGB=$P/models/xgb_model_durchlaufzeit.pkl
DATA=$P/data/rct_bpic17_channel_delta50
RUN=".venv/bin/python -u ../prescriptive/scripts/main.py --outcome bpic17=durchlaufzeit --delta 0.5 --rct-cases 10000 --rct-runs 10"
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory'
set_value() {
    python3 - "$P/catalog.yaml" "$1" "$2" <<'PY'
import re, sys
path, key, value = sys.argv[1:4]
text = open(path, encoding="utf-8").read()
new, n = re.subn(rf"^    {key}: .*$", f"    {key}: {value}".replace("\\", "\\\\"), text, count=1, flags=re.M)
assert n == 1
open(path, "w", encoding="utf-8").write(new)
PY
}
EMPLOYEES='["Mitarbeiter: W_Call incomplete files", "Mitarbeiter: W_Complete application", "Mitarbeiter: W_Handle leads", "Mitarbeiter: W_Validate application", "Mitarbeiter: W_Call after offers (resume)", "Mitarbeiter: W_Call incomplete files (resume)", "Mitarbeiter: W_Complete application (resume)", "Mitarbeiter: W_Validate application (resume)"]'
THREE='["Mitarbeiter: W_Call incomplete files", "Mitarbeiter: W_Complete application", "Mitarbeiter: W_Handle leads"]'
[ ! -e "$DATA" ] || { echo "######## $DATA ist belegt, Abbruch" | tee -a "$LOG"; exit 1; }
restore() {
    set_value mitarbeiterwahl "$THREE"; set_value nur_mitarbeiter false
    [ ! -e "$DATA" ] || mv "$DATA" "${DATA}_NurMitarbeiter"
    mv "$XGB.vor_NurMitarbeiter" "$XGB"
    echo "######## Katalog und Standardmodell wiederhergestellt ($(date +%H:%M:%S))" | tee -a "$LOG"
}
cp -p "$XGB" "$XGB.vor_NurMitarbeiter"
trap restore EXIT
set_value mitarbeiterwahl "$EMPLOYEES"; set_value nur_mitarbeiter true
cd $ROOT/process-simulation-engine
echo "######## Simulation und Training ($(date +%H:%M:%S))" | tee -a "$LOG"
$RUN --steps rct train models --estimators xgb --retrain-estimators --workers 6 --out "$OUT" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch in Simulation oder Training" | tee -a "$LOG"; exit 1; }
cp -p "$XGB" "$P/models/xgb_model_durchlaufzeit_NurMitarbeiter.pkl"
echo "######## online ($(date +%H:%M:%S))" | tee -a "$LOG"
$RUN --steps online --online-policies bank random xgb --online-seeds 10 --online-workers 6 --online-cases 4000 --online-warmup 700 --online-cooldown 1300 --out "$OUT" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch online" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
