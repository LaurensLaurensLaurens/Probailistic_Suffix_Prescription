#!/bin/bash
# Nur die acht Mitarbeiterpunkte unter hoher Ankunftslast (Faktor 4): 50.000 RCT-Fälle (2 Läufe zu je 25.000, delta 0.5), randomisiert nur dort, channel läuft wie die Bank. XGBoost darauf trainieren,
# online Bank, Zufall, XGBoost, 5 Seeds, je 5.000 gewertete Fälle mit 30 Tagen Warm- und Auslauf. Outcome: Durchlaufzeit. Es bleiben: data/rct_bpic17_channel_delta50_Last4,
# models/xgb_model_durchlaufzeit_Last4.pkl, scripts/results_100k/Last4, data/rct50k_last4.log. Danach stehen Katalog und Standardmodell wie vorher.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_100k/Last4
LOG=$P/data/rct50k_last4.log
XGB=$P/models/xgb_model_durchlaufzeit.pkl
DATA=$P/data/rct_bpic17_channel_delta50
RUN=".venv/bin/python -u ../prescriptive/scripts/main.py --outcome bpic17=durchlaufzeit --delta 0.5 --rct-cases 25000 --rct-runs 2"
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
    set_value mitarbeiterwahl "$THREE"; set_value nur_mitarbeiter false; set_value ankunft_faktor 1
    [ ! -e "$DATA" ] || mv "$DATA" "${DATA}_Last4"
    mv "$XGB.vor_Last4" "$XGB"
    echo "######## Katalog und Standardmodell wiederhergestellt ($(date +%H:%M:%S))" | tee -a "$LOG"
}
cp -p "$XGB" "$XGB.vor_Last4"
trap restore EXIT
set_value mitarbeiterwahl "$EMPLOYEES"; set_value nur_mitarbeiter true; set_value ankunft_faktor 4
cd $ROOT/process-simulation-engine
echo "######## Simulation und Training ($(date +%H:%M:%S))" | tee -a "$LOG"
$RUN --steps rct train models --estimators xgb --retrain-estimators --workers 2 --out "$OUT" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch in Simulation oder Training" | tee -a "$LOG"; exit 1; }
cp -p "$XGB" "$P/models/xgb_model_durchlaufzeit_Last4.pkl"
echo "######## online ($(date +%H:%M:%S))" | tee -a "$LOG"
$RUN --steps online --online-policies bank random xgb --online-seeds 5 --online-workers 8 --online-cases 24200 --online-warmup 9600 --online-cooldown 9600 --out "$OUT" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch online" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
