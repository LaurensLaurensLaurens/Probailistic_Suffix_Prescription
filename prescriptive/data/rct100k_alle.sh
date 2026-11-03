#!/bin/bash
# Wartet auf die Einzelläufe (rct100k_lauf.sh), dann ein gemeinsamer Lauf: 100.000 RCT-Fälle, randomisiert an allen acht Mitarbeiterpunkten und channel, ein XGBoost-Modell
# dafür, online Bank, Zufall, XGBoost: einmal Wahl an allen neun Punkten, einmal nur an den acht Mitarbeiterpunkten. Ablage: data/rct_bpic17_channel_delta50_Alle,
# models/xgb_model_durchlaufzeit_Alle.pkl, scripts/results_100k/Alle. Danach wie vorher: drei Punkte im Katalog, altes Modell als Standard.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
LOG=$P/data/rct100k_lauf.log
until grep -q "Alles fertig" "$LOG"; do
    grep -q "Abbruch\|Traceback" "$LOG" && { echo "######## Einzellauf abgebrochen, gemeinsamer Lauf startet nicht"; exit 1; }
    sleep 20
done
set_points() {
    python3 - "$P/catalog.yaml" "$1" <<'EOF'
import re, sys
path, points = sys.argv[1], sys.argv[2]
text = open(path, encoding="utf-8").read()
new, n = re.subn(r"^    mitarbeiterwahl: \[.*\]$", "    mitarbeiterwahl: " + points, text, flags=re.M)
assert n == 1
open(path, "w", encoding="utf-8").write(new)
EOF
}
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr'
OUT=$P/scripts/results_100k/Alle
cd $ROOT/process-simulation-engine
echo "######## Alle: Simulation und Training ($(date +%H:%M:%S))"
set_points '["Mitarbeiter: W_Call incomplete files", "Mitarbeiter: W_Complete application", "Mitarbeiter: W_Handle leads", "Mitarbeiter: W_Validate application", "Mitarbeiter: W_Call after offers (resume)", "Mitarbeiter: W_Call incomplete files (resume)", "Mitarbeiter: W_Complete application (resume)", "Mitarbeiter: W_Validate application (resume)"]'
.venv/bin/python -u ../prescriptive/scripts/main.py --outcome bpic17=durchlaufzeit --steps rct train models --estimators xgb --retrain-estimators \
    --delta 0.5 --rct-cases 10000 --rct-runs 10 --workers 10 --out "$OUT" 2>&1 | grep --line-buffered -v -E "$FILTER"
if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    echo "######## Alle: online, alle neun Punkte ($(date +%H:%M:%S))"
    .venv/bin/python -u ../prescriptive/scripts/main.py --outcome bpic17=durchlaufzeit --steps online --delta 0.5 --rct-cases 10000 --rct-runs 10 \
        --online-policies bank random xgb --online-seeds 10 --online-workers 14 --out "$OUT" 2>&1 | grep --line-buffered -v -E "$FILTER"
    echo "######## Alle: online, nur Mitarbeiterpunkte ($(date +%H:%M:%S))"
    PRESCRIPTIVE_OUTCOME_BPIC17=durchlaufzeit .venv/bin/python -u ../prescriptive/data/rct100k_alle_online.py "$OUT/nur_mitarbeiter" 2>&1 | grep --line-buffered -v -E "$FILTER"
    mv "$P/data/rct_bpic17_channel_delta50" "$P/data/rct_bpic17_channel_delta50_Alle"
    cp -p "$P/models/xgb_model_durchlaufzeit.pkl" "$P/models/xgb_model_durchlaufzeit_Alle.pkl"
else
    echo "######## Abbruch im gemeinsamen Lauf"
fi
set_points '["Mitarbeiter: W_Call incomplete files", "Mitarbeiter: W_Complete application", "Mitarbeiter: W_Handle leads"]'
cp -p "$P/models/xgb_model_durchlaufzeit_30k.pkl" "$P/models/xgb_model_durchlaufzeit.pkl"
echo "######## Gemeinsamer Lauf fertig ($(date +%H:%M:%S))"
