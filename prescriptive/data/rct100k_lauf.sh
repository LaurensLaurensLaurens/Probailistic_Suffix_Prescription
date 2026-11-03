#!/bin/bash
# Je Mitarbeiterpunkt ein eigener Lauf: 100.000 RCT-Fälle (10 Läufe zu je 10.000, delta 0.5), randomisiert nur dieser Punkt und channel, XGBoost darauf
# trainieren und bewerten, dann online Bank, Zufall, XGBoost, die Policy wählt nur an diesem Punkt. Outcome: Durchlaufzeit.
# lauf.mitarbeiterwahl im Katalog wird je Lauf gesetzt und am Ende auf alle drei zurückgesetzt. Je Punkt bleiben: data/rct_bpic17_channel_delta50_<tag>,
# models/xgb_model_durchlaufzeit_<tag>.pkl, scripts/results_100k/<tag>. Danach wieder das alte Modell (xgb_model_durchlaufzeit_30k.pkl) als Standard.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
set_points() {
    python3 - "$P/catalog.yaml" "$1" <<'EOF'
import re, sys
path, points = sys.argv[1], sys.argv[2]
text = open(path, encoding="utf-8").read()
new, n = re.subn(r"^    mitarbeiterwahl: \[.*\]$", "    mitarbeiterwahl: " + points.replace("\\", "\\\\"), text, flags=re.M)
assert n == 1
open(path, "w", encoding="utf-8").write(new)
EOF
}
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr'
cd $ROOT/process-simulation-engine
for entry in "W_Validate_resume|Mitarbeiter: W_Validate application (resume)"; do
    tag=${entry%%|*}; point=${entry#*|}; out=$P/scripts/results_100k/$tag
    echo "######## Lauf $tag: Simulation und Training ($(date +%H:%M:%S))"
    set_points "[\"$point\"]"
    .venv/bin/python -u ../prescriptive/scripts/main.py --outcome bpic17=durchlaufzeit --steps rct train models --estimators xgb --retrain-estimators \
        --delta 0.5 --rct-cases 10000 --rct-runs 10 --workers 10 --out "$out" 2>&1 | grep --line-buffered -v -E "$FILTER"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch in Lauf $tag"; break; }
    echo "######## Lauf $tag: online ($(date +%H:%M:%S))"
    .venv/bin/python -u ../prescriptive/scripts/main.py --outcome bpic17=durchlaufzeit --steps online --delta 0.5 --rct-cases 10000 --rct-runs 10 \
        --online-policies bank random xgb --online-points "bpic17=$point" --online-seeds 10 --online-workers 14 --out "$out" 2>&1 | grep --line-buffered -v -E "$FILTER"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch online in Lauf $tag"; break; }
    mv "$P/data/rct_bpic17_channel_delta50" "$P/data/rct_bpic17_channel_delta50_$tag"
    cp -p "$P/models/xgb_model_durchlaufzeit.pkl" "$P/models/xgb_model_durchlaufzeit_$tag.pkl"
    echo "######## Lauf $tag fertig ($(date +%H:%M:%S))"
done
set_points '["Mitarbeiter: W_Call incomplete files", "Mitarbeiter: W_Complete application", "Mitarbeiter: W_Handle leads"]'
cp -p "$P/models/xgb_model_durchlaufzeit_30k.pkl" "$P/models/xgb_model_durchlaufzeit.pkl"
echo "######## Alles fertig ($(date +%H:%M:%S))"
