#!/bin/bash
# Online-Auswertung der drei CATE-ERM-Modelle, gemeinsames Gamma (Standard 0.1).
# Vorher trainieren (aus process-simulation-engine/):
# PSP_CATALOG=../prescriptive/data/catalog_neue_punkte.yaml .venv/bin/python -u \
#   ../prescriptive/scripts/train_cate_real.py --tag real_alle --learner s x cf --cases 0 --erm-gamma 0.1
# Wie der EV-Vergleich: Seeds 0..9, 3000 Ankünfte, Warmup 700, Cooldown 1300, 1000 gewertete Fälle.
set -euo pipefail
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P="$ROOT/prescriptive"
GAMMA="${GAMMA:-0.1}"
TAG="${TAG:-real_alle}"
OUT="${OUT:-$P/scripts/results_100k/neue_punkte_online_cate_erm_gamma$GAMMA}"
if [ -e "$OUT" ]; then
    echo "Ergebnisordner existiert bereits: $OUT; für einen neuen Lauf OUT setzen" >&2
    exit 1
fi
for learner in s x cf; do
    test -f "$P/models/cate_erm_model_${learner}_gamma${GAMMA}_durchlaufzeit_${TAG}.pkl"
done
mkdir -p "$OUT"
cp "$P/data/catalog_neue_punkte.yaml" "$OUT/catalog.yaml"
export PSP_CATALOG="$OUT/catalog.yaml"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
cd "$ROOT/process-simulation-engine"
.venv/bin/python -u ../prescriptive/scripts/main.py --datasets bpic17 --steps online --seed 0 \
    --online-policies s_learner_erm x_learner_erm causal_forest_erm \
    --erm-gamma "$GAMMA" \
    --cate-model \
        "s_learner_erm=$P/models/cate_erm_model_s_gamma${GAMMA}_durchlaufzeit_${TAG}.pkl" \
        "x_learner_erm=$P/models/cate_erm_model_x_gamma${GAMMA}_durchlaufzeit_${TAG}.pkl" \
        "causal_forest_erm=$P/models/cate_erm_model_cf_gamma${GAMMA}_durchlaufzeit_${TAG}.pkl" \
    --online-seeds 10 --online-cases 3000 --online-warmup 700 --online-cooldown 1300 \
    --online-workers "${WORKERS:-4}" --out "$OUT" > "$OUT/online.log" 2>&1
