#!/bin/bash
# Corrected (d): 300 suffix draws per option, matching the other seed-0 runs.
set -euo pipefail
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P="$ROOT/prescriptive"
OUT="$P/scripts/results_100k/d_option_encoder_suffixe300_cases1000_seed0_20261113/d_korrigiert"
LOG="$P/data/PSP_online300_d_option_encoder_seed0_20261113.log"
MODEL="$P/models/psp_model_durchlaufzeit_real_alle_s60_klassen_option_encoder.pkl"
if [ -e "$OUT" ] || [ -e "$LOG" ]; then
    echo "Output or log already exists; refusing to overwrite." >&2
    exit 1
fi
test -f "$MODEL"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT/process-simulation-engine"
exec .venv/bin/python -u "$P/scripts/run_token_oe_ue_seed.py" \
    --seed 0 --samples 300 --scored-cases 1000 --warmup 700 --cooldown 1300 \
    --model "$MODEL" --without-transition-rules --out "$OUT" > "$LOG" 2>&1
