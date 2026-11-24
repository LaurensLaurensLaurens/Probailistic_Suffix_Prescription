#!/bin/bash
# Same 300-draw seed after fixing transition lookup for unseen time classes.
set -euo pipefail
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P="$ROOT/prescriptive"
OUT="$P/scripts/results_100k/neue_punkte_online_token_oe_ue_suffixe300_seed0_v2"
LOG="$P/data/PSP_online300_real_alle_s60_token_oe_ue_seed0_v2.log"
if [ -e "$OUT" ] || [ -e "$LOG" ]; then
    echo "Output or log already exists; refusing to overwrite." >&2
    exit 1
fi
cd "$ROOT/process-simulation-engine"
exec .venv/bin/python -u "$P/scripts/run_token_oe_ue_seed.py" --seed 0 --out "$OUT" > "$LOG" 2>&1
