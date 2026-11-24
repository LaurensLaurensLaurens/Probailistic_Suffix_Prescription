#!/bin/bash
# Variante (e): wie das (d)-Training, aber Zeitklasse bedingt auf Aktivität|Lifecycle.
# Optionsmodus erst NACH dem Training setzen: dieselbe Probengewichtung wie bei (d),
# bei der Auswertung ein gezogener Optionsabstand je Suffix und Option im Encoder.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
P="$ROOT/prescriptive"
TAG=${TAG:-real_alle_s60_klassen_event_oe}
LOG="$P/data/PSP_$TAG.log"
MODEL="$P/models/psp_model_durchlaufzeit_$TAG.pkl"
export PSP_CATALOG="$P/data/catalog_neue_punkte.yaml"
cd "$ROOT/process-simulation-engine"

if [[ -e "$MODEL" || -e "$P/models/psp_training_$TAG" ]]; then
    echo "Modell oder Checkpoints für $TAG gibt es schon; anderen TAG wählen."
    exit 1
fi

.venv/bin/python -u ../prescriptive/scripts/train_psp_real.py \
    --tag "$TAG" --cases 0 --epochs "${EPOCHS:-15}" --learning-rate 1e-4 \
    --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future \
    --time-from gaps --gap-condition-on-event --option-in-encoder-after-training \
    2>&1 | tee "$LOG"

CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/psp_diagnose.py \
    --model "$MODEL" --out "$P/scripts/results_100k/psp_diagnose_$TAG" \
    2>&1 | tee -a "$LOG"
