#!/bin/bash
# Das PSP auf dem echten Log neu trainieren, mit den neuen Interventionspunkten als Wahlpunkte (dreifach gewichtet):
# angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads (Katalog-Kopie
# data/catalog_neue_punkte.yaml über PSP_CATALOG, scripts/train_psp_real.py --catalog-points). Sonst wie
# PSP_real_w3_10k: 10.000 Fälle, 100 Epochen, Lernrate 1e-5, GPU. Danach online wie data/bpic17_neue_punkte.sh, mit diesem
# Modell (Ordner scripts/results_100k/neue_punkte_online_psp_neu). Es bleiben models/psp_model_durchlaufzeit_real_w3_10k_np.pkl,
# models/psp_training_real_w3_10k_np/ (last.pt jede Epoche), data/PSP_real_w3_10k_np.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
TAG=real_w3_10k_np
LOG=$P/data/PSP_$TAG.log
MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
XGB=$P/models/xgb_model_durchlaufzeit_xgb_real_10k_neuepunkte.pkl
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte.yaml
cd $ROOT/process-simulation-engine
echo "######## Training ($(date +%H:%M:%S))" | tee -a "$LOG"
.venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --cases ${CASES:-10000} --epochs ${EPOCHS:-100} --learning-rate 1e-5 \
    --point-weight 3 --catalog-points 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ -e "$MODEL" ] || { echo "######## Abbruch im Training ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Online ($(date +%H:%M:%S))" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/main.py --delta 0.5 --rct-cases 10000 --rct-runs 10 --steps online \
    --online-policies psp --psp-model "$MODEL" --online-seeds 10 --online-cases 3000 --online-warmup 700 --online-cooldown 1300 \
    --online-workers 12 --out "$P/scripts/results_100k/neue_punkte_online_psp_neu" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch online ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
