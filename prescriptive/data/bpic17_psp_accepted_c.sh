#!/bin/bash
# PSP Variante (c) auf den Wahlpunkten nach dem Screening auf O_Accepted (2026-11-12, screening/results_accepted):
# annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads, neues_angebot_beim_nachfassen (Katalog-Kopie
# data/catalog_accepted_punkte.yaml). Sonst wie data/bpic17_psp_real_stufe2c.sh: alle Trainingsfälle, 15 Epochen, Decoder 60,
# Lernrate 1e-4, Wahlpunkte dreifach, statische Merkmale ohne Zukunftswissen, Durchlaufzeit aus der Summe der Abstände,
# normale Abstände, GPU. Nur das Training: Diagnose und Rauschtest kennen die neuen Punkte und das Ziel O_Accepted noch nicht.
# Es bleiben models/psp_model_durchlaufzeit_<TAG>.pkl, models/psp_training_<TAG>/, data/PSP_<TAG>.log
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
TAG=${TAG:-real_alle_s60_normal_accepted}
LOG=$P/data/PSP_$TAG.log
MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
FILTER='FutureWarning|applied = df|cuInit|hasattr|pin_memory|super\(\).__init__'  # ohne cuda: sonst fehlt "CUDA out of memory"
export PSP_CATALOG=$P/data/catalog_accepted_punkte.yaml
cd $ROOT/process-simulation-engine
echo "######## Training $TAG ($(date +%H:%M:%S))" | tee -a "$LOG"
[ -e "$MODEL" ] || .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --cases 0 --epochs ${EPOCHS:-15} --learning-rate 1e-4 \
    --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps $EXTRA 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ -e "$MODEL" ] || { echo "######## Abbruch im Training ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
