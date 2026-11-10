#!/bin/bash
# PSP-Variante mit Durchlaufzeit aus den Abständen (wie data/bpic17_psp_variante_abstaende.sh), aber 100 statt 30 gezogene
# Suffixe je Option: wie viel des Rückstands liegt am Rauschen der Ziehung? Online an den neuen Punkten, alte Zuteilung
# (Katalog-Kopie data/catalog_neue_punkte.yaml), vergleichbar mit den bisherigen Läufen.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
LOG=$P/data/PSP_variante_100ziehungen.log
MODEL=$P/models/psp_model_durchlaufzeit_real_np_s40_ohne_zukunft_abstaende.pkl
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte.yaml
cd $ROOT/process-simulation-engine
echo "######## Online, 100 Ziehungen ($(date +%H:%M:%S))" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/main.py --delta 0.5 --rct-cases 10000 --rct-runs 10 --steps online \
    --online-policies psp --psp-model "$MODEL" --psp-samples 100 --online-seeds 10 --online-cases 3000 --online-warmup 700 --online-cooldown 1300 \
    --online-workers 12 --out "$P/scripts/results_100k/neue_punkte_online_variante_100ziehungen" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch online ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
