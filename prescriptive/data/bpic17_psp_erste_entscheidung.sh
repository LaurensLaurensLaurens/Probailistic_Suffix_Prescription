#!/bin/bash
# PSP online mit Einschränkung der Policy (2026-11-10, mit dem Nutzer abgestimmt): bei angebot_nachlegen entscheidet das PSP
# nur bei der ersten Entscheidung je Fall, danach wie die Bank (psp_policy first_only, main.py --psp-first-only). Sonst wie
# data/bpic17_psp_variante_100ziehungen.sh: neue Punkte, alte Zuteilung (Katalog-Kopie), 100 Ziehungen, gleiche Seeds.
# MODEL (Tag), OUT (Ordner unter scripts/results_100k), WORKERS aus der Umgebung.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
TAG=${TAG:-real_np_s40_ohne_zukunft_abstaende}
OUT=${OUT:-neue_punkte_online_${TAG}_erste_entscheidung}
LOG=$P/data/PSP_${OUT}.log
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte.yaml
cd $ROOT/process-simulation-engine
echo "######## Online $OUT ($(date +%H:%M:%S))" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/main.py --delta 0.5 --rct-cases 10000 --rct-runs 10 --steps online \
    --online-policies psp --psp-model "$P/models/psp_model_durchlaufzeit_$TAG.pkl" --psp-samples 100 --psp-first-only angebot_nachlegen \
    --online-seeds 10 --online-cases 3000 --online-warmup 700 --online-cooldown 1300 --online-workers ${WORKERS:-12} \
    --out "$P/scripts/results_100k/$OUT" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
