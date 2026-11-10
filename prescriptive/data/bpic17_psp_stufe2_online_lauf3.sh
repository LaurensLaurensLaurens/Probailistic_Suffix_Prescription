#!/bin/bash
# Online-Tests der PSP-Stufe 2 (2026-11-10) an den neuen Punkten, alte Zuteilung (Katalog-Kopie data/catalog_neue_punkte.yaml),
# gleiche Seeds wie neue_punkte_online (Bank, Regel, XGB) und die 100-Ziehungen-Variante, also gepaart vergleichbar:
#   1. (b) mit Restzeit-Kopf, bewertet über den Kopf (--psp-value head, 30 Dropout-Masken)
#   2. (b) bewertet über die gezogenen Suffixe, 100 Ziehungen
#   3. (a) ohne Kopf, über die gezogenen Suffixe, 100 Ziehungen
# Hier nur Lauf 3 (zurückgestellt am 2026-11-10), WORKERS aus der Umgebung. Log: data/PSP_stufe2_online.log
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
LOG=$P/data/PSP_stufe2_online.log
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte.yaml
cd $ROOT/process-simulation-engine
run() {  # run <Ordner> <Modell-Tag> <weitere Argumente>
    local out=$1 tag=$2; shift 2
    echo "######## Online $out ($(date +%H:%M:%S))" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/main.py --delta 0.5 --rct-cases 10000 --rct-runs 10 --steps online \
        --online-policies psp --psp-model "$P/models/psp_model_durchlaufzeit_$tag.pkl" "$@" --online-seeds 10 --online-cases 3000 \
        --online-warmup 700 --online-cooldown 1300 --online-workers ${WORKERS:-12} --out "$P/scripts/results_100k/$out" 2>&1 \
        | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || echo "######## Abbruch online $out ($(date +%H:%M:%S))" | tee -a "$LOG"
}


run neue_punkte_online_stufe2_ohne_kopf_suffixe100 real_alle_s60_loggaps --psp-samples 100
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
