#!/bin/bash
# Online-Test der PSP-Variante (c) (2026-11-11): TAG real_alle_s60_normal (alle Trainingsfälle, Decoder 60, normale Abstände,
# Durchlaufzeit aus der Summe der Abstände, statische Merkmale ohne Zukunftswissen), bewertet über die gezogenen Suffixe,
# 100 Ziehungen, ohne Einschränkung. Neue Punkte, alte Zuteilung (Katalog-Kopie data/catalog_neue_punkte.yaml), gleiche Seeds wie
# neue_punkte_online (Bank, Regel, XGB), die 100-Ziehungen-Variante s40 und Lauf 2/3, also gepaart vergleichbar.
# WORKERS aus der Umgebung (Standard 12). Log: data/PSP_stufe2c_online.log
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
LOG=$P/data/PSP_stufe2c_online.log
OUT=neue_punkte_online_stufe2c_suffixe100
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte.yaml
cd $ROOT/process-simulation-engine
echo "######## Online $OUT ($(date +%H:%M:%S))" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/main.py --delta 0.5 --rct-cases 10000 --rct-runs 10 --steps online \
    --online-policies psp --psp-model "$P/models/psp_model_durchlaufzeit_real_alle_s60_normal.pkl" --psp-samples 100 --online-seeds 10 \
    --online-cases 3000 --online-warmup 700 --online-cooldown 1300 --online-workers ${WORKERS:-12} --out "$P/scripts/results_100k/$OUT" 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || echo "######## Abbruch online $OUT ($(date +%H:%M:%S))" | tee -a "$LOG"
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
