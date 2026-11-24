#!/bin/bash
# Online-Test eines PSP mit 300 Ziehungen je Option (2026-11-12), Policy psp (Erwartungswert über die gezogenen Suffixe,
# policies/ev_policy/psp_policy.py), ohne Einschränkung. Gleiche Seeds, Fallzahlen und alte Zuteilung wie der Test von (c) mit
# 100 Ziehungen (data/bpic17_psp_stufe2c_online.sh) und neue_punkte_online_cate, also gepaart vergleichbar: Katalog-Kopie
# data/catalog_neue_punkte_zuteilung_random.yaml (Stand des letzten Commits; Bank und Regel damit am 12.11. exakt wie am 11.11.).
# TAG (Modell models/psp_model_durchlaufzeit_<TAG>.pkl), OUT (Ordner in scripts/results_100k) und WORKERS aus der Umgebung.
# Log: data/PSP_online300_<TAG>.log
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
[ -n "$TAG" ] && [ -n "$OUT" ] || { echo "TAG und OUT fehlen"; exit 1; }
LOG=$P/data/PSP_online300_$TAG.log
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
echo "######## Online $OUT, Modell $TAG, 300 Ziehungen, ${WORKERS:-5} Prozesse ($(date +%H:%M:%S))" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/main.py --delta 0.5 --rct-cases 10000 --rct-runs 10 --steps online \
    --online-policies psp --psp-model "$P/models/psp_model_durchlaufzeit_$TAG.pkl" --psp-samples 300 --online-seeds 10 \
    --online-cases 3000 --online-warmup 700 --online-cooldown 1300 --online-workers ${WORKERS:-5} --out "$P/scripts/results_100k/$OUT" 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || echo "######## Abbruch online $OUT ($(date +%H:%M:%S))" | tee -a "$LOG"
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
