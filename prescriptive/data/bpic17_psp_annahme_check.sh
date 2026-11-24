#!/bin/bash
# Offline-Prüfung der Policy psp_annahme (2026-11-12): sieht das PSP Unterschiede in der Angebotsannahme? Je Modell
# scripts/psp_annahme_check.py, die drei vorhandenen vollen Modelle parallel (CPU). Logs data/PSP_annahme_check_<tag>.log,
# Ergebnisse scripts/results_100k/psp_annahme_<tag>. TAGS aus der Umgebung möglich.
P=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_accepted_punkte.yaml CUDA_VISIBLE_DEVICES=''
cd $P/../process-simulation-engine
for tag in ${TAGS:-real_alle_s60_normal real_alle_s60_normal_tf1 real_alle_s60_token_oe}; do
    .venv/bin/python -u ../prescriptive/scripts/psp_annahme_check.py --model $P/models/psp_model_durchlaufzeit_$tag.pkl --workers 3 --threads 2 \
        --out $P/scripts/results_100k/psp_annahme_$tag 2>&1 | grep --line-buffered -v -E "$FILTER" > $P/data/PSP_annahme_check_$tag.log &
done
wait
echo "######## alle fertig $(date +%H:%M)" >> $P/data/PSP_annahme_check_real_alle_s60_normal.log
