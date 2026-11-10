#!/bin/bash
# Validierung des Screenings im Simulator (scripts/validate_screening.py, screening/validation.py): alle Kandidaten, die ein
# Wahlpunkt des Simulators sind (12 Punkte wie PSP_real_w3_10k_online, Katalog-Kopie über PSP_CATALOG), je Lauf 20.000 Fälle.
# Ergebnisse in scripts/results_100k/screening_validierung, Log data/screening_validierung.log. Räumt bei Speichermangel als erster.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_100k/screening_validierung
LOG=$P/data/screening_validierung.log
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_simulator_kpis_gesamt.yaml
cd $ROOT/process-simulation-engine
echo 1000 > /proc/self/oom_score_adj
echo "######## Validierung des Screenings ($(date +%H:%M:%S))" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/validate_screening.py --out "$OUT" --n-cases ${CASES:-20000} --workers ${WORKERS:-4} "$@" 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
