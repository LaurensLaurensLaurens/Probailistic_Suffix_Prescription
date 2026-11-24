#!/bin/bash
# Treue des Simulators an den neuen Interventionspunkten (Katalog-Kopie data/catalog_neue_punkte.yaml): Screening-Validierung
# (scripts/validate_screening.py, 20.000 Fälle je Lauf) und die Prescriptive-KPIs (main.py --steps kpis, 10.000 Fälle).
# Ergebnisse in scripts/results_100k/neue_punkte_validierung und neue_punkte_kpis, Log data/neue_punkte_treue.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
LOG=$P/data/neue_punkte_treue.log
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte.yaml
cd $ROOT/process-simulation-engine
echo 1000 > /proc/self/oom_score_adj
echo "######## Validierung ($(date +%H:%M:%S))" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/validate_screening.py --out $P/scripts/results_100k/neue_punkte_validierung --n-cases 20000 --workers 4 \
    --candidates angebot_nachlegen annehmen_bei_konzept annehmen_nach_vervollstaendigung handle_leads 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
echo "######## KPIs ($(date +%H:%M:%S))" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/main.py --steps kpis --n-cases 10000 --out $P/scripts/results_100k/neue_punkte_kpis 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
