#!/bin/bash
# Screening aller realistischen Entscheidungen der Bank (main.py --steps screening) auf dem echten Log, mit dem angepassten
# Screening (screening/: Pool frei, Merkmale der Zuteilung, S-Learner ab 6 Optionen, Effekte je Mitarbeiter). Kandidaten aus
# data/catalog_screening_alle.yaml (über PSP_CATALOG; catalog.yaml bleibt unberührt): 6 Control-Flow-Punkte, 10
# Mitarbeiterpunkte, 3 Angebotsvariablen am ersten O_Create Offer, Reaktionszeit auf den Antrag.
# Mitarbeiter mit mindestens 200 Fällen eine eigene Option, die übrigen zusammen other. Ergebnisse in
# scripts/results_100k/screening_alle (prescription/screening/), Log data/screening_alle.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_100k/screening_alle
LOG=$P/data/screening_alle.log
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_screening_alle.yaml
cd $ROOT/process-simulation-engine
echo 1000 > /proc/self/oom_score_adj
echo "######## Screening aller Entscheidungen ($(date +%H:%M:%S))" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/main.py --steps screening --pool frei --min-cases-resources 200 --no-screening-details \
    --intervention "bpic17=Mitarbeiter: W_Complete application" --out "$OUT" "$@" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
