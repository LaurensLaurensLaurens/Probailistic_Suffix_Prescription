#!/bin/bash
# PSP (c) mit Zählern, Zeit je Event und durchgehend Teacher Forcing 1 (2026-11-17): wie bpic17_psp_real_zaehler_zeit.sh, dazu
#   --teacher-forcing 1.0   konsistentes Trainingsziel (kein Versatz zwischen eigener Eingabe und Ziel → gültige Abläufe)
#   --counters        Zähler je Aktivität|Lifecycle-Token als Eingabe in Encoder und Decoder
#   --time-on-event   die Zeitköpfe sehen das Event desselben Schritts
#   --option-in-encoder-after-training   Option als letztes Prefix-Event mit vom Decoder gezogenem eigenem Abstand
#                                        (erst nach dem Training gesetzt: Probengewichtung wie bei (c))
# Katalog: eingefrorene Kopie data/catalog_neue_punkte_zuteilung_random.yaml (dieselben Wahlpunkte wie bei (c)).
# Danach die übliche Diagnose. Begleiter (Checkpoint je Epoche, Prüfreihe nach Epoche 3, 6, 9, 12 und am Ende):
# scripts/results_100k/zaehler_zeit_tf1_20261117/begleiter.py
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
TAG=real_alle_s60_normal_zaehler_zeit_tf1
LOG=$P/data/PSP_$TAG.log
MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
echo "######## Training ($(date +%H:%M:%S))" | tee -a "$LOG"
[ -e "$MODEL" ] || .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --cases 0 --epochs 15 --learning-rate 1e-4 \
    --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps \
    --counters --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ -e "$MODEL" ] || { echo "######## Abbruch im Training ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Diagnose ($(date +%H:%M:%S))" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/psp_diagnose.py --model "$MODEL" --out "$P/scripts/results_100k/psp_diagnose_$TAG" 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
