#!/bin/bash
# PSP (c) mit Zählern und Zeit je Event (2026-11-16): Einstellungen exakt wie Variante (c) (data/bpic17_psp_real_stufe2c.sh:
# alle Trainingsfälle, Decoder 60, normale Abstände, Durchlaufzeit aus der Summe der Abstände, statische Merkmale ohne
# Zukunftswissen, Wahlpunkte dreifach, Lernrate 1e-4, 15 Epochen, ursprünglicher Teacher-Forcing-Plan, Seed 42), dazu
#   --counters        Zähler je Aktivität|Lifecycle-Token als Eingabe in Encoder und Decoder
#   --time-on-event   die Zeitköpfe sehen das Event desselben Schritts
#   --option-in-encoder-after-training   Option als letztes Prefix-Event mit vom Decoder gezogenem eigenem Abstand
#                                        (erst nach dem Training gesetzt: Probengewichtung wie bei (c))
# Katalog: eingefrorene Kopie data/catalog_neue_punkte_zuteilung_random.yaml (dieselben Wahlpunkte wie bei (c)).
# Danach die übliche Diagnose. Begleiter (Checkpoint je Epoche, Prüfreihe nach Epoche 3, 9 und am Ende):
# scripts/results_100k/zaehler_zeit_20261116/begleiter.py
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
TAG=real_alle_s60_normal_zaehler_zeit
LOG=$P/data/PSP_$TAG.log
MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
echo "######## Training ($(date +%H:%M:%S))" | tee -a "$LOG"
[ -e "$MODEL" ] || .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --cases 0 --epochs 15 --learning-rate 1e-4 \
    --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps \
    --counters --time-on-event --option-in-encoder-after-training 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ -e "$MODEL" ] || { echo "######## Abbruch im Training ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Diagnose ($(date +%H:%M:%S))" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/psp_diagnose.py --model "$MODEL" --out "$P/scripts/results_100k/psp_diagnose_$TAG" 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
