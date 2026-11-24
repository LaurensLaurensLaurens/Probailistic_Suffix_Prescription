#!/bin/bash
# PSP-Stufe 3, Pilot (2026-11-11): kleine Vergleichstrainings (3.000 Fälle, 6 Epochen, sonst wie Stufe 2: Decoder 60, Lernrate 1e-4,
# Wahlpunkte dreifach, statische Merkmale ohne Zukunftswissen, Durchlaufzeit aus der Summe der Abstände, Katalog-Kopie
# data/catalog_neue_punkte.yaml), um vor den langen Läufen zu prüfen, was Teacher Forcing 1.0, das Token mit Abstandsklasse und
# die Option als erstes Event des Suffix bringen. TAG und EXTRA aus der Umgebung, z. B.
#   TAG=pilot_klassen_tf0 EXTRA="--gap-classes"
#   TAG=pilot_klassen_tf1 EXTRA="--gap-classes --teacher-forcing 1.0"
#   TAG=pilot_token_tf1   EXTRA="--gap-tokens --teacher-forcing 1.0 --option-in-decoder"
#   TAG=pilot_normal_tf1  EXTRA="--teacher-forcing 1.0"
# Es bleiben models/psp_model_durchlaufzeit_<TAG>.pkl, models/psp_training_<TAG>/, data/PSP_<TAG>.log
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
LOG=$P/data/PSP_$TAG.log
FILTER='FutureWarning|applied = df|cuInit|hasattr|pin_memory|super\(\).__init__'  # ohne cuda: sonst fehlt die Meldung "CUDA out of memory"
export PSP_CATALOG=$P/data/catalog_neue_punkte.yaml
cd $ROOT/process-simulation-engine
echo "######## Training $TAG: $EXTRA ($(date +%H:%M:%S))" | tee -a "$LOG"
.venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --cases ${CASES:-3000} --epochs ${EPOCHS:-6} --learning-rate 1e-4 \
    --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps $EXTRA 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
