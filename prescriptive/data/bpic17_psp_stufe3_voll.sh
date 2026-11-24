#!/bin/bash
# PSP-Stufe 3, volle Trainings (2026-11-11): wie Variante (c) (data/bpic17_psp_real_stufe2c.sh: alle Trainingsfälle, 15 Epochen,
# Decoder 60, Lernrate 1e-4, Wahlpunkte dreifach, statische Merkmale ohne Zukunftswissen, Durchlaufzeit aus der Summe der Abstände,
# Katalog-Kopie data/catalog_neue_punkte.yaml), dazu EXTRA aus der Umgebung. Danach die Auswertung der Stufe 3
# (data/bpic17_psp_stufe3_pilot_auswertung.sh: Suffixgüte, Zerlegung Pfad/Zeit, Rauschen an den Wahlpunkten); online erst nach deren Prüfung.
#   TAG=real_alle_s60_token_oe   EXTRA="--gap-tokens --option-in-encoder" FREI=7000   # Token mit Abstandsklasse, Option im Encoder
#   TAG=real_alle_s60_normal_tf1 EXTRA="--teacher-forcing 1.0"            FREI=5000   # wie (c), nur Teacher Forcing 1,0
# FREI: so viel GPU-Speicher (MiB) muss beim Start frei sein, sonst Abbruch (die GPU wird geteilt). WARTEN: Sekunden vor dem Start.
# Es bleiben models/psp_model_durchlaufzeit_<TAG>.pkl, models/psp_training_<TAG>/, data/PSP_<TAG>.log,
# data/PSP_voll_<TAG>_*.log, scripts/results_100k/voll_<TAG>_suffixguete, voll_<TAG>_zerlegung, psp_rauschen_voll_<TAG>_<TAG>
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
[ -n "$TAG" ] || { echo "TAG fehlt"; exit 1; }
LOG=$P/data/PSP_$TAG.log
MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
FILTER='FutureWarning|applied = df|cuInit|hasattr|pin_memory|super\(\).__init__'  # ohne cuda: sonst fehlt die Meldung "CUDA out of memory"
export PSP_CATALOG=$P/data/catalog_neue_punkte.yaml
cd $ROOT/process-simulation-engine
sleep ${WARTEN:-0}
frei=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$frei" -lt "${FREI:-7000}" ]; then
    echo "######## Abbruch $TAG: nur $frei MiB GPU-Speicher frei, nötig ${FREI:-7000} ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1
fi
echo "######## Training $TAG: $EXTRA; GPU frei $frei MiB ($(date +%H:%M:%S))" | tee -a "$LOG"
[ -e "$MODEL" ] || .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --cases 0 --epochs ${EPOCHS:-15} --learning-rate 1e-4 \
    --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps $EXTRA 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ -e "$MODEL" ] || { echo "######## Abbruch im Training ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Auswertung ($(date +%H:%M:%S))" | tee -a "$LOG"
RUNDE=voll_$TAG ALLE="$TAG=$(basename $MODEL)" bash ../prescriptive/data/bpic17_psp_stufe3_pilot_auswertung.sh
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
