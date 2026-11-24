#!/bin/bash
# ZZ-TF0,5 (2026-11-18, Wunsch des Nutzers): fertiges ZZ-TF1 laden (--init-from) und 6 Epochen mit Teacher-Forcing-Boden 0,5
# nachtrainieren. Quote je Epoche max(0,5; 1 − Epoche/3) = 1,0 / 0,67 / 0,5 / 0,5 / 0,5 / 0,5 (Trainer: Abfall über die halbe
# Epochenzahl). Bei 0,5 bekommt der Decoder je Schritt mit Wahrscheinlichkeit 1/2 das echte vorige Event, sonst sein eigenes argmax
# (wie bisher, kein Sampling). Ziel: Mittelweg zwischen ZZ-TF1 (gültige Abläufe, kaum Fernwirkung) und ZZ (Fernwirkung, ungültig).
# LR 1e-4 wie ZZ-TF1 (Regimewechsel, Plateau-Scheduler griff dort nie). Sonst alle Einstellungen wie ZZ-TF1.
# Startet erst, wenn das Restzeit-Hilfsziel-Training fertig ist (GPU), läuft parallel zum RCT-Lauf.
# Danach: Prüfreihe (Label "ZZ-TF0,5"), Storno-Zeitpunkt, Suffixgüte. Ungültige Übergänge rechnet register_erstellen.py aus der Prüfreihe.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_100k/tf_boden05_20261118
BASE=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1.pkl
WARTEN_AUF=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_restzeit.pkl
TAG=${TAG:-real_alle_s60_normal_zaehler_zeit_tf05}
CASES=${CASES:-0}
EPOCHS=${EPOCHS:-6}
LABEL="ZZ-TF0,5"
MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
LOG=$P/data/PSP_$TAG.log
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
note() { echo "$(date '+%Y-%m-%d %H:%M') $1" >> $OUT/fortschritt.log; }
if [ "$CASES" = "0" ]; then
    note "wartet auf das Ende des Restzeit-Hilfsziel-Trainings"
    until [ -e "$WARTEN_AUF" ] || ! tmux has-session -t psp_restzeit 2>/dev/null; do sleep 120; done
fi
note "Nachtraining mit Teacher-Forcing-Boden 0,5 beginnt ($EPOCHS Epochen, Fälle $CASES)"
[ -e "$MODEL" ] || .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --cases $CASES --epochs $EPOCHS --learning-rate 1e-4 \
    --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps \
    --counters --time-on-event --teacher-forcing 0.5 --option-in-encoder-after-training --init-from "$BASE" 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ -e "$MODEL" ] || { note "Abbruch im Training"; exit 1; }
[ "$CASES" = "0" ] || { note "Rauchtest fertig ($MODEL)"; exit 0; }
note "Training fertig, Prüfreihe beginnt"
.venv/bin/python -u $OUT/pruefreihe.py "zz_tf05=$LABEL=$MODEL" > $OUT/pruefreihe.log 2>&1
note "Prüfreihe fertig, Storno-Messung"
CUDA_VISIBLE_DEVICES="" .venv/bin/python $P/scripts/results_100k/zaehler_uhren_tf1_20261117/storno_abstand.py 300 20 "$LABEL=$MODEL" > $OUT/storno.log 2>&1
note "Storno-Messung fertig, Suffixgüte"
CUDA_VISIBLE_DEVICES="" nice -n 10 .venv/bin/python -u ../prescriptive/scripts/psp_suffixguete.py --psp "$LABEL=$MODEL" --prefixes 1000 --workers 4 --threads 2 --out $OUT/suffixguete > $OUT/logs/suffixguete.log 2>&1
note "Alles fertig"
