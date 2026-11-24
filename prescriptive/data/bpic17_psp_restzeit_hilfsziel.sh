#!/bin/bash
# ZZ-TF1 + Restzeit-Kopf als reines Hilfsziel (2026-11-18, Wunsch des Nutzers): fertiges ZZ-TF1 laden (--init-from), Restzeit-Kopf
# (log(1 + Restzeit in Tagen) bis Fallende aus dem Encoder-Zustand, attenuiert, Gewicht 1,0) dazu und 5 Epochen auf allen Prefixen
# nachtrainieren (LR 3e-5, sonst Einstellungen wie ZZ-TF1). Bewertet wird WEITER NUR über gezogene Suffixe (value_from bleibt "suffix");
# der Kopf soll nur den Encoder zwingen, Information über die Gesamtdauer (Leads, Angebote, Fristen) festzuhalten.
# Danach: Prüfreihe (Label "ZZ-TF1 + Restzeit-Hilfsziel"), Storno-Zeitpunkt, Suffixgüte. Vergleich mit ZZ-TF1 fertig.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_100k/restzeit_hilfsziel_20261118
BASE=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1.pkl
TAG=real_alle_s60_normal_zaehler_zeit_tf1_restzeit
MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
LOG=$P/data/PSP_$TAG.log
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
note() { echo "$(date '+%Y-%m-%d %H:%M') $1" >> $OUT/fortschritt.log; }
note "Nachtraining mit Restzeit-Hilfsziel beginnt (5 Epochen)"
[ -e "$MODEL" ] || .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --cases 0 --epochs 5 --learning-rate 3e-5 \
    --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps \
    --counters --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training --remaining-head --init-from "$BASE" 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ -e "$MODEL" ] || { note "Abbruch im Training"; exit 1; }
note "Training fertig, Prüfreihe beginnt"
.venv/bin/python -u $OUT/pruefreihe.py "zz_restzeit=ZZ-TF1 + Restzeit-Hilfsziel=$MODEL" > $OUT/pruefreihe.log 2>&1
note "Prüfreihe fertig, Storno-Messung"
CUDA_VISIBLE_DEVICES="" .venv/bin/python $P/scripts/results_100k/zaehler_uhren_tf1_20261117/storno_abstand.py 300 20 "ZZ-TF1 + Restzeit-Hilfsziel=$MODEL" > $OUT/storno.log 2>&1
note "Storno-Messung fertig, Suffixgüte"
CUDA_VISIBLE_DEVICES="" nice -n 10 .venv/bin/python -u ../prescriptive/scripts/psp_suffixguete.py --psp "ZZ-TF1 + Restzeit-Hilfsziel=$MODEL" --prefixes 1000 --workers 4 --threads 2 --out $OUT/suffixguete > $OUT/logs/suffixguete.log 2>&1
note "Alles fertig"
