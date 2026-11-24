#!/bin/bash
# Zeitkopf mit eigenen Parametern je Event auf den fertigen T-Learner-Armen (2026-11-18, Nutzer: „ja und abbrechen“ – β-NLL-Lauf
# nach Epoche 3 abgebrochen). Anlass: im trainierten Netz verschiebt das gemeinsame Event-Embedding der Zeitköpfe fast nur die
# log-Varianz (Mittel-Versatz vor A_Cancelled ~0 d, log-Varianz +1,6): der Kopf sagt dort 3,6 statt ~20 Tage bei σ ~22 Tagen;
# β-NLL kann das Verhältnis von Mittel- und Varianz-Gradient nicht ändern (befunde.md 18.11. 20:25).
# Je Arm (Aktion, Referenz) des T-Learners (warm vom ZZ-TF1, t_learner_20261117): Zeitköpfe um eigene Parameter je Event erweitern
# (psp_model time_event_head: "offset" = Versatz je Event je Ausgabe, "slope" = zusätzlich Steigung je Event auf h; mit null begonnen)
# und NUR die Zeitköpfe nachtrainieren (head_only, Rest eingefroren: Ereignisfolgen bleiben), auf denselben Wahlpunkt-Proben wie der
# Arm, Teacher Forcing 1, attenuierter Verlust wie bisher. LR 1e-3, Versätze 3e-2; Aktion 30 / Referenz 12 Epochen.
# Danach: Hülle, Sonde beider Arme (1500 Fälle), Storno-Messung, Prüfreihe, Online Aufbau F 12 Seeds (300 Ziehungen, first_only).
# Gemeinsame Dateien (storno_abstand.csv, staende.json, f_zwoelf_auswertung) unter flock, Online-Tests nacheinander.
# Aufruf (tmux): bash bpic17_psp_zeitkopf_je_event.sh slope|offset
VARIANT=${1:?Variante slope oder offset}
case $VARIANT in
  slope) NAME=steigung; LABEL="T-Learner Kopf je Event (Steigung)"; ARMKEY=tlks ;;
  offset) NAME=versatz; LABEL="T-Learner Kopf je Event (Versatz)"; ARMKEY=tlko ;;
  *) echo "Variante slope oder offset"; exit 1 ;;
esac
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
R=$P/scripts/results_100k
OUT=$R/zeitkopf_je_event_20261118
HULL=$P/models/psp_model_durchlaufzeit_t_learner_kopf_$NAME.pkl
FILTER='FutureWarning|applied = df|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
mkdir -p $OUT/logs
[ $VARIANT = offset ] && echo 1000 > /proc/self/oom_score_adj  # Vergleichsvariante zuerst opfern
note() { echo "$(date '+%Y-%m-%d %H:%M') $NAME: $1" >> $OUT/fortschritt.log; }
for ARM in aktion referenz; do
  SOURCE=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_$ARM.pkl
  TAG=real_alle_s60_normal_zaehler_zeit_tf1_t_${ARM}_kopf_$NAME
  LOG=$P/data/PSP_$TAG.log
  MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
  EPOCHS=$([ $ARM = aktion ] && echo 30 || echo 12)
  note "Training $ARM beginnt ($EPOCHS Epochen, nur Zeitköpfe)"
  [ -e "$MODEL" ] || .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --cases 0 --epochs $EPOCHS \
      --learning-rate 1e-3 --offset-learning-rate 3e-2 --time-event-head $VARIANT --head-only \
      --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps \
      --counters --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training \
      --only-points --only-points-validation --arm $ARM --init-from "$SOURCE" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
  [ -e "$MODEL" ] || { note "Abbruch im Training $ARM"; exit 1; }
  note "Training $ARM fertig"
done
A=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_aktion_kopf_$NAME.pkl
B=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_referenz_kopf_$NAME.pkl
CUDA_VISIBLE_DEVICES="" .venv/bin/python ../prescriptive/policies/ev_policy/t_learner_policy.py "$A" "$B" "$HULL" >> $OUT/fortschritt.log 2>&1 \
    || { note "Abbruch: Hülle"; exit 1; }
note "Hülle gebaut, Sonde der Arme"
SONDE_OUT=$OUT/sonde_$NAME CUDA_VISIBLE_DEVICES="" .venv/bin/python -u $R/zeitkopf_sonde_20261118/sonde.py 1500 200 20 \
    "Aktion ($NAME)=$A" "Referenz ($NAME)=$B" > $OUT/logs/sonde_$NAME.log 2>&1
.venv/bin/python $OUT/sonde_zeile.py $OUT/sonde_$NAME 2>&1 | while read -r line; do note "$line"; done
note "Storno-Messung"
flock $OUT/auswertung.lock env CUDA_VISIBLE_DEVICES="" .venv/bin/python $R/zaehler_uhren_tf1_20261117/storno_abstand.py 300 20 \
    "T-Learner Aktion ($NAME)=$A" "T-Learner Referenz ($NAME)=$B" > $OUT/storno_$NAME.log 2>&1
note "Storno-Messung fertig, Prüfreihe"
flock $OUT/auswertung.lock .venv/bin/python -u $OUT/pruefreihe.py "t_kopf_$NAME=$LABEL=$HULL" > $OUT/logs/pruefreihe_$NAME.log 2>&1
note "Prüfreihe fertig, Online-Test 12 Seeds (wartet ggf. auf die andere Variante)"
flock $OUT/online.lock .venv/bin/python -u $R/modellvergleich_20261116/f_zwoelf_seeds.py --arms ${ARMKEY}_fest300 --seeds 0-11 --workers 6 \
    > $OUT/logs/online_$NAME.log 2>&1
flock $OUT/auswertung.lock .venv/bin/python -u $R/modellvergleich_20261116/f_zwoelf_auswerten.py > $OUT/logs/online_auswertung_$NAME.log 2>&1
note "Online-Test fertig (modellvergleich_20261116/f_zwoelf_auswertung.md)"
note "Alles fertig"
