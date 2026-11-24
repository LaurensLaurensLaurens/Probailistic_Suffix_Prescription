#!/bin/bash
# T-Learner mit Suffix-Verlust + GEPAARTER Kalibrierung je (Wahlpunkt, Option) (2026-11-20, Nutzer: „ja“ zur Korrektur). Befund 01:45:
# die Kalibrierung von svk (gegen das globale Gruppenmittel) bestraft auch die Streuung der Fallmittel und der Ziehungen. Hier
# calibration_mode paired: Zielwert = echtes Mittel derselben Prefixe im Batch, Quadrat aus zwei Ziehungshälften. Sonst wie
# data/bpic17_psp_suffix_kalibrierung.sh (warm von den _sv-Armen, c = 3, Aktion 10 / Referenz 5 Epochen). Danach Hülle + Kopie mit
# keyed_noise, Messung je Gruppe (je_punkt.py svp, dazu svk und sv neu wegen der Spalten zur Streuung der Fallmittel), Wirkungsdiagnose
# (wirkung_kalibrierung_gepaart/) + Prognose, Prüfreihe, Online-Test nur mit Datei suffix_verlust_20261119/online_ok_svp.
# Aufruf (tmux): bash bpic17_psp_suffix_kalibrierung_gepaart.sh
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
V=$P/scripts/results_100k/modellvergleich_20261116
OUT=$P/scripts/results_100k/suffix_verlust_20261119
HULL=$P/models/psp_model_durchlaufzeit_t_learner_svp.pkl
HULL_CRN=$P/models/psp_model_durchlaufzeit_t_learner_svp_crn.pkl
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
note() { echo "$(date '+%Y-%m-%d %H:%M') svp: $1" >> $OUT/fortschritt.log; }
for ARM in aktion referenz; do
  BASE=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_${ARM}_sv.pkl
  TAG=real_alle_s60_normal_zaehler_zeit_tf1_t_${ARM}_svp
  LOG=$P/data/PSP_$TAG.log
  MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
  EPOCHS=$([ $ARM = aktion ] && echo 10 || echo 5)
  note "Training $ARM beginnt ($EPOCHS Epochen)"
  echo "######## Training $ARM ($(date +%H:%M:%S))" | tee -a "$LOG"
  [ -e "$MODEL" ] || .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --cases 0 --epochs $EPOCHS --learning-rate 3e-5 \
      --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps \
      --counters --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training \
      --only-points --only-points-validation --arm $ARM --init-from "$BASE" --sequence-loss --sequence-calibration 3 --sequence-calibration-mode paired \
      2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
  [ -e "$MODEL" ] || { note "Abbruch im Training $ARM"; exit 1; }
  note "Training $ARM fertig: $(grep -E 'Kalibrierung je Gruppe' "$LOG" | tail -1)"
done
CUDA_VISIBLE_DEVICES="" .venv/bin/python ../prescriptive/policies/ev_policy/t_learner_policy.py \
    $P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_aktion_svp.pkl \
    $P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_referenz_svp.pkl "$HULL" >> $OUT/fortschritt.log 2>&1 || { note "Abbruch: Hülle"; exit 1; }
CUDA_VISIBLE_DEVICES="" .venv/bin/python - "$HULL" "$HULL_CRN" >> $OUT/fortschritt.log 2>&1 <<'EOF' || { note "Abbruch: Kopie keyed_noise"; exit 1; }
import sys
sys.path.insert(0, "..")
from prescriptive.helpers.datasets import add_paths
add_paths()
from prescriptive.estimators.psp_model import PSPModel
hull = PSPModel.load(sys.argv[1])
for m in (hull, hull.aktion):
    m.params = {**(m.params or {}), "keyed_noise": True}
hull.save(sys.argv[2])
print(f"Kopie mit keyed_noise gespeichert: {sys.argv[2]}")
EOF
note "Hülle und Kopie gebaut, Messung je Gruppe beginnt"
.venv/bin/python -u $OUT/je_punkt.py 400 30 svp 2>&1 | grep -v -E "$FILTER" > $OUT/je_punkt_svp_400.log && note "Messung je Gruppe fertig" || note "Messung je Gruppe FEHLER"
for OLDSET in svk sv; do .venv/bin/python -u $OUT/je_punkt.py 400 30 $OLDSET 2>&1 | grep -v -E "$FILTER" > $OUT/je_punkt_${OLDSET}_400.log && note "Messung je Gruppe $OLDSET neu (mit Streuung) fertig" || note "Messung je Gruppe $OLDSET FEHLER"; done
export WIRKUNG_LABEL="T-Learner Suffix-Verlust + gepaarte Kalibrierung" WIRKUNG_OUT=wirkung_kalibrierung_gepaart WIRKUNG_EXTRA=$OUT/wirkung/ziehungen.pkl:$OUT/wirkung_kalibrierung/ziehungen.pkl
.venv/bin/python -u $OUT/wirkung.py "$HULL" > $OUT/wirkung_kalibrierung_gepaart.log 2>&1 && note "Wirkungsdiagnose fertig" || note "Wirkungsdiagnose FEHLER"
.venv/bin/python -u $OUT/prognose.py > $OUT/prognose_kalibrierung_gepaart.log 2>&1
note "Prognose: $(grep -E 'gepaarte Kalibrierung' $OUT/wirkung_kalibrierung_gepaart/prognose.md | tr -s ' ')"
unset WIRKUNG_LABEL WIRKUNG_OUT WIRKUNG_EXTRA
.venv/bin/python -u $OUT/pruefreihe.py "t_svp_crn=T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event)=$HULL_CRN" > $OUT/pruefreihe_kalibrierung_gepaart.log 2>&1
note "Prüfreihe fertig"
if [ ! -e $OUT/online_ok_svp ]; then note "Online-Test wartet auf Zustimmung (Datei online_ok_svp fehlt)"; exit 0; fi
note "Online-Test 12 Seeds beginnt (tlsvp_fest300)"
.venv/bin/python -u $V/f_zwoelf_seeds.py --arms tlsvp_fest300 --seeds 0-11 --workers 12 > $OUT/online_tlsvp.log 2>&1
note "Online-Test beendet (Code $?, $(grep -c COMPLETE $OUT/online_tlsvp.log) Seeds)"
.venv/bin/python -u $V/f_zwoelf_auswerten.py > $OUT/online_auswertung_svp.log 2>&1
.venv/bin/python -u $V/abweichungen_regel.py > $OUT/abweichungen_svp.log 2>&1
note "$(grep -E '^ *tlsvp_fest300 ' $V/f_zwoelf_auswertung.md | head -1)"
note "$(grep -E 'schneller als tlsvp_fest300' $V/f_zwoelf_auswertung.md | tr -s ' ' | tr '\n' ';')"
note "Alles fertig"
