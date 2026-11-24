#!/bin/bash
# T-Learner mit Suffix-Verlust (2026-11-19, Nutzer: „führe das aus ja“). Beide Arme warm vom fertigen T-Learner-Arm (ZZ-TF1), gleiche
# Einstellungen wie data/bpic17_psp_t_learner.sh, zusätzlich --sequence-loss (CRPS der frei gezogenen Restzeit gegen die echte Restzeit,
# Gewicht 1, je Batch 32 Prefixe × 16 Ziehungen); Aktion 10 Epochen (~17.000 Proben), Referenz 5 Epochen (~58.000 Proben), nacheinander.
# Danach: Hülle, Kopie mit keyed_noise (wie tlcrn), Wirkungsdiagnose (suffix_verlust_20261119/wirkung.py, gleiche Zustände wie
# crn_schluessel.py) und Prüfreihe (Register). Kein Online-Test (braucht eigenes OK).
# Aufruf (tmux): bash bpic17_psp_suffix_verlust.sh
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_100k/suffix_verlust_20261119
HULL=$P/models/psp_model_durchlaufzeit_t_learner_sv.pkl
HULL_CRN=$P/models/psp_model_durchlaufzeit_t_learner_sv_crn.pkl
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
note() { echo "$(date '+%Y-%m-%d %H:%M') $1" >> $OUT/fortschritt.log; }
for ARM in aktion referenz; do
  BASE=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_$ARM.pkl
  TAG=real_alle_s60_normal_zaehler_zeit_tf1_t_${ARM}_sv
  LOG=$P/data/PSP_$TAG.log
  MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
  EPOCHS=$([ $ARM = aktion ] && echo 10 || echo 5)
  note "Training $ARM beginnt ($EPOCHS Epochen)"
  echo "######## Training $ARM ($(date +%H:%M:%S))" | tee -a "$LOG"
  [ -e "$MODEL" ] || .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --cases 0 --epochs $EPOCHS --learning-rate 3e-5 \
      --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps \
      --counters --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training \
      --only-points --only-points-validation --arm $ARM --init-from "$BASE" --sequence-loss 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
  [ -e "$MODEL" ] || { note "Abbruch im Training $ARM"; exit 1; }
  note "Training $ARM fertig: $(grep -E 'Suffix-Verlust' "$LOG" | tail -1)"
done
CUDA_VISIBLE_DEVICES="" .venv/bin/python ../prescriptive/policies/ev_policy/t_learner_policy.py \
    $P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_aktion_sv.pkl \
    $P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_referenz_sv.pkl "$HULL" >> $OUT/fortschritt.log 2>&1 || { note "Abbruch: Hülle"; exit 1; }
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
note "Hülle und Kopie mit keyed_noise gebaut, Wirkungsdiagnose beginnt"
.venv/bin/python -u $OUT/wirkung.py "$HULL" > $OUT/wirkung.log 2>&1 && note "Wirkungsdiagnose fertig" || note "Wirkungsdiagnose FEHLER"
.venv/bin/python -u $OUT/pruefreihe.py "t_sv_crn=T-Learner Suffix-Verlust (Zufallszahlen je Event)=$HULL_CRN" > $OUT/pruefreihe.log 2>&1
note "Alles fertig"
