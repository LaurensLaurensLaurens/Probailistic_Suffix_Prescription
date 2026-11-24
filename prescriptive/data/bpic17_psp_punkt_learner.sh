#!/bin/bash
# S-Learner je Wahlpunkt (2026-11-22, Nutzer: „ja, bereite das so vor“). 4 Netze, je warm vom ZZ-TF1 (echter Log), nur Proben am Options-
# Event IHRES Wahlpunkts, beide Optionen (--point ohne --option). Stufe A 20 Epochen, Stufe B Suffix-Verlust 5 Epochen, LR 3e-5, keine
# Kalibrierung; zwei Stränge, Stufe B nacheinander (flock, GPU-Speicher), --load-chunks 4. Startet erst, wenn der Option-Learner-Lauf
# (option_learner_20261122) fertig ist. Danach Online-Arm eintragen, Hülle (option_learner_policy, beide Optionen → dasselbe Netz, gemeinsam
# gezogen) + keyed_noise, Wirkungsdiagnose + Prognose, Online Aufbau F 12 Seeds (plsv_fest300).
# Aufruf (tmux): bash bpic17_psp_punkt_learner.sh
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_100k/punkt_learner_20261122
OL=$P/scripts/results_100k/option_learner_20261122
SV=$P/scripts/results_100k/suffix_verlust_20261119
V=$P/scripts/results_100k/modellvergleich_20261116
BASE=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1.pkl
HULL=$P/models/psp_model_durchlaufzeit_punkt_learner_sv.pkl
HULL_CRN=$P/models/psp_model_durchlaufzeit_punkt_learner_sv_crn.pkl
AWKFILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super[(][)][.]__init__'
COMMON="--cases 0 --learning-rate 3e-5 --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps --counters --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training --only-points --only-points-validation --load-chunks 4"
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
note() { echo "$(date '+%Y-%m-%d %H:%M') $1" >> $OUT/fortschritt.log; }
note "wartet auf das Ende des Option-Learner-Laufs"
until grep -qE "Alles fertig|Abbruch im Training|Abbruch: Hülle" <(sed -n '/Online-Test 12 Seeds beginnt (olsv_fest300)/,$p' $OL/fortschritt.log); do sleep 120; done
note "Option-Learner fertig, Training beginnt"
# Wahlpunkt | Kürzel
STRANG1="annehmen_bei_konzept|konzept
annehmen_nach_vervollstaendigung|nachv"
STRANG2="angebot_nachlegen|nachlegen
handle_leads|leads"
strang() {
  while IFS='|' read -r PUNKT KURZ; do
    [ -z "$PUNKT" ] && continue
    for STUFE in A B; do
      if [ $STUFE = A ]; then TAG=pl_$KURZ; INIT=$BASE; EP=20; EXTRA=""; SPERRE=""; else TAG=pl_${KURZ}_sv; INIT=$P/models/psp_model_durchlaufzeit_pl_$KURZ.pkl; EP=5; EXTRA="--sequence-loss"; SPERRE="flock $OUT/gpu_sv.lock"; fi
      MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
      [ -e "$MODEL" ] || rm -rf "$P/models/psp_training_$TAG"
      note "$TAG beginnt ($PUNKT, beide Optionen, $EP Epochen)"
      [ -e "$MODEL" ] || $SPERRE .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --epochs $EP $COMMON --point "$PUNKT" \
          --init-from "$INIT" $EXTRA 2>&1 | awk -v f="$AWKFILTER" '/Error/ || $0 !~ f { print; fflush() }' >> "$P/data/PSP_$TAG.log"
      [ -e "$MODEL" ] || { note "Abbruch: $TAG"; return 1; }
      note "$TAG fertig ($(grep -E 'nur die [0-9,]+ von [0-9,]+ Trainingsproben' $P/data/PSP_$TAG.log | tail -2 | sed -E 's/.*nur die ([0-9,]+) von.*/\1/' | tr '\n' '/'); Validierung $(grep 'Validation Loss for Scheduler' $P/data/PSP_$TAG.log | awk '{print $NF}' | sed -n '1p;$p' | tr '\n' ' '))"
    done
  done <<< "$1"
}
strang "$STRANG1" & PA=$!
sleep 120
strang "$STRANG2" & PB=$!
wait $PA; A=$?; wait $PB; B=$?
[ $A = 0 ] && [ $B = 0 ] || { note "Abbruch im Training ($A / $B)"; exit 1; }
CUDA_VISIBLE_DEVICES="" .venv/bin/python - "$HULL" "$HULL_CRN" "$P/models" <<'PYEOF' >> $OUT/fortschritt.log 2>&1 || { note "Abbruch: Hülle"; exit 1; }
import sys
sys.path.insert(0, "..")
from prescriptive.helpers.datasets import add_paths
add_paths()
from prescriptive.policies.ev_policy import option_learner_policy as ol
m = sys.argv[3]
punkte = {"annehmen_bei_konzept": ("konzept", ["A_Accepted|complete", "W_Complete application|start"]),
          "annehmen_nach_vervollstaendigung": ("nachv", ["A_Accepted|complete", "W_Complete application|resume"]),
          "angebot_nachlegen": ("nachlegen", ["O_Create Offer|complete", "O_Sent (mail and online)|complete"]),
          "handle_leads": ("leads", ["W_Handle leads|start", "W_Handle leads|withdraw"])}
paare = {(p, t): f"{m}/psp_model_durchlaufzeit_pl_{k}_sv.pkl" for p, (k, tokens) in punkte.items() for t in tokens}
hull = ol.bauen(paare, sys.argv[1])
for net in [hull] + list({id(n): n for n in hull.netze.values()}.values()):
    net.params = {**(net.params or {}), "keyed_noise": True}
hull.save(sys.argv[2])
print(f"Punkt-Learner-Hülle {sys.argv[1]} und Kopie mit keyed_noise {sys.argv[2]} ({len({id(n) for n in hull.netze.values()})} Netze)")
PYEOF
note "Hülle und Kopie gebaut, Wirkungsdiagnose beginnt"
export WIRKUNG_LABEL="S-Learner je Wahlpunkt + Suffix-Verlust" WIRKUNG_OUT=wirkung_punkt_learner WIRKUNG_EXTRA=$SV/wirkung/ziehungen.pkl:$SV/wirkung_kalibrierung/ziehungen.pkl:$SV/wirkung_option_learner/ziehungen.pkl
.venv/bin/python -u $SV/wirkung.py "$HULL" > $OUT/wirkung.log 2>&1 && note "Wirkungsdiagnose fertig" || note "Wirkungsdiagnose FEHLER"
.venv/bin/python -u $SV/prognose.py > $OUT/prognose.log 2>&1
note "Prognose: $(grep -E 'S-Learner je Wahlpunkt' $SV/wirkung_punkt_learner/prognose.md | tr -s ' ')"
unset WIRKUNG_LABEL WIRKUNG_OUT WIRKUNG_EXTRA
.venv/bin/python $OUT/arm_eintragen.py >> $OUT/fortschritt.log 2>&1 || { note "Abbruch: Arm eintragen"; exit 1; }
note "Online-Test 12 Seeds beginnt (plsv_fest300)"
.venv/bin/python -u $V/f_zwoelf_seeds.py --arms plsv_fest300 --seeds 0-11 --workers 12 > $OUT/online_plsv.log 2>&1
note "Online-Test beendet (Code $?, $(grep -c COMPLETE $OUT/online_plsv.log) Seeds)"
.venv/bin/python -u $V/f_zwoelf_auswerten.py > $OUT/online_auswertung.log 2>&1
.venv/bin/python -u $V/abweichungen_regel.py > $OUT/abweichungen.log 2>&1
note "$(grep -E '^ *plsv_fest300 ' $V/f_zwoelf_auswertung.md | head -1)"
note "$(grep -E 'schneller als plsv_fest300' $V/f_zwoelf_auswertung.md | tr -s ' ' | tr '\n' ';')"
note "Alles fertig"
