#!/bin/bash
# Ein PSP je Wahlpunkt und Option (2026-11-22, Nutzer: „Basismodell ZZ-TF1 und dann die T-Learner jeweils aufteilen pro Option?
# Kalibrierung weglassen?“ – ja). 8 Netze, je warm vom ZZ-TF1 (echter Log), nur Proben am Options-Event ihres Wahlpunkts
# (--point / --option), Einstellungen wie die T-Learner-Arme: Stufe A wie t_learner (Aktions-Optionen 40, Referenz-Optionen 20 Epochen,
# LR 3e-5), Stufe B Suffix-Verlust wie sv (10 bzw. 5 Epochen), KEINE Kalibrierung. Zwei Stränge parallel auf der GPU (Aktion / Referenz),
# --load-chunks 4 (Speicher). 2026-11-22 12:35: Stufe B nur eine zur Zeit (flock), zwei parallele Suffix-Verlust-Trainings sprengten die 24 GB GPU. Danach Hülle (policies/ev_policy/option_learner_policy.py) + Kopie mit keyed_noise, Wirkungsdiagnose +
# Prognose (wie svk), Online-Test Aufbau F 12 Seeds (Arm olsv_fest300).
# Aufruf (tmux): bash bpic17_psp_option_learner.sh
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_100k/option_learner_20261122
SV=$P/scripts/results_100k/suffix_verlust_20261119
V=$P/scripts/results_100k/modellvergleich_20261116
BASE=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1.pkl
HULL=$P/models/psp_model_durchlaufzeit_option_learner_sv.pkl
HULL_CRN=$P/models/psp_model_durchlaufzeit_option_learner_sv_crn.pkl
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
AWKFILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super[(][)][.]__init__'
COMMON="--cases 0 --learning-rate 3e-5 --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps --counters --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training --only-points --only-points-validation --load-chunks 4"
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
note() { echo "$(date '+%Y-%m-%d %H:%M') $1" >> $OUT/fortschritt.log; }
# Wahlpunkt | Token | Kürzel | Epochen Stufe A | Epochen Stufe B
AKTION="annehmen_bei_konzept|A_Accepted|complete|konzept_annehmen|40|10
annehmen_nach_vervollstaendigung|A_Accepted|complete|nachv_annehmen|40|10
angebot_nachlegen|O_Create Offer|complete|nachlegen_weiteres|40|10
handle_leads|W_Handle leads|start|leads_rueckruf|40|10"
REFERENZ="annehmen_bei_konzept|W_Complete application|start|konzept_bearbeiten|20|5
annehmen_nach_vervollstaendigung|W_Complete application|resume|nachv_weiter|20|5
angebot_nachlegen|O_Sent (mail and online)|complete|nachlegen_senden|20|5
handle_leads|W_Handle leads|withdraw|leads_zurueck|20|5"
strang() {
  while IFS='|' read -r PUNKT AKT LC KURZ EA EB; do
    [ -z "$PUNKT" ] && continue
    TOKEN="$AKT|$LC"
    for STUFE in A B; do
      if [ $STUFE = A ]; then TAG=ol_$KURZ; INIT=$BASE; EP=$EA; EXTRA=""; else TAG=ol_${KURZ}_sv; INIT=$P/models/psp_model_durchlaufzeit_ol_$KURZ.pkl; EP=$EB; EXTRA="--sequence-loss"; fi
      MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
      # Checkpoint-Ordner eines abgebrochenen Laufs (kein fertiges Modell) entfernen, sonst bricht train_psp_real ab
      [ -e "$MODEL" ] || rm -rf "$P/models/psp_training_$TAG"
      note "$TAG beginnt ($PUNKT / $TOKEN, $EP Epochen)"
      # Stufe B (Suffix-Verlust) belegt ≈ 12 GB GPU-Speicher: nur eine zur Zeit (Sperre); Fehlerzeilen nicht ausfiltern
      if [ $STUFE = B ]; then SPERRE="flock $OUT/gpu_sv.lock"; else SPERRE=""; fi
      [ -e "$MODEL" ] || $SPERRE .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --epochs $EP $COMMON --point "$PUNKT" --option "$TOKEN" \
          --init-from "$INIT" $EXTRA 2>&1 | awk -v f="$AWKFILTER" '/Error/ || $0 !~ f { print; fflush() }' >> "$P/data/PSP_$TAG.log"
      [ -e "$MODEL" ] || { note "Abbruch: $TAG"; return 1; }
      note "$TAG fertig ($(grep -E 'nur die [0-9,]+ von [0-9,]+ Trainingsproben' $P/data/PSP_$TAG.log | tail -2 | sed -E 's/.*nur die ([0-9,]+) von.*/\1/' | tr '\n' '/'))"
    done
  done <<< "$1"
}
strang "$AKTION" & PA=$!
sleep 120  # versetzt starten (Laden braucht viel Speicher)
strang "$REFERENZ" & PR=$!
wait $PA; A=$?; wait $PR; R=$?
[ $A = 0 ] && [ $R = 0 ] || { note "Abbruch im Training (Aktion $A, Referenz $R)"; exit 1; }
PAARE=""
while IFS='|' read -r PUNKT AKT LC KURZ EA EB; do PAARE="$PAARE|$PUNKT=$AKT|$LC=$P/models/psp_model_durchlaufzeit_ol_${KURZ}_sv.pkl"; done <<< "$AKTION
$REFERENZ"
IFS='|' read -r -a _ <<< ""
CUDA_VISIBLE_DEVICES="" .venv/bin/python - "$HULL" "$HULL_CRN" <<PYEOF >> $OUT/fortschritt.log 2>&1 || { note "Abbruch: Hülle"; exit 1; }
import sys
sys.path.insert(0, "..")
from prescriptive.helpers.datasets import add_paths
add_paths()
from prescriptive.policies.ev_policy import option_learner_policy as ol
from prescriptive.estimators.psp_model import PSPModel
eintraege = [e for e in """$PAARE""".split("|") if e]
paare = {}
i = 0
while i < len(eintraege):  # Einträge "Punkt=Aktivität", "Lifecycle=Pfad"
    punkt, akt = eintraege[i].split("=", 1); lc, pfad = eintraege[i + 1].split("=", 1)
    paare[(punkt, f"{akt}|{lc}")] = pfad; i += 2
hull = ol.bauen(paare, sys.argv[1])
for m in [hull] + list(hull.netze.values()):
    m.params = {**(m.params or {}), "keyed_noise": True}
hull.save(sys.argv[2])
print(f"Option-Learner-Hülle {sys.argv[1]} und Kopie mit keyed_noise {sys.argv[2]} ({len(paare)} Netze)")
PYEOF
note "Hülle und Kopie gebaut, Wirkungsdiagnose beginnt"
export WIRKUNG_LABEL="Netz je Wahlpunkt und Option + Suffix-Verlust" WIRKUNG_OUT=wirkung_option_learner WIRKUNG_EXTRA=$SV/wirkung/ziehungen.pkl:$SV/wirkung_kalibrierung/ziehungen.pkl
.venv/bin/python -u $SV/wirkung.py "$HULL" > $OUT/wirkung.log 2>&1 && note "Wirkungsdiagnose fertig" || note "Wirkungsdiagnose FEHLER"
.venv/bin/python -u $SV/prognose.py > $OUT/prognose.log 2>&1
note "Prognose: $(grep -E 'Netz je Wahlpunkt' $SV/wirkung_option_learner/prognose.md | tr -s ' ')"
unset WIRKUNG_LABEL WIRKUNG_OUT WIRKUNG_EXTRA
note "Online-Test 12 Seeds beginnt (olsv_fest300)"
.venv/bin/python -u $V/f_zwoelf_seeds.py --arms olsv_fest300 --seeds 0-11 --workers 12 > $OUT/online_olsv.log 2>&1
note "Online-Test beendet (Code $?, $(grep -c COMPLETE $OUT/online_olsv.log) Seeds)"
.venv/bin/python -u $V/f_zwoelf_auswerten.py > $OUT/online_auswertung.log 2>&1
.venv/bin/python -u $V/abweichungen_regel.py > $OUT/abweichungen.log 2>&1
note "$(grep -E '^ *olsv_fest300 ' $V/f_zwoelf_auswertung.md | head -1)"
note "$(grep -E 'schneller als olsv_fest300' $V/f_zwoelf_auswertung.md | tr -s ' ' | tr '\n' ';')"
note "Alles fertig"
