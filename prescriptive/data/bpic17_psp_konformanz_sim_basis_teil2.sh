#!/bin/bash
# Fortsetzung von bpic17_psp_konformanz_sim_basis.sh (2026-11-23 13:35). Das erste Skript wurde beendet; das von ihm gestartete Basistraining
# (simv3_s60_zaehler_zeit_tf1) und seine Kontrolle konformanz_ol300 (eigener Prozess) laufen weiter. Grund: der Filter in Schritt 2 verglich
# Rohnamen der Aktivitäten mit dem Basisvokabular aus Tokens Aktivität|Lifecycle und hätte alle Fälle entfernt. Hier über PSPModel.view
# (dieselbe Aufbereitung wie im Training) und mit Sicherung (höchstens 1 % der Fälle dürfen wegfallen). Schritte 3–4 unverändert.
# Aufruf (tmux): bash bpic17_psp_konformanz_sim_basis_teil2.sh
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
K=$P/scripts/results_100k/kuenstliche_effekte_20261120
SB=$P/scripts/results_100k/sim_basis_20261123
LOG1=$K/log_effekte_v3/log_raw.pkl
LOG4=$K/log_effekte_v3_x4/log_raw.pkl
LOG4F=$SB/log_x4_basisvokabular.pkl
BASE_TAG=simv3_s60_zaehler_zeit_tf1
BASE=$P/models/psp_model_durchlaufzeit_$BASE_TAG.pkl
HULL=$P/models/psp_model_durchlaufzeit_option_learner_simv3.pkl
HULL_CRN=$P/models/psp_model_durchlaufzeit_option_learner_simv3_schlank_crn.pkl
AWKFILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super[(][)][.]__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
mkdir -p $SB
cd $ROOT/process-simulation-engine
note() { echo "$(date '+%Y-%m-%d %H:%M') $1" >> $SB/fortschritt.log; }
note "Teil 2 übernimmt (korrigierter Log-Filter); wartet auf die laufende Basis"

# 1. Auf die laufende Basis warten; bei MemoryError ein Versuch mit 20.000 Fällen
while pgrep -f "tag $BASE_TAG " > /dev/null; do sleep 120; done
if [ ! -e "$BASE" ] && tail -50 "$P/data/PSP_$BASE_TAG.log" | grep -q MemoryError; then
  rm -rf "$P/models/psp_training_$BASE_TAG"
  note "Basis: MemoryError, neuer Versuch mit 20.000 Fällen"
  .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $BASE_TAG --cases 20000 --epochs 10 --learning-rate 1e-4 \
      --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps \
      --counters --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training --log-pickle $LOG1 2>&1 \
      | awk -v f="$AWKFILTER" '/Error/ || $0 !~ f { print; fflush() }' >> "$P/data/PSP_$BASE_TAG.log"
fi
touch "$SB/basis_ende"
[ -e "$BASE" ] || { note "Abbruch: Basis"; exit 1; }
note "Basis fertig ($(grep -o "'Validierungsverlust letzte Epoche': [0-9.]*" $P/data/PSP_$BASE_TAG.log | tail -1))"

# 2. 4-facher Log ohne Fälle mit Kategorien außerhalb des Basisvokabulars (Vergleich in der Aufbereitung des Trainings: PSPModel.view)
CUDA_VISIBLE_DEVICES="" .venv/bin/python - "$BASE" "$LOG4" "$LOG4F" <<'PYEOF' >> $SB/fortschritt.log 2>&1 || { note "Abbruch: Log-Filter"; exit 1; }
import sys
sys.path.insert(0, "..")
from prescriptive.helpers.datasets import add_paths
add_paths()
import pandas as pd
from preprocessing import preprocess
from prescriptive.estimators.psp_model import PSPModel
base = PSPModel.load(sys.argv[1])
raw = pd.read_pickle(sys.argv[2])
log = preprocess("bpic17", raw, verbose=False).reset_index(drop=True)
view = base.view(log)
case = "case:concept:name"
schlecht = set()
for spalte, enc in base.encoder_decoder.categorical_encoders.items():
    if spalte not in view.columns:
        continue
    bekannt = {str(v) for v in enc.categories_[0] if not (isinstance(v, float) and v != v)}
    werte = view[spalte].astype(str)
    neu = ~werte.isin(bekannt) & ~werte.isin(["nan", "None", "NaN"])
    faelle = set(view.loc[neu, case].astype(str))
    if faelle:
        print(f"  {spalte}: {sorted(set(werte[neu]))[:8]} in {len(faelle)} Fällen")
    schlecht |= faelle
anteil = len(schlecht) / raw[case].nunique()
assert anteil < 0.01, f"zu viele Fälle mit unbekannten Kategorien: {len(schlecht)} ({anteil:.1%}) – Filter prüfen"
raw[~raw[case].astype(str).isin(schlecht)].to_pickle(sys.argv[3])
print(f"Log x4 mit Basisvokabular: {len(schlecht)} von {raw[case].nunique():,} Fällen entfernt")
PYEOF
note "Log x4 gefiltert: $(tail -1 $SB/fortschritt.log)"

# 3. 8 Netze
COMMON="--cases 0 --learning-rate 1e-4 --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps --counters --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training --only-points --only-points-validation --load-chunks 16 --log-pickle $LOG4F"
AKTION="annehmen_bei_konzept|A_Accepted|complete|konzept_annehmen|10
annehmen_nach_vervollstaendigung|A_Accepted|complete|nachv_annehmen|10
angebot_nachlegen|O_Create Offer|complete|nachlegen_weiteres|10
handle_leads|W_Handle leads|start|leads_rueckruf|10"
REFERENZ="annehmen_bei_konzept|W_Complete application|start|konzept_bearbeiten|5
annehmen_nach_vervollstaendigung|W_Complete application|resume|nachv_weiter|5
angebot_nachlegen|O_Sent (mail and online)|complete|nachlegen_senden|5
handle_leads|W_Handle leads|withdraw|leads_zurueck|5"
strang() {
  while IFS='|' read -r PUNKT AKT LC KURZ EP; do
    [ -z "$PUNKT" ] && continue
    TAG=simv3ol_$KURZ
    MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
    for VERSUCH in 1 2 3; do
      [ -e "$MODEL" ] && break
      rm -rf "$P/models/psp_training_$TAG"
      note "$TAG beginnt ($PUNKT / $AKT|$LC, $EP Epochen, Versuch $VERSUCH)"
      .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --epochs $EP $COMMON --point "$PUNKT" --option "$AKT|$LC" \
          --init-from "$BASE" --vocab-from "$BASE" 2>&1 | awk -v f="$AWKFILTER" '/Error/ || $0 !~ f { print; fflush() }' >> "$P/data/PSP_$TAG.log"
      [ -e "$MODEL" ] && break
      if [ $VERSUCH -lt 3 ] && tail -50 "$P/data/PSP_$TAG.log" | grep -q MemoryError; then note "$TAG: MemoryError, neuer Versuch in 10 Minuten"; sleep 600; else break; fi
    done
    [ -e "$MODEL" ] || { note "Abbruch: $TAG ($(grep -m1 -E 'Error' $P/data/PSP_$TAG.log | cut -c1-150))"; return 1; }
    note "$TAG fertig ($(grep -E 'nur die [0-9,]+ von [0-9,]+ Trainingsproben' $P/data/PSP_$TAG.log | tail -2 | sed -E 's/.*nur die ([0-9,]+) von.*/\1/' | tr '\n' '/'))"
  done <<< "$1"
}
strang "$AKTION" & PA=$!
sleep 900
strang "$REFERENZ" & PR=$!
wait $PA; A=$?; wait $PR; R=$?
[ $A = 0 ] && [ $R = 0 ] || { note "Abbruch im Training der Netze (Aktion $A, Referenz $R)"; exit 1; }

# 4. Hülle, schlanke Kopie mit keyed_noise, Diagnose, Online
PAARE=()
while IFS='|' read -r PUNKT AKT LC KURZ EP; do PAARE+=("$PUNKT=$AKT|$LC=$P/models/psp_model_durchlaufzeit_simv3ol_$KURZ.pkl"); done <<< "$AKTION
$REFERENZ"
CUDA_VISIBLE_DEVICES="" .venv/bin/python ../prescriptive/policies/ev_policy/option_learner_policy.py "$HULL" "${PAARE[@]}" >> $SB/fortschritt.log 2>&1 || { note "Abbruch: Hülle"; exit 1; }
CUDA_VISIBLE_DEVICES="" .venv/bin/python - "$HULL" "$HULL_CRN" <<'PYEOF' >> $SB/fortschritt.log 2>&1 || { note "Abbruch: schlanke Kopie"; exit 1; }
import sys
sys.path.insert(0, "..")
from prescriptive.helpers.datasets import add_paths
add_paths()
import pandas as pd
from prescriptive.estimators.psp_model import PSPModel
hull = PSPModel.load(sys.argv[1])
for m in [hull] + list(dict.fromkeys(hull.netze.values())):
    if isinstance(getattr(m.encoder_decoder, "event_log", None), pd.DataFrame):
        m.encoder_decoder.event_log = m.encoder_decoder.event_log.head(0).copy()  # nur im Training gebraucht (huelle_schlank.py)
    m.params = {**(m.params or {}), "keyed_noise": True}
hull.save(sys.argv[2])
print(f"schlanke Kopie mit keyed_noise: {sys.argv[2]}")
PYEOF
note "Hülle gebaut, Diagnose beginnt"
KU_MODELL=$HULL_CRN KU_NAME=konformanz_unsicherheit_simbasis .venv/bin/python -u $K/diagnose/konformanz_unsicherheit.py 6 60 > $SB/diagnose.log 2>&1 \
    && note "Diagnose fertig: $(grep -E '^Gesamt' $K/diagnose/konformanz_unsicherheit_simbasis.md)" || note "Diagnose FEHLER"
until grep -q "Kontrolle konformanz_ol300 beendet" $SB/fortschritt.log; do sleep 120; done  # Kontrolle fertig (Speicher, CPU)
note "Online-Auswertung konformanz_sim beginnt (12 Seeds, 300 Ziehungen)"
.venv/bin/python -u $K/online_konformanz.py konformanz_sim 0-11 12 > $K/online_konformanz_sim.log 2>&1
note "Online-Auswertung beendet (Code $?): $(grep -E '^FERTIG' $K/online_konformanz_sim.log | head -1)"
note "Alles fertig"
