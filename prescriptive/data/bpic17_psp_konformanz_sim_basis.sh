#!/bin/bash
# Option-Learner für Conformance mit einer Basis NUR auf dem Simulator (2026-11-23, Nutzer: „ja“ – „also trainieren auf den Simulatordaten?“).
# Bisher (k3x4ol) kam die Basis ZZ-TF1 vom echten Log; die verzerrten Teilgruppen (konzept ≥/< 11k, leads < 11k) hängen am Betrag, den die
# Basis des echten Logs an diesen Stellen übergeht. Ablauf:
# 1. Basis ZZ-TF1 (Einstellungen wie bpic17_psp_real_zaehler_zeit_tf1.sh, alle Prefixe) auf dem 1-fachen v3-Log (32.000 Fälle; mehr geht wegen
#    ulimit -v 56 GB nicht), 10 statt 15 Epochen (auf dem echten Log Plateau ab Epoche 9–12). Bei MemoryError neuer Versuch mit --cases 20000.
# 2. Der 4-fache Log enthält ≈ 10 Fälle mit Kategorien, die die Basis nicht kennt (1 Aktivität, 5 Mitarbeiter) → vocab_from bräche ab. Diese
#    Fälle werden entfernt (Log x4 mit Basisvokabular, Zahl der entfernten Fälle im Fortschritt).
# 3. 8 Netze je (Wahlpunkt, Option) wie bpic17_psp_konformanz_option_learner.sh, aber warm von der neuen Basis (--init-from / --vocab-from),
#    LR 1e-4, Aktions-Optionen 10 / Referenz-Optionen 5 Epochen, nur Stufe A, --load-chunks 16, zwei Stränge.
# 4. Hülle option_learner_simv3.pkl, schlanke Kopie mit keyed_noise (_schlank_crn), Diagnose konformanz_unsicherheit (100 Ziehungen wie bisher),
#    Online-Arm konformanz_sim (keyed_noise, 300 Ziehungen, 12 Seeds).
# Kontrolle parallel (CPU, 6 Prozesse, startet nach Epoche 1 der Basis): konformanz_ol300 = bisherige Netze k3x4ol mit keyed_noise und 300 Ziehungen
#    – trennt den Einfluss der neuen Basis von dem der Kopplung und der Ziehungszahl.
# Aufruf (tmux): bash bpic17_psp_konformanz_sim_basis.sh
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

# Kontrolle konformanz_ol300 im Hintergrund, sobald Epoche 1 der Basis gespeichert ist (Spitze beim Laden der Basis vorbei)
(
  until [ -e "$P/models/psp_training_$BASE_TAG/last.pt" ] || [ -e "$BASE" ] || [ -e "$SB/basis_ende" ]; do sleep 120; done
  note "Kontrolle konformanz_ol300 beginnt (6 Prozesse)"
  .venv/bin/python -u $K/online_konformanz.py konformanz_ol300 0-11 6 > $K/online_konformanz_ol300.log 2>&1
  note "Kontrolle konformanz_ol300 beendet (Code $?): $(grep -E '^FERTIG' $K/online_konformanz_ol300.log | head -1)"
) &
KONTROLLE=$!

# 1. Basis
for VERSUCH in 1 2; do
  [ -e "$BASE" ] && break
  FAELLE=$([ $VERSUCH = 1 ] && echo 0 || echo 20000)
  rm -rf "$P/models/psp_training_$BASE_TAG"
  note "Basis $BASE_TAG beginnt (10 Epochen, --cases $FAELLE, Versuch $VERSUCH, GPU: $(.venv/bin/python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null))"
  .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $BASE_TAG --cases $FAELLE --epochs 10 --learning-rate 1e-4 \
      --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps \
      --counters --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training --log-pickle $LOG1 2>&1 \
      | awk -v f="$AWKFILTER" '/Error/ || $0 !~ f { print; fflush() }' >> "$P/data/PSP_$BASE_TAG.log"
  [ -e "$BASE" ] && break
  tail -50 "$P/data/PSP_$BASE_TAG.log" | grep -q MemoryError || break
  note "Basis: MemoryError, neuer Versuch mit 20.000 Fällen"
done
touch "$SB/basis_ende"  # Basis beendet (auch bei Abbruch): Kontrolle darf spätestens jetzt starten
[ -e "$BASE" ] || { note "Abbruch: Basis"; wait $KONTROLLE; exit 1; }
note "Basis fertig ($(grep -o "'Validierungsverlust letzte Epoche': [0-9.]*" $P/data/PSP_$BASE_TAG.log | tail -1))"

# 2. 4-facher Log ohne Fälle mit Kategorien außerhalb des Basisvokabulars
CUDA_VISIBLE_DEVICES="" .venv/bin/python - "$BASE" "$LOG4" "$LOG4F" <<'PYEOF' >> $SB/fortschritt.log 2>&1 || { note "Abbruch: Log-Filter"; wait $KONTROLLE; exit 1; }
import sys
sys.path.insert(0, "..")
from prescriptive.helpers.datasets import add_paths
add_paths()
import pandas as pd
from preprocessing import preprocess
from prescriptive.estimators.psp_model import PSPModel
base = PSPModel.load(sys.argv[1])
raw = pd.read_pickle(sys.argv[2])
log = preprocess("bpic17", raw, verbose=False)
case = "case:concept:name"
schlecht = set()
for spalte, enc in base.encoder_decoder.categorical_encoders.items():
    if spalte not in log.columns:
        continue
    bekannt = {str(v) for v in enc.categories_[0] if not (isinstance(v, float) and v != v)}
    werte = log[spalte].dropna().astype(str)
    neu = ~werte.isin(bekannt) & ~werte.isin(["nan", "None"])
    faelle = set(log.loc[werte.index[neu], case].astype(str))
    if faelle:
        print(f"  {spalte}: {sorted(set(werte[neu]))[:8]} in {len(faelle)} Fällen")
    schlecht |= faelle
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
[ $A = 0 ] && [ $R = 0 ] || { note "Abbruch im Training der Netze (Aktion $A, Referenz $R)"; wait $KONTROLLE; exit 1; }

# 4. Hülle, schlanke Kopie mit keyed_noise, Diagnose, Online
PAARE=()
while IFS='|' read -r PUNKT AKT LC KURZ EP; do PAARE+=("$PUNKT=$AKT|$LC=$P/models/psp_model_durchlaufzeit_simv3ol_$KURZ.pkl"); done <<< "$AKTION
$REFERENZ"
CUDA_VISIBLE_DEVICES="" .venv/bin/python ../prescriptive/policies/ev_policy/option_learner_policy.py "$HULL" "${PAARE[@]}" >> $SB/fortschritt.log 2>&1 || { note "Abbruch: Hülle"; wait $KONTROLLE; exit 1; }
CUDA_VISIBLE_DEVICES="" .venv/bin/python - "$HULL" "$HULL_CRN" <<'PYEOF' >> $SB/fortschritt.log 2>&1 || { note "Abbruch: schlanke Kopie"; wait $KONTROLLE; exit 1; }
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
wait $KONTROLLE  # Kontrolle muss fertig sein (Speicher, CPU)
note "Online-Auswertung konformanz_sim beginnt (12 Seeds, 300 Ziehungen)"
.venv/bin/python -u $K/online_konformanz.py konformanz_sim 0-11 12 > $K/online_konformanz_sim.log 2>&1
note "Online-Auswertung beendet (Code $?): $(grep -E '^FERTIG' $K/online_konformanz_sim.log | head -1)"
note "Alles fertig"
