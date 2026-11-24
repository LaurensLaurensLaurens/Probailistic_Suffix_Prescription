#!/bin/bash
# Option-Learner auf dem Simulator-Log mit Effekten v3, Ziel Conformance (2026-11-22, Nutzer: „ja, und dieses Netz dann trainieren auf die
# Simulatordaten mit Verhindern von Conformance Deviations?“). Rezept wie bpic17_psp_option_learner.sh (8 Netze, je Wahlpunkt und Option,
# --point / --option, keine Kalibrierung), Daten und Einstellungen wie bpic17_psp_konformanz_v3x4c.sh (4-facher v3-Log, 128.000 Fälle):
# - jedes Netz warm vom T-Learner-Arm k3x4 seiner Seite (kennt den Simulator-Log schon; Aktions-Optionen vom Aktions-, Referenz-Optionen
#   vom Referenz-Arm), --vocab-from derselbe Arm, LR 1e-4, Aktions-Optionen 10 / Referenz-Optionen 5 Epochen (wie k3x4)
# - nur Stufe A: KEIN Suffix-Verlust (der wirkt nur auf die Zeiten, die KonformanzPolicy bewertet nur die gezogenen Events)
# - --load-chunks 16 (≈ 8.000 Fälle je Stück), damit zwei Stränge gleichzeitig laden können; bei MemoryError bis zu 2 neue Versuche
# Danach Hülle option_learner_k3x4.pkl, Diagnose je Teilgruppe (konformanz_signal.py, k3x4 gegen k3x4ol auf denselben Zuständen) und
# Online-Auswertung Arm konformanz_ol (Aufbau F, 12 Seeds, gepaart mit den vorhandenen bank / regel / orakel / konformanz_x4).
# Aufruf (tmux): bash bpic17_psp_konformanz_option_learner.sh
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_100k/kuenstliche_effekte_20261120
LOGP=$OUT/log_effekte_v3_x4/log_raw.pkl
HULL=$P/models/psp_model_durchlaufzeit_option_learner_k3x4.pkl
ARM_AKTION=$P/models/psp_model_durchlaufzeit_k3x4_zaehler_zeit_tf1_t_aktion.pkl
ARM_REFERENZ=$P/models/psp_model_durchlaufzeit_k3x4_zaehler_zeit_tf1_t_referenz.pkl
AWKFILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super[(][)][.]__init__'
COMMON="--cases 0 --learning-rate 1e-4 --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps --counters --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training --only-points --only-points-validation --load-chunks 16 --log-pickle $LOGP"
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
note() { echo "$(date '+%Y-%m-%d %H:%M') k3x4ol: $1" >> $OUT/fortschritt.log; }
# Wahlpunkt | Aktivität | Lifecycle | Kürzel | Epochen
AKTION="annehmen_bei_konzept|A_Accepted|complete|konzept_annehmen|10
annehmen_nach_vervollstaendigung|A_Accepted|complete|nachv_annehmen|10
angebot_nachlegen|O_Create Offer|complete|nachlegen_weiteres|10
handle_leads|W_Handle leads|start|leads_rueckruf|10"
REFERENZ="annehmen_bei_konzept|W_Complete application|start|konzept_bearbeiten|5
annehmen_nach_vervollstaendigung|W_Complete application|resume|nachv_weiter|5
angebot_nachlegen|O_Sent (mail and online)|complete|nachlegen_senden|5
handle_leads|W_Handle leads|withdraw|leads_zurueck|5"
strang() {  # $1 Liste, $2 Arm (warm und Vokabular)
  while IFS='|' read -r PUNKT AKT LC KURZ EP; do
    [ -z "$PUNKT" ] && continue
    TAG=k3x4ol_$KURZ
    MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
    for VERSUCH in 1 2 3; do
      [ -e "$MODEL" ] && break
      rm -rf "$P/models/psp_training_$TAG"  # Checkpoint-Ordner eines abgebrochenen Versuchs, sonst bricht train_psp_real ab
      note "$TAG beginnt ($PUNKT / $AKT|$LC, $EP Epochen, Versuch $VERSUCH)"
      .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --epochs $EP $COMMON --point "$PUNKT" --option "$AKT|$LC" \
          --init-from "$2" --vocab-from "$2" 2>&1 | awk -v f="$AWKFILTER" '/Error/ || $0 !~ f { print; fflush() }' >> "$P/data/PSP_$TAG.log"
      [ -e "$MODEL" ] && break
      if [ $VERSUCH -lt 3 ] && tail -50 "$P/data/PSP_$TAG.log" | grep -q MemoryError; then note "$TAG: MemoryError, neuer Versuch in 10 Minuten"; sleep 600; else break; fi
    done
    [ -e "$MODEL" ] || { note "Abbruch: $TAG"; return 1; }
    note "$TAG fertig ($(grep -E 'nur die [0-9,]+ von [0-9,]+ Trainingsproben' $P/data/PSP_$TAG.log | tail -2 | sed -E 's/.*nur die ([0-9,]+) von.*/\1/' | tr '\n' '/'))"
  done <<< "$1"
}
[ -e "$LOGP" ] && [ -e "$ARM_AKTION" ] && [ -e "$ARM_REFERENZ" ] || { note "Abbruch: Log oder k3x4-Arme fehlen"; exit 1; }
strang "$AKTION" "$ARM_AKTION" & PA=$!
sleep 900  # versetzt starten: das Laden braucht viel Arbeitsspeicher
strang "$REFERENZ" "$ARM_REFERENZ" & PR=$!
wait $PA; A=$?; wait $PR; R=$?
[ $A = 0 ] && [ $R = 0 ] || { note "Abbruch im Training (Aktion $A, Referenz $R)"; exit 1; }
PAARE=()
while IFS='|' read -r PUNKT AKT LC KURZ EP; do PAARE+=("$PUNKT=$AKT|$LC=$P/models/psp_model_durchlaufzeit_k3x4ol_$KURZ.pkl"); done <<< "$AKTION
$REFERENZ"
CUDA_VISIBLE_DEVICES="" .venv/bin/python ../prescriptive/policies/ev_policy/option_learner_policy.py "$HULL" "${PAARE[@]}" >> $OUT/fortschritt.log 2>&1 || { note "Abbruch: Hülle"; exit 1; }
note "Hülle gebaut, Diagnose je Teilgruppe beginnt (k3x4 gegen k3x4ol)"
SIGNAL_MODELLE=k3x4,k3x4ol SIGNAL_NAME=konformanz_signal_ol .venv/bin/python -u $OUT/diagnose/konformanz_signal.py 6 60 > $OUT/diagnose/konformanz_signal_ol.log 2>&1 \
    && note "Diagnose fertig" || note "Diagnose FEHLER"
note "Online-Auswertung konformanz_ol beginnt"
.venv/bin/python -u $OUT/online_konformanz.py konformanz_ol 0-11 12 > $OUT/online_konformanz_ol.log 2>&1
note "Online-Auswertung beendet (Code $?): $(grep -E '^FERTIG' $OUT/online_konformanz_ol.log | head -1)"
note "Alles fertig"
