#!/bin/bash
# Statische Merkmale in jedem Decoder-Schritt (2026-11-24, Nutzer: „ja“ zum Vorschlag nach der Sonde diagnose/sonde_betrag.md: der Betrag
# verblasst im Decoder nach 6–15 Schritten, bevor die Effekt-Events gezogen werden). psp_model static_in_decoder (Gewichte mit 0 begonnen,
# geprüft: statik_decoder_20261124/test_statik.py – mit Null-Gewichten bitgenau wie k3x4ol, Speichern/Laden gleich; Probelauf ok).
# Rezept wie k3x4ol (bpic17_psp_konformanz_option_learner.sh), jedes Netz warm von SEINEM k3x4ol-Netz (--init-from/--vocab-from),
# --static-in-decoder, LR 1e-4, neue Abbildung LR 1e-3 (--static-learning-rate), 5 Epochen je Netz.
# 1. die beiden konzept-Netze (parallel) → Zwischenhülle (k3x4ol mit den zwei neuen konzept-Netzen, schlank, keyed_noise)
# 2. Prüfung an konzept: konzept_effekt_gezogen.py und sonde_betrag.py auf der Zwischenhülle. Tor: Gruppenunterschied
#    (A_Complete doppelt bei „vervollständigen“ < 11k − ≥ 11k, oder W_Validate fehlt bei „annehmen“ ≥ 11k − < 11k) ≥ 10 Punkte
#    (Wahrheit 30 / 41, k3x4ol 2 / 1). Sonst Ende.
# 3. die übrigen 6 Netze nacheinander (ein Strang: die Pipeline der Simulator-Basis trainiert ab ~5 Uhr selbst 2 Stränge), jeweils erst
#    starten, wenn ≥ 20 GB frei sind und < 3 Trainings laufen.
# 4. Hülle k3x4sd (schlank, keyed_noise); Online-Arm konformanz_sd (12 Seeds, 300 Ziehungen, 8 Prozesse) erst nach „Alles fertig“ der
#    Simulator-Basis (Speicher), gepaart mit konformanz_ol300.
# Aufruf (tmux): bash bpic17_psp_statik_decoder.sh
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
K=$P/scripts/results_100k/kuenstliche_effekte_20261120
SD=$P/scripts/results_100k/statik_decoder_20261124
SB=$P/scripts/results_100k/sim_basis_20261123
LOGP=$K/log_effekte_v3_x4/log_raw.pkl
HULL_ZW=$P/models/psp_model_durchlaufzeit_option_learner_k3x4sd_konzept.pkl
HULL_ZW_CRN=$P/models/psp_model_durchlaufzeit_option_learner_k3x4sd_konzept_schlank_crn.pkl
HULL=$P/models/psp_model_durchlaufzeit_option_learner_k3x4sd.pkl
HULL_CRN=$P/models/psp_model_durchlaufzeit_option_learner_k3x4sd_schlank_crn.pkl
AWKFILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super[(][)][.]__init__'
COMMON="--cases 0 --learning-rate 1e-4 --static-in-decoder --static-learning-rate 1e-3 --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps --counters --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training --only-points --only-points-validation --load-chunks 16 --log-pickle $LOGP"
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
echo 1000 > /proc/self/oom_score_adj 2>/dev/null  # bei Speichermangel zuerst diese Läufe (vererbt sich an die Kinder)
note() { echo "$(date '+%Y-%m-%d %H:%M') $1" >> $SD/fortschritt.log; }
frei() { free -g | awk '/Mem:/ {print $7}'; }
trainings() { pgrep -fc "scripts/train_psp_real.py" ; }

netz() {  # $1 Wahlpunkt, $2 Aktivität|Lifecycle, $3 Kürzel (wie k3x4ol), $4 Epochen, $5 1 = auf Speicher/Trainings warten
  local TAG=k3x4sd_$3 MODEL=$P/models/psp_model_durchlaufzeit_k3x4sd_$3.pkl SRC=$P/models/psp_model_durchlaufzeit_k3x4ol_$3.pkl
  [ -e "$SRC" ] || { note "Abbruch: $SRC fehlt"; return 1; }
  for VERSUCH in 1 2 3; do
    [ -e "$MODEL" ] && break
    if [ "$5" = 1 ]; then
      while [ "$(frei)" -lt 20 ] || [ "$(trainings)" -ge 3 ]; do sleep 120; done
    fi
    rm -rf "$P/models/psp_training_$TAG"
    note "$TAG beginnt ($1 / $2, $4 Epochen, Versuch $VERSUCH, frei $(frei) GB, Trainings $(trainings))"
    .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --epochs $4 $COMMON --point "$1" --option "$2" \
        --init-from "$SRC" --vocab-from "$SRC" 2>&1 | awk -v f="$AWKFILTER" '/Error/ || $0 !~ f { print; fflush() }' >> "$P/data/PSP_$TAG.log"
    [ -e "$MODEL" ] && break
    if [ $VERSUCH -lt 3 ] && tail -50 "$P/data/PSP_$TAG.log" | grep -q MemoryError; then note "$TAG: MemoryError, neuer Versuch in 10 Minuten"; sleep 600; else break; fi
  done
  [ -e "$MODEL" ] || { note "Abbruch: $TAG"; return 1; }
  note "$TAG fertig"
}

huelle() {  # $1 Hülle, $2 schlanke Kopie mit keyed_noise, $3 Präfix der konzept-Netze, $4 Präfix der übrigen Netze
  local PAARE=(
    "annehmen_bei_konzept=A_Accepted|complete=$P/models/psp_model_durchlaufzeit_$3_konzept_annehmen.pkl"
    "annehmen_bei_konzept=W_Complete application|start=$P/models/psp_model_durchlaufzeit_$3_konzept_bearbeiten.pkl"
    "annehmen_nach_vervollstaendigung=A_Accepted|complete=$P/models/psp_model_durchlaufzeit_$4_nachv_annehmen.pkl"
    "annehmen_nach_vervollstaendigung=W_Complete application|resume=$P/models/psp_model_durchlaufzeit_$4_nachv_weiter.pkl"
    "angebot_nachlegen=O_Create Offer|complete=$P/models/psp_model_durchlaufzeit_$4_nachlegen_weiteres.pkl"
    "angebot_nachlegen=O_Sent (mail and online)|complete=$P/models/psp_model_durchlaufzeit_$4_nachlegen_senden.pkl"
    "handle_leads=W_Handle leads|start=$P/models/psp_model_durchlaufzeit_$4_leads_rueckruf.pkl"
    "handle_leads=W_Handle leads|withdraw=$P/models/psp_model_durchlaufzeit_$4_leads_zurueck.pkl")
  CUDA_VISIBLE_DEVICES="" .venv/bin/python ../prescriptive/policies/ev_policy/option_learner_policy.py "$1" "${PAARE[@]}" >> $SD/fortschritt.log 2>&1 || return 1
  CUDA_VISIBLE_DEVICES="" .venv/bin/python - "$1" "$2" <<'PYEOF' >> $SD/fortschritt.log 2>&1
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
}

# 1. konzept-Netze
note "Beginn (frei $(frei) GB, Trainings $(trainings))"
netz annehmen_bei_konzept "A_Accepted|complete" konzept_annehmen 5 0 & PA=$!
sleep 600  # versetzt starten: das Laden braucht viel Arbeitsspeicher
netz annehmen_bei_konzept "W_Complete application|start" konzept_bearbeiten 5 0 & PB=$!
wait $PA; A=$?; wait $PB; B=$?
[ $A = 0 ] && [ $B = 0 ] || { note "Abbruch im Training der konzept-Netze ($A / $B)"; exit 1; }

# 2. Prüfung an konzept
huelle "$HULL_ZW" "$HULL_ZW_CRN" k3x4sd k3x4ol || { note "Abbruch: Zwischenhülle"; exit 1; }
note "Zwischenhülle gebaut, Prüfung an konzept beginnt"
DIAG_HUELLE=$HULL_ZW_CRN DIAG_NAME=konzept_effekt_gezogen_sd .venv/bin/python -u $K/diagnose/konzept_effekt_gezogen.py 120 30 > $SD/konzept_effekt.log 2>&1 || { note "Prüfung konzept_effekt FEHLER"; exit 1; }
DIAG_HUELLE=$HULL_ZW_CRN DIAG_NAME=sonde_betrag_sd .venv/bin/python -u $K/diagnose/sonde_betrag.py 400 5 > $SD/sonde.log 2>&1 || note "Sonde FEHLER (weiter)"
TOR=$(.venv/bin/python - $K/diagnose/konzept_effekt_gezogen_sd.csv <<'PYEOF'
import sys
import pandas as pd
d = pd.read_csv(sys.argv[1])
g = d[d.quelle == "gezogen"].set_index(["option", "gruppe"])
doppelt = g.loc[("W_Complete application|start", "< 11k"), "a_complete_doppelt"] - g.loc[("W_Complete application|start", "≥ 11k"), "a_complete_doppelt"]
validate = g.loc[("A_Accepted", "≥ 11k"), "validate_fehlt"] - g.loc[("A_Accepted", "< 11k"), "validate_fehlt"]
print(f"{'BESTANDEN' if max(doppelt, validate) >= 0.10 else 'NICHT'} doppelt {100 * doppelt:+.1f} / validate {100 * validate:+.1f} Punkte (Wahrheit +30 / +41, k3x4ol +2 / +1)")
PYEOF
)
note "Tor: $TOR"
case "$TOR" in BESTANDEN*) ;; *) note "Ende: Tor nicht bestanden"; exit 0 ;; esac

# 3. übrige Netze, ein Strang
while IFS='|' read -r PUNKT AKT LC KURZ; do
  [ -z "$PUNKT" ] && continue
  netz "$PUNKT" "$AKT|$LC" "$KURZ" 5 1 || { note "Abbruch im Training"; exit 1; }
done <<'LISTE'
annehmen_nach_vervollstaendigung|A_Accepted|complete|nachv_annehmen
annehmen_nach_vervollstaendigung|W_Complete application|resume|nachv_weiter
angebot_nachlegen|O_Create Offer|complete|nachlegen_weiteres
angebot_nachlegen|O_Sent (mail and online)|complete|nachlegen_senden
handle_leads|W_Handle leads|start|leads_rueckruf
handle_leads|W_Handle leads|withdraw|leads_zurueck
LISTE

# 4. Hülle und Online
huelle "$HULL" "$HULL_CRN" k3x4sd k3x4sd || { note "Abbruch: Hülle"; exit 1; }
note "Hülle k3x4sd gebaut; Online wartet auf das Ende der Simulator-Basis-Pipeline"
until grep -q "Alles fertig\|Abbruch" $SB/fortschritt.log; do sleep 300; done
note "Online konformanz_sd beginnt (12 Seeds, 300 Ziehungen, 8 Prozesse)"
.venv/bin/python -u $K/online_konformanz.py konformanz_sd 0-11 8 > $K/online_konformanz_sd.log 2>&1
note "Online beendet (Code $?): $(grep -E '^FERTIG' $K/online_konformanz_sd.log | head -1)"
note "Alles fertig"
