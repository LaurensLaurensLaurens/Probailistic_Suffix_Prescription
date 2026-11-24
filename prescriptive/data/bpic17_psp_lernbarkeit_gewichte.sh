#!/bin/bash
# Lernbarkeitstest mit festen Aufgabengewichten (2026-11-24, Nutzer: „Ja mach a“) nach der Gradienten-Zerlegung
# (statik_decoder_20261124/gradienten.md: der Zeitverlust case_elapsed_time bestimmt die Gradienten am Betrag und zieht gegen den
# Effekt-Übergang A_Complete → A_Complete). Gleicher Aufbau wie sn (bpic17_psp_lernbarkeit_statik.sh): Netz „vervollständigen“ an konzept,
# warm von k3x4ol_konzept_bearbeiten, nur die Statik-Abbildung (mit 0 begonnen, LR 1e-2) lernt, 2 Epochen; dazu psp_model task_weights:
#   sw10  Event-Verlust 10, Zeitverluste 1 (--task-weights 10,1)
#   sw0   nur der Event-Verlust (--task-weights 1,0; obere Schranke, die Zeitköpfe sind ohnehin eingefroren)
# Danach je eine Hülle (k3x4ol mit dem neuen Netz, schlank, keyed_noise) und die Prüfung:
#   statik_decoder_20261124/uebergang_lehrer.py (k3x4ol, sn, sw10, sw0) → gewichte_uebergang.md
#   kuenstliche_effekte_20261120/diagnose/konzept_effekt_gezogen.py je Hülle → konzept_effekt_gezogen_sw10/_sw0.md
# Trainings starten erst, wenn ≥ 20 GB frei sind und < 3 Trainings laufen (Simulator-Basis teil3 trainiert parallel 2 Stränge).
# Aufruf (tmux): bash bpic17_psp_lernbarkeit_gewichte.sh
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
K=$P/scripts/results_100k/kuenstliche_effekte_20261120
SD=$P/scripts/results_100k/statik_decoder_20261124
LOGP=$K/log_effekte_v3_x4/log_raw.pkl
SRC=$P/models/psp_model_durchlaufzeit_k3x4ol_konzept_bearbeiten.pkl
OL_CRN=$P/models/psp_model_durchlaufzeit_option_learner_k3x4_schlank_crn.pkl
AWKFILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super[(][)][.]__init__'
COMMON="--cases 0 --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps --counters --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training --only-points --only-points-validation --load-chunks 16 --log-pickle $LOGP --static-in-decoder --static-only --static-learning-rate 1e-2"
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
echo 1000 > /proc/self/oom_score_adj 2>/dev/null  # bei Speichermangel zuerst diese Läufe
note() { echo "$(date '+%Y-%m-%d %H:%M') $1" >> $SD/gewichte.log; }
frei() { free -g | awk '/Mem:/ {print $7}'; }
trainings() { pgrep -fc "scripts/train_psp_real.py" ; }

netz() {  # $1 Kürzel (sw10 / sw0), $2 Lernrate, $3 weitere Schalter
  local TAG=k3x4$1_konzept_bearbeiten MODEL=$P/models/psp_model_durchlaufzeit_k3x4$1_konzept_bearbeiten.pkl
  for VERSUCH in 1 2 3; do
    [ -e "$MODEL" ] && break
    while [ "$(frei)" -lt 20 ] || [ "$(trainings)" -ge 3 ]; do sleep 120; done
    rm -rf "$P/models/psp_training_$TAG"
    note "$TAG beginnt (2 Epochen, Versuch $VERSUCH, frei $(frei) GB, Trainings $(trainings))"
    .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --epochs 2 --learning-rate $2 $COMMON $3 \
        --point annehmen_bei_konzept --option "W_Complete application|start" --init-from "$SRC" --vocab-from "$SRC" 2>&1 \
        | awk -v f="$AWKFILTER" '/Error/ || $0 !~ f { print; fflush() }' >> "$P/data/PSP_$TAG.log"
    [ -e "$MODEL" ] && break
    if [ $VERSUCH -lt 3 ] && tail -50 "$P/data/PSP_$TAG.log" | grep -q MemoryError; then note "$TAG: MemoryError, neuer Versuch in 10 Minuten"; sleep 600; else break; fi
  done
  [ -e "$MODEL" ] || { note "Abbruch: $TAG"; return 1; }
  note "$TAG fertig"
}

huelle() {  # $1 Kürzel: k3x4ol mit dem neuen vervollständigen-Netz, schlank, keyed_noise
  local H=$P/models/psp_model_durchlaufzeit_option_learner_k3x4$1_konzept.pkl
  local PAARE=(
    "annehmen_bei_konzept=A_Accepted|complete=$P/models/psp_model_durchlaufzeit_k3x4ol_konzept_annehmen.pkl"
    "annehmen_bei_konzept=W_Complete application|start=$P/models/psp_model_durchlaufzeit_k3x4$1_konzept_bearbeiten.pkl"
    "annehmen_nach_vervollstaendigung=A_Accepted|complete=$P/models/psp_model_durchlaufzeit_k3x4ol_nachv_annehmen.pkl"
    "annehmen_nach_vervollstaendigung=W_Complete application|resume=$P/models/psp_model_durchlaufzeit_k3x4ol_nachv_weiter.pkl"
    "angebot_nachlegen=O_Create Offer|complete=$P/models/psp_model_durchlaufzeit_k3x4ol_nachlegen_weiteres.pkl"
    "angebot_nachlegen=O_Sent (mail and online)|complete=$P/models/psp_model_durchlaufzeit_k3x4ol_nachlegen_senden.pkl"
    "handle_leads=W_Handle leads|start=$P/models/psp_model_durchlaufzeit_k3x4ol_leads_rueckruf.pkl"
    "handle_leads=W_Handle leads|withdraw=$P/models/psp_model_durchlaufzeit_k3x4ol_leads_zurueck.pkl")
  CUDA_VISIBLE_DEVICES="" .venv/bin/python ../prescriptive/policies/ev_policy/option_learner_policy.py "$H" "${PAARE[@]}" >> $SD/gewichte.log 2>&1 || return 1
  CUDA_VISIBLE_DEVICES="" .venv/bin/python - "$H" "${H%.pkl}_schlank_crn.pkl" <<'PYEOF' >> $SD/gewichte.log 2>&1
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
w = hull.netze[("annehmen_bei_konzept", "W_Complete application|start")].net.decoder.input_proj.static_projection.weight.detach()
print(f"schlanke Kopie mit keyed_noise: {sys.argv[2]}; Statik-Abbildung Norm {w.norm():.3f}, Spalten "
      f"{[round(x, 3) for x in w.norm(dim=0).tolist()]} (letzte = Betrag)")
PYEOF
}

note "Beginn (frei $(frei) GB, Trainings $(trainings))"
netz sw10 1e-4 "--task-weights 10,1" || exit 1
huelle sw10 || { note "Abbruch: Hülle sw10"; exit 1; }
netz sw0 1e-4 "--task-weights 1,0" || exit 1
huelle sw0 || { note "Abbruch: Hülle sw0"; exit 1; }
note "Hüllen gebaut, Prüfung beginnt"
H=$P/models/psp_model_durchlaufzeit_option_learner_k3x4
UL_HUELLEN="k3x4ol=$OL_CRN;k3x4sn=${H}sn_konzept_schlank_crn.pkl;k3x4sw10=${H}sw10_konzept_schlank_crn.pkl;k3x4sw0=${H}sw0_konzept_schlank_crn.pkl" \
    UL_NAME=gewichte_uebergang .venv/bin/python -u $SD/uebergang_lehrer.py 120 10 > $SD/gewichte_uebergang.log 2>&1 \
    && note "Teacher Forcing fertig" || note "Teacher Forcing FEHLER"
for V in sw10 sw0; do
  DIAG_HUELLE=${H}${V}_konzept_schlank_crn.pkl DIAG_NAME=konzept_effekt_gezogen_$V \
      .venv/bin/python -u $K/diagnose/konzept_effekt_gezogen.py 120 30 > $SD/konzept_effekt_$V.log 2>&1 && note "gezogen $V fertig" || note "gezogen $V FEHLER"
done
note "Alles fertig"
