#!/bin/bash
# XGBoost auf dem echten Log (scripts/train_xgb_real.py, 10.000 Fälle wie das PSP vom echten Log) und Online-Auswertung im
# Simulator: Bank, Zufall, xgb; Fenster und Seeds wie PSP_real_w3_10k_online (3000 Ankünfte, Warmlauf 700, Auslauf 1300,
# 10 Seeds), also gepaart mit den PSP-Läufen.
# MODE=gesamt: 12 Punkte wie bpic17_psp_real_online.sh (channel+angebot_nachlegen und 3 weitere Control-Flow-Punkte, 8 Mitarbeiter)
# MODE=channel: wie bpic17_psp_real_online_channel.sh (am O_Created nur der Kanal, ohne angebot_nachlegen)
# Der Katalog des Laufs ist eine Kopie (data/catalog_<TAG>.yaml über PSP_CATALOG), prescriptive/catalog.yaml und
# models/xgb_model*.pkl bleiben unberührt. Es bleiben: models/xgb_model_durchlaufzeit_<TAG>.pkl,
# scripts/results_100k/<TAG>_online, data/<TAG>.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
MODE=${MODE:-gesamt}; CASES=${CASES:-10000}
case $MODE in
    gesamt) TAG=${TAG:-xgb_real_10k}; CONTROL='["handle_leads", "annehmen_nach_vervollstaendigung", "angebot_nachlegen", "neues_angebot_beim_nachfassen"]' ;;
    channel) TAG=${TAG:-xgb_real_10k_channel}; CONTROL='["handle_leads", "annehmen_nach_vervollstaendigung", "neues_angebot_beim_nachfassen"]' ;;
    *) echo "MODE gesamt oder channel"; exit 1 ;;
esac
ONLINE=${ONLINE:-3000}; WARM=${WARM:-700}; COOL=${COOL:-1300}; SEEDS=${SEEDS:-10}; WORKERS=${WORKERS:-6}
MODEL=$P/models/xgb_model_durchlaufzeit_$TAG.pkl
OUT=$P/scripts/results_100k/${TAG}_online
LOG=$P/data/$TAG.log
CATALOG=$P/data/catalog_$TAG.yaml
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory|super\(\).__init__'
python3 - "$P/catalog.yaml" "$CATALOG" "$CONTROL" <<'PY' || { echo "######## Katalog nicht erstellt, Abbruch" | tee -a "$LOG"; exit 1; }
import re, sys
source, target, control = sys.argv[1:4]
text = open(source, encoding="utf-8").read()
start = text.index("\nbpic17:\n") + len("\nbpic17:\n")
end = re.compile(r"^\S", re.M).search(text, start).start()
values = {"mitarbeiterwahl": '["Mitarbeiter: W_Call incomplete files", "Mitarbeiter: W_Complete application", "Mitarbeiter: W_Handle leads", '
                             '"Mitarbeiter: W_Validate application", "Mitarbeiter: W_Call after offers (resume)", "Mitarbeiter: W_Call incomplete files (resume)", '
                             '"Mitarbeiter: W_Complete application (resume)", "Mitarbeiter: W_Validate application (resume)"]',
          "control_flow": control}
part = text[start:end]
for key, value in values.items():
    part, n = re.subn(rf"^    {key}: \[.*\]$", lambda m: f"    {key}: {value}", part, flags=re.M)
    assert n == 1, key
open(target, "w", encoding="utf-8").write(text[:start] + part + text[end:])
PY
export PSP_CATALOG=$CATALOG
cd $ROOT/process-simulation-engine
echo "######## XGBoost auf dem echten Log, $MODE, $CASES Fälle ($(date +%H:%M:%S))" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/train_xgb_real.py --tag $TAG --cases $CASES 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch im Training ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Online: $ONLINE Ankünfte, Warmlauf $WARM, Auslauf $COOL, $SEEDS Seeds ($(date +%H:%M:%S))" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/main.py --delta 0.5 --rct-cases 10000 --rct-runs 10 \
    --steps online --online-policies bank random xgb --xgb-model "$MODEL" --online-seeds $SEEDS --online-cases $ONLINE \
    --online-warmup $WARM --online-cooldown $COOL --online-workers $WORKERS --out "$OUT" "$@" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch online ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
