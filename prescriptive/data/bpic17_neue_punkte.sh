#!/bin/bash
# Die neuen Interventionspunkte (Screening 2026-11-06, Outcome Durchlaufzeit): angebot_nachlegen (statt channel Hauptintervention),
# annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads; keine Mitarbeiterpunkte (Katalog-Kopie
# data/catalog_neue_punkte.yaml über PSP_CATALOG). 1. XGBoost auf dem echten Log für diese Punkte (scripts/train_xgb_real.py).
# 2. Online: Bank, Zufall, PSP vom echten Log (models/psp_model_durchlaufzeit_real_w3_10k_neuepunkte.pkl: dasselbe Modell, der
# Punkt annehmen_bei_konzept eingetragen), XGBoost vom echten Log und die Regel des Screenings (policies/rule_policy.py);
# Fenster und Seeds wie PSP_real_w3_10k_online. Ergebnisse in scripts/results_100k/neue_punkte_online, Log data/neue_punkte.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
LOG=$P/data/neue_punkte.log
OUT=$P/scripts/results_100k/neue_punkte_online
PSP=$P/models/psp_model_durchlaufzeit_real_w3_10k_neuepunkte.pkl
XGB=$P/models/xgb_model_durchlaufzeit_xgb_real_10k_neuepunkte.pkl
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte.yaml
cd $ROOT/process-simulation-engine
echo "######## XGBoost auf dem echten Log, neue Punkte ($(date +%H:%M:%S))" | tee -a "$LOG"
[ -e "$XGB" ] || CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/train_xgb_real.py --tag xgb_real_10k_neuepunkte --cases 10000 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ -e "$XGB" ] || { echo "######## Abbruch im Training ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Online ($(date +%H:%M:%S))" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/main.py --delta 0.5 --rct-cases 10000 --rct-runs 10 --steps online \
    --online-policies bank random psp xgb regel --psp-model "$PSP" --xgb-model "$XGB" --online-seeds 10 --online-cases 3000 \
    --online-warmup 700 --online-cooldown 1300 --online-workers ${WORKERS:-12} --out "$OUT" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch online ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
