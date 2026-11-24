#!/bin/bash
# Online-Vergleich der kausalen Schätzer nach Bozorgi et al. (2023) mit dem PSP (2026-11-12): S-Learner, X-Learner und
# Causal Forest (estimators/cate_model.py, policies/ev_policy/cate_policy.py; Modelle aus scripts/train_cate_real.py --tag
# real_alle, alle Trainingsfälle des echten Logs), Schwelle 0, ohne Einschränkung. Dazu als Gegenprobe im selben Lauf die
# Regel und XGBoost (echter Log, 10.000 Fälle). Neue Punkte, alte Zuteilung (Katalog-Kopie data/catalog_neue_punkte.yaml),
# gleiche Seeds und Fallzahlen wie neue_punkte_online und der Test der PSP-Variante (c) (data/bpic17_psp_stufe2c_online.sh),
# also gepaart vergleichbar. WORKERS aus der Umgebung (Standard 6). Log: data/CATE_online.log
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
LOG=$P/data/CATE_online.log
OUT=${OUT:-neue_punkte_online_cate}
TAG=${TAG:-real_alle}
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte.yaml
cd $ROOT/process-simulation-engine
echo "######## Online $OUT ($(date +%H:%M:%S))" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/main.py --delta 0.5 --rct-cases 10000 --rct-runs 10 --steps online \
    --online-policies regel xgb s_learner x_learner causal_forest --xgb-model "$P/models/xgb_model_durchlaufzeit_xgb_real_10k_neuepunkte.pkl" \
    --cate-model "s_learner=$P/models/cate_model_s_durchlaufzeit_$TAG.pkl" "x_learner=$P/models/cate_model_x_durchlaufzeit_$TAG.pkl" \
                 "causal_forest=$P/models/cate_model_cf_durchlaufzeit_$TAG.pkl" $EXTRA \
    --online-seeds 10 --online-cases 3000 --online-warmup 700 --online-cooldown 1300 --online-workers ${WORKERS:-6} --out "$P/scripts/results_100k/$OUT" 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || echo "######## Abbruch online $OUT ($(date +%H:%M:%S))" | tee -a "$LOG"
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
