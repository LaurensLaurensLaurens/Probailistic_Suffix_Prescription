#!/bin/bash
# Online-Auswertung (normale psp-Policy) mit dem PSP vom echten Log (scripts/train_psp_real.py, models/psp_model_durchlaufzeit_real_w3_10k.pkl):
# Bank, Zufall und psp an allen 12 Wahlpunkten (8 Mitarbeiter, 4 Control Flow), Fenster und Seeds wie PSP_lr1e4_online
# (psp_nacht_online.sh: 3000 Ankünfte, Warmlauf 700, Auslauf 1300, 10 Seeds), damit vergleichbar. Das Modell kommt über
# --psp-model; models/psp_model*.pkl und der Katalog werden nicht angefasst (laufen parallel zu bpic17_psp_w3_10k.sh, das
# den Katalog auf die 12 Punkte gestellt hat; der Katalog wird hier nur geprüft).
# Weitere Argumente gehen an main.py, z. B. --online-points "bpic17=handle_leads" (Ablation: psp nur an diesen Punkten, sonst Bank).
# Zum Probelauf: TAG, ONLINE, WARM, COOL, SEEDS aus der Umgebung (z. B. TAG=PSP_real_online_test ONLINE=60 WARM=10 COOL=10 SEEDS=1).
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
TAG=${TAG:-PSP_real_w3_10k_online}; ONLINE=${ONLINE:-3000}; WARM=${WARM:-700}; COOL=${COOL:-1300}; SEEDS=${SEEDS:-10}; WORKERS=${WORKERS:-12}
MODEL=$P/models/psp_model_durchlaufzeit_real_w3_10k.pkl
OUT=$P/scripts/results_100k/$TAG
LOG=$P/data/$TAG.log
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory|super\(\).__init__'
[ -e "$MODEL" ] || { echo "######## $MODEL fehlt, Abbruch" | tee -a "$LOG"; exit 1; }
# Die 12 Wahlpunkte müssen im Katalog stehen (der Simulator liest sie beim Start)
grep -q '^    mitarbeiterwahl: \[.*W_Validate application (resume)' "$P/catalog.yaml" && grep -q '^    control_flow: \[.*neues_angebot_beim_nachfassen' "$P/catalog.yaml" \
    || { echo "######## Katalog hat nicht die 12 Wahlpunkte, Abbruch" | tee -a "$LOG"; exit 1; }
cd $ROOT/process-simulation-engine
echo "######## PSP vom echten Log online: $ONLINE Ankünfte, Warmlauf $WARM, Auslauf $COOL, $SEEDS Seeds ($(date +%H:%M:%S))" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/main.py --delta 0.5 --rct-cases 10000 --rct-runs 10 \
    --steps online --online-policies bank random psp --psp-model "$MODEL" --online-seeds $SEEDS --online-cases $ONLINE \
    --online-warmup $WARM --online-cooldown $COOL --online-workers $WORKERS --out "$OUT" "$@" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch online ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
