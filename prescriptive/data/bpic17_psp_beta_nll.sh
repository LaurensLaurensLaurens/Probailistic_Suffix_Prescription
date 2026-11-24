#!/bin/bash
# β-NLL (2026-11-18, Wunsch des Nutzers): fertiges ZZU-TF1 (Zähler + Uhren + Zeit je Event + Teacher Forcing 1) laden (--init-from)
# und mit β-NLL der Zeitköpfe nachtrainieren (--beta-nll 0.5; Seitzer et al. 2022: jeder Term des attenuierten Verlusts mal
# stop_grad(σ^(2β))). Anlass: Zeitkopf-Sonde (zeitkopf_sonde_20261118): an echten A_Cancelled-Schritten sagt der Kopf ~4 Tage bei
# σ ~23 Tagen voraus (echt ~25), obwohl eine lineare Sonde auf demselben Zustand ~24 Tage trifft – der Mittelwert der langen
# Wartezeiten wird wegen 1/σ² kaum gelernt. Sonst alle Einstellungen wie ZZU-TF1, LR 1e-4, 5 Epochen.
# Loss Attenuation, Dropout-Masken und gezogene Suffixe (Event + Zeit je Schritt) bleiben unverändert.
# Begleiter (je Epoche Checkpoint + kleine Sonde): scripts/results_100k/beta_nll_20261118/begleiter.py
# Danach: Prüfreihe (Label "β-NLL 0,5"), Storno-Zeitpunkt, volle Sonde, Suffixgüte.
# Rauchtest: CASES=300 EPOCHS=1 TAG=<anderer Tag> (ohne Warmstart, kleineres Vokabular).
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_100k/beta_nll_20261118
BASE=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_uhren_tf1.pkl
TAG=${TAG:-real_alle_s60_normal_zaehler_uhren_tf1_bnll05}
CASES=${CASES:-0}
EPOCHS=${EPOCHS:-5}
BETA=${BETA:-0.5}
LABEL="β-NLL 0,5"
MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
LOG=$P/data/PSP_$TAG.log
FILTER='FutureWarning|applied = df|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
mkdir -p $OUT/logs
note() { echo "$(date '+%Y-%m-%d %H:%M') $1" >> $OUT/fortschritt.log; }
INIT=""
[ "$CASES" = "0" ] && INIT="--init-from $BASE"
note "Nachtraining mit β-NLL $BETA beginnt ($EPOCHS Epochen, Fälle $CASES, Tag $TAG)"
[ -e "$MODEL" ] || .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --cases $CASES --epochs $EPOCHS --learning-rate 1e-4 \
    --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps \
    --counters --clocks --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training --beta-nll $BETA $INIT 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ -e "$MODEL" ] || { note "Abbruch im Training"; exit 1; }
[ "$CASES" = "0" ] || { note "Rauchtest fertig ($MODEL)"; exit 0; }
note "Training fertig, Prüfreihe beginnt"
.venv/bin/python -u $OUT/pruefreihe.py "bnll05=$LABEL=$MODEL" > $OUT/pruefreihe.log 2>&1
note "Prüfreihe fertig, Storno-Messung"
CUDA_VISIBLE_DEVICES="" .venv/bin/python $P/scripts/results_100k/zaehler_uhren_tf1_20261117/storno_abstand.py 300 20 "$LABEL=$MODEL" > $OUT/storno.log 2>&1
note "Storno-Messung fertig, volle Sonde"
SONDE_OUT=$OUT/sonde_final .venv/bin/python -u $P/scripts/results_100k/zeitkopf_sonde_20261118/sonde.py 1500 200 20 "$LABEL=$MODEL" > $OUT/logs/sonde_final.log 2>&1
note "Sonde fertig, Suffixgüte"
CUDA_VISIBLE_DEVICES="" nice -n 10 .venv/bin/python -u ../prescriptive/scripts/psp_suffixguete.py --psp "$LABEL=$MODEL" --prefixes 1000 --workers 4 --threads 2 --out $OUT/suffixguete > $OUT/logs/suffixguete.log 2>&1
note "Alles fertig"
