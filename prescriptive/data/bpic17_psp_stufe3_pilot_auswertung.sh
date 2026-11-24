#!/bin/bash
# Auswertung der Stufe-3-Pilotläufe (2026-11-11): Suffixgüte, Zerlegung Pfad/Zeit und Rauschen an den Wahlpunkten.
# Nur untereinander vergleichbar (3.000 Trainingsfälle, 6 Epochen). *_od: Kopie mit params option_as "decoder" (nachträglich gesetzt).
# Aufruf: tmux new-session -d -s pilot_eval -c process-simulation-engine "bash ../prescriptive/data/bpic17_psp_stufe3_pilot_auswertung.sh"
# Runde 1 (Standard): ohne Umgebung. Weitere Runden: RUNDE=<name> ALLE="kurz=datei.pkl ..." (SG: Modelle für die Suffixgüte, Standard ALLE)
P=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive
R=$P/scripts/results_100k
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte.yaml CUDA_VISIBLE_DEVICES=''
M=psp_model_durchlaufzeit_pilot
RUNDE=${RUNDE:-pilot}
if [ -z "$ALLE" ]; then
    ALLE="k_tf0=${M}_klassen_tf0.pkl k_tf1=${M}_klassen_tf1.pkl k_tf1_od=${M}_klassen_tf1_od.pkl t_tf1_od=${M}_token_tf1.pkl n_tf1=${M}_normal_tf1.pkl n_tf1_od=${M}_normal_tf1_od.pkl"
    SG="k_tf0=${M}_klassen_tf0.pkl k_tf1=${M}_klassen_tf1.pkl t_tf1=${M}_token_tf1.pkl n_tf1=${M}_normal_tf1.pkl"
fi
SG=${SG:-$ALLE}

.venv/bin/python -u ../prescriptive/scripts/psp_suffixguete.py --psp $SG --prefixes 1000 --workers 4 --threads 2 --out $R/${RUNDE}_suffixguete 2>&1 \
    | grep --line-buffered -v -E "$FILTER" > $P/data/PSP_${RUNDE}_suffixguete.log &

.venv/bin/python -u ../prescriptive/scripts/psp_zerlegung.py --psp $ALLE --workers 8 --threads 1 --out $R/${RUNDE}_zerlegung 2>&1 \
    | grep --line-buffered -v -E "$FILTER" > $P/data/PSP_${RUNDE}_zerlegung.log &

for e in $ALLE; do
    tag=${e%%=*}; model=${e#*=}
    .venv/bin/python -u ../prescriptive/scripts/psp_rauschen.py --model $P/models/$model --workers 4 --threads 2 --out $R/psp_rauschen_${RUNDE}_$tag 2>&1 \
        | grep --line-buffered -v -E "$FILTER" > $P/data/PSP_${RUNDE}_rauschen_$tag.log
done
wait
echo "######## Auswertung fertig $(date +%H:%M)" >> $P/data/PSP_${RUNDE}_zerlegung.log
