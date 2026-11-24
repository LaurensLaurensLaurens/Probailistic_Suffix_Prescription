#!/bin/bash
# Stufe-3-Pilotmodelle mit der Option im Prefix und vom Decoder gezogenem Abstand (params option_as "encoder", Kopien *_oe,
# nachträglich gesetzt; 2026-11-11): Rauschen/Richtung an den Wahlpunkten und Zerlegung Pfad/Zeit. Ohne neues Training.
# Aufruf: tmux new-session -d -s pilot_oe -c process-simulation-engine "TAGS='token_tf0 ...' bash ../prescriptive/data/bpic17_psp_stufe3_pilot_oe.sh"
P=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive
R=$P/scripts/results_100k
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte.yaml CUDA_VISIBLE_DEVICES=''
TAGS=${TAGS:-"token_tf0 token_tf05 token_tf1 klassen_tf0 klassen_tf1 normal_tf1"}
RUNDE=${RUNDE:-pilot_oe}
ALLE=""
for t in $TAGS; do ALLE="$ALLE ${t}_oe=psp_model_durchlaufzeit_pilot_${t}_oe.pkl"; done
.venv/bin/python -u ../prescriptive/scripts/psp_zerlegung.py --psp $ALLE --workers 8 --threads 1 --out $R/${RUNDE}_zerlegung 2>&1 \
    | grep --line-buffered -v -E "$FILTER" > $P/data/PSP_${RUNDE}_zerlegung.log &
k=0
for t in $TAGS; do
    .venv/bin/python -u ../prescriptive/scripts/psp_rauschen.py --model $P/models/psp_model_durchlaufzeit_pilot_${t}_oe.pkl --workers 4 --threads 2 \
        --out $R/psp_rauschen_pilot_${t}_oe 2>&1 | grep --line-buffered -v -E "$FILTER" > $P/data/PSP_pilot_rauschen_${t}_oe.log &
    k=$((k + 1)); [ $((k % 2)) -eq 0 ] && wait -n 2>/dev/null
done
wait
echo "######## Auswertung fertig $(date +%H:%M)" >> $P/data/PSP_${RUNDE}_zerlegung.log
