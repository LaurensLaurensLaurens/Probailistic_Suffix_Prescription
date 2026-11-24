#!/bin/bash
# Zeitkopf je Event (Steigung) mit β-NLL β = 1 auf den T-Learner-Armen (2026-11-19, Nutzer: „okay dann mach das“; Online-Test der
# Versatz-Variante dafür gestrichen). Anlass: an angebot_nachlegen schätzt die Steigungsvariante den Nachteil eines Zusatzangebots auf
# 0,37 d (Wahrheit 1,55), es fehlt ~1 d Kundenantwort nach zwei Angeboten (nachlegen_wartezeiten.md). Die Wartezeit steckt in Abständen
# mit gemischter Länge (W_Call after offers|ate_abort: 28 % < 1 h, 67 % > 1 d) – wie bei A_Cancelled tragen dort die sicheren kurzen
# Fälle das 1/σ²-Gewicht. β-NLL (Seitzer et al. 2022) gewichtet jeden Term mit stop_grad(σ^(2β)): mit β = 1 lernt der Mittelwert mit
# ungewichtetem Fehler, σ wird weiter mit dem attenuierten Verlust gelernt. Sonst genau wie bpic17_psp_zeitkopf_je_event.sh slope
# (Ein-Faktor-Vergleich): vom T-Learner-Arm, Steigung je Event, nur Zeitköpfe, LR 1e-3 / Versätze 3e-2, Aktion 30 / Referenz 12 Epochen.
# Danach: Hülle, Sonde, Storno-Messung, Prüfreihe, Wartezeiten nach dem Nachlegen, SCHRANKE: Online nur, wenn der geschätzte
# Vorteil der Regel-Option an nachlegen (1. Besuch, punkte_alle) ≥ 1,0 d (Steigung 0,37, Wahrheit 2,11 RCT / 1,55 Bank-Fortsetzung);
# Online Aufbau F 12 Seeds (Arm tlkb_fest300, wartet bis er in f_zwoelf_seeds.py eingetragen ist), Auswertung, Suffixgüte.
# Aufruf (tmux): bash bpic17_psp_zeitkopf_bnll.sh
NAME=steigung_bnll1; LABEL="T-Learner Kopf je Event (Steigung, β-NLL 1)"; ARMKEY=tlkb
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
R=$P/scripts/results_100k
V=$R/modellvergleich_20261116
OUT=$R/zeitkopf_je_event_20261118
HULL=$P/models/psp_model_durchlaufzeit_t_learner_kopf_$NAME.pkl
FILTER='FutureWarning|applied = df|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
mkdir -p $OUT/logs
note() { echo "$(date '+%Y-%m-%d %H:%M') $NAME: $1" >> $OUT/fortschritt.log; }
for ARM in aktion referenz; do
  SOURCE=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_$ARM.pkl
  TAG=real_alle_s60_normal_zaehler_zeit_tf1_t_${ARM}_kopf_$NAME
  LOG=$P/data/PSP_$TAG.log
  MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
  EPOCHS=$([ $ARM = aktion ] && echo 30 || echo 12)
  note "Training $ARM beginnt ($EPOCHS Epochen, nur Zeitköpfe, β-NLL 1)"
  [ -e "$MODEL" ] || .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --cases 0 --epochs $EPOCHS \
      --learning-rate 1e-3 --offset-learning-rate 3e-2 --time-event-head slope --head-only --beta-nll 1.0 \
      --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps \
      --counters --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training \
      --only-points --only-points-validation --arm $ARM --init-from "$SOURCE" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
  [ -e "$MODEL" ] || { note "Abbruch im Training $ARM"; exit 1; }
  note "Training $ARM fertig"
done
A=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_aktion_kopf_$NAME.pkl
B=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_referenz_kopf_$NAME.pkl
CUDA_VISIBLE_DEVICES="" .venv/bin/python ../prescriptive/policies/ev_policy/t_learner_policy.py "$A" "$B" "$HULL" >> $OUT/fortschritt.log 2>&1 \
    || { note "Abbruch: Hülle"; exit 1; }
note "Hülle gebaut, Sonde der Arme"
SONDE_OUT=$OUT/sonde_$NAME CUDA_VISIBLE_DEVICES="" .venv/bin/python -u $R/zeitkopf_sonde_20261118/sonde.py 1500 200 20 \
    "Aktion ($NAME)=$A" "Referenz ($NAME)=$B" > $OUT/logs/sonde_$NAME.log 2>&1
.venv/bin/python $OUT/sonde_zeile.py $OUT/sonde_$NAME 2>&1 | while read -r line; do note "$line"; done
note "Storno-Messung"
flock $OUT/auswertung.lock env CUDA_VISIBLE_DEVICES="" .venv/bin/python $R/zaehler_uhren_tf1_20261117/storno_abstand.py 300 20 \
    "T-Learner Aktion ($NAME)=$A" "T-Learner Referenz ($NAME)=$B" > $OUT/storno_$NAME.log 2>&1
note "Storno-Messung fertig, Prüfreihe"
flock $OUT/auswertung.lock .venv/bin/python -u $OUT/pruefreihe.py "t_kopf_$NAME=$LABEL=$HULL" > $OUT/logs/pruefreihe_$NAME.log 2>&1
.venv/bin/python $V/nachlegen_wartezeiten.py nachlegen_erklaerung_t_kopf_$NAME > $OUT/logs/wartezeiten_$NAME.log 2>&1
GATE=$(.venv/bin/python - "$V/punkte_alle_t_kopf_$NAME/vergleich.csv" "$V/nachlegen_wartezeiten.csv" "$LABEL" <<'EOF'
import sys
import pandas as pd
points, waits, label = sys.argv[1:4]
v = pd.read_csv(points)
v = v[(v.Gruppe == "nachlegen 1. Besuch") & (v.Quelle != "Simulator")].iloc[0]
w = pd.read_csv(waits).set_index(["quelle", "opt"])
d = w.xs("extra", level="opt") - w.xs("senden", level="opt")
line = d.loc[label] if label in d.index else None
extra = "" if line is None else (f"; extra − senden: bis letztes O_Sent {line['bis letztes O_Sent']:+.2f} d, Kundenantwort "
                                 f"{line['letztes O_Sent bis Validierung']:+.2f}, Nachfass-Start → Validierung {line['Nachfass-Start bis Validierung']:+.2f} "
                                 "(echt +0,83 / +0,64 / +1,45; Steigung +0,24 / −0,43 / −0,25)")
ok = v["Vorteil Regel-Option"] >= 1.0
print(f"{'ja' if ok else 'nein'}|Prüfreihe nachlegen 1. Besuch: Vorteil Regel-Option {v['Vorteil Regel-Option']:.2f} d "
      f"(Steigung 0,37, Wahrheit 2,11 / 1,55), richtig {v['Anteil Zustände richtig (100 Ziehungen)']:.0%}, MC-SE {v['MC-SE je Entscheidung']:.2f}{extra}")
EOF
)
note "${GATE#*|}"
if [ "${GATE%%|*}" != ja ]; then
  note "Schranke (≥ 1,0 d an nachlegen) nicht erreicht: kein Online-Test"
else
  note "Schranke erreicht, Online-Test 12 Seeds (wartet auf Eintrag des Arms und auf den Online-Slot)"
  until grep -q "\"$ARMKEY\"" $V/f_zwoelf_seeds.py; do sleep 120; done
  flock $OUT/online.lock .venv/bin/python -u $V/f_zwoelf_seeds.py --arms ${ARMKEY}_fest300 --seeds 0-11 --workers 6 \
      > $OUT/logs/online_$NAME.log 2>&1
  flock $OUT/auswertung.lock .venv/bin/python -u $V/f_zwoelf_auswerten.py > $OUT/logs/online_auswertung_$NAME.log 2>&1
  note "Online-Test fertig (modellvergleich_20261116/f_zwoelf_auswertung.md)"
fi
note "Suffixgüte beginnt"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/psp_suffixguete.py --prefixes 1000 \
    --psp "T-Learner Aktion Kopf je Event (Steigung, β-NLL 1)=$A" "T-Learner Referenz Kopf je Event (Steigung, β-NLL 1)=$B" \
    --out $OUT/suffixguete_$NAME > $OUT/logs/suffixguete_$NAME.log 2>&1
note "Suffixgüte fertig (Code $?)"
note "Alles fertig"
