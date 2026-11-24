#!/bin/bash
# T-Learner auf der β-NLL-Basis (2026-11-18, Nutzer: „weiter“). Wartet auf das Ende der β-NLL-Pipeline (bpic17_psp_beta_nll.sh),
# prüft die Schranke (results_100k/beta_nll_20261118/schranke.py: Kopf-Mittel an A_Cancelled ≥ 10 d, MAE alle Schritte ≤ 0,70 d) und
# baut dann den T-Learner mit demselben Rezept wie bpic17_psp_t_learner.sh, nur warm vom β-Modell und mit dessen Einstellungen
# (Uhren, β-NLL 0,5): Aktion 40 / Referenz 20 Epochen, LR 3e-5, nur Wahlpunkt-Proben des Arms. Danach Hülle, Prüfreihe
# (t_learner_20261117/pruefreihe.py, Label "T-Learner (β-NLL)"), Storno-Messung und Sonde beider Arme, Online Aufbau F 12 Seeds
# (Arm tlbnll_fest300: 300 Ziehungen, first_only), Auswertung f_zwoelf_auswerten.py. Nebenjob: oom_score_adj 1000.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
R=$P/scripts/results_100k
BASE=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_uhren_tf1_bnll05.pkl
HULL=$P/models/psp_model_durchlaufzeit_t_learner_bnll05.pkl
OUT=$R/beta_nll_20261118
TL=$R/t_learner_20261117
LABEL="T-Learner (β-NLL)"
FILTER='FutureWarning|applied = df|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
echo 1000 > /proc/self/oom_score_adj
note() { echo "$(date '+%Y-%m-%d %H:%M') T-Learner: $1" >> $OUT/fortschritt.log; }
note "wartet auf das Ende der β-NLL-Pipeline"
until grep -qE "Alles fertig|Abbruch im Training" $OUT/fortschritt.log || ! tmux has-session -t psp_bnll 2>/dev/null; do sleep 300; done
GATE=$(.venv/bin/python $OUT/schranke.py); STATUS=$?
note "$GATE"
[ $STATUS -eq 0 ] || { note "Halt nach der Schranke, kein T-Learner"; exit 0; }
for ARM in aktion referenz; do
  TAG=real_alle_s60_normal_zaehler_uhren_tf1_bnll05_t_$ARM
  LOG=$P/data/PSP_$TAG.log
  MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
  EPOCHS=$([ $ARM = aktion ] && echo 40 || echo 20)
  note "Training $ARM beginnt ($EPOCHS Epochen)"
  [ -e "$MODEL" ] || .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --cases 0 --epochs $EPOCHS --learning-rate 3e-5 \
      --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps \
      --counters --clocks --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training --beta-nll 0.5 \
      --only-points --only-points-validation --arm $ARM --init-from "$BASE" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
  [ -e "$MODEL" ] || { note "Abbruch im Training $ARM"; exit 1; }
  note "Training $ARM fertig"
done
A=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_uhren_tf1_bnll05_t_aktion.pkl
B=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_uhren_tf1_bnll05_t_referenz.pkl
CUDA_VISIBLE_DEVICES="" .venv/bin/python ../prescriptive/policies/ev_policy/t_learner_policy.py "$A" "$B" "$HULL" >> $OUT/fortschritt.log 2>&1 \
    || { note "Abbruch: Hülle"; exit 1; }
note "Hülle gebaut, Prüfreihe beginnt"
.venv/bin/python -u $TL/pruefreihe.py "t_bnll05=$LABEL=$HULL" > $OUT/pruefreihe_t_learner.log 2>&1
note "Prüfreihe fertig, Storno-Messung und Sonde der Arme"
CUDA_VISIBLE_DEVICES="" .venv/bin/python $R/zaehler_uhren_tf1_20261117/storno_abstand.py 300 20 "T-Learner β Aktion=$A" "T-Learner β Referenz=$B" \
    > $OUT/storno_t_learner.log 2>&1
SONDE_OUT=$OUT/sonde_t_learner .venv/bin/python -u $R/zeitkopf_sonde_20261118/sonde.py 1500 200 20 "T-Learner β Aktion=$A" "T-Learner β Referenz=$B" \
    > $OUT/logs/sonde_t_learner.log 2>&1
note "Online-Test 12 Seeds beginnt"
.venv/bin/python -u $R/modellvergleich_20261116/f_zwoelf_seeds.py --arms tlbnll_fest300 --seeds 0-11 --workers 6 > $OUT/online_12_seeds.log 2>&1
.venv/bin/python -u $R/modellvergleich_20261116/f_zwoelf_auswerten.py > $OUT/online_auswertung.log 2>&1
note "Online-Test fertig (f_zwoelf_auswertung.md)"
