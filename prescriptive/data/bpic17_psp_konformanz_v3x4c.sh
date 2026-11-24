#!/bin/bash
# 4-fache Datenmenge für die T-Learner-Arme auf dem v3-Log (2026-11-21, Nutzer: „mach das“). 128.000 Fälle scheiterten am Prozess-Speicherlimit
# (ulimit -v 56 GB, auch 3- und 2-fach): der Loader legt jeden Prefix als Python-Liste an. Daher --load-chunks 8 (psp_model load_chunks: in
# Stücken von 16.000 Fällen kodieren, je Stück auf die Wahlpunkt-Proben filtern, zusammenfügen; gleiche Encoder und Aufteilung).
# Sonst wie bpic17_psp_konformanz_v3.sh: Arme warm von den Armen des echten Logs (--vocab-from), LR 1e-4, Aktion 10 / Referenz 5 Epochen,
# GPU; Hülle t_learner_k3x4.pkl; Online-Auswertung nur Arm konformanz_x4 (12 Seeds, gepaart mit bank / regel / orakel).
# Aufruf (tmux): bash bpic17_psp_konformanz_v3x4c.sh
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_100k/kuenstliche_effekte_20261120
FILTER='FutureWarning|applied = df|cuda|CUDA|cuInit|hasattr|pin_memory|super\(\).__init__'
export PSP_CATALOG=$P/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
note() { echo "$(date '+%Y-%m-%d %H:%M') k3x4c: $1" >> $OUT/fortschritt.log; }
for N in 4; do
  LOGP=$OUT/log_effekte_v3_x$N/log_raw.pkl
  if [ ! -e "$LOGP" ]; then
    .venv/bin/python - $N <<'PYEOF' >> $OUT/fortschritt.log 2>&1 || { note "Abbruch: Teil-Log x$N"; exit 1; }
import sys, json
from pathlib import Path
import pandas as pd
n = int(sys.argv[1]); quelle = Path("/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive/scripts/results_100k/kuenstliche_effekte_20261120/log_effekte_v3_x4")
ziel = quelle.parent/f"log_effekte_v3_x{n}"; ziel.mkdir(exist_ok=True)
keep = {f"K{r}_" for r in range(8 * n)}
ev = pd.read_pickle(quelle/"log_raw.pkl"); ev = ev[ev["case:concept:name"].str.split("_").str[0].add("_").isin(keep)]
ev.to_pickle(ziel/"log_raw.pkl")
ef = pd.read_csv(quelle/"effekte.csv"); ef[ef["case"].str.split("_").str[0].add("_").isin(keep)].to_csv(ziel/"effekte.csv", index=False)
(ziel/"info.json").write_text(json.dumps({"quelle": str(quelle), "laeufe": 8 * n, "cases": int(ev["case:concept:name"].nunique()), "events": int(len(ev))}, indent=2))
print(f"Teil-Log x{n}: {ev['case:concept:name'].nunique():,} Fälle, {len(ev):,} Events")
PYEOF
  fi
  COMMON="--cases 0 --point-weight 3 --catalog-points --suffix-size 60 --statics-without-future --time-from gaps --counters --time-on-event --teacher-forcing 1.0 --option-in-encoder-after-training --log-pickle $LOGP"
  OK=1
  for ARM in aktion referenz; do
    BASE=$P/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_${ARM}.pkl
    TAG=k3x${N}_zaehler_zeit_tf1_t_$ARM
    MODEL=$P/models/psp_model_durchlaufzeit_$TAG.pkl
    EPOCHS=$([ $ARM = aktion ] && echo 10 || echo 5)
    note "x$N: Training $ARM beginnt ($EPOCHS Epochen, GPU: $(.venv/bin/python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null))"
    [ -e "$MODEL" ] || .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag $TAG --epochs $EPOCHS --learning-rate 1e-4 $COMMON \
        --only-points --only-points-validation --arm $ARM --init-from "$BASE" --vocab-from "$BASE" --load-chunks 8 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$P/data/PSP_$TAG.log"
    if [ ! -e "$MODEL" ]; then note "x$N: Abbruch im Training $ARM ($(grep -c MemoryError $P/data/PSP_$TAG.log) MemoryError)"; OK=0; break; fi
    note "x$N: Training $ARM fertig"
  done
  [ $OK = 1 ] && break
done
[ $OK = 1 ] || { note "Abbruch"; exit 1; }
HULL=$P/models/psp_model_durchlaufzeit_t_learner_k3x$N.pkl
CUDA_VISIBLE_DEVICES="" .venv/bin/python ../prescriptive/policies/ev_policy/t_learner_policy.py \
    $P/models/psp_model_durchlaufzeit_k3x${N}_zaehler_zeit_tf1_t_aktion.pkl $P/models/psp_model_durchlaufzeit_k3x${N}_zaehler_zeit_tf1_t_referenz.pkl "$HULL" >> $OUT/fortschritt.log 2>&1 || { note "Abbruch: Hülle"; exit 1; }
note "x$N: Hülle gebaut, Online-Auswertung konformanz_x$N beginnt"
.venv/bin/python -u $OUT/online_konformanz.py konformanz_x$N 0-11 12 > $OUT/online_konformanz_x$N.log 2>&1
note "x$N: Online-Auswertung beendet (Code $?): $(grep -E '^FERTIG' $OUT/online_konformanz_x$N.log | head -1)"
note "Alles fertig"
