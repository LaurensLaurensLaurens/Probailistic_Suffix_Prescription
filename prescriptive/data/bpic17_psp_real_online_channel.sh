#!/bin/bash
# Wie bpic17_psp_real_online.sh (normale psp-Policy, PSP vom echten Log, Fenster und Seeds wie PSP_lr1e4_online), aber am
# O_Created nur die Wahl des Kanals: O_Sent (mail and online) oder O_Sent (online only). angebot_nachlegen ist kein Wahlpunkt
# mehr (O_Create Offer nach O_Created zieht der Simulator selbst, wie die Bank); die anderen 3 Control-Flow-Punkte und die
# 8 Mitarbeiterpunkte bleiben, also 12 Punkte: channel, handle_leads, annehmen_nach_vervollstaendigung,
# neues_angebot_beim_nachfassen und 8 Mitarbeiter. Das Modell kennt den Punkt als channel+angebot_nachlegen; es nimmt ihn
# auch für channel allein (PSPModel.knows: selbes Entscheidungs-Event, Optionen bekannt).
# Der Katalog des Laufs ist eine Kopie (data/catalog_<TAG>.yaml, über PSP_CATALOG), prescriptive/catalog.yaml bleibt unberührt.
# Weitere Argumente gehen an main.py. Zum Probelauf: TAG, ONLINE, WARM, COOL, SEEDS, WORKERS aus der Umgebung.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
TAG=${TAG:-PSP_real_w3_10k_online_channel}; ONLINE=${ONLINE:-3000}; WARM=${WARM:-700}; COOL=${COOL:-1300}; SEEDS=${SEEDS:-10}; WORKERS=${WORKERS:-12}
MODEL=$P/models/psp_model_durchlaufzeit_real_w3_10k.pkl
OUT=$P/scripts/results_100k/$TAG
LOG=$P/data/$TAG.log
CATALOG=$P/data/catalog_$TAG.yaml
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory|super\(\).__init__'
[ -e "$MODEL" ] || { echo "######## $MODEL fehlt, Abbruch" | tee -a "$LOG"; exit 1; }
# Katalog des Laufs: lauf.mitarbeiterwahl die 8 Punkte, lauf.control_flow ohne angebot_nachlegen
python3 - "$P/catalog.yaml" "$CATALOG" <<'PY' || { echo "######## Katalog nicht erstellt, Abbruch" | tee -a "$LOG"; exit 1; }
import re, sys
source, target = sys.argv[1:3]
text = open(source, encoding="utf-8").read()
start = text.index("\nbpic17:\n") + len("\nbpic17:\n")
end = re.compile(r"^\S", re.M).search(text, start).start()
values = {"mitarbeiterwahl": '["Mitarbeiter: W_Call incomplete files", "Mitarbeiter: W_Complete application", "Mitarbeiter: W_Handle leads", '
                             '"Mitarbeiter: W_Validate application", "Mitarbeiter: W_Call after offers (resume)", "Mitarbeiter: W_Call incomplete files (resume)", '
                             '"Mitarbeiter: W_Complete application (resume)", "Mitarbeiter: W_Validate application (resume)"]',
          "control_flow": '["handle_leads", "annehmen_nach_vervollstaendigung", "neues_angebot_beim_nachfassen"]'}
part = text[start:end]
for key, value in values.items():
    part, n = re.subn(rf"^    {key}: \[.*\]$", lambda m: f"    {key}: {value}", part, flags=re.M)
    assert n == 1, key
open(target, "w", encoding="utf-8").write(text[:start] + part + text[end:])
PY
export PSP_CATALOG=$CATALOG
cd $ROOT/process-simulation-engine
echo "######## PSP vom echten Log online, am O_Created nur der Kanal: $ONLINE Ankünfte, Warmlauf $WARM, Auslauf $COOL, $SEEDS Seeds ($(date +%H:%M:%S))" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/main.py --delta 0.5 --rct-cases 10000 --rct-runs 10 \
    --steps online --online-policies bank random psp --psp-model "$MODEL" --online-seeds $SEEDS --online-cases $ONLINE \
    --online-warmup $WARM --online-cooldown $COOL --online-workers $WORKERS --out "$OUT" "$@" 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch online ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"
