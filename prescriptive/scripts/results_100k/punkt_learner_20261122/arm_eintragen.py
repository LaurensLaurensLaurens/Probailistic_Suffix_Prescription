"""Online-Arm plsv_fest300 (S-Learner je Wahlpunkt) in den Auswerteskripten eintragen (2026-11-22). Erst ausführen, wenn kein Online-Test
mit diesen Skripten läuft (von data/bpic17_psp_punkt_learner.sh nach dem Ende des Option-Learner-Laufs aufgerufen). Mehrfach aufrufbar."""
from pathlib import Path
R = Path(__file__).resolve().parents[1]


def rep(path, a, b):
    p = R/path
    t = p.read_text()
    if "plsv" in t:
        return
    assert t.count(a) == 1, (path, a[:60])
    p.write_text(t.replace(a, b))


rep("modellvergleich_20261116/f_zwoelf_seeds.py",
    '          "olsv": ROOT/"prescriptive/models/psp_model_durchlaufzeit_option_learner_sv_crn.pkl"}  # Netz je (Punkt, Option), Suffix-Verlust',
    '          "olsv": ROOT/"prescriptive/models/psp_model_durchlaufzeit_option_learner_sv_crn.pkl",  # Netz je (Punkt, Option), Suffix-Verlust\n'
    '          "plsv": ROOT/"prescriptive/models/psp_model_durchlaufzeit_punkt_learner_sv_crn.pkl"}  # S-Learner je Wahlpunkt, Suffix-Verlust')
p = R/"modellvergleich_20261116/f_zwoelf_seeds.py"
t = p.read_text()
if '"plsv_fest300": 45' not in t:
    t = t.replace('"olsv_fest300": 45, ', '"olsv_fest300": 45, "plsv_fest300": 45, ', 1)
    t = t.replace("  olsv_fest300   ein PSP je Wahlpunkt und Option (8 Netze, warm vom ZZ-TF1) mit Suffix-Verlust, ohne Kalibrierung, keyed_noise (22.11.)\n",
                  "  olsv_fest300   ein PSP je Wahlpunkt und Option (8 Netze, warm vom ZZ-TF1) mit Suffix-Verlust, ohne Kalibrierung, keyed_noise (22.11.)\n"
                  "  plsv_fest300   S-Learner je Wahlpunkt (4 Netze, beide Optionen, warm vom ZZ-TF1) mit Suffix-Verlust, ohne Kalibrierung (22.11.)\n", 1)
    p.write_text(t)
p = R/"modellvergleich_20261116/f_zwoelf_auswerten.py"
t = p.read_text()
if "plsv" not in t:
    line = [l for l in t.splitlines() if l.startswith("ARMS = [")][0]
    t = t.replace(line, line.replace('"olsv_fest300"', '"olsv_fest300", "plsv_fest300"', 1))
    a = '("regel", "olsv_fest300"), ("tlsvk_fest300", "olsv_fest300"), ("tlsv_fest300", "olsv_fest300")'
    assert t.count(a) == 1
    t = t.replace(a, a + ',\n                 ("regel", "plsv_fest300"), ("tlsvk_fest300", "plsv_fest300"), ("olsv_fest300", "plsv_fest300")')
    p.write_text(t)
rep("modellvergleich_20261116/abweichungen_regel.py", '"tlsvh_fest300", "olsv_fest300"]', '"tlsvh_fest300", "olsv_fest300", "plsv_fest300"]')
rep("psp_register/online_sammeln.py",
    '             "olsv_fest300": ("Netz je Wahlpunkt und Option + Suffix-Verlust (Zufallszahlen je Event)", "Zufallszahlen je Ereignis (keyed_noise), first_only, 300 Ziehungen fest", 300)}',
    '             "olsv_fest300": ("Netz je Wahlpunkt und Option + Suffix-Verlust (Zufallszahlen je Event)", "Zufallszahlen je Ereignis (keyed_noise), first_only, 300 Ziehungen fest", 300),\n'
    '             "plsv_fest300": ("S-Learner je Wahlpunkt + Suffix-Verlust (Zufallszahlen je Event)", "Zufallszahlen je Ereignis (keyed_noise), first_only, 300 Ziehungen fest", 300)}')
print("Arm plsv_fest300 eingetragen")
