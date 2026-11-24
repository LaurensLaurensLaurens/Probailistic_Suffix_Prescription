"""Zentrales Ergebnisregister der BPIC17-PSP-Modelle (angelegt 2026-11-16, wird bei jedem neuen Ergebnis neu erzeugt).

Sammelt aus den vorhandenen Ergebnisordnern die einheitlich gemessenen Kennzahlen je Modellstand und schreibt
  register.csv        eine Zeile je Modellstand (maschinenlesbar, Grundlage für die Suche nach der besten Konfiguration)
  register.md         dieselben Werte als lesbare Tabellen, dazu Wahrheit/Referenzen und Online-Ergebnisse
  verlustkurven.csv   Trainings-/Validierungsverluste je Epoche aus allen data/PSP_*.log
Deutungen und Schlussfolgerungen stehen von Hand in befunde.md (nicht erzeugt, nicht überschreiben).
Neues Modell: Eintrag in MODELLE ergänzen; Prüfstände des Zähler-Trainings (zaehler_zeit_20261116) und Piloten werden
automatisch gefunden. Liest nur aus den Ergebnisordnern (außer: fehlende kennzahlen.csv einer Nachlegen-Auswertung wird
mit nachlegen_auswerten.py erzeugt). Aufruf: <venv>/python register_erstellen.py
"""
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
R = OUT.parent
P = R.parents[1]
V = R/"modellvergleich_20261116"
Z = R/"zaehler_zeit_20261116"
G = R/"psp_gesamtuebersicht_20261114"
PUNKTE = ["annehmen_bei_konzept", "annehmen_nach_vervollstaendigung", "angebot_nachlegen", "handle_leads"]
KURZ = {"annehmen_bei_konzept": "konzept", "annehmen_nach_vervollstaendigung": "nach_v", "angebot_nachlegen": "nachlegen",
        "handle_leads": "leads"}
GRUPPEN = {"annehmen_bei_konzept": "konzept", "annehmen_nach_vervollstaendigung": "nach_v", "nachlegen 1. Besuch": "nachlegen1",
           "nachlegen ab 2. Besuch": "nachlegen2", "handle_leads": "leads"}
SUFFIX_G = G/"suffix_vollmodelle_1000_testprefixe.csv"
STANDARD = "alle Trainingsfälle (20.482/4.726), Decoder 60, 15 Ep., LR 1e-4, Wahlpunkte ×3, TF-Plan max(0, 1−ep/7,5)"

# label, Tag (models/psp_model_durchlaufzeit_<tag>.pkl), Konfiguration, Optionssignal-Ordner,
# (Endung der Vergleichsordner in modellvergleich_20261116, Beschriftung dort), (Suffixgüte-CSV, Beschriftung), Trainingslog/Epoche
MODELLE = [
    dict(label="(a)", tag="real_alle_s60_loggaps", konfig=f"{STANDARD}; Log-Abstände; Option +1 ms",
         rauschen=V/"rauschen_a", suffix=(SUFFIX_G, "(a), Log-Gaps")),
    dict(label="(b)", tag="real_alle_s60_loggaps_rest", konfig="wie (a) + Restzeitkopf (Kopfbewertung verboten, hier Suffixe)",
         rauschen=V/"rauschen_b"),
    dict(label="(c)", tag="real_alle_s60_normal", konfig=f"{STANDARD}; normale Abstände, Durchlaufzeit = Summe der Abstände; "
         "statische Merkmale ohne Zukunft; Option +1 ms", rauschen=R/"psp_rauschen_real_alle_s60_normal",
         vergleich=("", "(c)"), suffix=(V/"suffixguete/suffixguete.csv", "(c)")),
    dict(label="(d) 1 ms", tag="real_alle_s60_klassen", konfig=f"{STANDARD}; Zeitklassen (Klassenkopf); Option +1 ms",
         rauschen=R/"psp_rauschen_real_alle_s60_klassen", suffix=(SUFFIX_G, "(d), Zeitklassen")),
    dict(label="D", tag="real_alle_s60_klassen_option_encoder", konfig="(d)-Netz, Option als letztes Prefix-Event mit eigenem "
         "gezogenem Abstand (option_as encoder)", rauschen=V/"rauschen_d_option_encoder", vergleich=("", "D-Netz")),
    dict(label="OE", tag="real_alle_s60_token_oe", konfig=f"{STANDARD}; gemeinsames Token Aktivität|Lifecycle|Zeitklasse; "
         "Option im Encoder", rauschen=R/"psp_rauschen_voll_real_alle_s60_token_oe_real_alle_s60_token_oe",
         vergleich=("", "OE"), suffix=(SUFFIX_G, "OE, Event+Zeitklasse als Token")),
    dict(label="(e)", tag="real_alle_s60_klassen_event_oe", konfig="Zeitklasse bedingt auf das Event desselben Schritts, Option im Encoder",
         rauschen=V/"rauschen_e", suffix=(V/"suffixguete/suffixguete.csv", "(e)")),
    dict(label="(c)-TF1", tag="real_alle_s60_normal_tf1", konfig="wie (c), Teacher Forcing immer 1",
         rauschen=R/"psp_rauschen_voll_real_alle_s60_normal_tf1_real_alle_s60_normal_tf1", suffix=(SUFFIX_G, "(c), Teacher Forcing 1")),
    dict(label="D-TF1", tag="real_alle_s60_klassen_tf1_option_encoder", konfig="wie D, Teacher Forcing immer 1",
         rauschen=V/"rauschen_d_tf1", suffix=(V/"suffixguete/suffixguete.csv", "D-TF1")),
    dict(label="OE-TF1 Ep. 26", tag="(Checkpoint) oe_tf1_100_training_20261115", konfig="wie OE, TF immer 1, 100 Ep. geplant, "
         "bei Ep. ~27 gestoppt; Stand Ep. 26", rauschen=V/"rauschen_oe_tf1_letzte", vergleich=("_oe_tf1", "OE-TF1 Ep. 26"),
         suffix=(V/"suffixguete_oe_tf1/suffixguete.csv", "OE-TF1 Ep. 26"), modell=V/"oe_tf1_schnappschuss/psp_oe_tf1_letzte.pkl", stand=26, verluste=("oe_tf1_100", 26)),
    dict(label="OE-TF1 Ep. 1", tag="(Checkpoint) oe_tf1_100_training_20261115", konfig="wie OE-TF1, Stand Ep. 1 (bester Val-Verlust)",
         rauschen=V/"rauschen_oe_tf1_beste_val", vergleich=("_oe_tf1", "OE-TF1 Ep. 1"),
         suffix=(V/"suffixguete_oe_tf1/suffixguete.csv", "OE-TF1 Ep. 1"), modell=V/"oe_tf1_schnappschuss/psp_oe_tf1_beste_val.pkl", stand=1, verluste=("oe_tf1_100", 1)),
    dict(label="s40", tag="real_np_s40_ohne_zukunft", konfig="10k Fälle, Decoder 40, 20 Ep., LR 1e-4, Abstände, statische ohne Zukunft",
         rauschen=R/"psp_rauschen_s40", suffix=(SUFFIX_G, "s40, Gaps")),
    dict(label="10k neue Punkte", tag="real_w3_10k_np", konfig="10k Fälle, Decoder 5, 100 Ep., LR 1e-5 (ursprüngliches Training)",
         rauschen=V/"rauschen_10k_neue_punkte"),
]


# Trainings mit Begleiter (Prüfstände je Epoche): Ordner, Kurzname in Dateien, Beschriftung, Trainings-Tag, Beschreibung
BEGLEITET = [(Z, "zz", "ZZ", "real_alle_s60_normal_zaehler_zeit",
              "(c) + Zähler je Aktivität|Lifecycle in Encoder/Decoder + Zeitköpfe sehen das Event des Schritts; Option im Encoder nach dem Training"),
             (R/"zaehler_zeit_tf1_20261117", "zztf1", "ZZ-TF1", "real_alle_s60_normal_zaehler_zeit_tf1",
              "wie ZZ, aber durchgehend Teacher Forcing 1 (konsistentes Trainingsziel)"),
             (R/"zaehler_uhren_tf1_20261117", "zzu", "ZZU-TF1", "real_alle_s60_normal_zaehler_uhren_tf1",
              "wie ZZ-TF1, dazu Uhren: Tage seit dem letzten Auftreten je Aktivität|Lifecycle-Token als Eingabe"),
             (R/"nur_wahlpunkte_20261117", "wp", "ZZ-TF1-WP", "real_alle_s60_normal_zaehler_zeit_tf1_wp",
              "wie ZZ-TF1, aber nur die Trainingsproben am Options-Event eines Wahlpunkts (only_points, ~75.000 statt ~933.000), 30 Epochen geplant, bei Ep. 18 abgebrochen"),
             (R/"nur_wahlpunkte150_20261117", "wp150", "ZZ-TF1-WP150", "real_alle_s60_normal_zaehler_zeit_tf1_wp150",
              "wie ZZ-TF1-WP, Training und Validierung nur am Options-Event (only_points + only_points_validation), 150 Epochen (≈ 10 ZZ-TF1-Epochen an Schritten)")]
EPOCHEN_FINAL = {"wp": 30, "wp150": 150}  # sonst 15


def zaehler_staende():
    """Prüfstände der begleiteten Trainings (ZZ, ZZ-TF1), automatisch aus ihren Ordnern."""
    found = []
    for folder_run, short, name, tag, text in BEGLEITET:
        for folder in sorted(folder_run.glob("rauschen_*"), key=lambda f: (f.name == "rauschen_final", int(re.sub(r"\D", "", f.name) or 0))):
            stand = folder.name.removeprefix("rauschen_")
            epoch = EPOCHEN_FINAL.get(short, 15) if stand == "final" else int(stand.removeprefix("ep"))
            found.append(dict(label=f"{name} {stand}", tag=tag + ("" if stand == "final" else f" (Checkpoint {stand})"),
                              konfig=f"{text}; Stand Epoche {epoch}", rauschen=folder, vergleich=(f"_{short}_{stand}", f"{name} {stand}"),
                              verluste=(tag, epoch), ordner=folder_run, stand=epoch,
                              suffix=(folder_run/f"suffixguete_{stand}/suffixguete.csv", f"{name} {stand}"),
                              modell=P/f"models/psp_model_durchlaufzeit_{tag}.pkl" if stand == "final"
                              else folder_run/f"modelle/psp_{short}_{stand}.pkl"))
    return found


def beschraenkt_staende():
    """Varianten mit beschränktem Ziehen (psp_beschraenkt.py, 2026-11-17), aus beschraenkt_20261117/staende.json."""
    import json
    folder = R/"beschraenkt_20261117"
    try:
        known = json.loads((folder/"staende.json").read_text())
    except (FileNotFoundError, ValueError):
        return []
    return [dict(label=entry["label"], tag=f"(beschränkt) {tag}", modell=entry["modell"], rauschen=folder/f"rauschen_{tag}",
                 vergleich=(f"_{tag}", entry["label"]), konfig="Kopie mit beschränktem Ziehen (psp_beschraenkt.py): Übergänge zwischen Events "
                 "und Lifecycle-Folgen innerhalb von Aktivitäten wie in den Trainingsfällen, Fallende erst nach A_Pending/Denied/Cancelled"
                 + ("; Gumbel-Kopplung der Eventwahl" if "Gumbel" in entry["label"] else ""))
            for tag, entry in known.items()]


def t_learner_staende():
    """T-Learner-PSP (policies/ev_policy/t_learner_policy.py, 2026-11-17), aus t_learner_20261117/staende.json."""
    import json
    out = []
    texts = {"t_learner_20261117": "T-Learner: zwei PSPs (Aktion / Referenz), warm vom fertigen ZZ-TF1, je nachtrainiert nur auf den "
             "Wahlpunkt-Proben ihrer Gruppe (LR 3e-5, Aktion 40 / Referenz 20 Epochen); jede Option zieht aus ihrem Modell; "
             "Gedächtnistests ohne Option messen nur das Referenz-Modell",
             "rct_tlearner_20261118": "trainiert auf Simulator-RCT-Daten des korrigierten Simulators (8 × 4.000 Fälle, Optionen 50/50, "
             "Nachlegen nur 1. Besuch randomisiert), sonst wie ZZ-TF1 bzw. T-Learner; Testprefixe aus dem echten Log",
             "restzeit_hilfsziel_20261118": "ZZ-TF1 fertig + Restzeit-Kopf als reines Hilfsziel im Training (5 Epochen nachtrainiert, LR 3e-5, "
             "warm vom ZZ-TF1); Bewertung weiter nur über gezogene Suffixe",
             "tf_boden05_20261118": "ZZ-TF0,5: fertiges ZZ-TF1 6 Epochen nachtrainiert mit Teacher-Forcing-Boden 0,5 (Quote 1,0 / 0,67 / "
             "dann 0,5; frei laufende Schritte speisen argmax ein), LR 1e-4, sonst wie ZZ-TF1",
             "beta_nll_20261118": "β-NLL 0,5: fertiges ZZU-TF1 5 Epochen nachtrainiert mit β-NLL der Zeitköpfe (jeder Term des attenuierten "
             "Verlusts mal stop_grad(σ^(2β)), β = 0,5; Seitzer et al. 2022), LR 1e-4, sonst wie ZZU-TF1",
             "suffix_verlust_20261119": "T-Learner (ZZ-TF1), beide Arme warm nachtrainiert mit zusätzlichem Suffix-Verlust (sequence_loss: CRPS der "
             "frei gezogenen Restzeit gegen die echte Restzeit, je Batch 32 Prefixe × 16 Ziehungen, Gewicht 1; Aktion 10 / Referenz 5 Epochen, "
             "LR 3e-5, Teacher Forcing 1); Bewertung mit Zufallszahlen je Event (keyed_noise); Stand „+ Kalibrierung“: warm von diesen Armen "
             "weiter mit Kalibrierung je (Wahlpunkt, Option) im Suffix-Verlust (Gewicht 3, Prefixe je Gruppe gleich häufig; Aktion 10 / Referenz 5 Epochen); "
             "Stand „+ gepaarte Kalibrierung“ (20.11.): wie „+ Kalibrierung“, aber Zielwert = echtes Mittel derselben Prefixe im Batch und "
             "Quadrat der Abweichung aus zwei unabhängigen Ziehungshälften (kein Schrumpfdruck gegen fallabhängige Effekte); "
             "Stand „+ Kalibrierung Hälften“ (20.11.): Prefixe einer Gruppe im Batch in zwei disjunkte Hälften, Strafe (Mittel 1 − "
             "Gruppenmittel) · (Mittel 2 − Gruppenmittel), sonst wie „+ Kalibrierung“",
             "zeitkopf_je_event_20261118":"T-Learner (ZZ-TF1), Zeitköpfe beider Arme um eigene Parameter je Event erweitert (Versatz je Event je Ausgabe für Mittel und log-Varianz getrennt; Variante Steigung zusätzlich Steigung je Event auf h) und nur die Zeitköpfe nachtrainiert (Rest eingefroren, Wahlpunkt-Proben des Arms, Teacher Forcing 1, LR 1e-3 / Versätze 3e-2, Aktion 30 / Referenz 12 Epochen)"}
    for name, text in texts.items():
        folder = R/name
        try:
            known = json.loads((folder/"staende.json").read_text())
        except (FileNotFoundError, ValueError):
            continue
        out += [dict(label=entry["label"], tag=f"({name}) {tag}", modell=entry["modell"], rauschen=folder/f"rauschen_{tag}",
                     vergleich=(f"_{tag}", entry["label"]), konfig=text) for tag, entry in known.items()]
    return out


# Optionssignal-Ordner der Piloten -> Modelldatei (aus data/bpic17_psp_stufe3_pilot_auswertung.sh, ..._pilot_oe.sh und den
# Aufrufen RUNDE=pilot2/pilot3, siehe Gedächtnis psp-stage3); *_od / *_oe: Kopien mit option_as decoder / encoder
PILOT_MODELLE = {"k_tf0": "klassen_tf0", "k_tf0_od": "klassen_tf0_od", "k_tf1": "klassen_tf1", "k_tf1_od": "klassen_tf1_od",
                 "t_tf1_od": "token_tf1", "n_tf1": "normal_tf1", "n_tf1_od": "normal_tf1_od", "2_t_tf0": "token_tf0",
                 "2_t_tf05": "token_tf05", "3_t_tf05_kx": "token_tf05_kontext", "3_t_tf1_kx": "token_tf1_kontext"}


def piloten():
    out = []
    for f in sorted(R.glob("psp_rauschen_pilot*")):
        if not (f/"rauschen.csv").exists():
            continue
        name = f.name.removeprefix("psp_rauschen_pilot").strip("_")
        model = PILOT_MODELLE.get(name, name)
        out.append(dict(label="Pilot " + name, tag="pilot_" + model, konfig="Pilot: 3.000 Fälle, 6 Ep. (k/klassen = Zeitklassen, "
                        "t/token = Token mit Zeitklasse, n/normal = normale Abstände; tf = TF-Untergrenze; od/oe = Option im Decoder/Encoder; "
                        "kx/kontext = Decoder sieht Encoder-Zustand)", rauschen=f, pilot=True))
    return out


# Konfiguration aus den gespeicherten Modellen (params, Netz, Trainingszusammenfassung); Zwischenspeicher je Datei und Änderungszeit
STANDARD_PARAMS = {"teacher_forcing": 0.0, "point_weight": 1, "lifecycle_tokens": True, "time_from": "(Original: Fallalter)"}
FESTE_SPALTEN = {"epochs": "epochen_geplant", "learning_rate": "lernrate", "batch_size": "batch", "regularization_term": "regularisierung",
                 "suffix_size": "decoderlaenge", "point_weight": "wahlpunkt_gewicht", "teacher_forcing": "tf_untergrenze",
                 "time_from": "zeit_aus", "lifecycle_tokens": "lifecycle_tokens", "statics": "statische_merkmale", "train_cases": "faelle_soll"}
NICHT_ZEIGEN = {"checkpoints", "threads", "device"}


# Trainingslogs, die nicht als data/PSP_<tag>.log vorliegen (Läufe der anderen Sitzung in eigenen Ordnern)
LOGS = {"real_alle_s60_klassen_event_oe": R/"e_suffixe30_cases200_seed0_20261113/pipeline.log",
        "real_alle_s60_klassen_tf1_option_encoder": R/"d_tf1_training_20261114/pipeline.log",
        "oe_tf1_100": R/"oe_tf1_100_training_20261115/pipeline.log"}


def startinfo(tag):
    """Exakte Trainingsparameter laut Trainingslog (Zeile '######## Training: {...}') und die Startskripte in data/, die den Tag nennen."""
    out = {}
    if tag.startswith("(Checkpoint) oe_tf1"):
        tag = "oe_tf1_100"
    if tag == "real_alle_s60_klassen_option_encoder":  # D: kein eigenes Training, Netz von (d) mit option_as encoder
        out = startinfo("real_alle_s60_klassen")
        out["startskripte"] = "kein eigenes Training: Kopie von (d) 1 ms, params option_as = encoder nachträglich gesetzt"
        return out
    if not tag or tag.startswith("("):
        return out
    tag = tag.split(" (")[0]
    log = LOGS.get(tag, P/f"data/PSP_{tag}.log")
    if log.exists():
        line = next((l for l in log.read_text(errors="ignore").splitlines() if l.startswith("######## Training: {")), None)
        if line:
            out["trainingsparameter_log"] = re.sub(r"'checkpoints': '[^']*', ", "", line.removeprefix("######## Training: "))
        out["trainingslog"] = str(log.relative_to(P.parent))
    scripts = [f.name for f in sorted((P/"data").glob("*.sh")) if re.search(rf"(?<![\w]){re.escape(tag)}(?![\w])", f.read_text(errors="ignore"))]
    if scripts:
        out["startskripte"] = ", ".join(scripts)
    return out


def modellpfad(spec):
    if spec.get("modell"):
        return Path(spec["modell"])
    tag = spec.get("tag", "")
    return P/f"models/psp_model_durchlaufzeit_{tag}.pkl" if tag and not tag.startswith("(") else None


def konfigurationen(specs):
    import json
    cache_file = OUT/"konfigurationen_cache.json"
    try:
        cache = json.loads(cache_file.read_text())
    except (FileNotFoundError, ValueError):
        cache = {}
    loaded = False
    result = {}
    for spec in specs:
        path = modellpfad(spec)
        if path is None or not path.exists():
            continue
        key = f"v2|{path}|{path.stat().st_mtime:.0f}"
        if key not in cache:
            if not loaded:
                import os
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
                sys.path.insert(0, str(P.parent))
                from prescriptive.helpers.datasets import add_paths
                add_paths()
                loaded = True
            from prescriptive.estimators.psp_model import PSPModel
            model = PSPModel.load(str(path))
            params = {**STANDARD_PARAMS, **model.params}
            entry = {FESTE_SPALTEN[k]: params.get(k) for k in FESTE_SPALTEN}
            entry["statische_merkmale"] = "+".join(sum(map(list, params["statics"]), [])) if params.get("statics") else "alle (mit Zukunftswissen)"
            entry["weitere_params"] = "; ".join(f"{k}={v}" for k, v in sorted(params.items()) if k not in FESTE_SPALTEN and k not in NICHT_ZEIGEN)
            net = model.net
            entry.update(hidden=getattr(net, "hidden_size", None), schichten=getattr(net, "num_layers", None),
                         dropout=getattr(net, "dropout", None), seed=getattr(model, "seed", None), modelldatei=str(path.relative_to(P.parent)))
            summary = next(iter((model.summary or {}).values()), {}) if isinstance(model.summary, dict) else {}
            entry.update(faelle_training=summary.get("Trainingsfälle"), faelle_validierung=summary.get("Validierungsfälle"),
                         proben=summary.get("Proben"), proben_validierung=summary.get("Proben Validierung"),
                         proben_wahlpunkt=summary.get("Proben am Wahlpunkt"), fensterbreite=summary.get("window_size"),
                         epochen_trainiert=summary.get("Epochen"),
                         val_bester=summary.get("bester Validierungsverlust"), val_letzte=summary.get("Validierungsverlust letzte Epoche"))
            # Netzaufbau: Merkmale mit Vokabulargröße, Ausgaben, Eingabebreiten, Parameterzahl
            cats, nums = net.data_set_categories
            statics_cat, statics_num = net.static_data_set_categories
            entry.update(encoder_merkmale=", ".join([f"{c[0]}({c[1]})" for c in cats] + [c[0] for c in nums]),
                         decoder_merkmale=" + ".join(", ".join(x) for x in net.dec_feat),
                         statische_eingaben=", ".join([f"{c[0]}({c[1]})" for c in statics_cat] + [c[0] for c in statics_num]),
                         ausgaben=", ".join(f"{k}({v})" for d in net.output_sizes for k, v in d.items()),
                         eingabe_encoder=getattr(net, "input_size_enc", None), eingabe_decoder=getattr(net, "input_size_dec", None),
                         parameter_anzahl=int(sum(x.numel() for x in net.parameters())),
                         wahlpunkte=", ".join(model.points), ziehungen_standard=getattr(model, "samples_per_option", None),
                         test_faelle=len(getattr(model, "test_cases", []) or []), lifecycle_als_token=getattr(model, "lifecycle_tokens", None),
                         modell_datum=time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)))
            cache[key] = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in entry.items()}
        result[spec["label"]] = cache[key]
    cache_file.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    return result


SCREENING = R/"screening_alle/prescription/screening/control_flow/bpic17_screening_control_flow_detailliert.csv"


def aipw_effekte():
    t = pd.read_csv(SCREENING).set_index("Kandidat")
    return {p: float(t.at[p, "Effekt (AIPW)"]) for p in PUNKTE}


def rauschen(folder):
    try:
        t = pd.read_csv(folder/"rauschen.csv", index_col=0)
    except FileNotFoundError:
        return {}
    out = {}
    for p in PUNKTE:
        if p in t.columns:
            k = KURZ[p]
            for row, key in (("Vorzeichen wie AIPW (alle)", "aipw_vorzeichen"), ("Diff Mittel (Tage)", "optionsdiff"),
                             ("Diff Betrag Median", "diff_betrag_median"), ("|z| Median", "z_median"), ("Anteil |z| > 2", "anteil_z_gt2"),
                             ("Vorzeichen wie AIPW (|z| > 2)", "vorzeichen_z_gt2"), ("Anteil Aktion nach Mittel", "anteil_aktion"),
                             ("Regel 2 SE: Aktion", "regel2se_aktion"), ("Regel 2 SE: Referenz", "regel2se_referenz"),
                             ("nötige Ziehungen für |z|=2 (Median)", "noetige_ziehungen"), ("SE gepaart Median", "se_gepaart"),
                             ("Korrelation der Optionen Median", "korrelation")):
                if row in t.index and pd.notna(t.at[row, p]):
                    out[f"{key}_{k}"] = float(t.at[row, p])
            if f"optionsdiff_{k}" in out:  # gegen den AIPW-Effekt des Screenings (wie in der Wahrheitszeile)
                out[f"effekt_anteil_{k}"] = out[f"optionsdiff_{k}"]/aipw_effekte()[p]
    for key, row in (("se_gepaart", "SE gepaart Median"), ("korrelation_optionen", "Korrelation der Optionen Median"),
                     ("dlz_gezogen", "Durchlaufzeit gezogen (Mittel)"), ("dlz_sd_gezogen", "SD der gezogenen Durchlaufzeit (Tage)"),
                     ("dlz_echt", "Durchlaufzeit echt (Mittel)")):
        if row in t.index:
            out[key] = float(t.loc[row, [p for p in PUNKTE if p in t.columns]].astype(float).mean())
    return out


def vergleich(spec):
    if not spec.get("vergleich"):
        return {}
    ending, label = spec["vergleich"]
    out = {}
    try:
        t = pd.read_csv(V/f"punkte_alle{ending}/vergleich.csv")
        t = t[t.Quelle == label]
        for _, row in t.iterrows():
            g = GRUPPEN[row.Gruppe]
            out[f"sim_vorteil_{g}"] = row["Vorteil Regel-Option"]
            out[f"sim_richtig_{g}"] = row["Anteil Zustände richtig (100 Ziehungen)"]
        out["sim_mcse"] = t["MC-SE je Entscheidung"].mean()
    except FileNotFoundError:
        pass
    folder = V/f"nachlegen_erklaerung{ending}"
    if (folder/"verlaeufe.pkl").exists() and not (folder/"kennzahlen.csv").exists():
        subprocess.run([sys.executable, str(V/"nachlegen_auswerten.py"), folder.name], check=True, stdout=subprocess.DEVNULL)
    try:
        k = pd.read_csv(folder/"kennzahlen.csv", index_col=[0, 1])
        d = k.loc[(label, "extra")]-k.loc[(label, "senden")]
        out.update(nachlegen_restzeit=d["Restzeit"], nachlegen_versendet=d["versendet"], nachlegen_nachfassen=d["Nachfassen bis Validierung"],
                   nachlegen_ohne_ende=k.loc[label, "ohne"].mean())
    except (FileNotFoundError, KeyError):
        pass
    try:
        a = pd.read_csv(V/f"angebote_einfuegen{ending}/ziehungen.csv")
        w = a[a.modell == label].pivot_table(index="case", columns="variante", values="rest")
        out["angebot_eingefuegt_rest"] = (w["B"]-w["A"]).mean()
    except (FileNotFoundError, KeyError):
        pass
    try:
        lg = pd.read_csv(V/f"leads_gedaechtnis{ending}/ziehungen.csv")
        m = lg[lg.modell == label].groupby("gruppe").rest.mean()
        out["lead_gedaechtnis"] = m["Lead bearbeitet"]-m["Lead zurückgezogen"]
    except (FileNotFoundError, KeyError):
        pass
    return out


# Suffixgüte für Modelle ohne eigenen Eintrag in MODELLE (alle auf denselben 1.000 Testprefixen, 100 Ziehungen)
SUFFIX_EXTRA = {"D": (SUFFIX_G, "(d), Zeitklassen", "wie (d) 1 ms: gleiches Netz, option_as wirkt nur bei der Optionsbewertung"),
                "Pilot k_tf0": (R/"pilot_suffixguete/suffixguete.csv", "k_tf0"), "Pilot k_tf1": (R/"pilot_suffixguete/suffixguete.csv", "k_tf1"),
                "Pilot t_tf1_od": (R/"pilot_suffixguete/suffixguete.csv", "t_tf1"), "Pilot n_tf1": (R/"pilot_suffixguete/suffixguete.csv", "n_tf1"),
                "Pilot 2_t_tf0": (R/"pilot2_suffixguete/suffixguete.csv", "t_tf0"), "Pilot 2_t_tf05": (R/"pilot2_suffixguete/suffixguete.csv", "t_tf05"),
                "Pilot 3_t_tf1_kx": (R/"pilot3_suffixguete/suffixguete.csv", "t_tf1_kx"),
                "Pilot 3_t_tf05_kx": (R/"pilot3_suffixguete/suffixguete.csv", "t_tf05_kx")}
SUFFIX_SPALTEN = {"Schritt 1": "suffix_s1", "Schritt 2": "suffix_s2", "Schritt 3": "suffix_s3", "Schritt 5": "suffix_s5", "Schritt 10": "suffix_s10",
                  "Schritt 20": "suffix_s20", "DLS": "suffix_dls", "Suffixlänge-MAE": "suffix_laenge_mae",
                  "Abstand Schritt 1 MAE (h)": "suffix_erster_abstand_mae_h", "Restzeit MAE (Tage)": "restzeit_mae",
                  "Restzeit Fehler (Tage)": "restzeit_bias", "Restzeit CRPS (Tage)": "restzeit_crps", "Abdeckung 80 %": "abdeckung80",
                  "Restzeit gezogen (Tage)": "restzeit_gezogen", "Restzeit echt (Tage)": "restzeit_echt"}


def suffix(spec):
    source = spec.get("suffix") or SUFFIX_EXTRA.get(spec["label"])
    if not source or not Path(source[0]).exists():
        return {}
    path, label = source[0], source[1]
    t = pd.read_csv(path)
    column = "Variante" if "Variante" in t.columns else "Modell"
    row = t[t[column] == label]
    if row.empty:
        return {}
    row = row.iloc[0]
    out = {name: row[col] for col, name in SUFFIX_SPALTEN.items() if col in row.index}
    out["suffix_quelle"] = str(Path(path).relative_to(R)) + (f" ({source[2]})" if len(source) > 2 else "")
    return out


def diagnose(spec):
    """psp_diagnose: an 100 echten Entscheidungen je Punkt Fehler und Korrelation der gezogenen gegen die echte Durchlaufzeit."""
    tag = spec.get("tag", "")
    path = R/f"psp_diagnose_{tag}/diagnose.csv"
    if spec["label"] == "D":
        path = R/"psp_diagnose_real_alle_s60_klassen/diagnose.csv"
    if not path.exists():
        return {}
    t = pd.read_csv(path, index_col=0)
    cols = [p for p in PUNKTE if p in t.columns]
    get = lambda row: float(t.loc[row, cols].astype(float).mean()) if row in t.index else np.nan
    return {"diag_mae_faktisch": get("Abfrage: Fehler faktisch (MAE, Tage)"), "diag_korrelation_faktisch": get("Abfrage: Korrelation faktisch"),
            "diag_diff_nach_5_schritten": get("Abfrage: Diff nach 5 Schritten"), "diag_suffixlaenge": get("Abfrage: Suffixlänge")}


# Zerlegung des Optionsvorteils in Pfad und Zeit (psp_zerlegung.py): Spaltennamen der Dateien -> Register-Beschriftung
ZERLEGUNG = {"psp_zerlegung": {"(c)": "(c)", "(d)": "(d) 1 ms"}, "voll_real_alle_s60_normal_tf1_zerlegung": {"real_alle_s60_normal_tf1": "(c)-TF1"},
             "voll_real_alle_s60_token_oe_zerlegung": {"real_alle_s60_token_oe": "OE"}, "pilot_zerlegung": None, "pilot_oe_zerlegung": None,
             "pilot_oe2_zerlegung": None, "pilot2_zerlegung": {"t_tf05": "Pilot 2_t_tf05", "t_tf0": "Pilot 2_t_tf0"},
             "pilot3_zerlegung": {"t_tf1_kx": "Pilot 3_t_tf1_kx", "t_tf05_kx": "Pilot 3_t_tf05_kx"}}


def _prozente(text):
    return [float(x) for x in re.findall(r"([\d.]+)%", str(text))]


def zerlegungen():
    result = {}
    for folder, names in ZERLEGUNG.items():
        path = R/folder/"zerlegung.csv"
        if not path.exists():
            continue
        t = pd.read_csv(path, header=[0, 1], index_col=0)
        for (point, model) in t.columns:
            label = "Wahrheit/Referenz" if model.startswith("echt") else (names or {}).get(model, f"Pilot {model}")
            entry = result.setdefault(label, {})
            k = KURZ.get(point, point)
            for row, key in (("Vorteil Aktion gesamt (Tage)", "zerl_vorteil"), ("davon Pfad", "zerl_pfad"), ("davon Zeit", "zerl_zeit")):
                if row in t.index and pd.notna(t.at[row, (point, model)]):
                    entry[f"{key}_{k}"] = float(t.at[row, (point, model)])
            if "Anteil keins: Aktion / Referenz" in t.index:
                entry.setdefault("_ohne", []).extend(_prozente(t.at["Anteil keins: Aktion / Referenz", (point, model)]))
    for entry in result.values():
        values = entry.pop("_ohne", [])
        if values:
            entry["zerl_ohne_ende"] = np.mean(values)/100
    return result


def verlustkurven():
    rows = []
    for log in sorted((P/"data").glob("PSP_*.log")) + [f for f in LOGS.values() if f.exists()]:
        name = log.stem.removeprefix("PSP_") if log.parent.name == "data" else next(k for k, v in LOGS.items() if v == log)
        run = 0
        for line in log.read_text(errors="ignore").splitlines():
            m = re.match(r"Epoch \[(\d+)/(\d+)\]", line)
            if m:
                epoch = int(m[1])
                run += epoch == 1
                tf = re.search(r"Teacher forcing ratio: ([\d.]+)", line)
                lr = re.search(r"Learning Rate: ([\d.e-]+)", line)
                rows.append(dict(log=name, lauf=run, epoche=epoch, epochen=int(m[2]), lr=float(lr[1]) if lr else np.nan,
                                 tf=float(tf[1]) if tf else np.nan))
                continue
            for key, pattern in (("train_att", "Avg Attenuated Training Loss: "), ("val_std", "Avg Standard Validation Loss: "),
                                 ("val_att", "Avg Attenuated Validation Loss: ")):
                if pattern in line and rows and rows[-1]["log"] == name:
                    try:
                        rows[-1][key] = float(line.split(pattern)[1].split()[0])
                    except ValueError:
                        pass
    return pd.DataFrame(rows)


def wahrheit():
    out = {"label": "Wahrheit/Referenz", "pilot": False, "konfig": "Simulator-RCT (Vorteil Regel-Option), Simulator mit Bank danach (Nachlegen), "
           "echter Log (Lead-Gedächtnis, Durchlaufzeit)"}
    t = pd.read_csv(V/"punkte_alle/vergleich.csv")
    for _, row in t[t.Quelle == "Simulator"].iterrows():
        out[f"sim_vorteil_{GRUPPEN[row.Gruppe]}"] = row["Vorteil Regel-Option"]
    k = pd.read_csv(V/"nachlegen_erklaerung/kennzahlen.csv", index_col=[0, 1])
    sim = "Simulator (Bank danach)"
    d = k.loc[(sim, "extra")]-k.loc[(sim, "senden")]
    out.update(nachlegen_restzeit=d["Restzeit"], nachlegen_versendet=d["versendet"], nachlegen_nachfassen=d["Nachfassen bis Validierung"],
               nachlegen_ohne_ende=k.loc[sim, "ohne"].mean())
    pre = pd.read_csv(V/"leads_gedaechtnis/prefixe.csv").groupby("gruppe").real_rest.mean()
    out["lead_gedaechtnis"] = pre["Lead bearbeitet"]-pre["Lead zurückgezogen"]
    out.update(rauschen(R/"psp_rauschen_real_alle_s60_normal"))
    out = {k: v for k, v in out.items() if k in ("label", "pilot", "konfig", "dlz_echt") or k.startswith(("sim_", "nachlegen_", "lead_"))}
    # AIPW auf dem echten Log (Screening 2026-11-06): Effekt der Aktion mit 95-%-KI, Policy-Gewinn einer Einzelfall-Policy gegenüber
    # der besten festen Regel, BLP-Heterogenitätstest, E-Wert, Anteil Fälle mit anderem Vorzeichen
    screening = R/"screening_alle/prescription/screening/control_flow/bpic17_screening_control_flow_detailliert.csv"
    t = pd.read_csv(screening).set_index("Kandidat")
    for p in PUNKTE:
        k = KURZ[p]
        out[f"optionsdiff_{k}"] = out[f"aipw_effekt_{k}"] = float(t.at[p, "Effekt (AIPW)"])
        out[f"effekt_anteil_{k}"] = 1.0
        for col, key in (("Effekt KI unten", "aipw_ki_unten"), ("Effekt KI oben", "aipw_ki_oben"), ("Policy-Gewinn", "aipw_policy_gewinn"),
                         ("BLP p", "aipw_blp_p"), ("E-Wert", "aipw_e_wert"), ("anderes Vorzeichen", "aipw_anderes_vorzeichen"),
                         ("Überlappung", "aipw_ueberlappung")):
            try:
                out[f"{key}_{k}"] = float(str(t.at[p, col]).replace("%", "").replace(",", ".")) / (100 if "%" in str(t.at[p, col]) else 1)
            except (ValueError, KeyError):
                pass
    return out


GANZZAHLIG = {"epoche_stand", "epochen_geplant", "epochen_trainiert", "batch", "decoderlaenge", "wahlpunkt_gewicht", "faelle_training",
              "faelle_validierung", "faelle_soll", "proben", "hidden", "schichten", "seed", "lauf", "epochen", "ziehungen", "seeds", "faelle_je_seed", "noetige_ziehungen_konzept", "noetige_ziehungen_nach_v", "noetige_ziehungen_nachlegen", "noetige_ziehungen_leads", "proben_validierung", "proben_wahlpunkt", "fensterbreite", "test_faelle", "eingabe_encoder", "eingabe_decoder", "parameter_anzahl", "ziehungen_standard"}
WISSENSCHAFTLICH = {"lernrate", "regularisierung"}


def tabelle(frame, columns, formats):
    """Markdown-Tabelle ohne Zusatzpakete; formats: Spalte -> 'p' (Prozent), 'd2' (2 Nachkommastellen), 's' (Text)."""
    names = [c for c in columns if c in frame.columns]
    lines = ["| " + " | ".join(names) + " |", "|" + "|".join("---" for _ in names) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for c in names:
            v = row[c]
            if isinstance(v, float) and np.isnan(v):
                cells.append("–")
            elif formats.get(c) == "p":
                cells.append(f"{100*v:.0f} %")
            elif c in GANZZAHLIG and isinstance(v, (int, float, np.integer, np.floating)):
                cells.append(f"{v:.0f}")
            elif c in WISSENSCHAFTLICH and isinstance(v, (int, float, np.integer, np.floating)):
                cells.append(f"{v:.0e}")
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:.2f}")
            else:
                cells.append(str(v).replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    specs = MODELLE + zaehler_staende() + beschraenkt_staende() + t_learner_staende() + piloten()
    losses = verlustkurven()
    losses.to_csv(OUT/"verlustkurven.csv", index=False)
    configs = konfigurationen(specs)
    decompositions = zerlegungen()
    # Gültigkeit der gezogenen Abläufe: Übergänge zwischen Events und Lifecycles innerhalb einer Aktivität (uebergaenge_pruefen.py)
    try:
        sys.path.insert(0, str(OUT))
        from uebergaenge_pruefen import pruefen
        transitions = pruefen(verbose=False)
    except Exception as error:
        transitions = pd.DataFrame({"Quelle": [f"(Fehler: {error})"]})
    names = {"D-Netz": "D"}
    validity = {names.get(r["Quelle"], r["Quelle"]): {"ungueltig_zwischen_je100": r.get("Verstöße zwischen je 100 Events"),
                                                      "ungueltig_innerhalb_je100": r.get("Verstöße innerhalb je 100 Events"),
                                                      "suffixe_mit_ungueltigem_uebergang": r.get("mit Verstoß zwischen Events"),
                                                      "suffixe_mit_ungueltigem_lifecycle": r.get("mit Verstoß innerhalb Aktivität")}
                for r in transitions.to_dict("records")}
    from online_sammeln import online_ergebnisse
    online = online_ergebnisse()
    online.to_csv(OUT/"online.csv", index=False)
    online["kurz"] = online.aufbau.str[0]
    rows = []
    for spec in specs:
        row = {"label": spec["label"], "tag": spec.get("tag", ""), "konfig": spec["konfig"], "pilot": spec.get("pilot", False)}
        row.update(configs.get(spec["label"], {}))
        row.update(startinfo(spec.get("tag", "")))
        row["epoche_stand"] = spec.get("stand", row.get("epochen_trainiert"))
        row.update(rauschen(spec["rauschen"]))
        row.update(vergleich(spec))
        row.update(suffix(spec))
        row.update(diagnose(spec))
        row.update(decompositions.get(spec["label"], {}))
        row.update(validity.get(spec["label"], {}))
        # bestes Online-Ergebnis je Aufbau (nur fertige Modelle; Zwischenstände wurden nicht online gemessen)
        if not spec.get("stand"):
            allowed = online[(online.label == spec["label"]) & ~online.variante.astype(str).str.contains("direkte Restzeitbewertung")]
            for letter, part in allowed.groupby("kurz"):
                best = part.loc[part.gewinn_tage.idxmax()]
                row[f"online_{letter}_gewinn"] = best.gewinn_tage
                row[f"online_{letter}_variante"] = best.variante
        if spec.get("verluste"):
            log, epoch = spec["verluste"]
            part = losses[(losses.log == log) & (losses.epoche == epoch)]
            if not part.empty:
                row.update(val_std=part.iloc[-1].get("val_std"), val_att=part.iloc[-1].get("val_att"), train_att=part.iloc[-1].get("train_att"))
        rows.append(row)
    truth = wahrheit()
    truth.update({k: v for k, v in decompositions.get("Wahrheit/Referenz", {}).items() if k.startswith(("zerl_vorteil", "zerl_pfad", "zerl_zeit"))})
    frame = pd.DataFrame([truth]+rows)
    frame.to_csv(OUT/"register.csv", index=False)

    full = frame[~frame.pilot.astype(bool)]
    pct = {c: "p" for c in frame.columns if c.startswith(("aipw_vorzeichen", "sim_richtig", "suffix_s", "abdeckung", "nachlegen_ohne"))}
    md = [f"# Ergebnisregister BPIC17-PSP (erzeugt {time.strftime('%Y-%m-%d %H:%M')})", "",
          "Erzeugt von `register_erstellen.py` aus den Ergebnisordnern; Deutungen in `befunde.md`; maschinenlesbar in `register.csv`, "
          "Verluste je Epoche in `verlustkurven.csv`. Erste Zeile jeder Tabelle = Wahrheit/Referenz, sofern vorhanden.", "",
          "**Messungen** (alle mit Durchlaufzeit als Ziel, Katalog `catalog_neue_punkte_zuteilung_random.yaml`):",
          "- Optionssignal (`psp_rauschen.py`): 200 echte Testentscheidungen je Wahlpunkt, 100 Ziehungen je Option; Anteil mit "
          "Vorzeichen wie AIPW auf dem echten Log (AIPW: konzept +5,2, nach V. +3,0, nachlegen −1,4, leads −2,2 Tage; Vorzeichen-Konvention "
          "der Datei), gepaarter SE je Entscheidung, Korrelation der Optionen (gemeinsame Zufallszahlen), gezogene vs. echte Durchlaufzeit.",
          "- Simulator-Wahrheit (`punkte_alle.py`): 60 Simulatorzustände je Gruppe, Vorteil der Regel-Option in Tagen und Anteil "
          "Zustände mit richtigem Vorzeichen, MC-SE je Entscheidung.",
          "- Gedächtnis: `nachlegen_erklaerung.py` (1. Besuch Nachlegen, extra minus senden: Restzeit, zusätzlich versendete Angebote, "
          "Nachfassen bis Validierung), `angebote_einfuegen.py` (zweites Angebot in echte Prefixe eingefügt: Restzeit B−A; richtige Richtung: "
          "positiv), `leads_gedaechtnis.py` (Restzeit ab erstem O_Sent, Lead bearbeitet minus zurückgezogen; echt +2,83 ± 1,11).",
          "- Suffixgüte (`psp_suffixguete.py`): 1.000 Testprefixe, 100 Ziehungen.", "",
          "## Konfigurationen", "", tabelle(full, ["label", "tag", "konfig"], {}), "",
          "## Trainingseinstellungen (aus den gespeicherten Modellen ausgelesen)", "",
          "Teacher-Forcing-Plan des Trainers je Epoche: max(tf_untergrenze, 1 − Epoche/(0,5 · epochen_geplant)); tf_untergrenze 1 = immer "
          "Teacher Forcing. epoche_stand = ausgewerteter Stand (bei Checkpoints die Epoche, sonst das fertige Modell). weitere_params = "
          "alle übrigen gesetzten Parameter (Zeitdarstellung, Option, Zähler usw.); leer = Standard (normale Abstände, Option +1 ms "
          "nach dem Entscheidungspunkt). Checkpoints haben keine Trainingszusammenfassung (Fallzahlen wie beim fertigen Lauf).", "",
          tabelle(full, ["label", "epoche_stand", "epochen_geplant", "epochen_trainiert", "lernrate", "batch", "regularisierung",
                         "decoderlaenge", "wahlpunkt_gewicht", "tf_untergrenze", "zeit_aus", "faelle_training", "faelle_validierung",
                         "hidden", "schichten", "dropout", "seed", "statische_merkmale", "val_bester", "val_letzte"], {}), "",
          tabelle(full, ["label", "weitere_params", "modelldatei", "modell_datum"], {}), "",
          "### Netzaufbau und Daten (aus dem gespeicherten Netz)", "",
          "Merkmale mit Vokabulargröße in Klammern (concept:name = Aktivität|Lifecycle-Token, bei OE mit Zeitklasse); fensterbreite = "
          "Prefixfenster; proben_wahlpunkt = Proben an den Wahlpunkten (mit wahlpunkt_gewicht gewichtet); ziehungen_standard = "
          "Ziehungen je Option, wenn die Auswertung nichts anderes setzt.", "",
          tabelle(full, ["label", "encoder_merkmale", "decoder_merkmale", "statische_eingaben", "ausgaben", "eingabe_encoder",
                         "eingabe_decoder", "parameter_anzahl"], {}), "",
          tabelle(full, ["label", "fensterbreite", "proben", "proben_validierung", "proben_wahlpunkt", "test_faelle", "wahlpunkte",
                         "ziehungen_standard"], {}), "",
          "### Exakter Trainingsaufruf (Parameter laut Trainingslog, Startskripte in prescriptive/data)", "",
          tabelle(full, ["label", "trainingsparameter_log", "startskripte", "trainingslog"], {}), "",
          "### Für alle mit PSPModel.fit trainierten Modelle gleich (Code-Stand, prescriptive/estimators/psp_model.py `fit`)", "",
          "- Optimierer AdamW (weight_decay 0), Lernrate wie oben; ReduceLROnPlateau (factor 0,1, patience 15, min_lr 1e-8) auf dem "
          "Validierungsverlust; Batchgröße, Regularisierung, Epochen wie oben; Mischen der Proben an.",
          "- Verlust: Loss Attenuation (gelernte Varianz je Ausgabe), GradNorm (alpha 1,5, eigene Lernrate 1e-4, eine Aufgabe je Ausgabe).",
          "- Teacher Forcing je Epoche von 1,0 linear abnehmend bis tf_untergrenze (Trainer des U-ED-LSTM); Masken für Auffüllung und EOS an.",
          "- Netz: U-ED-LSTM mit MC-Dropout (Variational Dropout, gleiche Maske über die Zeitschritte), Gewichte je Epoche in last.pt.",
          "- Ältere Modelle (10k, s40) können mit einem früheren Code-Stand trainiert sein; ihre Parameter oben stammen aus dem Modell selbst.", "",
          "## Optionssignal offline", "",
          tabelle(full, ["label", "aipw_vorzeichen_konzept", "aipw_vorzeichen_nach_v", "aipw_vorzeichen_nachlegen", "aipw_vorzeichen_leads",
                         "se_gepaart", "korrelation_optionen", "dlz_gezogen", "dlz_sd_gezogen", "dlz_echt"], pct), "",
          "## Effektgrößen offline (gleiche 200 Testentscheidungen je Punkt, 100 Ziehungen)", "",
          "optionsdiff = mittlerer Optionsunterschied des Modells (Aktion minus Referenz, Tage, Vorzeichen wie AIPW); effekt_anteil = "
          "optionsdiff / AIPW-Effekt (1 = volle Effektgröße, 0 = kein Signal, negativ = falsches Vorzeichen); diff_betrag_median = typischer "
          "Betrag des Unterschieds je Entscheidung. Erste Zeile: AIPW auf dem echten Log (Screening). Simulator-Wahrheit (RCT) im nächsten "
          "Abschnitt: konzept 2,79, nach V. 2,40, nachlegen 2,11 / 0,42, leads 1,32 Tage Vorteil der Regel-Option.", "",
          tabelle(full, ["label"] + [f"{a}_{p}" for p in ("konzept", "nach_v", "nachlegen", "leads") for a in ("optionsdiff", "effekt_anteil",
                                                                                                      "diff_betrag_median")], {}), "",
          "AIPW-Details (echter Log): Effekt mit 95-%-KI, Policy-Gewinn einer Einzelfall-Policy (DR-Learner) gegenüber der besten "
          "festen Regel (negativ = Einzelfallentscheidung lohnt nicht), BLP-p (Heterogenität), E-Wert, Überlappung.", "",
          tabelle(frame.iloc[:1], ["label"] + [f"{a}_{p}" for p in ("konzept", "nach_v", "nachlegen", "leads") for a in
                                               ("aipw_effekt", "aipw_ki_unten", "aipw_ki_oben", "aipw_policy_gewinn", "aipw_blp_p", "aipw_e_wert")], {}), "",
          "## Entscheidungssicherheit offline (z = Optionsunterschied / gepaarter SE je Entscheidung)", "",
          "anteil_z_gt2 = Anteil Entscheidungen mit |z| > 2 (sicher); vorzeichen_z_gt2 = davon Vorzeichen wie AIPW; anteil_aktion = Anteil "
          "Entscheidungen für die Aktion nach dem Mittel; noetige_ziehungen = Median der Ziehungen, die für |z| = 2 nötig wären.", "",
          tabelle(full, ["label"] + [f"{a}_{p}" for p in ("konzept", "nach_v", "nachlegen", "leads") for a in
                                     ("anteil_z_gt2", "vorzeichen_z_gt2", "anteil_aktion", "noetige_ziehungen")],
                  {c: "p" for c in frame.columns if c.startswith(("anteil_z_gt2", "vorzeichen_z_gt2", "anteil_aktion"))}), "",
          "## Gegen die Simulator-Wahrheit (Vorteil Regel-Option, Tage / Anteil richtig)", "",
          tabelle(full, ["label", "sim_vorteil_konzept", "sim_richtig_konzept", "sim_vorteil_nach_v", "sim_richtig_nach_v",
                         "sim_vorteil_nachlegen1", "sim_richtig_nachlegen1", "sim_vorteil_nachlegen2", "sim_richtig_nachlegen2",
                         "sim_vorteil_leads", "sim_richtig_leads", "sim_mcse"], pct), "",
          "## Gedächtnis (Angebote, Leads)", "",
          tabelle(full, ["label", "nachlegen_restzeit", "nachlegen_versendet", "nachlegen_nachfassen", "nachlegen_ohne_ende",
                         "angebot_eingefuegt_rest", "lead_gedaechtnis"], pct), "",
          "## Suffixgüte (1.000 Testprefixe, 100 Ziehungen)", "",
          "suffix_sK = Anteil Prefixe, bei denen das häufigste gezogene Token an Position K stimmt; DLS = mittlere normierte "
          "Damerau-Levenshtein-Ähnlichkeit der Einzelziehungen; Restzeit ab dem letzten Prefix-Event: MAE, Bias (gezogen − echt), CRPS, "
          "Anteil echter Restzeiten im 80-%-Intervall.", "",
          tabelle(full, ["label", "suffix_s1", "suffix_s2", "suffix_s3", "suffix_s5", "suffix_s10", "suffix_s20", "suffix_dls",
                         "suffix_laenge_mae"], pct), "",
          tabelle(full, ["label", "suffix_erster_abstand_mae_h", "restzeit_mae", "restzeit_bias", "restzeit_crps", "abdeckung80",
                         "restzeit_gezogen", "restzeit_echt", "suffix_quelle"], pct), "",
          "## Gültigkeit der gezogenen Abläufe (uebergaenge_pruefen.py; Verläufe ab dem 1. Besuch Nachlegen)", "",
          "Übergangsmodelle aus 80 % der echten Fälle: zwischen Events (Token → nächstes Token, wie die Maske des Simulators) und "
          "innerhalb einer Aktivität (letzter Lifecycle derselben Aktivität → nächster, z. B. suspend → resume; A_Pending nur einmal). "
          "Die übrigen 20 % echten Fälle zeigen die Fehlalarm-Rate (ungesehen, aber echt).", "",
          tabelle(transitions, list(transitions.columns), {c: "p" for c in transitions.columns if c.startswith("mit Verstoß")}), "",
          "## Diagnose an echten Entscheidungen (psp_diagnose, 100 je Punkt, Mittel über die vier Punkte)", "",
          "diag_mae_faktisch / diag_korrelation_faktisch: gezogene Durchlaufzeit der tatsächlich gewählten Option gegen die echte; "
          "diag_diff_nach_5_schritten: Optionsunterschied nach 5 Schritten; diag_suffixlaenge: mittlere gezogene Suffixlänge.", "",
          tabelle(full, ["label", "diag_mae_faktisch", "diag_korrelation_faktisch", "diag_diff_nach_5_schritten", "diag_suffixlaenge"], {}), "",
          "## Zerlegung des Optionsvorteils in Pfad und Zeit (psp_zerlegung, echte Testentscheidungen)", "",
          "zerl_vorteil = Vorteil der Aktion (Tage), davon zerl_pfad (andere Abfolge) und zerl_zeit (andere Abstände); erste Zeile = "
          "beobachtete Gruppenunterschiede im echten Log (nicht kausal). zerl_ohne_ende = Anteil Ziehungen ohne Endzustand.", "",
          tabelle(frame, ["label"] + [f"zerl_{a}_{p}" for p in ("konzept", "nach_v", "nachlegen", "leads") for a in ("vorteil", "pfad", "zeit")]
                  + ["zerl_ohne_ende"], {"zerl_ohne_ende": "p"}).replace("| nan |", "| – |"), "",
          "## Verluste der Prüfstände (Epoche des Stands)", "",
          tabelle(full, ["label", "train_att", "val_std", "val_att"], {}), ""]
    pilots = frame[frame.pilot.astype(bool)]
    if not pilots.empty:
        md += ["## Piloten (3.000 Fälle, 6 Epochen): Optionssignal", "",
               tabelle(pilots, ["label", "aipw_vorzeichen_konzept", "aipw_vorzeichen_nach_v", "aipw_vorzeichen_nachlegen", "aipw_vorzeichen_leads",
                                "se_gepaart", "korrelation_optionen"], pct), "",
               "Einstellungen der Piloten:", "",
               tabelle(pilots, ["label", "epochen_trainiert", "lernrate", "decoderlaenge", "tf_untergrenze", "faelle_training", "zeit_aus",
                                "weitere_params", "modelldatei"], {}), ""]
    md += ["## Online-Ergebnisse (alle Läufe, Langformat in online.csv mit Quelle je Zeile)", "",
           "gewinn_tage = Durchlaufzeit Bank − Policy im selben Aufbau (positiv = schneller). Nur innerhalb eines Aufbaus vergleichen: "
           "Simulatorstand, Zuteilung, Fallzahl und Ziel unterscheiden sich. In register.csv steht je Modell und Aufbau die beste Variante "
           "(online_<Buchstabe>_gewinn / _variante). (b) „direkte Restzeitbewertung“ nutzt den Restzeitkopf (für PSP nicht mehr erlaubt).", ""]
    letters = sorted(c.removeprefix("online_").removesuffix("_gewinn") for c in frame.columns if c.startswith("online_") and c.endswith("_gewinn"))
    names = {a[0]: a for a in online.aufbau.dropna()}
    md += ["Beste erlaubte Variante je Modell und Aufbau (Gewinn in Tagen; Aufbauten: " + "; ".join(f"{k} = {names[k]}" for k in letters) + "):", "",
           tabelle(full[full[[f"online_{k}_gewinn" for k in letters]].notna().any(axis=1)], ["label"] + [f"online_{k}_gewinn" for k in letters], {}), ""]
    for aufbau, part in online.groupby("aufbau", sort=True):
        md += [f"### {aufbau}", "", tabelle(part.sort_values("gewinn_tage", ascending=False),
                                            ["label", "variante", "ziel", "ziehungen", "seeds", "faelle_je_seed", "dlz_tage", "bank_tage",
                                             "gewinn_tage", "se_tage", "causal_forest_gewinn_gleiche_seeds", "angebote_je_fall", "kosten_eur",
                                             "bank_kosten_eur"], {}), ""]
    from simulator_sammeln import simulator_kennzahlen, markdown as simulator_markdown
    simulator = simulator_kennzahlen()
    simulator.to_csv(OUT/"simulator.csv", index=False)
    md += simulator_markdown(simulator, tabelle)
    runs = losses.dropna(subset=["val_std"]).groupby(["log", "lauf"]).agg(epochen=("epoche", "max"), val_std_letzte=("val_std", "last"),
                                                                           val_std_min=("val_std", "min"), val_att_letzte=("val_att", "last"))
    md += ["## Trainingsläufe (aus data/PSP_*.log; Einzelwerte je Epoche in verlustkurven.csv)", "",
           tabelle(runs.reset_index(), ["log", "lauf", "epochen", "val_std_letzte", "val_std_min", "val_att_letzte"], {}), ""]
    (OUT/"register.md").write_text("\n".join(md))
    try:  # Bild EXTRAZWISCHENERGEBNISSE/2017/suffix_vorhersage_bpic17.png mit allen Läufen aktuell halten
        import suffix_tabelle_bild
        import online_tabelle_bild
        import aipw_richtung_bild
        suffix_tabelle_bild.main()
        online_tabelle_bild.main()
        aipw_richtung_bild.main()
    except Exception as error:
        print(f"Suffix-Bild nicht erneuert: {error}")
    print(f"register.csv ({len(frame)} Zeilen), register.md, verlustkurven.csv ({len(losses)} Epochen) geschrieben")


if __name__ == "__main__":
    main()
