"""Online-Ergebnisse aller BPIC17-Läufe in einem Langformat (eine Zeile je Policy-Variante und Aufbau), für register_erstellen.py.

Gewinn = Durchlaufzeit der Bank minus Durchlaufzeit der Policy im selben Aufbau (positiv = schneller als die Bank). Die Aufbauten
sind untereinander NICHT vergleichbar (anderer Simulatorstand, Zuteilung, Fallzahl, Ziehungen); verglichen wird nur innerhalb
eines Aufbaus. Neue Läufe: unten in QUELLEN bzw. in einer der Funktionen ergänzen.
"""
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
R = OUT.parent
P = R.parents[1]
G = R/"psp_gesamtuebersicht_20261114"
U = R/"psp_online_uebersicht_20261113"
V = R/"modellvergleich_20261116"
E = P/"policies/ev_policy_event_costs/results"

TAGS = {"real_w3_10k_neuepunkte": "10k Original", "real_w3_10k_np": "10k neue Punkte", "real_np_s40_ohne_zukunft": "s40",
        "real_np_s40_ohne_zukunft_abstaende": "s40", "real_alle_s60_loggaps": "(a)", "real_alle_s60_loggaps_rest": "(b)",
        "real_alle_s60_normal": "(c)"}
POLICIES = {"bank": "Bank", "regel": "Regel (AIPW)", "xgb": "XGB", "s_learner": "S-Learner", "x_learner": "X-Learner",
            "causal_forest": "Causal Forest", "causal_forest_erm": "Causal Forest", "s_learner_erm": "S-Learner", "x_learner_erm": "X-Learner"}


def _row(label, variante, aufbau, tage, bank, se=np.nan, quelle="", ziel="Durchlaufzeit", ziehungen=np.nan, seeds=np.nan, faelle=np.nan, **extra):
    return dict(label=label, variante=variante, aufbau=aufbau, ziel=ziel, ziehungen=ziehungen, seeds=seeds, faelle_je_seed=faelle,
                dlz_tage=tage, bank_tage=bank, gewinn_tage=bank-tage, se_tage=se, quelle=str(Path(quelle).relative_to(P.parent)) if quelle else "",
                **extra)


def alt_10_seeds(rows):
    path = G/"online_10_seeds.csv"
    for r in pd.read_csv(path).itertuples():
        se = (r.uplift_ci95_high-r.uplift_ci95_low)/(2*1.96)*r.bank_cycle_days/100
        rows.append(_row(TAGS.get(r.model_tag, r.model_tag), r.variant, "A: älterer Simulator, 10 Seeds × 1.000 Fälle", r.cycle_days,
                         r.bank_cycle_days, se, r.source, ziehungen=r.samples, seeds=r.seeds, faelle=r.scored_cases_per_seed,
                         p90=r.p90_days))


def cate_12_11(rows):
    path = R/"neue_punkte_online_cate/vergleich_gepaart.csv"
    t = pd.read_csv(path)
    bank = float(t.loc[t.Policy == "bank", "Durchlaufzeit (Tage)"].iloc[0])
    for r in t.itertuples():
        if r.Policy == "bank":
            continue
        m = re.match(r"([+-][\d.]+) \(SE ([\d.]+)", str(r[3]))
        label = "(c)" if r.Policy.startswith("psp") else POLICIES.get(r.Policy, r.Policy)
        rows.append(_row(label, r.Policy, "A: älterer Simulator, 10 Seeds × 1.000 Fälle (Lauf 12.11. mit CATE)", r[2], bank,
                         float(m[2]) if m else np.nan, path, ziehungen=100 if label == "(c)" else np.nan, seeds=10, faelle=1000, p90=r[6]))


def erm(rows):
    path = R/"neue_punkte_online_psp_erm_gamma0.1_suffixe100_bank/vergleich_alle_erm.csv"
    t = pd.read_csv(path)
    bank = float(t.loc[t.Policy == "bank", "Durchlaufzeit (Tage)"].iloc[0])
    for r in t.itertuples():
        if r.Policy == "bank":
            continue
        label = "(c)" if r.Policy.startswith("psp") else POLICIES.get(r.Policy, r.Policy)
        rows.append(_row(label, r.Policy, "B: gelernte Zuteilung, 10 Seeds × 1.000 Fälle, ERM-Ziel γ=0,1", r[5], bank, quelle=path,
                         ziel="ERM γ=0,1 (Durchlaufzeit)", ziehungen=100 if label == "(c)" else np.nan, seeds=10, faelle=1000))


def json_row(folder):
    psp, bank = folder/"psp_seed0_row.json", folder/"bank_seed0_row.json"
    if not (psp.exists() and bank.exists()):
        return None
    a, b = json.loads(psp.read_text()), json.loads(bank.read_text())
    return a["Durchlaufzeit (Tage)"], b["Durchlaufzeit (Tage)"], a.get("Fälle")


def seed0_200(rows):
    path = G/"online_200_faelle_30_ziehungen.csv"
    for r in pd.read_csv(path).itertuples():
        label = "(c)" if r.variant.startswith("(c)") else "D" if r.variant.startswith("(d)") else "OE"
        rows.append(_row(label, r.variant, "C: 1 Seed × 200 Fälle (Seed 0), 30 Ziehungen", r.cycle_days, r.bank_cycle_days, quelle=r.source,
                         ziehungen=30, seeds=1, faelle=200, p90=r.p90_days))
    for label, folder in (("(c)-TF1", R/"c_tf1_suffixe30_cases200_seed0_20261114/c_tf1"), ("(e)", R/"e_suffixe30_cases200_seed0_20261113/e")):
        got = json_row(folder)
        if got:
            rows.append(_row(label, folder.name, "C: 1 Seed × 200 Fälle (Seed 0), 30 Ziehungen", got[0], got[1], quelle=folder/"psp_seed0_row.json",
                             ziehungen=30, seeds=1, faelle=got[2]))


def seed0_1000(rows):
    path = G/"online_1000_faelle_300_ziehungen.csv"
    for r in pd.read_csv(path).itertuples():
        label = "D" if r.variant.startswith("(d)") else "OE"
        rows.append(_row(label, r.variant, "D: 1 Seed × 1.000 Fälle (Seed 0), 300 Ziehungen", r.cycle_days, r.bank_cycle_days, quelle=r.source,
                         ziehungen=300, seeds=1, faelle=1000, p90=r.p90_days))
    folder = R/"neue_punkte_online_token_oe_ue_suffixe300_seed0_v2"
    got = json_row(folder)
    if got:
        rows.append(_row("OE", "OE mit Unsicherheitsrückfall (token_oe_ue)", "D: 1 Seed × 1.000 Fälle (Seed 0), 300 Ziehungen", got[0], got[1],
                         quelle=folder/"psp_seed0_row.json", ziehungen=300, seeds=1, faelle=got[2]))


def seeds_300(rows):
    path = U/"online_300_fertige_seeds.csv"
    t = pd.read_csv(path)
    t["gewinn"] = t.bank_days-t.cycle_days_approx
    t["cf"] = t.bank_days-t.cf_days
    for variant, part in t.groupby("variant"):
        label = "(c)" if variant.startswith("(c)") else "OE"
        rows.append(_row(label, variant, "E: ältere Zuteilung, fertige Seeds × 1.000 Fälle, 300 Ziehungen (Werte gerundet)",
                         part.cycle_days_approx.mean(), part.bank_days.mean(), part.gewinn.std()/np.sqrt(len(part)), path, ziehungen=300,
                         seeds=len(part), faelle=1000, causal_forest_gewinn_gleiche_seeds=part.cf.mean()))


def korrigiert(rows):
    path = V/"tabelle_korrigierter_simulator.csv"
    t = pd.read_csv(path)
    bank = float(t.loc[t.Policy == "Bank", "Tage"].iloc[0])
    names = {"XGB Zeit": ("XGB", "Zeit"), "CF Kosten": ("Causal Forest", "Ziel Gesamtkosten")}
    for r in t.itertuples():
        if r.Policy == "Bank":
            continue
        if r.Policy in names:
            label, variant = names[r.Policy]
        elif r.Policy.startswith("(c)"):
            label, variant = "(c)", r.Policy.removeprefix("(c) ")+" (Original-Netz, Decoder-Zeiten)"
        else:
            label, variant = r.Policy.split()[0], r.Policy.split(" ", 1)[1]+" (zustandsbewusste Kopie: Arbeitszustand + empirische Abstandstabelle)"
        rows.append(_row(label, variant, "F: korrigierter Simulator, 3 Seeds × 200 Fälle, 100 Ziehungen", r.Tage, bank, r.SE, path,
                         ziehungen=100 if label in ("(c)", "D", "OE") else np.nan, seeds=3, faelle=200, angebote_je_fall=r[-2]))


def _angebote(folders):
    """Mittlere Zahl O_Created je bewertetem Fall über die Lauf-Ordner (events.pkl + case_costs.csv); NaN, wenn nichts da ist."""
    values = []
    for run in folders:
        if (run/"events.pkl").exists() and (run/"case_costs.csv").exists():
            events = pd.read_pickle(run/"events.pkl")
            cases = pd.read_csv(run/"case_costs.csv", index_col=0).index
            events = events[events["case:concept:name"].isin(cases)]
            values.append(float((events["concept:name"] == "O_Created").sum() / len(cases)))
    return float(np.mean(values)) if values else np.nan


def korrigiert_cate(rows):
    """Aufbau F: Causal Forest und X-Learner mit Ziel Durchlaufzeit, feste Regel (cf_im_korrigierten_simulator.py, 17.11.)."""
    path = V/"cf_im_korrigierten_simulator/runs.csv"
    if not path.exists():
        return
    t = pd.read_csv(path)
    names = {"causal_forest_zeit": ("Causal Forest", "Ziel Durchlaufzeit"), "x_learner_zeit": ("X-Learner", "Ziel Durchlaufzeit"),
             "regel": ("Regel (AIPW)", "feste AIPW-Regel")}
    for arm, part in t.groupby("arm"):
        gain = part.bank_cycle_days - part.cycle_days
        label, variant = names.get(arm, (arm, arm))
        rows.append(_row(label, variant, "F: korrigierter Simulator, 3 Seeds × 200 Fälle, 100 Ziehungen", part.cycle_days.mean(),
                         part.bank_cycle_days.mean(), gain.std() / np.sqrt(len(gain)), path, seeds=len(part), faelle=200,
                         angebote_je_fall=_angebote([path.parent/f"{arm}_seed{seed}" for seed in part.seed])))


def korrigiert_first_only(rows):
    """Aufbau F: (c) mit first_only = angebot_nachlegen (nur 1. Besuch entscheidet, danach Bank; c_first_only_im_korrigierten_simulator.py,
    17.11.). Nur fertige Arme mit allen 3 Seeds; Angebote je Fall aus events.pkl der bewerteten Fälle."""
    names = {"fest100": "nur 1. Besuch Nachlegen (first_only), fest 100", "adaptiv_ohne": "nur 1. Besuch Nachlegen (first_only), adaptiv ohne",
             "adaptiv_mit": "nur 1. Besuch Nachlegen (first_only), adaptiv mit",
             "fest300": "nur 1. Besuch Nachlegen (first_only), 300 Ziehungen fest"}  # Kurzformen wie in online_tabelle_bild.kurz
    for folder in (V/"c_first_only_im_korrigierten_simulator", V/"c_first_only_300_im_korrigierten_simulator"):
        if (folder/"runs.csv").exists():
            _first_only_rows(rows, folder, names)


def _first_only_rows(rows, folder, names):
    path = folder/"runs.csv"
    t = pd.read_csv(path)
    for arm, part in t.groupby("arm"):
        if len(part) < 3:
            continue
        gain = part.bank_cycle_days - part.cycle_days
        rows.append(_row("(c)", names.get(arm, arm), "F: korrigierter Simulator, 3 Seeds × 200 Fälle, 100 Ziehungen", part.cycle_days.mean(),
                         part.bank_cycle_days.mean(), gain.std() / np.sqrt(len(gain)), path, ziehungen=int(arm[4:]) if arm.startswith("fest") else 100,
                         seeds=len(part), faelle=200, angebote_je_fall=_angebote([folder/f"{arm}_seed{seed}" for seed in part.seed])))


def zwoelf_seeds(rows):
    """Aufbau L = F auf 12 Seeds × 200 Fälle (17.11., f_zwoelf_seeds.py / f_zwoelf_auswerten.py): Bank, Regel, (c) 100 / 300 fest
    mit first_only, später ZZ-TF1. Gewinn gepaart je Seed gegen die Bank, SE über die Seeds."""
    path = V/"f_zwoelf_auswertung.csv"
    if not path.exists():
        return
    t = pd.read_csv(path)
    bank = float(t.loc[t.arm == "bank", "dlz_tage"].iloc[0])
    names = {"regel": ("Regel (AIPW)", "feste AIPW-Regel", np.nan), "c_fest100": ("(c)", "first_only, fest 100", 100),
             "c_fest300": ("(c)", "first_only, 300 Ziehungen fest", 300), "zztf1_fest100": ("ZZ-TF1", "first_only, fest 100", 100),
             "tl_fest300": ("T-Learner (ZZ-TF1)", "first_only, 300 Ziehungen fest", 300),
             "tl_fest500": ("T-Learner (ZZ-TF1)", "first_only, 500 Ziehungen fest", 500),
             "zzrct_fest100": ("ZZ-TF1 (RCT)", "auf Simulator-RCT-Daten trainiert, first_only, fest 100", 100),
             "tlrct_fest300": ("T-Learner (RCT)", "auf Simulator-RCT-Daten trainiert, first_only, 300 Ziehungen fest", 300),
             "tlbnll_fest300": ("T-Learner (β-NLL)", "warm vom β-NLL-Modell (ZZU-TF1 + β-NLL 0,5), first_only, 300 Ziehungen fest", 300),
             "tlks_fest300": ("T-Learner Kopf je Event (Steigung)", "Zeitkopf je Event (Versatz + Steigung), nur Kopf nachtrainiert, first_only, 300 Ziehungen fest", 300),
             "tlko_fest300": ("T-Learner Kopf je Event (Versatz)", "Zeitkopf je Event (nur Versatz), nur Kopf nachtrainiert, first_only, 300 Ziehungen fest", 300),
             "tlkb_fest300": ("T-Learner Kopf je Event (Steigung, β-NLL 1)", "Zeitkopf je Event (Versatz + Steigung), nur Kopf mit β-NLL 1 nachtrainiert, first_only, 300 Ziehungen fest", 300),
             "tlcrn_fest300": ("T-Learner (ZZ-TF1)", "Zufallszahlen je Ereignis (keyed_noise), first_only, 300 Ziehungen fest", 300),
             "tlsv_fest300": ("T-Learner Suffix-Verlust (Zufallszahlen je Event)", "Zufallszahlen je Ereignis (keyed_noise), first_only, 300 Ziehungen fest", 300),
             "tlsvk_fest300": ("T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event)", "Zufallszahlen je Ereignis (keyed_noise), first_only, 300 Ziehungen fest", 300),
             "tlsvp_fest300": ("T-Learner Suffix-Verlust + gepaarte Kalibrierung (Zufallszahlen je Event)", "Zufallszahlen je Ereignis (keyed_noise), first_only, 300 Ziehungen fest", 300),
             "tlsvh_fest300": ("T-Learner Suffix-Verlust + Kalibrierung Hälften (Zufallszahlen je Event)", "Zufallszahlen je Ereignis (keyed_noise), first_only, 300 Ziehungen fest", 300),
             "olsv_fest300": ("Netz je Wahlpunkt und Option + Suffix-Verlust (Zufallszahlen je Event)", "Zufallszahlen je Ereignis (keyed_noise), first_only, 300 Ziehungen fest", 300),
             "plsv_fest300": ("S-Learner je Wahlpunkt + Suffix-Verlust (Zufallszahlen je Event)", "Zufallszahlen je Ereignis (keyed_noise), first_only, 300 Ziehungen fest", 300),
             "olsvcf_fest300": ("Netz je Wahlpunkt und Option + Suffix-Verlust, Causal Forest bei Unsicherheit", "keyed_noise, first_only, 300 Ziehungen fest; |Δ| < 1,96·SE → Causal Forest", 300)}
    for r in t.itertuples():
        if r.arm not in names:
            continue
        label, variant, draws = names[r.arm]
        rows.append(_row(label, variant, "L: korrigierter Simulator, 12 Seeds × 200 Fälle, first_only (17.11.)", r.dlz_tage, bank,
                         r.gewinn_se, path, ziehungen=draws, seeds=r.seeds, faelle=200, angebote_je_fall=r.angebote_je_fall))


def eventkosten(rows):
    for folder, label, variant in ((E/"vergleich_200_faelle_30_ziehungen_seed0_20261113/online", None, "30 Ziehungen"),
                                   (E/"psp_d_200_faelle_300_ziehungen_seed0_20261114_neustart/online", "D", "300 Ziehungen"),
                                   (E/"psp_d_ue_200_faelle_300_ziehungen_seed0_20261114/online", "D", "300 Ziehungen, Unsicherheitsrückfall"),
                                   (E/"psp_d_ue_200_faelle_500_ziehungen_seed0_20261114/online", "D", "500 Ziehungen, Unsicherheitsrückfall")):
        path = folder/"comparison.csv"
        if not path.exists():
            continue
        t = pd.read_csv(path)
        bank = t[t.Policy == "bank"].iloc[0]
        for _, r in t[t.Policy != "bank"].iterrows():
            name = "D" if r.Policy == "psp" else POLICIES.get(r.Policy, r.Policy)
            draws = int(re.search(r"(\d+) Ziehungen", variant)[1]) if name == "D" else np.nan
            rows.append(_row(name, f"{r.Policy}, {variant}" if name == "D" else r.Policy, "G: Eventkosten-Ziel, 1 Seed × 200 Fälle (Seed 0)",
                             r["Durchlaufzeit (Tage)"], bank["Durchlaufzeit (Tage)"], quelle=path, ziel="Eventkosten (€)", ziehungen=draws,
                             seeds=1, faelle=200, kosten_eur=r["Outcome je Fall"]*-1, bank_kosten_eur=bank["Outcome je Fall"]*-1))


S = P/"scripts/results"  # Läufe vor dem 30.10. (eigener Ordner)


def laeufe(rows, path, aufbau, policies, bank_path=None, **extra):
    """Gepaart je Seed aus einer *_online_laeufe.csv: Gewinn = Bank − Policy, SE über die Seeds. policies: Name → (Label, Variante)."""
    t = pd.read_csv(path)
    wide = t.pivot_table(index="seed", columns="Policy", values="Durchlaufzeit (Tage)")
    if "bank" not in wide.columns:
        wide["bank"] = pd.read_csv(bank_path).pivot_table(index="seed", columns="Policy", values="Durchlaufzeit (Tage)")["bank"]
    cases = int(t["Fälle"].mean())
    for policy, (label, variant) in policies.items():
        if policy not in wide.columns:
            continue
        gain = (wide["bank"] - wide[policy]).dropna()
        rows.append(_row(label, variant, aufbau, wide[policy].mean(), wide["bank"].mean(), gain.std() / np.sqrt(len(gain)), path,
                         seeds=len(gain), faelle=cases, **extra))


def alte_aufbauten(rows):
    R1 = R
    zwoelf = "H: 12 Interventionspunkte (Kontrollfluss + 8 Mitarbeiterpunkte), älterer Simulator, 10 Seeds × 1.000 Fälle (Anfang Nov.)"
    bank12 = R1/"PSP_lr1e4_xgb/prescription/XGBoost/bpic17_online_laeufe.csv"
    laeufe(rows, bank12, zwoelf, {"random": ("Zufall", "zufällige Wahl"), "xgb": ("XGB", "XGB, 5.000 Fälle")})
    laeufe(rows, R1/"PSP_lr1e4_online/prescription/PSP/bpic17_online_laeufe.csv", zwoelf,
           {"psp": ("PSP Nachtlauf", "PSP 5.000 Fälle, 25 Ep., 1e-4, Decoder 5")}, bank12, ziehungen=30)
    laeufe(rows, R1/"PSP_real_w3_10k_online/prescription/PSP/bpic17_online_laeufe.csv", zwoelf,
           {"psp": ("10k Original", "PSP 10.000 Fälle, 1e-5, Decoder 5")}, bank12, ziehungen=30)
    laeufe(rows, R1/"PSP_real_w3_10k_online_channel/prescription/PSP/bpic17_online_laeufe.csv", zwoelf,
           {"psp": ("10k Original", "PSP 10k, Kanal statt Kanal+Nachlegen")}, bank12, ziehungen=30)
    laeufe(rows, R1/"PSP_real_w3_10k_online_nur_angebot/prescription/PSP/channel_angebot_nachlegen/bpic17_online_laeufe.csv", zwoelf,
           {"psp": ("10k Original", "PSP 10k, nur Kanal+Nachlegen")}, bank12, ziehungen=30)
    laeufe(rows, R1/("PSP_real_w3_10k_online_nur_controlflow/prescription/PSP/channel_angebot_nachlegen_handle_leads_annehmen_nach_"
                     "vervollstaendigung_neues_angebot_beim_nachfassen/bpic17_online_laeufe.csv"), zwoelf,
           {"psp": ("10k Original", "PSP 10k, nur die 4 Kontrollfluss-Punkte")}, bank12, ziehungen=30)
    laeufe(rows, S/"prescription/XGBoost/bpic17_online_laeufe.csv", "I: Kanal + 2 Mitarbeiterpunkte, 5 Seeds × 2.000 Fälle (28.10.)",
           {"random": ("Zufall", "zufällige Wahl"), "xgb": ("XGB", "XGB")})
    laeufe(rows, S/"prescription/LSTM/bpic17_online_laeufe.csv", "I: Kanal + 2 Mitarbeiterpunkte, 5 Seeds × 2.000 Fälle (28.10.)",
           {"lstm": ("LSTM", "LSTM")})
    laeufe(rows, S/"prescription/PSP/Mitarbeiter_W_Call_incomplete_files/bpic17_online_laeufe.csv",
           "J: nur Mitarbeiterwahl W_Call incomplete files, 5 Seeds × 2.000 Fälle", {"random": ("Zufall", "zufällige Wahl"), "psp": ("PSP", "PSP")})
    laeufe(rows, S/"prescription/PSP_Vorausschau/Mitarbeiter_W_Call_incomplete_files/bpic17_online_laeufe.csv",
           "J: nur Mitarbeiterwahl W_Call incomplete files, 5 Seeds × 2.000 Fälle", {"psp_lookahead": ("PSP", "PSP mit Vorausschau")})
    laeufe(rows, S/"bpic17_online_laeufe.csv", "K: Mitarbeiterwahl, Simulator-RCT-Modell, 5 Seeds × 2.000 Fälle (27.10.)",
           {"random_resource": ("Zufall", "zufälliger Mitarbeiter"), "xgb_rct_lc": ("XGB", "XGB auf Simulator-RCT trainiert")})


def online_ergebnisse():
    rows = []
    for function in (alt_10_seeds, cate_12_11, erm, seed0_200, seed0_1000, seeds_300, korrigiert, korrigiert_cate, korrigiert_first_only, zwoelf_seeds, eventkosten, alte_aufbauten):
        try:
            function(rows)
        except Exception as error:  # eine kaputte Quelle soll das Register nicht verhindern
            rows.append(dict(label="(Fehler)", variante=function.__name__, aufbau=str(error)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(online_ergebnisse().drop(columns=["quelle"]).to_string())
