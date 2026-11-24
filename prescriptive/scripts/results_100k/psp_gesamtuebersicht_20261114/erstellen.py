"""Consolidate existing BPIC17 PSP results; no training or simulation is run."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
R = OUT.parent
ROOT = OUT.parents[3]
SOURCES = set()


def csv(path):
    SOURCES.add(path)
    return pd.read_csv(path)


def js(path):
    SOURCES.add(path)
    return json.loads(path.read_text())


def number(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",") if pd.notna(value) else "—"


def percent(value):
    return number(100 * value, 1) + " %" if pd.notna(value) else "—"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def event_table(data):
    headers = ["Modell", "Längen-MAE ↓", "DLS ↑", "S1 ↑", "S2 ↑", "S3 ↑", "S5 ↑", "S10 ↑", "S20 ↑"]
    return table(headers, [[row["Variante"], number(row["Suffixlänge-MAE"]), number(row["DLS"], 3),
                           *[percent(row[f"Schritt {k}"]) for k in (1, 2, 3, 5, 10, 20)]]
                          for _, row in data.iterrows()])


def time_table(data):
    columns = ["Abstand Schritt 1 MAE (h)", "Restzeit MAE (Tage)", "Restzeit Fehler (Tage)", "Restzeit CRPS (Tage)"]
    return table(["Modell", "Erster Abstand MAE (h) ↓", "Restzeit-MAE (d) ↓", "Bias (d)", "CRPS (d) ↓", "80-%-Abdeckung"],
                 [[row["Variante"], *[number(row[c]) for c in columns], percent(row["Abdeckung 80 %"])]
                  for _, row in data.iterrows()])


def online_row(name, folder):
    progress = js(folder / "progress.json")
    policy = js(folder / "psp_seed0_row.json")
    bank = js(folder / "bank_seed0_row.json")
    assert progress["status"] == "complete" and policy["Fälle"] == progress["scored_cases"]
    assert policy["seed"] == bank["seed"] == 0
    p, b = policy["Durchlaufzeit (Tage)"], bank["Durchlaufzeit (Tage)"]
    return {"variant": name, "samples": progress["samples_per_option"], "seeds": 1,
            "scored_cases_per_seed": policy["Fälle"], "cycle_days": p, "outcome": -p,
            "bank_cycle_days": b, "uplift_percent": 100 * (b - p) / b,
            "p90_days": policy["Durchlaufzeit P90 (Tage)"], "p95_days": policy["Durchlaufzeit P95 (Tage)"],
            "cvar90_days": policy["CVaR 90 % (Tage)"], "source": str(folder)}, bank["_days"]


def outcomes_table(data):
    return table(["Variante", "Ziehungen/Option", "Seeds × Fälle", "Tage ↓", "Bank (Tage)", "Verbesserung ggü. Bank"],
                 [[row["variant"], int(row["samples"]), f"{int(row['seeds'])} × {int(row['scored_cases_per_seed'])}",
                   number(row["cycle_days"], 3), number(row["bank_cycle_days"], 3), number(row["uplift_percent"]) + " %"]
                  for _, row in data.iterrows()])


def main():
    combined = csv(R / "psp_online_uebersicht_20261113/suffix_und_zeitguete.csv")
    names = ["Semi-Markov (Referenz)", "s40, Gaps", "(a), Log-Gaps", "(c), normal", "(d), Zeitklassen",
             "OE, Event+Zeitklasse als Token", "(c), Teacher Forcing 1"]
    assert len(combined) == len(names)
    combined.insert(0, "Variante", names)
    # Verify the actual underlying rows and the common held-out prefix set.
    reference = None
    for source in combined["source"].unique():
        source = Path(source)
        original = csv(source)
        prefix = csv(source.with_name("prefixe.csv"))
        for model, rows in prefix.groupby("Modell", sort=False):
            signature = set(zip(rows["Fall"].astype(str), rows["Prefixlänge"].astype(int)))
            if reference is None:
                reference = signature
            assert signature == reference and len(rows) == 1000, (source, model)
        for _, row in combined[combined["source"] == str(source)].iterrows():
            match = original[original["Modell"] == row["Modell"]].iloc[0]
            for col in ("DLS", "Suffixlänge-MAE", "Restzeit MAE (Tage)"):
                assert np.isclose(match[col], row[col])
    combined.to_csv(OUT / "suffix_vollmodelle_1000_testprefixe.csv", index=False)
    legacy = csv(R / "suffix_vorhersage_stufe2.csv")
    legacy.insert(0, "Variante", legacy["Modell"])
    legacy.to_csv(OUT / "suffix_aeltere_200_prefixe.csv", index=False)
    pilots = []
    pilot_names = {"k_tf0": "Pilot Klassen, TF 0", "k_tf1": "Pilot Klassen, TF 1", "t_tf1": "Pilot Token, TF 1",
                   "n_tf1": "Pilot normal, TF 1", "t_tf05": "Pilot Token, TF 0,5", "t_tf0": "Pilot Token, TF 0",
                   "t_tf1_kx": "Pilot Token, TF 1 + Encoderkontext", "t_tf05_kx": "Pilot Token, TF 0,5 + Encoderkontext"}
    for directory in ("pilot_suffixguete", "pilot2_suffixguete", "pilot3_suffixguete"):
        data = csv(R / directory / "suffixguete.csv")
        data = data[data["Modell"] != "Semi-Markov-Kette"].copy()
        data.insert(0, "Variante", data["Modell"].map(pilot_names))
        data["source"] = str(R / directory / "suffixguete.csv")
        prefix = csv(R / directory / "prefixe.csv")
        for model, rows in prefix.groupby("Modell", sort=False):
            assert set(zip(rows["Fall"].astype(str), rows["Prefixlänge"].astype(int))) == reference
        pilots.append(data)
    pilots = pd.concat(pilots, ignore_index=True)
    pilots.to_csv(OUT / "suffix_piloten_1000_testprefixe.csv", index=False)
    historical = csv(R / "psp_online_uebersicht_20261113/online_bpic17_vier_punkte.csv")
    historical.to_csv(OUT / "online_10_seeds.csv", index=False)
    mini_paths = {
        "(c), 1 ms": R / "c_shared_gap_suffixe30_paired200_seed0/original_1ms",
        "(c), ein Abstand je Option": R / "c_shared_gap_suffixe30_paired200_seed0/normal_einmal_pro_option",
        "(d), Abstand je Suffix": R / "d_option_encoder_suffixe30_cases200_seed0_20261113/d_korrigiert",
        "OE, Abstand je Suffix": R / "token_oe_suffixe30_vergleich_c_seed0_20261113/oe_ohne_sperre",
        "OE, ein Abstand je Option": R / "token_oe_shared_gap_suffixe30_cases200_seed0_20261113/oe_ein_abstand_neustart",
        "OE + Übergangssperren": R / "token_oe_suffixe30_paired200_seed0_v2/mit_sperre"}
    mini_rows, mini_bank = [], None
    for name, path in mini_paths.items():
        row, bank = online_row(name, path)
        if mini_bank is None:
            mini_bank = bank
        assert mini_bank == bank
        mini_rows.append(row)
    mini = pd.DataFrame(mini_rows)
    mini.to_csv(OUT / "online_200_faelle_30_ziehungen.csv", index=False)
    large_rows, large_bank = [], None
    for name, path in {
        "(d), Abstand je Suffix": R / "d_option_encoder_suffixe300_cases1000_seed0_20261113/d_korrigiert",
        "OE + Übergangssperren": R / "neue_punkte_online_token_oe_ue_suffixe300_seed0_v2"}.items():
        row, bank = online_row(name, path)
        if large_bank is None:
            large_bank = bank
        assert large_bank == bank
        large_rows.append(row)
    large = pd.DataFrame(large_rows)
    large.to_csv(OUT / "online_1000_faelle_300_ziehungen.csv", index=False)
    cost_folder = ROOT / "prescriptive/policies/ev_policy_event_costs/results/vergleich_200_faelle_30_ziehungen_seed0_20261113"
    cost_table = csv(cost_folder / "comparison.csv").set_index("Policy")
    e_status = js(R / "e_suffixe30_cases200_seed0_20261113/status.json")
    e_log = R / "e_suffixe30_cases200_seed0_20261113/pipeline.log"
    SOURCES.add(e_log)
    e_epochs = [line.split("]", 1)[0].removeprefix("Epoch [") for line in e_log.read_text().splitlines() if line.startswith("Epoch [")]
    timestamp = datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds")
    lines = ["# Alle vorhandenen BPIC17-PSP-Auswertungen", "", f"Stand: {timestamp}.", "",
             "Dies ist eine Zusammenstellung vorhandener Ergebnisse. Es wurden keine neuen Modelle trainiert "
             "und keine neuen Simulationen gestartet. Vollmodelle, Pilotmodelle und ältere Stichproben sind getrennt.", "",
             "## Definitionen", "",
             "- Längen-/Event-MAE: mittlerer absoluter Fehler der mittleren gezogenen Zahl verbleibender Events.",
             "- Schritt k: häufigstes gezogenes Aktivität|Lifecycle-Token an Position k trifft das echte Token; "
             "nur Prefixe mit einem echten Event an dieser Position zählen.",
             "- DLS: mittlere normierte Damerau-Levenshtein-Ähnlichkeit über die einzelnen gezogenen Suffixe, 1 = identisch.",
             "- Restzeit-MAE, Bias und CRPS beziehen sich auf die Zeit ab dem letzten Prefix-Event. "
             "Bias = geschätzt minus echt. Kleinere MAE/CRPS sind besser; 80-%-Abdeckung sollte etwa 80 % erreichen.",
             "- Online-Outcome bei Durchlaufzeit: Y = −Tage. Zur besseren Lesbarkeit wird T positiv gezeigt, kleiner ist besser. "
             "Positive Verbesserungsprozente bedeuten kürzer als die jeweilige Bankreferenz.", "",
             "## Suffixgüte: Vollmodelle, gleiche 1.000 Testprefixe, je 100 Suffixziehungen", "",
             "Die Übereinstimmung der Prefixmengen wurde anhand der einzelnen Fall-ID/Prefixlängen-Paare überprüft.", "",
             event_table(combined), "", time_table(combined), "",
             "(d) bezeichnet hier das Basisnetz mit getrennten Event-/Zeitklassen-Köpfen. Die freie Suffixdiagnose "
             "wurde nicht erneut als eigene Diagnose an Interventionsprefixen der späteren Optionsabstand-Variante gemessen. "
             "OE mit Übergangssperren hat keine eigene solche Suffixdiagnose; dessen Werte werden nicht vom unbeschränkten OE übernommen.", "",
             "## Ältere Modelle und (b): 200 Prefixe aus dem gesamten Log", "",
             "Diese Auswertung kann Trainingsfälle enthalten und ist nicht direkt mit der oberen Testtabelle vergleichbar. "
             "Für (b) fehlt die entsprechende 1.000-Testprefix-Auswertung.", "", event_table(legacy), "",
             "## Online-Outcomes: abgeschlossene Vergleiche mit zehn Seeds", "",
             "Je Seed 1.000 bewertete Fälle, 700 Warmup, 1.300 Cooldown; zufällige Mitarbeiterzuteilung. "
             "Bank: 22,328 Tage. Prozentwerte sind Mittel der gepaarten Seed-Verbesserungen. "
             "95-%-Intervalle und Tail-Kennzahlen stehen in online_10_seeds.csv.", "", outcomes_table(historical), "",
             "## Online-Outcomes: gleicher Mini-Test mit 200 Fällen und 30 Ziehungen", "",
             "Ein Seed (0), identische Bank-Falldauern zwischen allen sechs Varianten, Bankmittel "
             f"{number(mini.iloc[0]['bank_cycle_days'], 3)} Tage.", "", outcomes_table(mini), "",
             "## Online-Outcomes: 1.000 Fälle und 300 Ziehungen, Seed 0", "",
             "Beide Varianten haben exakt dieselben Bank-Falldauern. Kein Zehn-Seed-Ergebnis.", "", outcomes_table(large), "",
             "## Weitere historische Zielgrößen und Zwischenstände", "",
             "- (c) mit 300 Suffixen: fünf fertig gewordene Seeds 0–4 des später gestoppten Laufs, "
             "ungefähr 19,88 Tage gegenüber 22,33 Banktagen. Nur gerundete Log-Zwischenstände.",
             "- OE ohne Übergangssperre mit 300 Suffixen: sieben fertig gewordene Seeds 0,1,3,4,5,6,8, "
             "ungefähr 21,30 Tage gegenüber 22,42 Banktagen. Andere Seedmenge als (c), daher nicht direkt dieselbe Stichprobe.",
             "- PSP-ERM auf (c), γ=0,1/Tag: 22,437 Tage gegenüber 22,600 Banktagen; empirisches entropisches Risiko "
             "63,487 gegenüber 62,364 Tagen. Zehn Seeds, je 1.000 Fälle; learned-Mitarbeiterzuteilung und anderes Optimierungsziel. "
             "Quelle: ../neue_punkte_online_psp_erm_gamma0.1_suffixe100_bank/auswertung.md.",
             f"- (d) mit Eventkosten als Ziel: {number(cost_table.loc['psp', 'Eventkosten je Fall (€)'])} €/Fall gegenüber "
             f"{number(cost_table.loc['bank', 'Eventkosten je Fall (€)'])} €/Fall der Bank, "
             "200 Fälle, 30 Ziehungen, Seed 0, konfigurierte Beispieltarife. Outcome ist hier negative Euro, "
             "nicht negative Tage. Dies ist eine andere Bewertung derselben Suffixgewichte, kein neu trainiertes PSP.", "",
             "## (e) und fehlende eigenständige Messungen", "",
             f"(e): Status {e_status['status']}; letzte im Log ausgewiesene Epoche "
             f"{e_epochs[-1] if e_epochs else 'noch keine'}. Es liegen noch keine fertigen Suffixgüte- oder Online-Ergebnisse vor.", "",
             "Für (c) mit geteiltem Optionsabstand, OE mit geteiltem Optionsabstand sowie OE mit Übergangssperren "
             "gibt es Online-Ergebnisse, aber keine neu gemessene vollständige Suffixgüte in dieser Zusammenstellung. "
             "Bei reinen Bewertungsvarianten bleiben die Basisgewichte gleich; das ersetzt keine gemessene Diagnose "
             "des jeweiligen Entscheidungsverfahrens. Pilot-TF-0,8 und einige *_od/*_oe-Kopien haben ebenfalls keine "
             "eigenständige vollständige Suffixgüte-Datei.", "",
             "## Pilotmodelle: 3.000 Trainingsfälle, sechs Epochen", "",
             "Gleiche 1.000 Testprefixe und je 100 Suffixe, aber erheblich kleineres Training als die Vollmodelle. "
             "k = getrennte Klassen, t = gemeinsames Event-/Zeitklassen-Token, n = normale Abstände, "
             "TF = unterste Teacher-Forcing-Quote, kx = Encoderkontext in jedem Decoderschritt. "
             "Für diese Piloten liegt keine entsprechende vollständige Online-Outcome-Tabelle vor.", "",
             event_table(pilots), "", time_table(pilots), "",
             "## Dateien", "",
             "Alle Tabellen stehen zusätzlich als CSV in diesem Ordner. manifest.json enthält die Quellenprüfsummen. "
             "Die ältere übergreifende Inventarliste einschließlich BPIC19/Repair steht unter "
             "../psp_online_uebersicht_20261113/online_alle_berichte.csv; diese anderen Datensätze sind nicht mit BPIC17 zu vermischen."]
    (OUT / "uebersicht.md").write_text("\n".join(lines) + "\n")
    # Record the historical sources used only for the explicitly qualified text.
    SOURCES.update([R / "psp_online_uebersicht_20261113/uebersicht.md",
                    R / "neue_punkte_online_psp_erm_gamma0.1_suffixe100_bank/auswertung.md"])
    (OUT / "manifest.json").write_text(json.dumps({"created_at": timestamp,
        "common_1000_test_prefixes_verified": True, "mini_bank_durations_identical": True,
        "large_bank_durations_identical": True,
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(SOURCES)}},
        indent=2, ensure_ascii=False) + "\n")
    print("Report:", OUT / "uebersicht.md")
    print("Full-model rows:", len(combined), "pilot rows:", len(pilots), "legacy rows:", len(legacy))
    print(combined[["Variante", "DLS", "Suffixlänge-MAE", "Restzeit MAE (Tage)"]].to_string(index=False))


if __name__ == "__main__":
    main()
