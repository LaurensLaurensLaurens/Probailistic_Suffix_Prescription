"""Summarize existing PSP evaluations without running new experiments."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from statistics import mean

OUT = Path(__file__).resolve().parent
R = OUT.parent
SCRIPTS = R.parent
P = SCRIPTS.parent
SOURCES = set()


def records(path):
    SOURCES.add(path)
    with path.open() as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    SOURCES.add(path)
    return json.loads(path.read_text())


def num(value):
    return float(value) if value not in (None, "") else None


def fmt(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}".replace(".", ",")


def link(path, label=None):
    return f"[{label or path.name}]({path.resolve()})"


def write_csv(name, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def online(folder, policy="psp", directory="PSP"):
    path = R / folder / "prescription" / directory / "bpic17_online.csv"
    rows = records(path)
    return next(row for row in rows if row["Policy"] == policy), next(row for row in rows if row["Policy"] == "bank"), path


def main():
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    inventory = []
    for folder in sorted(SCRIPTS.glob("results*")):
        for path in sorted(folder.rglob("*online.csv")):
            rows = records(path)
            bank = next((row for row in rows if row.get("Policy") == "bank"), {})
            for row in rows:
                if not row.get("Policy", "").startswith("psp"):
                    continue
                inventory.append({
                    "dataset": path.name.removesuffix("_online.csv"),
                    "experiment": str(path.parent.relative_to(SCRIPTS)), "policy": row["Policy"],
                    "scored_cases_per_seed": num(row.get("Fälle")), "seeds": num(row.get("Wiederholungen")),
                    "cycle_days": num(row.get("Durchlaufzeit (Tage)")), "outcome": num(row.get("Outcome je Fall")),
                    "bank_cycle_days": num(bank.get("Durchlaufzeit (Tage)")), "bank_outcome": num(bank.get("Outcome je Fall")),
                    "uplift_percent": num(row.get("Uplift ggü. Bank (%)")),
                    "uplift_ci95_low": num(row.get("Uplift 95-%-KI unten")),
                    "uplift_ci95_high": num(row.get("Uplift 95-%-KI oben")),
                    "points": "; ".join(key.split(": ", 1)[1] for key in row if key.startswith("Wahlen der Policy:")),
                    "source": str(path),
                })
    write_csv("online_alle_berichte.csv", inventory)

    # Same four BPIC17 control-flow points, random staff allocation, ten seeds.
    definitions = [
        ("Ursprüngliches Echtlog-PSP", "neue_punkte_online", 30, "real_w3_10k_neuepunkte", "30 Suffixe", "Ursprüngliches 10k-Modell, neuer Wahlpunkt eingetragen; ältere statische Merkmale können spätere Fallinformationen enthalten."),
        ("Auf neue Wahlpunkte neu trainiert", "neue_punkte_online_psp_neu", 30, "real_w3_10k_np", "30 Suffixe", "10k-Fälle, neue Wahlpunkte dreifach gewichtet; Decoder im ursprünglichen kurzen Training."),
        ("s40, Zeit aus vorhergesagtem Fallalter", "neue_punkte_online_real_np_s40_ohne_zukunft", 30, "real_np_s40_ohne_zukunft", "30 Suffixe", "Decoder 40, 10k-Fälle; statische Merkmale ohne Zukunftswissen."),
        ("s40, Summe der Zeitabstände", "neue_punkte_online_variante_abstaende", 30, "real_np_s40_ohne_zukunft_abstaende", "30 Suffixe", "Gleiche Modellfamilie; Outcome aus Summe nichtnegativer Abstände."),
        ("s40, Summe der Zeitabstände", "neue_punkte_online_variante_100ziehungen", 100, "real_np_s40_ohne_zukunft_abstaende", "100 Suffixe", "Mehr Ziehungen je Option."),
        ("s40, Angebot nur beim ersten Besuch wählen", "neue_punkte_online_real_np_s40_ohne_zukunft_abstaende_erste_entscheidung", 100, "real_np_s40_ohne_zukunft_abstaende", "100 Suffixe", "first_only betrifft angebot_nachlegen; spätere Besuche dort folgen der Bank."),
        ("(a), logarithmierte Abstände", "neue_punkte_online_stufe2_ohne_kopf_suffixe100", 100, "real_alle_s60_loggaps", "100 Suffixe", "Alle äußeren Trainingsfälle, Decoder 60, Log-Gaps; Gesamtzeit als Summe der Abstände."),
        ("(b), mit Restzeitkopf, Suffixbewertung", "neue_punkte_online_stufe2_kopf_suffixe100", 100, "real_alle_s60_loggaps_rest", "100 Suffixe", "Zusätzlicher Restzeitkopf trainiert, für die Wahl hier aber Suffixbewertung verwendet."),
        ("(b), direkte Restzeitbewertung", "neue_punkte_online_stufe2_kopf", 30, "real_alle_s60_loggaps_rest", "30 Dropout-Auswertungen des Kopfs", "Die Policy verwendet den Erwartungswert des Restzeitkopfs, statt Suffixzeiten zu summieren."),
        ("(c), normale Abstände", "neue_punkte_online_stufe2c_suffixe100", 100, "real_alle_s60_normal", "100 Suffixe", "Alle äußeren Trainingsfälle, Decoder 60, normale Abstände; Option am Zeitpunkt now + 1 ms eingesetzt."),
    ]
    main_rows = []
    bank_runs_path = R / "neue_punkte_online_stufe2c_suffixe100/prescription/PSP/bpic17_online_laeufe.csv"
    bank_runs = {int(row["seed"]): float(row["Durchlaufzeit (Tage)"]) for row in records(bank_runs_path) if row["Policy"] == "bank"}
    for label, folder, samples, tag, evaluation, note in definitions:
        row, bank, source = online(folder)
        assert float(row["Fälle"]) == 1000 and int(row["Wiederholungen"]) == 10
        assert float(bank["Durchlaufzeit (Tage)"]) == 22.328
        run_path = source.with_name("bpic17_online_laeufe.csv")
        run_rows = records(run_path)
        current_bank = {int(r["seed"]): float(r["Durchlaufzeit (Tage)"]) for r in run_rows if r["Policy"] == "bank"}
        assert current_bank == bank_runs, folder
        main_rows.append({"variant": label, "samples": samples, "evaluation": evaluation, "model_tag": tag,
                          "seeds": 10, "scored_cases_per_seed": 1000,
                          "cycle_days": float(row["Durchlaufzeit (Tage)"]), "outcome": float(row["Outcome je Fall"]),
                          "p90_days": float(row["Durchlaufzeit P90 (Tage)"]),
                          "p95_days": float(row["Durchlaufzeit P95 (Tage)"]),
                          "cvar90_days": float(row["CVaR 90 % (Tage)"]),
                          "bank_cycle_days": float(bank["Durchlaufzeit (Tage)"]),
                          "uplift_percent": float(row["Uplift ggü. Bank (%)"]),
                          "uplift_ci95_low": float(row["Uplift 95-%-KI unten"]),
                          "uplift_ci95_high": float(row["Uplift 95-%-KI oben"]), "note": note, "source": str(source)})
    write_csv("online_bpic17_vier_punkte.csv", main_rows)

    cf_path = R / "neue_punkte_online_cate/prescription/Causal_Forest/bpic17_online_laeufe.csv"
    cf_runs = {int(row["seed"]): float(row["Durchlaufzeit (Tage)"]) for row in records(cf_path) if row["Policy"] == "causal_forest"}
    c100_runs = {int(row["seed"]): float(row["Durchlaufzeit (Tage)"]) for row in records(bank_runs_path) if row["Policy"] == "psp"}
    partials, partial_seed_rows = [], []
    values = {}
    for tag, label in [("normal", "(c), 300 Suffixe"), ("token_oe", "OE, 300 Suffixe")]:
        path = P / f"data/PSP_online300_real_alle_s60_{tag}.log"
        SOURCES.add(path)
        vals = {int(s): -float(v) for s, v in re.findall(r"seed (\d+), psp: Outcome je Fall (-?[\d.]+)", path.read_text())}
        values[tag] = vals
        partials.append({"variant": label, "seeds": sorted(vals), "seed_count": len(vals), "scored_cases_per_seed": 1000,
                         "cycle_days_approx": mean(vals.values()), "bank_matched_days": mean(bank_runs[s] for s in vals),
                         "cf_matched_days": mean(cf_runs[s] for s in vals),
                         "uplift_matched_percent_approx": mean(100 * (bank_runs[s] - v) / bank_runs[s] for s, v in vals.items()),
                         "source": str(path), "status": "stopped; completed seeds only; log rounded to 0.1 day"})
        partial_seed_rows.extend({"variant": label, "seed": s, "cycle_days_approx": v,
                                  "bank_days": bank_runs[s], "cf_days": cf_runs[s]} for s, v in sorted(vals.items()))
    write_csv("online_300_fertige_seeds.csv", partial_seed_rows)

    mini_defs = [
        ("(c), 1 ms", R / "c_shared_gap_suffixe30_paired200_seed0/original_1ms"),
        ("(c), ein Normal-Abstand je Option", R / "c_shared_gap_suffixe30_paired200_seed0/normal_einmal_pro_option"),
        ("OE, ohne Sperre", R / "token_oe_suffixe30_vergleich_c_seed0_20261113/oe_ohne_sperre"),
        ("OE, beobachtete Übergänge", R / "token_oe_suffixe30_paired200_seed0_v2/mit_sperre"),
    ]
    minis, mini_bank = [], None
    for label, folder in mini_defs:
        row = read_json(folder / "psp_seed0_row.json")
        bank = read_json(folder / "bank_seed0_row.json")
        progress = read_json(folder / "progress.json")
        assert progress["status"] == "complete"
        if mini_bank is None:
            mini_bank = bank["_days"]
        assert bank["_days"] == mini_bank and row["Fälle"] == 200
        days, bank_days = row["Durchlaufzeit (Tage)"], bank["Durchlaufzeit (Tage)"]
        minis.append({"variant": label, "seed": 0, "samples": 30, "scored_cases": 200,
                      "cycle_days": days, "outcome": row["Outcome je Fall"], "bank_days": bank_days,
                      "uplift_percent": 100 * (bank_days - days) / bank_days,
                      "elapsed_seconds": progress["elapsed_seconds"], "source": str(folder)})
    write_csv("online_minitests.csv", minis)

    # These three evaluations use exactly the same 1,000 (case, prefix-length) pairs.
    quality_sources = [R / "suffixguete/suffixguete.csv", R / "voll_real_alle_s60_token_oe_suffixguete/suffixguete.csv",
                       R / "voll_real_alle_s60_normal_tf1_suffixguete/suffixguete.csv"]
    quality, expected_prefixes = [], None
    for path in quality_sources:
        prefix_rows = records(path.with_name("prefixe.csv"))
        prefix_sets = {}
        for row in prefix_rows:
            prefix_sets.setdefault(row["Modell"], set()).add((row["Fall"], row["Prefixlänge"]))
        for pairs in prefix_sets.values():
            assert len(pairs) == 1000
            if expected_prefixes is None:
                expected_prefixes = pairs
            assert pairs == expected_prefixes
        for row in records(path):
            if path != quality_sources[0] and row["Modell"] == "Semi-Markov-Kette":
                continue
            quality.append({**row, "prefixes": 1000, "draws_per_prefix": 100, "source": str(path)})
    write_csv("suffix_und_zeitguete.csv", quality)
    older_quality = records(R / "suffix_vorhersage_stufe2.csv")
    old_diagnosis = records(R / "psp_diagnose_real_alle_s60_loggaps_rest/diagnose.csv")
    SOURCES.add(P / "data/PSP_schritte_stufe2.log")

    option_rows = []
    for model, folder in [
        ("(c)", "psp_rauschen_real_alle_s60_normal"),
        ("OE", "psp_rauschen_voll_real_alle_s60_token_oe_real_alle_s60_token_oe"),
        ("(d), nur offline", "psp_rauschen_real_alle_s60_klassen"),
        ("(c) TF1, nur offline", "psp_rauschen_voll_real_alle_s60_normal_tf1_real_alle_s60_normal_tf1"),
    ]:
        path = R / folder / "rauschen.csv"
        table = records(path)
        measurements = {row[""]: row for row in table}
        for point in ["annehmen_bei_konzept", "annehmen_nach_vervollstaendigung", "angebot_nachlegen", "handle_leads"]:
            option_rows.append({"model": model, "point": point, "prefixes": 200, "samples": 100,
                                "mean_action_advantage_days": float(measurements["Diff Mittel (Tage)"][point]),
                                "paired_se_median_days": float(measurements["SE gepaart Median"][point]),
                                "sign_agreement_with_aipw": float(measurements["Vorzeichen wie AIPW (alle)"][point]),
                                "source": str(path)})
    write_csv("optionssignal_und_rauschen.csv", option_rows)
    long_progress = read_json(R / "neue_punkte_online_token_oe_ue_suffixe300_seed0_v2/progress.json")
    erm, erm_bank, erm_source = online("neue_punkte_online_psp_erm_gamma0.1_suffixe100_bank", "psp_erm", "PSP_ERM")
    SOURCES.add(R / "neue_punkte_online_psp_erm_gamma0.1_suffixe100_bank/auswertung.md")
    SOURCES.add(R / "token_oe_300_diagnose_20261113/auswertung.md")

    lines = [
        "**PSP: Online-Outcome, Suffixvorhersage und Zeitgüte**", "", f"Erstellt: {now}.", "",
        "Die Hauptübersicht betrifft BPIC17 und die vier zuletzt untersuchten Control-Flow-Punkte: "
        "angebot_nachlegen, annehmen_bei_konzept, annehmen_nach_vervollstaendigung, handle_leads. "
        "Ein vollständiges Inventar aller gefundenen PSP-Online-Berichte einschließlich älterer BPIC17-, BPIC19- und Repair-Versuche steht am Ende und in online_alle_berichte.csv. "
        "Mehrere Berichtsordner können denselben Lauf enthalten; ihre Anzahl ist keine Zahl unabhängiger Versuche.", "",
        "Outcome ist bei den BPIC17-Hauptvergleichen die negative Durchlaufzeit: größerer Outcome = kürzere Zeit. "
        "Die Tabellen zeigen positive Tage; kleiner ist besser. Online-Dauern werden aus dem Simulator-Log gemessen, "
        "nicht aus den vorhergesagten Suffixen. Die künstlichen 1 ms beeinflussen die Optionsbewertung. "
        "Ein daraus folgender Policy-Vorteil ist deshalb kein Rechenfehler der Online-Dauermessung.", "",
        "**Vollständig abgeschlossene Hauptvergleiche: zehn Seeds, je 1.000 gewertete Fälle**", "",
        "Je Lauf 3.000 Ankünfte, 700 Warmup- und 1.300 Cooldown-Fälle, Seed 0–9, zufällige Mitarbeiterzuteilung. "
        "Die gespeicherten Bank-Mittelwerte je Seed stimmen in diesen zehn Versuchen überein; die Bank liegt insgesamt bei 22,328 Tagen. "
        "Positive Verbesserung bedeutet kürzer als die Bank. Die Prozentwerte sind Mittel der gepaarten Seed-Uplifts; "
        "sie können vom Quotienten der Gesamtmittel geringfügig abweichen.", "",
        "| Variante | Bewertung je Option | Tage ↓ | Verbesserung ggü. Bank | 95-%-KI der Verbesserung | Quelle |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in main_rows:
        lines.append(f"| {row['variant']} | {row['evaluation']} | {fmt(row['cycle_days'],3)} | {fmt(row['uplift_percent'])} % | "
                     f"[{fmt(row['uplift_ci95_low'])}; {fmt(row['uplift_ci95_high'])}] % | {link(Path(row['source']),'CSV')} |")
    lines += ["", "Als Vergleich auf diesen zehn Seeds: Causal Forest 19,477 Tage, X-Learner 19,621, S-Learner 20,044, feste Regel 19,356. "
              "Der direkte Restzeitkopf ist mit 19,830 Tagen der beste PSP unter diesen vollständig abgeschlossenen Hauptvergleichen. "
              "Das ist eine Aussage über diesen Simulator und diese Konfiguration, keine Garantie für den echten Prozess.", "",
              "Auch die langen Fälle unterscheiden sich: Bank P95 47,415 / CVaR90 51,265 Tage; "
              "direkter Restzeitkopf (b) 44,195 / 48,964; (c) mit 100 Suffixen 45,918 / 50,953; "
              "(a) 49,002 / 53,763. CVaR90 ist der Mittelwert der längsten 10 % je Seed, danach über die Seeds gemittelt. "
              "Alle Tail-Werte der zehn Varianten stehen zusätzlich in online_bpic17_vier_punkte.csv.", "",
              "Die ursprünglichen PSPs verwendeten sehr kurzes Decodertraining und teilweise statische Merkmale aus dem vollständigen Fall. "
              "Bei s40 und später sind die statischen Merkmale auf LoanGoal, ApplicationType und RequestedAmount beschränkt. "
              "Die s60-Modelle verwenden alle 25.208 Fälle außerhalb der 6.301 zurückgehaltenen Testfälle, Decoder 60, "
              "dreifach gewichtete Wahlpunkte und regulär 15 Epochen. Die alten und neuen Online-Resultate bleiben historische Auswertungen ihrer jeweiligen Modell- und Codeversion.", "",
              "**300 Suffixe: fertig gewordene Seeds der später gestoppten Läufe**", "",
              "| Variante | Fertige Seeds | Tage, ungefähr | Bank auf diesen Seeds | CF auf diesen Seeds | Verbesserung zur Bank, ungefähr |",
              "|---|---|---:|---:|---:|---:|"]
    for row in partials:
        lines.append(f"| {row['variant']} | {', '.join(map(str,row['seeds']))} | {fmt(row['cycle_days_approx'])} | "
                     f"{fmt(row['bank_matched_days'])} | {fmt(row['cf_matched_days'])} | {fmt(row['uplift_matched_percent_approx'])} % |")
    common = sorted(values["normal"].keys() & values["token_oe"].keys())
    lines += ["", "Die 300er-Logs enthalten die Outcomes nur auf 0,1 Tage gerundet. Es gibt keine vollständige Zehn-Seed-Auswertung. "
              "Die Tabellenzeilen verwenden unterschiedliche Seed-Mengen und sind daher keine unmittelbar gleiche Stichprobe. "
              f"Auf den gemeinsamen Seeds {', '.join(map(str,common))} erreicht (c) ungefähr {fmt(mean(values['normal'][s] for s in common),3)} "
              f"und OE {fmt(mean(values['token_oe'][s] for s in common),3)} Tage. "
              f"Auf den fünf fertig gewordenen (c)-Seeds lagen 100 Suffixe bei {fmt(mean(c100_runs[s] for s in values['normal']),3)} "
              "gegenüber ungefähr 19,88 Tagen mit 300 Suffixen. Das ist ein gepaarter Zwischenstand, keine vollständige Wiederholung.", "",
              "**Gepaarte Mini-Tests: ein Seed, 30 Ziehungen, 200 gewertete Fälle**", "",
              "Alle vier Varianten: Seed 0, Fallfenster 700–899, 2.200 Ankünfte insgesamt, zufällige Mitarbeiterzuteilung. "
              "Die einzelnen Bank-Dauern aller 200 Fälle sind zwischen den vier Varianten exakt gleich; Bankmittel 20,374 Tage.", "",
              "| Variante | Tage ↓ | Verbesserung ggü. Bank | Quelle |", "|---|---:|---:|---|"]
    for row in minis:
        lines.append(f"| {row['variant']} | {fmt(row['cycle_days'],3)} | {fmt(row['uplift_percent'])} % | {link(Path(row['source'])/'psp_seed0_row.json','JSON')} |")
    lines += ["", "Beim neuen (c) bekommt jede Option einmal je Entscheidung einen eigenen gezogenen Zeitabstand; "
              "alle 30 Suffixe dieser Option teilen ihn. Die Normalparameter stammen aus den beobachteten Trainingsabständen "
              "je Wahlpunkt und Option, nicht aus einem zusätzlichen prefixabhängigen Zeitmodell. "
              "Die Verteilungen sind bei null trunkiert: Bei den stark schiefen Daten kann dies die erwarteten Abstände erhöhen. "
              "Die Unsicherheit des ersten Abstands wird bei dieser Variante nicht über 30 Zeitziehungen gemittelt. "
              "OE zieht dagegen je Suffix einen Optionsabstand über die vorhergesagte Zeitklasse und deren Zeitverteilung. "
              "Die Sperre in der vierten Zeile erlaubt nur die im äußeren Trainingssplit beobachteten Aktivität|Lifecycle-Paare und Fallenden.", "",
              "Diese konkrete Zeitvariante verschlechtert (c) um 1,013 Tage; die Übergangssperre verschlechtert OE um 0,542 Tage. "
              "Das sind zwei einzelne Mini-Tests. Sie belegen weder, dass eine korrekte Zeitmodellierung schadet, noch, "
              "dass der ursprüngliche Unterschied zwischen (c) und OE überwiegend auf dem Optionszeitpunkt beruht. "
              "Der erneute OE-Lauf reproduziert alle Entscheidungen und Falldauern des früheren OE-Laufs desselben Seeds exakt; "
              "er ist kein zusätzlicher unabhängiger Seed.", "",
              "**Suffixvorhersage und Zeitschätzung auf identischen Testprefixen**", "",
              "Die folgenden Werte stammen aus 1.000 identischen (Testfall, Prefixlänge)-Paaren aus den zurückgehaltenen echten Fällen, "
              "jeweils 100 Suffixe pro Prefix. Die Übereinstimmung der Prefixmengen wurde bei dieser Zusammenstellung überprüft. "
              "Das sind freie Fortsetzungen eines echten Prefix, keine Messung bekannter individueller Gegenfakten an einem Wahlpunkt.", "",
              "Treffer Schritt k: Das häufigste gezogene Token an Position k entspricht dem beobachteten Token; "
              "nur Prefixe mit einem echten Event an dieser Position werden gewertet. DLS ist die mittlere normierte Sequenzähnlichkeit "
              "der gezogenen Suffixe (1 = identisch). MAE und CRPS sind Fehlermaße: kleiner ist besser. "
              "Zeit-Bias = mittlere vorhergesagte Restzeit minus beobachtete Restzeit, also positiv = zu lang. "
              "Die Abdeckung eines 80-%-Intervalls sollte ungefähr 80 % erreichen.", "",
              "| Modell | Schritt 1 ↑ | Schritt 5 ↑ | Schritt 10 ↑ | DLS ↑ | Längen-MAE ↓ | Erster Abstand: MAE (h) ↓ | Restzeit-MAE (Tage) ↓ | Zeit-Bias (Tage) | CRPS (Tage) ↓ | 80-%-Abdeckung |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    labels = {"Semi-Markov-Kette":"Semi-Markov", "s40 (10.000 Fälle, Decoder 40)":"s40, Gaps", "(a) alle Fälle, Decoder 60, log-normale Abstände":"(a), Log-Gaps",
              "(c) alle Fälle, Decoder 60, normale Abstände":"(c), normal", "(d) wie (c) mit Zeitklassen":"(d), Zeitklassen; nur offline",
              "real_alle_s60_token_oe":"OE", "real_alle_s60_normal_tf1":"(c), Teacher Forcing 1; nur offline"}
    for row in quality:
        lines.append(f"| {labels[row['Modell']]} | " + " | ".join([
            fmt(100*float(row['Schritt 1']),1)+' %', fmt(100*float(row['Schritt 5']),1)+' %', fmt(100*float(row['Schritt 10']),1)+' %',
            fmt(float(row['DLS']),3), fmt(float(row['Suffixlänge-MAE'])), fmt(float(row['Abstand Schritt 1 MAE (h)'])),
            fmt(float(row['Restzeit MAE (Tage)'])), fmt(float(row['Restzeit Fehler (Tage)'])), fmt(float(row['Restzeit CRPS (Tage)'])),
            fmt(100*float(row['Abdeckung 80 %']),1)+' %']) + " |")
    lines += ["", "(c) hat fast keinen mittleren Restzeit-Bias, aber trotzdem rund 8,13 Tage mittleren absoluten Fehler. "
              "Sich ausgleichende Über- und Unterschätzungen ergeben keine genaue Vorhersage einzelner Fälle. "
              "OE ist bei längeren Eventfolgen und beim ersten Zeitabstand besser als (c), bei der Restzeit-MAE nur geringfügig besser. "
              "Das Modell mit Teacher Forcing 1 hat die beste DLS dieser Tabelle, aber schwächere Optionssignale als (c). "
              "Für (d) und Teacher Forcing 1 wurde kein abgeschlossener Online-Lauf an den vier aktuellen Punkten gefunden. "
              "Für das neue (c) mit geteiltem Optionsabstand und OE mit Übergangssperre gibt es noch keine eigenständige "
              "Suffix-/Zeitgüte-Auswertung dieser Art; Werte des unveränderten Basismodells dürfen nicht als neu gemessene Variante ausgegeben werden.", "",
              "Für (b) gibt es nur die ältere Eventauswertung mit 200 Prefixen aus dem gesamten echten Log: "
              "nächstes Event 84,5 %, DLS 0,266. Diese Stichprobe kann Trainingsfälle enthalten; sie ist keine "
              "saubere Auswertung auf ausschließlich zurückgehaltenen Fällen und nicht die obige 1.000-Prefix-Stichprobe. "
              "Eine separate Diagnose an je 100 Wahlpunkt-Prefixen zurückgehaltener Fälle "
              "ergibt für den direkten Restzeitkopf faktische Gesamtzeit-MAE von 9,25–10,73 Tagen, "
              "gegenüber 19,91–26,00 Tagen bei Suffixbewertung desselben Modells. Das sind andere Stichproben und Zielgrößen "
              "als die Restzeit-MAE in der obigen Tabelle.", "",
              f"Die ältere 200-Prefix-Tabelle ist hier vollständig enthalten, damit auch die ursprünglichen Online-Modelle "
              f"dokumentiert sind: {link(R/'suffix_vorhersage_stufe2.csv','Original-CSV')}. "
              "Ihre Werte sind wegen der möglichen Überschneidung mit Trainingsfällen vorsichtig zu interpretieren.", "",
              "| Ältere Eventauswertung, 200 Prefixe aus dem gesamten Log | Schritt 1 | Schritt 5 | DLS | Längen-MAE |",
              "|---|---:|---:|---:|---:|"]
    for row in older_quality:
        lines.append(f"| {row['Modell']} | {fmt(100*float(row['Schritt 1']),1)} % | "
                     f"{fmt(100*float(row['Schritt 5']),1)} % | {fmt(float(row['DLS']),3)} | {fmt(float(row['Suffixlänge-MAE']))} |")
    lines += ["",
              "**Optionssignal und Rauschen**", "",
              "Je Wahlpunkt 200 identische erste Entscheidungen aus echten Testfällen, hälftig beobachtete Aktion bzw. Referenz, "
              "100 Suffixe je Option. Positiver geschätzter Aktionsvorteil bedeutet kürzere Durchlaufzeit. "
              "Die Vorzeichenquote vergleicht mit der Richtung des AIPW-Effekts aus dem Screening; "
              "sie ist keine bekannte individuelle Entscheidungsgenauigkeit.", "",
              "| Aktion | (c): Vorteil (Tage) | OE: Vorteil (Tage) | (c): Vorzeichen wie AIPW | OE: Vorzeichen wie AIPW | (c): SE-Median (Tage) | OE: SE-Median (Tage) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for point in ["annehmen_bei_konzept", "annehmen_nach_vervollstaendigung", "angebot_nachlegen", "handle_leads"]:
        a = next(row for row in option_rows if row['model']=='(c)' and row['point']==point)
        b = next(row for row in option_rows if row['model']=='OE' and row['point']==point)
        lines.append(f"| {point} | {fmt(a['mean_action_advantage_days'])} | {fmt(b['mean_action_advantage_days'])} | "
                     f"{fmt(100*a['sign_agreement_with_aipw'],1)} % | {fmt(100*b['sign_agreement_with_aipw'],1)} % | "
                     f"{fmt(a['paired_se_median_days'])} | {fmt(b['paired_se_median_days'])} |")
    lines += ["", "OE hat hier rund 40–50 % höhere Standardfehler als (c). Beim Handle-leads-Punkt weicht außerdem "
              "die mittlere Richtung vom Screening ab. In einer separaten Wiederholungsprobe mit 48 Prefixen und jeweils "
              "300 Suffixen wechselte OE bei 11/48 Prefixen die Wahl zwischen zwei Zufallsstarts. "
              "Diese kleinen Proben erklären keine feste Anzahl verlorener Online-Tage.", "",
              "Die OE-Pfadprüfung fand bei ungefähr 99,3 % der gezogenen Suffixe mindestens einen im äußeren Trainingssplit "
              "unbeobachteten Übergang; das ist kein Anteil ungültiger Einzelschritte. Ungefähr 35,6 % der Suffixe enthielten "
              "keines der Events A_Pending, A_Denied, A_Cancelled. Auch das ist keine allgemeine Quote aller PSP-Modelle. "
              "Die Mini-Auswertung mit Übergangssperre zeigt bisher keine Verbesserung des Online-Outcomes. "
              f"Details: {link(R/'token_oe_300_diagnose_20261113/auswertung.md','OE-Diagnose')}.", "",
              "**Abweichendes Ziel: PSP-ERM**", "",
              "Zusätzlich abgeschlossen: (c) mit 100 Suffixen und entropischer Risikobewertung, γ = 0,1/Tag, zehn Seeds "
              "mit je 1.000 Fällen. Hier ist die Mitarbeiterzuteilung learned; der Lauf gehört daher nicht in die Rangliste "
              "der obigen Random-Zuteilung. Er optimiert zudem ein anderes Risiko-Ziel.", "",
              f"PSP-ERM: {fmt(float(erm['Durchlaufzeit (Tage)']),3)} Tage; zugehörige Bank "
              f"{fmt(float(erm_bank['Durchlaufzeit (Tage)']),3)} Tage. Empirisches ERM: 63,487 gegenüber 62,364 Tagen bei der Bank; "
              "kleiner wäre besser. Die gepaarte ERM-Differenz +1,124 Tage hat ein 95-%-KI von [−16,038; +18,285]; "
              "kein gesicherter ERM-Vorteil. Das Netz ist dasselbe (c), die Auswahlregel ist anders. "
              f"Quelle: {link(R/'neue_punkte_online_psp_erm_gamma0.1_suffixe100_bank/auswertung.md','ERM-Auswertung')}.", "",
              "**Noch ohne Ergebnis**", "",
              f"OE mit Übergangssperre und 300 Ziehungen, Seed 0: gespeicherter Status `{long_progress['status']}`, "
              f"{long_progress['decisions_completed']} Entscheidungen, zuletzt aktualisiert {long_progress['updated_at']}. "
              "Noch kein fertiger Online-Outcome. Der vorherige abgelöste Lauf und der fehlgeschlagene erste Sperren-Kurzlauf "
              "werden nicht als zusätzliche Ergebnisse gezählt. Die Annahmepunkt-Trainingsvariante hat ebenfalls keinen "
              "fertigen Online-Nachweis in den gefundenen Berichten.", "",
              "**Archiv: andere BPIC17-Wahlpunkte und ältere Datensätze**", "",
              "Die folgenden vollständigen Berichtszeilen sind zur Vollständigkeit aufgeführt. "
              "Wahlpunkte, Daten, Fallzahlen und teilweise Outcome unterscheiden sich. "
              "Kosten-Outcomes im Repair-Datensatz sind keine negativen Tage. "
              "Gleiche PSP-Ergebnisse in mehreren Vergleichsordnern können dieselben Läufe wiedergeben.", "",
              "| Bericht / Policy | Datensatz | Fälle/Seed | Seeds | Zeit (Tage) | Outcome | Verbesserung zum jeweiligen Bank-Outcome | Quelle |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    main_paths = {row['source'] for row in main_rows}
    mini_paths = {str(Path(row['source'])/'prescription/PSP/bpic17_online.csv') for row in minis}
    for row in inventory:
        if row['source'] in main_paths or row['source'] in mini_paths or row['source'] == str(erm_source):
            continue
        short = row['experiment'].replace('/prescription/', ' / ')
        lines.append(f"| {short} ({row['policy']}) | {row['dataset']} | {fmt(row['scored_cases_per_seed'],0)} | "
                     f"{fmt(row['seeds'],0)} | {fmt(row['cycle_days'],3)} | {fmt(row['outcome'],3)} | "
                     f"{fmt(row['uplift_percent'])} % | {link(Path(row['source']),'CSV')} |")
    lines += ["", "**Einordnung**", "",
              "Bei den vollständigen Hauptvergleichen ist der direkte Restzeitkopf der stärkste PSP; "
              "unter den dort vollständig ausgewerteten Suffix-Policies ist (c) stärker als die Log-Gap-Varianten. "
              "OE verbessert Teile der freien Suffixvorhersage, liefert aber schwächere und noisigere Optionsvergleiche. "
              "Gute Prognosen beobachteter Fortsetzungen, gute Zeitkalibrierung und gute Entscheidungen sind hier "
              "drei getrennt zu prüfende Eigenschaften. Weder CF-Effekte noch AIPW-Vorzeichen liefern eine bekannte "
              "individuelle Gegenfakten-Wahrheit.", "",
              "Alle Zahlen sind aus vorhandenen CSV-, JSON- und Logdateien zusammengeführt. "
              "Es wurden keine neuen Trainings oder Simulationen gestartet. Die Originaldateien wurden nicht geändert."]
    (OUT / "uebersicht.md").write_text("\n".join(lines) + "\n")
    metadata = {"created_at": now, "online_report_rows": len(inventory), "main_comparisons": len(main_rows),
                "partial_300": partials, "common_300_seeds": common,
                "quality_prefix_sets_identical": True, "mini_bank_case_durations_identical": True,
                "main_bank_seed_means_identical": True, "long_oe_masked_progress_snapshot": long_progress,
                "sources_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(SOURCES)}}
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(OUT/'uebersicht.md'), "inventory_rows": len(inventory),
                      "main_comparisons": len(main_rows), "quality_models": len(quality), "partial_300": partials},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
