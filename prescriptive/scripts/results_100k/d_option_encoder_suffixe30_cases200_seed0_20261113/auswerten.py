"""Compare the corrected D mini run with the existing paired C and OE runs."""
import csv
from datetime import datetime
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
SOURCES = {
    "d_korrigiert": OUT / "d_korrigiert",
    "c_original": ROOT / "c_shared_gap_suffixe30_paired200_seed0/original_1ms",
    "c_ein_abstand": ROOT / "c_shared_gap_suffixe30_paired200_seed0/normal_einmal_pro_option",
    "oe": ROOT / "token_oe_suffixe30_vergleich_c_seed0_20261113/oe_ohne_sperre",
}
LABELS = {
    "d_korrigiert": "(d), Optionsabstand pro Suffix gezogen",
    "c_original": "(c), ursprüngliche 1 ms",
    "c_ein_abstand": "(c), ein Normal-Abstand je Option für alle Suffixe",
    "oe": "OE, Optionsabstand pro Suffix gezogen",
}


def read(path):
    return json.loads(path.read_text())


def main():
    rows, banks, progresses = {}, {}, {}
    for key, folder in SOURCES.items():
        progresses[key] = read(folder / "progress.json")
        assert progresses[key]["status"] == "complete", (key, progresses[key])
        assert progresses[key]["seed"] == 0 and progresses[key]["samples_per_option"] == 30
        assert progresses[key]["scored_cases"] == 200
        assert not progresses[key]["transition_rules_expected"]
        rows[key] = read(folder / "psp_seed0_row.json")
        banks[key] = read(folder / "bank_seed0_row.json")
        assert rows[key]["Fälle"] == 200
        assert banks[key]["_days"] == banks["d_korrigiert"]["_days"]
    validation = read(OUT / "sampling_validation.json")
    assert validation["status"] == "passed" and validation["weights_unchanged"]
    days = {key: row["Durchlaufzeit (Tage)"] for key, row in rows.items()}
    bank_days = banks["d_korrigiert"]["Durchlaufzeit (Tage)"]
    summary = {
        "seed": 0, "samples_per_option": 30, "scored_cases": 200, "allocation": "random",
        "transition_rules": False, "bank_case_durations_identical": True,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "bank_days": bank_days, "policy_days": days,
        "d_improvement_vs_bank_days": bank_days - days["d_korrigiert"],
        "d_improvement_vs_bank_percent": 100 * (bank_days - days["d_korrigiert"]) / bank_days,
        "d_minus_oe_days": days["d_korrigiert"] - days["oe"],
        "d_minus_c_original_days": days["d_korrigiert"] - days["c_original"],
        "d_minus_c_single_gap_days": days["d_korrigiert"] - days["c_ein_abstand"],
        "d_elapsed_seconds": progresses["d_korrigiert"]["elapsed_seconds"],
        "d_decisions": progresses["d_korrigiert"]["decisions_completed"],
        "sampling_validation": "passed", "sources": {k: str(v) for k, v in SOURCES.items()},
    }
    (OUT / "vergleich.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    comparison_rows = [{"variant": key, "description": LABELS[key], "cycle_days": days[key],
                        "outcome": rows[key]["Outcome je Fall"], "bank_cycle_days": bank_days,
                        "uplift_percent": 100 * (bank_days - days[key]) / bank_days,
                        "scored_cases": 200, "seed": 0, "samples_per_option": 30} for key in SOURCES]
    with (OUT / "vergleich.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    lines = [
        "**(d): gezogener Optionsabstand je Suffixziehung**", "",
        "Seed 0; 30 Ziehungen pro Option; 200 gewertete Fälle (700–899); 700 Warmup- und 1.300 Cooldown-Fälle. "
        "Zufällige Mitarbeiterzuteilung, keine Übergangssperre. Die Bank-Dauern aller 200 Fälle stimmen "
        "mit den drei früheren Vergleichsläufen exakt überein.", "",
        "| Variante | Mittlere Durchlaufzeit (Tage) |", "|---|---:|",
        *[f"| {LABELS[key]} | {days[key]:.3f} |" for key in SOURCES], f"| Bank | {bank_days:.3f} |", "",
        f"(d) minus OE: {summary['d_minus_oe_days']:+.3f} Tage; "
        f"(d) minus ursprüngliches (c): {summary['d_minus_c_original_days']:+.3f} Tage. "
        "Positive Differenzen bedeuten längere Durchlaufzeit.", "",
        f"Laufzeit: {summary['d_elapsed_seconds'] / 60:.1f} Minuten; {summary['d_decisions']} Entscheidungen.", "",
        "**Umsetzung und Prüfung**", "",
        "Das vorhandene (d)-Modell wurde kopiert. Die einzige Konfigurationsänderung ist option_as = encoder. "
        "Es gab kein neues Training; gelernte Gewichte und das Originalmodell sind unverändert. "
        "Event und Zeitklasse bleiben getrennte Ausgaben von (d). Die vorhandene Ziehungslogik wird auch für OE verwendet.", "",
        "Alle Ziehungen starten mit derselben beobachteten Vorgeschichte. Für jede Option werden 30 erste Abstände "
        "aus dem Modell gezogen: Zeitklasse und dann ein Abstand aus der auf diese Klasse trunkierten Normalverteilung. "
        "Für jede Ziehung wird die Option samt Abstand, Fallalter, Wochentag, Tageszeit und Zeitklasse im erweiterten "
        "Encoder-Prefix eingesetzt. Von dort folgt der restliche Suffix. Der Optionsabstand zählt genau einmal zur bewerteten Dauer.", "",
        "Die Optionen verwenden gemeinsame Zufallszahlen, wie bei den bisherigen PSP-Vergleichen. "
        "Bei (d) hängen die erste Zeitklasse und die Gaussian-Parameter vor dem Options-Event nicht vom erzwungenen "
        "Kandidaten ab. Die gepaarten ersten Abstände können deshalb zwischen den Optionen gleich sein; "
        "über die 30 Suffixziehungen variieren sie. Der nachfolgende Encoder sieht die jeweilige Option und ihren Abstand.", "",
        "Die Prüfung am echten Modell bestätigte 30 unterschiedliche Abstände pro Option, unveränderte beobachtete "
        "Vorgeschichte, passende Zeitmerkmale und Zeitklassen sowie die einmalige Berücksichtigung des Optionsabstands. "
        "Durch das Anhängen des Events verschiebt sich lediglich die linke Auffüllung im festen Encoderfenster. "
        "Details: sampling_validation.json; Parameter und Dateiprüfsummen: run_manifest.json.", "",
        "**Wahlhäufigkeiten**", "",
        "| Kennzahl | (d), korrigiert | (c), 1 ms | (c), ein Abstand | OE |", "|---|---:|---:|---:|---:|",
    ]
    for key in rows["d_korrigiert"]:
        if key.startswith("Anteil ") and ": " in key:
            lines.append("| " + key + " | " + " | ".join(f"{100 * rows[variant][key]:.2f} %" for variant in SOURCES) + " |")
    lines.extend([
        "", "**Einordnung**", "",
        "Ein Seed mit 200 gewerteten Fällen und 30 Ziehungen ist ein Mini-Test. "
        "Die ursprüngliche (d)-Variante mit 1 ms wurde hier nicht zusätzlich online ausgeführt. "
        "Der Vergleich mit (c) und OE isoliert deshalb nicht die Wirkung der Zeitkorrektur bei (d). "
        "Die Online-Dauern stammen aus den tatsächlichen Simulator-Ereignissen, nicht aus den vorhergesagten Suffixzeiten.",
    ])
    (OUT / "auswertung.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
