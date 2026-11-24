"""Compare this OE repeat with the paired, completed C mini experiments."""
from datetime import datetime
import json
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
C = ROOT / "c_shared_gap_suffixe30_paired200_seed0"
SOURCES = {
    "c_original": C / "original_1ms",
    "c_normal": C / "normal_einmal_pro_option",
    "oe": OUT / "oe_ohne_sperre",
}
LABELS = {
    "c_original": "(c), ursprüngliche 1 ms",
    "c_normal": "(c), ein gezogener Normal-Abstand je Option",
    "oe": "token_oe, ohne Übergangssperre",
}


def read(path):
    return json.loads(path.read_text())


def main():
    rows, banks, progresses = {}, {}, {}
    for key, directory in SOURCES.items():
        progresses[key] = read(directory / "progress.json")
        assert progresses[key]["status"] == "complete", (key, progresses[key])
        assert progresses[key]["seed"] == 0
        assert progresses[key]["samples_per_option"] == 30
        assert progresses[key]["scored_cases"] == 200
        assert not progresses[key]["transition_rules_expected"]
        rows[key] = read(directory / "psp_seed0_row.json")
        banks[key] = read(directory / "bank_seed0_row.json")
        assert rows[key]["Fälle"] == 200
        assert banks[key]["_days"] == banks["c_original"]["_days"]

    days = {key: row["Durchlaufzeit (Tage)"] for key, row in rows.items()}
    original = days["c_original"]
    previous = ROOT / "token_oe_suffixe30_paired200_seed0_v2/ohne_sperre"
    previous_row = read(previous / "psp_seed0_row.json")
    choice_keys = ("case", "point", "choice", "simulation_time")
    new_choices = [json.loads(line) for line in (SOURCES["oe"] / "choices.jsonl").read_text().splitlines()]
    old_choices = [json.loads(line) for line in (previous / "choices.jsonl").read_text().splitlines()]
    same_choices = len(new_choices) == len(old_choices) and all(
        all(a[key] == b[key] for key in choice_keys) for a, b in zip(new_choices, old_choices)
    )
    summary = {
        "seed": 0, "samples_per_option": 30, "scored_cases": 200,
        "transition_rules": False, "bank_case_durations_identical": True,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "bank_days": banks["oe"]["Durchlaufzeit (Tage)"], "policy_days": days,
        "oe_minus_c_original_days": days["oe"] - original,
        "oe_minus_c_original_percent": 100 * (days["oe"] - original) / original,
        "oe_minus_c_normal_days": days["oe"] - days["c_normal"],
        "oe_decisions": len(new_choices),
        "oe_elapsed_seconds": progresses["oe"]["elapsed_seconds"],
        "previous_oe_choices_identical": same_choices,
        "previous_oe_case_durations_identical": rows["oe"]["_days"] == previous_row["_days"],
        "sources": {key: str(value) for key, value in SOURCES.items()},
    }
    (OUT / "vergleich.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# OE im Vergleich mit (c): Mini-Test", "",
        "Seed 0; 30 Ziehungen pro Option; 200 gewertete Fälle (700–899), 700 Warmup- und 1.300 Cooldown-Fälle.",
        "Alle Varianten verwenden zufällige Mitarbeiterzuteilung und keine Übergangssperre. "
        "Die Modelle und die Simulationskonfiguration des C-Vergleichs sind unverändert. "
        "Die Bank-Durchlaufzeiten aller 200 Fälle stimmen in allen drei Läufen exakt überein.", "",
        "| Variante | Mittlere Durchlaufzeit (Tage) |", "|---|---:|",
        f"| Bank | {summary['bank_days']:.3f} |",
        *[f"| {LABELS[key]} | {days[key]:.3f} |" for key in SOURCES], "",
        f"OE minus ursprüngliches (c): {summary['oe_minus_c_original_days']:+.3f} Tage "
        f"({summary['oe_minus_c_original_percent']:+.2f} %). Positiv bedeutet längere Durchlaufzeit.",
        f"OE minus (c) mit gezogenem Normal-Abstand: {summary['oe_minus_c_normal_days']:+.3f} Tage.", "",
        "## Wahlhäufigkeiten", "",
        "| Kennzahl | (c), 1 ms | (c), Normal-Abstand | OE |", "|---|---:|---:|---:|",
    ]
    for key in rows["c_original"]:
        if key.startswith("Anteil ") and ": " in key:
            lines.append("| " + key + " | " + " | ".join(f"{100 * rows[variant][key]:.2f} %" for variant in SOURCES) + " |")
    lines.extend([
        "", f"OE-Laufzeit: {summary['oe_elapsed_seconds'] / 60:.1f} Minuten; {len(new_choices)} Entscheidungen.",
        f"Reproduktion des früheren OE-Laufs mit demselben Seed: Entscheidungen identisch = {same_choices}; "
        f"einzelne Falldauern identisch = {summary['previous_oe_case_durations_identical']}.", "",
        "Die Durchlaufzeiten werden aus den Simulator-Ereignissen gemessen, nicht aus den PSP-Prognosen. "
        "Der erneute Lauf mit demselben Seed ist eine Reproduktionskontrolle und kein zusätzlicher unabhängiger Seed. "
        "Ein Seed und 200 Fälle erlauben keine abgesicherte Aussage über andere Seeds und erklären nicht, "
        "welcher Modellunterschied die Ergebnisdifferenz verursacht.",
    ])
    (OUT / "auswertung.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
