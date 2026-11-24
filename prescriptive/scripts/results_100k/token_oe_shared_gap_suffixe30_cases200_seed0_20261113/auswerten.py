"""Compare OE with one model-drawn option gap against the existing mini runs."""
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
SOURCES = {
    "oe_ein_abstand": OUT / "oe_ein_abstand_neustart",
    "oe_pro_suffix": ROOT / "token_oe_suffixe30_vergleich_c_seed0_20261113/oe_ohne_sperre",
    "d_pro_suffix": ROOT / "d_option_encoder_suffixe30_cases200_seed0_20261113/d_korrigiert",
    "c_original": ROOT / "c_shared_gap_suffixe30_paired200_seed0/original_1ms",
    "c_ein_abstand": ROOT / "c_shared_gap_suffixe30_paired200_seed0/normal_einmal_pro_option",
}
LABELS = {
    "oe_ein_abstand": "OE, ein Modell-Abstand je Option für alle 30 Suffixe",
    "oe_pro_suffix": "OE, ein Modell-Abstand je Suffixziehung",
    "d_pro_suffix": "(d), korrigierter Optionsabstand je Suffixziehung",
    "c_original": "(c), ursprüngliche 1 ms",
    "c_ein_abstand": "(c), ein Normal-Abstand je Option für alle 30 Suffixe",
}


def read(path):
    return json.loads(path.read_text())


def main():
    manifest = read(OUT / "run_manifest.json")
    hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
              for name, expected in manifest["sha256"].items()}
    assert all(hashes.values()), hashes
    validation = read(OUT / "sampling_validation.json")
    assert validation["status"] == "passed"
    for key in ("original_model_unchanged", "weights_unchanged", "observed_history_unchanged",
                "shared_gap_independent_of_suffix_count", "one_suffix_equals_default_oe_exactly"):
        assert validation[key], key
    assert validation["sampling_call_sizes"] == [1, 30, 1, 30]
    rows, banks, progresses = {}, {}, {}
    for key, folder in SOURCES.items():
        progress = progresses[key] = read(folder / "progress.json")
        assert progress["status"] == "complete", (key, progress)
        expected = {"seed": 0, "samples_per_option": 30, "scored_cases": 200,
                    "warmup": 700, "cooldown": 1300, "online_cases": 2200,
                    "transition_rules_expected": False}
        assert all(progress[name] == value for name, value in expected.items()), (key, progress)
        rows[key] = read(folder / "psp_seed0_row.json")
        banks[key] = read(folder / "bank_seed0_row.json")
        for row in (rows[key], banks[key]):
            assert row["Fälle"] == 200 and row["seed"] == 0 and len(row["_days"]) == 200
            assert math.isclose(sum(row["_days"]) / 200, row["Durchlaufzeit (Tage)"], abs_tol=1e-10)
        assert banks[key]["_days"] == banks["oe_ein_abstand"]["_days"], key
    assert progresses["oe_ein_abstand"]["shared_option_gap_expected"] is True
    assert progresses["oe_ein_abstand"]["model"] == manifest["model"]
    choices = [json.loads(line) for line in (SOURCES["oe_ein_abstand"] / "choices.jsonl").read_text().splitlines()]
    assert len(choices) == progresses["oe_ein_abstand"]["decisions_completed"]
    initial_choices = [json.loads(line) for line in (OUT / "oe_ein_abstand/choices.jsonl").read_text().splitlines()]
    compared = min(len(initial_choices), len(choices))
    replay_fields = ("case", "point", "choice", "simulation_time", "option_gap_draws")
    assert all(all(old[key] == new[key] for key in replay_fields)
               for old, new in zip(initial_choices[:compared], choices[:compared]))
    counts = Counter()
    for choice in choices:
        assert 700 <= int(choice["case"].rsplit("_", 1)[1]) < 900
        draw = choice["option_gap_draws"]
        assert draw["point"] == choice["point"] and draw["source"] == "decoder"
        assert draw["draws_per_case_and_option"] == 1
        assert choice["choice"] in draw["gaps_seconds"]
        for gaps in draw["gaps_seconds"].values():
            assert len(gaps) == 1 and math.isfinite(gaps[0]) and gaps[0] >= 0
        counts[f"{choice['point']}: {choice['choice']}"] += 1
    assert dict(counts) == progresses["oe_ein_abstand"]["choices_by_point"]
    for point in {choice["point"] for choice in choices}:
        assert rows["oe_ein_abstand"][f"Wahlen der Policy: {point}"] == sum(
            choice["point"] == point for choice in choices)
    days = {key: row["Durchlaufzeit (Tage)"] for key, row in rows.items()}
    bank_days = banks["oe_ein_abstand"]["Durchlaufzeit (Tage)"]
    summary = {
        "seed": 0, "samples_per_option": 30, "scored_cases": 200, "allocation": "random",
        "transition_rules": False, "bank_case_durations_identical": True,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "bank_days": bank_days, "policy_days": days,
        "oe_shared_minus_per_suffix_days": days["oe_ein_abstand"] - days["oe_pro_suffix"],
        "oe_shared_minus_d_corrected_days": days["oe_ein_abstand"] - days["d_pro_suffix"],
        "oe_shared_improvement_vs_bank_days": bank_days - days["oe_ein_abstand"],
        "oe_shared_improvement_vs_bank_percent": 100 * (bank_days - days["oe_ein_abstand"]) / bank_days,
        "elapsed_seconds": progresses["oe_ein_abstand"]["elapsed_seconds"],
        "decisions": len(choices), "all_decisions_have_one_gap_per_option": True,
        "initial_attempt_decisions_reproduced_exactly": compared,
        "sampling_validation": "passed", "source_hashes_unchanged": hashes,
        "sources": {key: str(folder) for key, folder in SOURCES.items()},
    }
    (OUT / "vergleich.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    comparison = [{"variant": key, "description": LABELS[key], "cycle_days": days[key],
                   "bank_cycle_days": bank_days, "scored_cases": 200, "seed": 0,
                   "samples_per_option": 30} for key in SOURCES]
    with (OUT / "vergleich.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    lines = [
        "**OE: ein gezogener Optionsabstand für alle 30 Suffixe**", "",
        "Seed 0; 30 Suffixziehungen pro Option; 200 gewertete Fälle (700–899); "
        "700 Warmup- und 1.300 Cooldown-Fälle. Zufällige Mitarbeiterzuteilung, keine Übergangssperre.", "",
        "| Variante | Mittlere Durchlaufzeit (Tage) |", "|---|---:|",
        *[f"| {LABELS[key]} | {days[key]:.3f} |" for key in SOURCES],
        f"| Bank | {bank_days:.3f} |", "",
        f"OE mit einem Abstand minus bisheriges OE: {summary['oe_shared_minus_per_suffix_days']:+.3f} Tage; "
        f"minus korrigiertes (d): {summary['oe_shared_minus_d_corrected_days']:+.3f} Tage. "
        "Positive Differenzen bedeuten längere Durchlaufzeit.", "",
        f"Laufzeit: {summary['elapsed_seconds'] / 60:.1f} Minuten; {len(choices)} Entscheidungen.", "",
        "**Umsetzung und Prüfung**", "",
        "Jede Entscheidung beginnt mit der beobachteten Vorgeschichte. Pro Option wird einmal aus dem vorhandenen "
        "OE-Modell gezogen: gemeinsames Event/Zeitklassen-Token unter der erzwungenen Option, dann ein Abstand aus "
        "der auf diese Zeitklasse trunkierten Normalverteilung. Dieser Abstand wird für alle 30 Fortsetzungen "
        "derselben Option wiederholt. Es wird kein neuer Abstand für jede Fortsetzung gezogen.", "",
        "Das erweiterte Encoder-Prefix enthält die Option mit diesem Abstand, Fallalter, Wochentag, Tageszeit und "
        "passender Zeitklasse. Alle 30 Fortsetzungen einer Option sehen dieses identische erweiterte Prefix. "
        "Die gemeinsame Ausgangsbasis zwischen den Optionen ist die beobachtete Vorgeschichte; die erweiterten "
        "Prefixe enthalten jeweils das eigene Options-Event. Der Optionsabstand zählt genau einmal zur bewerteten Dauer.", "",
        "Das OE-Modell wurde kopiert und ausschließlich shared_option_gap = true ergänzt. Kein neues Training; "
        "Originalmodell und gelernte Gewichte sind unverändert. Die Abstände stammen weiterhin aus dem OE-Decoder. "
        "Der frühere (c)-Versuch verwendete dagegen empirisch angepasste Normalverteilungen je Wahlpunkt und Option. "
        "Die Analogie zu (c) betrifft das Teilen eines Abstands durch die 30 Suffixe.", "",
        "Die bisherigen gemeinsamen Zufallszahlen zwischen den Optionen bleiben erhalten. Verschiedene Optionen "
        "können dadurch denselben Zahlenwert erhalten; pro Option erfolgt dennoch genau eine Modellziehung.", "",
        "Die Prüfung am echten Modell bestätigte die Aufrufgrößen 1/30/1/30, identische erweiterte Prefixe je Option, "
        "unveränderte beobachtete Historie und passende Zeitmerkmale. Mit einer einzigen Fortsetzung stimmen die "
        "Ergebnisse exakt mit dem bisherigen OE überein. Neun gezielte Unit-Tests bestanden. "
        f"Alle {len(choices)} protokollierten Online-Entscheidungen enthalten genau einen endlichen, nichtnegativen "
        "Abstand je Option. Die Bank-Dauern aller 200 Fälle stimmen in sämtlichen Vergleichsläufen exakt überein. "
        "Die Dateiprüfsummen aus run_manifest.json stimmen nach dem Lauf unverändert.", "",
        "**Wahlhäufigkeiten**", "",
        "| Kennzahl | OE, ein Abstand | OE, pro Suffix | (d), korrigiert | (c), 1 ms | (c), ein Abstand |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in rows["oe_ein_abstand"]:
        if key.startswith("Anteil ") and ": " in key:
            lines.append("| " + key.replace("|", "\\|") + " | " + " | ".join(f"{100 * rows[variant][key]:.2f} %" for variant in SOURCES) + " |")
    lines.extend([
        "", "**Einordnung**", "",
        "Dies ist ein Mini-Test mit einem Seed. Unterschiede zwischen den Varianten sind damit nicht statistisch "
        "abgesichert. Die Durchlaufzeiten stammen aus den tatsächlichen Simulator-Ereignissen, nicht aus den "
        "vorhergesagten Suffixzeiten. Details der Prüfung stehen in sampling_validation.json, die gezogenen Abstände "
        "je Entscheidung in oe_ein_abstand_neustart/choices.jsonl, die Konfiguration in run_manifest.json. "
        "Der erste Versuch unter oe_ein_abstand blieb nach 147 Entscheidungen ohne Abschluss stehen. "
        "Er ist aufbewahrt; ausgewertet wird ausschließlich der vollständige Neustart mit identischer Konfiguration.",
        f"Die ersten {compared} Entscheidungen einschließlich der gezogenen Abstände wurden im Neustart exakt reproduziert.",
    ])
    (OUT / "auswertung.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
