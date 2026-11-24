"""Summarize the isolated D draw diagnosis as reviewable German artifacts."""
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prescriptive.scripts.diagnose_d_sampling_20261115 import OUT, REFERENCE, CASE, ACT, LC, TS
from prescriptive.policies.ev_policy_event_costs.costs import EventCosts
import numpy as np
import pandas as pd


def main():
    read = lambda name: json.loads((OUT / name).read_text())
    assert read("replay_validation.json")["status"] == "passed"
    assert read("fork_validation.json")["status"] == "passed"
    assert read("original_sampling_reproduction.json")["status"] == "passed"
    selected = set(read("selected_decisions.json"))
    estimates = pd.read_csv(OUT / "psp_estimates.csv")
    draws = pd.read_csv(OUT / "psp_draws.csv")
    branches = pd.read_csv(OUT / "simulator_branches.csv")
    assert len(draws) == 36000 and len(estimates) == 72 and len(branches) == 72
    assert set(draws.id) == set(branches.id) == selected and len(selected) == 12
    paired = estimates.groupby(["id", "point", "option"], sort=True).mean_cost.mean().rename("psp_mean_cost").to_frame()
    actual = branches.groupby(["id", "point", "option"], sort=True).total_eur.agg(["mean", "std"])
    paired = paired.join(actual.rename(columns={"mean": "sim_mean_cost", "std": "sim_sd_cost"}))
    paired["ratio"] = paired.psp_mean_cost / paired.sim_mean_cost
    paired.to_csv(OUT / "prediction_vs_simulator.csv")
    prefixes = {r["id"]: r for r in pickle.loads((OUT / "prefixes.pkl").read_bytes())}
    costs = EventCosts(REFERENCE / "event_costs.yaml")
    raw = pd.read_pickle(ROOT / "prescriptive/data/cache/bpic17_raw.pkl")
    raw = raw[[CASE, ACT, LC, TS]].copy()
    raw[ACT] = raw[ACT].astype(str).str.strip()
    raw = raw.sort_values([CASE, TS], kind="stable")
    raw["gap"] = pd.to_datetime(raw[TS]).groupby(raw[CASE], observed=True).diff().dt.total_seconds().fillna(0)
    raw["token"] = raw[ACT] + "|" + raw[LC].astype(str)
    edges = np.array([1, 600, 4*3600, 86400, 3*86400, 7*86400, 14*86400, 30*86400])
    raw["class"] = np.searchsorted(edges, raw.gap.to_numpy(), side="right")
    seen = set(zip(raw.token, raw["class"]))
    event_rows, intervals, path_rows = [], [], []
    for rid in sorted(selected):
        record = prefixes[rid]
        for oi, option in enumerate(record["options"]):
            suffixes = pickle.loads((OUT / f"suffixes_{rid:03d}_{oi}.pkl").read_bytes())
            for k, suffix in enumerate(suffixes):
                active = dict(costs.state(record["events"]).opened)
                at, unseen = record["now"], 0
                for step, (token, _, gap) in enumerate(suffix):
                    at += gap
                    a, _, lc = token.rpartition("|")
                    missing = (token, int(np.searchsorted(edges, gap, side="right"))) not in seen
                    unseen += missing
                    event_rows.append({"id": rid, "option": oi, "draw": k, "token": token, "gap": gap,
                                       "unseen_event_class": missing})
                    kind = None
                    if lc in ("start", "resume"):
                        if a in active:
                            began, kind = active[a], "reopen"
                        active[a] = at
                    elif lc in ("suspend", "complete", "ate_abort", "withdraw") and a in active:
                        began, kind = active.pop(a), "close"
                    if kind:
                        hours = (at-began)/3600
                        intervals.append({"id": rid, "option": oi, "draw": k, "step": step,
                                          "activity": a, "kind": kind, "hours": hours,
                                          "cost": hours*costs.tariff(a)["eur_per_work_hour"]})
                for a, began in active.items():
                    hours = (at-began)/3600
                    intervals.append({"id": rid, "option": oi, "draw": k, "step": len(suffix),
                                      "activity": a, "kind": "unclosed", "hours": hours,
                                      "cost": hours*costs.tariff(a)["eur_per_work_hour"]})
                path_rows.append({"id": rid, "option": oi, "draw": k, "any_unseen_event_class": bool(unseen)})
    sampled_events, work, paths = pd.DataFrame(event_rows), pd.DataFrame(intervals), pd.DataFrame(path_rows)
    work.to_csv(OUT / "sampled_work_intervals.csv", index=False)
    sampled_events.groupby("token").agg(n=("gap", "size"), median_gap=("gap", "median"),
            longer_4h=("gap", lambda x: (x>4*3600).mean()),
            unseen=("unseen_event_class", "mean")).to_csv(OUT / "sampled_gap_by_token.csv")
    paired_noise = {}
    for key in ("total_eur", "days"):
        rows = []
        for (rid, repeat), group in draws.groupby(["id", "repeat"]):
            wide = group.pivot(index="draw", columns="option", values=key)
            delta = wide.iloc[:, 0]-wide.iloc[:, 1]
            se = delta.std(ddof=1)/np.sqrt(len(delta))
            rows.append({"id": rid, "repeat": repeat, "diff": delta.mean(), "se": se, "z": abs(delta.mean())/se})
        table = pd.DataFrame(rows)
        table.to_csv(OUT / f"paired_uncertainty_{key}.csv", index=False)
        paired_noise[key] = {"median_se": table.se.median(), "fraction_abs_z_above_2": table.z.gt(2).mean(),
                "fraction_prefixes_with_choice_flips": table.groupby("id")["diff"].agg(lambda x: x.min()<0<x.max()).mean()}
    real = read("real_lifecycle_summary.json")
    original = pd.read_csv(OUT / "original_seed_psp_estimates.csv")
    case750 = pd.read_csv(OUT / "case750_branches.csv")
    gap750 = read("case750_first_gap_decomposition.json")
    old750 = original[original.id.eq(102)].set_index("option")
    first750 = case750[case750.repeat.eq(0)].set_index("option")
    re750 = pd.read_csv(OUT / "case750_psp_estimates.csv").pivot(index="repeat", columns="option", values="mean_cost")
    base = REFERENCE / "online"
    bank = pd.read_csv(base / "bank_seed0/case_costs.csv").set_index(CASE)
    psp = pd.read_csv(base / "psp_seed0/case_costs.csv").set_index(CASE)
    delta = psp.total_eur-bank.total_eur
    stats = {"prefixes": 12, "psp_suffixes": len(draws), "simulator_branches": len(branches),
         "psp_mean_remaining_cost": draws.total_eur.mean(), "sim_mean_remaining_cost": branches.total_eur.mean(),
         "psp_mean_remaining_days": draws.days.mean(), "sim_mean_remaining_days": branches.remaining_days.mean(),
         "psp_mean_active_hours": draws.active_hours.mean(), "psp_time_cost_share": draws.time_eur.sum()/draws.total_eur.sum(),
         "missing_business_end_fraction": draws.missing_business_end.mean(),
         "open_sessions_at_end_fraction": draws.open_at_end.gt(0).mean(), "cap_fraction": draws.at_cap.mean(),
         "unseen_event_class_fraction": sampled_events.unseen_event_class.mean(),
         "paths_with_unseen_event_class_fraction": paths.any_unseen_event_class.mean(),
         "identical_option_first_gaps_all_36_repeats": bool(estimates.same_first_gaps_for_both_options.all()),
         "sampled_work_over4h_fraction": work.hours.gt(4).mean(),
         "sampled_work_over24h_fraction": work.hours.gt(24).mean(),
         "real_work_over4h_fraction": real["over_4h_fraction"],
         "sampled_time_cost_by_interval_kind": work.groupby("kind").cost.sum().to_dict(),
         "noise": paired_noise, "d500_minus_bank_per_case": delta.mean(), "d500_minus_bank_200_total": delta.sum(),
         "case750_original_psp_difference_start_minus_accept": float(old750.loc['W_Complete application|start', 'mean_cost']-old750.loc['A_Accepted', 'mean_cost']),
         "case750_original_psp_paired_se": float(old750.paired_se.iloc[0]),
         "case750_original_future_sim_costs": first750.total_eur.to_dict(),
         "case750_start_chosen_in_new_repeats": int((re750['W_Complete application|start']<re750['A_Accepted']).sum())}
    (OUT / "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2)+"\n")
    # Standalone scientific figure; compare the same conditional states and work durations.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    per_id = paired.groupby("id")[["psp_mean_cost", "sim_mean_cost"]].mean()
    x = np.arange(len(per_id))
    axes[0].plot(x, per_id.psp_mean_cost, 'o-', label="D: 3 × 500 Suffixe je Option")
    axes[0].plot(x, per_id.sim_mean_cost, 's-', label="Simulator: 3 Fortsetzungen je Option")
    axes[0].set(yscale='log', ylabel="Verbleibende Kosten (€), logarithmisch", xlabel="Entscheidungszustand",
                title="Gleiche Zustände, stark verschiedene Kosten")
    axes[0].set_xticks(x, per_id.index.astype(str))
    axes[0].legend(fontsize=8)
    true_work = pd.read_csv(OUT / "real_work_intervals.csv")
    for name, values in (("Echter Log", true_work.hours), ("D-Suffixe (erste Wiederholung)", work.hours)):
        values = np.sort(np.maximum(values.to_numpy(), 1e-6))
        axes[1].plot(values, 1-(np.arange(len(values))+.5)/len(values), label=name)
    axes[1].set(xscale='log', yscale='log', xlabel="Arbeitsabschnitt (Stunden)", ylabel="Anteil längerer Abschnitte",
                title="D legt lange Wartezeiten in offene Arbeitssitzungen", ylim=(1e-4, 1))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(OUT / "diagnose.png", dpi=180)
    plt.close(fig)
    lines = ["# Warum D nicht besser abschneidet: Ziehverhalten und Simulator", "",
        "Untersucht wurde die bisherige D-Variante mit Übergangssperre und einzeln gezogenem Optionsabstand. "
        "Referenz: Eventkosten, 200 gewertete Fälle, 500 Ziehungen je Option, Seed 0. "
        "Das aktuell trainierende D mit konstantem TF 1 ist ein anderes, hier nicht bewertetes Modell.", "",
        "## Befund", "",
        "Beim Eventkosten-Ziel liegt das klarste systematische Problem im PSP: D kombiniert lange Zeitabstände mit "
        "laufenden Arbeitssitzungen und erzeugt dadurch massiv überhöhte Arbeitskosten. Hinzu kommt eine konkrete "
        "Schwäche der Simulator-/Policy-Schnittstelle: Optionen werden ohne Prüfung bereits offener Sitzungen angeboten. "
        "Monte-Carlo-Rauschen kann dann eine seltene, besonders teure Wahl auslösen. "
        "Beim Durchlaufzeit-Ziel sind die Kostenfehler weniger sichtbar; dort bleiben schwache und verrauschte "
        "Optionsunterschiede sowie unvollständige Pfade als gemessene Probleme.", "",
        "## Wie geprüft wurde", "",
        "Alle 582 historischen Entscheidungen wurden mit ihren gespeicherten Wahlen erneut simuliert. "
        "Alle 7.954 gewerteten Events, Mitarbeiter, Zeitstempel und Fallkosten stimmen überein. "
        "Aus jedem der vier Wahlpunkte wurden drei verschiedene Fälle mit Seed 20261115 ausgewählt: 12 Zustände. "
        "Je Zustand und Option wurden dreimal 500 D-Suffixe gezogen, insgesamt 36.000 Suffixe. "
        "Zusätzlich wurde die Simulator-Engine am exakten Wahlzustand gesichert. Jede der zwölf Kopien "
        "reproduziert bei gespeicherter Fortsetzung den ursprünglichen Ziel-Fall exakt.", "",
        "Für den Optionsvergleich wurde danach jeweils genau eine Aktion erzwungen und die gesamte Engine mit "
        "normalem Bankverhalten fortgesetzt, einschließlich anderer Fälle, Dienstplan und Mitarbeiterkonkurrenz. "
        "Je Option gibt es drei Fortsetzungen: einmal mit dem gesicherten Zufallszustand, zweimal mit "
        "neu gezogenen künftigen Zufallszahlen. Die drei Fortsetzungen sind eine kleine Diagnose, keine präzise "
        "Schätzung individueller kausaler Effekte und kein neuer 200-Fälle-Online-Benchmark.", "",
        "Außerdem wurden für 13 Entscheidungen die ursprünglichen D-Zufallszustände rekonstruiert. "
        "Alle 13 neu berechneten Wahlen stimmen mit dem historischen 500er-Lauf überein. "
        "Modell, Simulator, Trainingscode und laufende Trainings wurden nicht verändert.", "",
        "## 1. Kosten werden durch falsche Arbeitszeiten verzerrt", "",
        f"Über beide Optionen der zwölf Zustände schätzt D im Mittel **{stats['psp_mean_remaining_cost']:.2f} €** "
        f"verbleibende Kosten. Die 72 Simulator-Fortsetzungen ergeben im Mittel **{stats['sim_mean_remaining_cost']:.2f} €**. "
        f"D zieht durchschnittlich **{stats['psp_mean_active_hours']:.1f} aktive Arbeitsstunden**; "
        f"**{100*stats['psp_time_cost_share']:.2f} %** seiner geschätzten Kosten sind Zeitkosten. "
        "Die Kostenrechnung wurde je Suffix gegen die vorhandene Outcome-Bewertung geprüft. "
        "Der Fehler entsteht in den gezogenen Ereignissen und Zeiten, bevor daraus Kosten berechnet werden.", "",
        "Zum Größenvergleich: Die unveränderten Tarife ergeben im gesamten echten Log 168,18 € und 0,782 aktive Stunden "
        "je vollständigem Fall. Die Bank im Online-Lauf liegt bei 174,03 € und 0,844 Stunden. "
        "Diese beiden Gesamtfall-Mittel sind Kontextwerte, keine mit den zwölf Entscheidungszuständen gepaarte Stichprobe.", "",
        "Konkreter D-Pfad: Application_703, Option `W_Handle leads|start`, Wiederholung 0, Ziehung 77. "
        "Dieser Pfad liegt ungefähr bei den medianen Kosten der 500 Suffixe dieser Option. "
        "Nach `W_Call after offers|start` und einem fast sofortigen `A_Complete` folgen bis zum "
        "`W_Call after offers|suspend` **103,39 Stunden**. Allein diese Arbeitssitzung kostet **4.135,65 €** "
        "bei 40 €/h. Der gesamte reale Log hat in dieser Kosten-/Sitzungsdefinition keinen Arbeitsabschnitt "
        "über 19,12 Stunden; Median 1,63 Minuten, 99. Perzentil 45,59 Minuten.", "",
        "Die Gesamt-Restzeit wirkt dabei weniger auffällig: "
        f"D {stats['psp_mean_remaining_days']:.2f} Tage gegenüber {stats['sim_mean_remaining_days']:.2f} Tagen "
        "im kleinen Simulatorvergleich. Eine ungefähr plausible Gesamtdauer garantiert keine plausible Aufteilung "
        "in Arbeitszeit und Pause.", "",
        "## 2. Getrennte Köpfe: Die erzwungene Option ändert ihren eigenen ersten Abstand nicht", "",
        "In allen 36 geprüften D-Ziehungsaufrufen sind die 500 ersten Optionsabstände für beide Optionen "
        "elementweise identisch. Das ist durch den D-Code erklärbar: Event und Zeitklasse werden aus demselben "
        "bisherigen Zustand getrennt gezogen; die Zeitklasse und numerische Zeitverteilung sehen das gerade "
        "erzwungene aktuelle Event nicht. Mit gemeinsamen Zufallszahlen ergibt das dieselben Abstände. "
        "Erst beim erneuten Encodieren des Prefixes beeinflusst die Option die folgenden Ereignisse.", "",
        "Beispiel: ID 123 (Application_762, `annehmen_bei_konzept`). Die Simulator-Verteilung des ersten "
        "reinen Modellabstands liegt für `start` bei durchschnittlich 15,76 Stunden, für `A_Accepted` bei "
        "5,90 Minuten (je 10.000 Ziehungen, vor zusätzlicher Mitarbeiterwartezeit). D zieht für beide "
        "Optionen exakt dieselbe Abstandsfolge. Damit fehlt schon im ersten Schritt ein Teil des echten "
        "Optionsunterschieds des ausgewerteten Simulators.", "",
        "## 3. Die Übergangssperre erfüllt ihre Aufgabe, aber keine vollständige Ablaufprüfung", "",
        f"In {100*stats['paths_with_unseen_event_class_fraction']:.1f} % der 12.000 gespeicherten Suffixe "
        f"der ersten Wiederholung kommt mindestens eine Kombination aus Event und Zeitklasse vor, die im gesamten "
        f"echten Log nie vorkommt ({100*stats['unseen_event_class_fraction']:.1f} % aller gezogenen Events). "
        "Die Sperre prüft Aktivität|Lifecycle → Aktivität|Lifecycle sowie erlaubte Endtokens. "
        "Sie prüft weder die Zeitklasse noch den Zustand aller Arbeitssitzungen.", "",
        f"**{100*stats['missing_business_end_fraction']:.2f} %** der 36.000 Suffixe enden ohne "
        "`A_Pending`, `A_Denied` oder `A_Cancelled`, obwohl auch der Prefix kein solches Event enthält. "
        "Im gesamten echten Log fehlen diese geschäftlichen Abschlussmarker bei 0,31 % der Fälle. "
        f"**{100*stats['open_sessions_at_end_fraction']:.2f} %** der Suffixe lassen eine Arbeitssitzung offen; "
        "im echten Gesamtlog sind am Fallende keine geöffneten Abschnitte übrig. "
        f"Nur **{100*stats['cap_fraction']:.4f} %** der Suffixe erreichen die Längengrenze. "
        "Das hier gemessene vorzeitige Ende liegt damit überwiegend an gezogenem EOS, nicht am Längenlimit.", "",
        "`withdraw` oder `ate_abort` ohne vorherigen Beginn sind im echten Log häufig und können das "
        "Zurückziehen einer nur geplanten Aufgabe bedeuten. Die Gesamtzahl solcher Endevents allein "
        "wurde deshalb nicht als Beweis ungültiger Pfade verwendet.", "",
        "## 4. Fall 750: Ein konkretes Zusammenspiel von PSP-Rauschen und Simulatorzustand", "",
        "Historischer Prefix: `W_Complete application|start` → `A_Concept`. "
        "Die Arbeitssitzung ist bereits offen. Trotzdem erlaubt der Wahlpunkt wieder "
        "`W_Complete application|start`. Dieser Zustand tritt bei 36 der 162 Annahmeentscheidungen auf. "
        "D wählt in 35 davon `A_Accepted`; in Fall 750 wählt es erneut `start`.", "",
        f"Die rekonstruierten Originalschätzungen mit 500 Ziehungen sind: erneuter Start "
        f"**{old750.loc['W_Complete application|start','mean_cost']:.2f} €**, Annahme "
        f"**{old750.loc['A_Accepted','mean_cost']:.2f} €**. Der vermeintliche Startvorteil ist "
        f"**{-stats['case750_original_psp_difference_start_minus_accept']:.2f} €**, sein gepaarter "
        f"Monte-Carlo-Standardfehler aber **{stats['case750_original_psp_paired_se']:.2f} €**. "
        f"In fünf neuen 500er-Ziehungen gewinnt der Start nur {stats['case750_start_chosen_in_new_repeats']}-mal. "
        "Die ursprüngliche Wahl ist also instabil.", "",
        "Der Simulator legt zwischen `A_Concept` und erneutem `start` 20,99 Stunden. "
        "Da die alte Sitzung offen ist, werden dafür 944,56 € aktive Arbeit berechnet. "
        "Die Engine behandelt den erneuten Beginn nicht als neue Mitarbeiterzuordnung und "
        "beendet die alte Sitzung nicht vor dieser Wartezeit. Das passt zur aktuellen Kostenlogik, "
        "aber zeigt eine fehlende Zustandsprüfung an der Interventionsschnittstelle.", "",
        "Die erste Zeitverteilung lässt sich hier getrennt prüfen: In 10.000 Ziehungen aus dem "
        f"gesicherten Simulatorzustand liegt der erste Abstand für erneuten Start im Mittel bei "
        f"{gap750['sim_first_gap_means_seconds']['W_Complete application|start']/3600:.2f} Stunden, "
        f"für Annahme bei {gap750['sim_first_gap_means_seconds']['A_Accepted']/60:.2f} Minuten. "
        "Da die Arbeit schon offen ist, entspricht dieser Unterschied allein "
        f"**{gap750['missing_first_gap_cost_delta_eur']:.2f} €** zusätzlichen Arbeitskosten für den Start. "
        "D zieht den ersten Abstand für beide Optionen gleich und lässt diesen Unterschied weg. "
        "Hält man alle späteren D-Events und Abstände fest und ersetzt nur diesen ersten "
        "erwarteten Kostendifferenzterm durch den Simulatorwert, kippt die ursprüngliche D-Wahl: "
        f"Start minus Annahme wird von {gap750['original_psp_cost_delta_start_minus_accept']:.2f} € "
        f"zu **{gap750['delta_with_only_first_gap_mean_correction']:.2f} €**. "
        "Das ist eine rechnerische Zerlegung dieses Zustands, kein getesteter neuer Policy-Lauf.", "",
        "Im gesicherten Zustand und mit anschließender Bankfortsetzung kosten die beiden Varianten "
        f"**{first750.loc['W_Complete application|start','total_eur']:.2f} €** (Start) und "
        f"**{first750.loc['A_Accepted','total_eur']:.2f} €** (Annahme). "
        "Auch in den zwei weiteren Fortsetzungen ist Annahme günstiger; die einzelnen "
        "Fortsetzungen stehen in `case750_branches.csv`. Die 944,56 € dieses einen Arbeitsabschnitts "
        f"sind größer als der gesamte Kostennachteil des ursprünglichen 200-Fälle-Laufs "
        f"({stats['d500_minus_bank_200_total']:.2f} €). Das zeigt die Empfindlichkeit des Benchmarks, "
        "beweist aber nicht, dass eine einzelne Änderung alle anderen Verläufe unverändert ließe.", "",
        "## 5. Mehr Ziehungen lösen verschiedene Probleme unterschiedlich gut", "",
        "Kosten: Bei den zwölf zufällig ausgewählten Zuständen ist die Kostenrangfolge in drei "
        "Wiederholungen jeweils stabil. Die massive Verzerrung bleibt dabei bestehen. "
        "Mehr Ziehungen reduzieren Zufallsrauschen, reparieren aber weder fehlende Sitzungszustände "
        "noch die falsche Kopplung von Event und Zeit. Fall 750 ist ein zusätzlicher gezielt untersuchter "
        "Ausreißer und zeigt, dass einzelne wichtige Kostenentscheidungen trotzdem stark rauschen.", "",
        "Durchlaufzeit: Bei gleicher D-Suffixbasis wechselt die bevorzugte Zeitoption in "
        f"**{100*paired_noise['days']['fraction_prefixes_with_choice_flips']:.0f} %** der zwölf Zustände "
        "zwischen den drei 500er-Ziehungen. Der mediane gepaarte Standardfehler eines "
        f"Zeitunterschieds beträgt **{paired_noise['days']['median_se']:.2f} Tage**. "
        f"Nur **{100*paired_noise['days']['fraction_abs_z_above_2']:.1f} %** der 36 Vergleiche "
        "liegen weiter als zwei Monte-Carlo-Standardfehler von null entfernt. "
        "Das ist Unsicherheit der PSP-Schätzung, kein Konfidenzintervall des echten Interventionseffekts.", "",
        "## Einordnung und nächste Schritte", "",
        "Der Simulator ist kein gesicherter Wahrheitsmaßstab für reale Interventionen: Auch er lernt aus "
        "Beobachtungsdaten. Seine durchschnittlichen Kosten liegen näher am echten Log als die D-Prognosen, "
        "doch er erzeugt ebenfalls seltene unplausibel lange Arbeitsabschnitte, etwa 30,29 Stunden "
        "in Bank-Fall 766. Die 200-Fälle-Auswertung mit einem Seed und 176,98 € gegenüber 174,03 € "
        "reicht nicht für die Aussage, D sei grundsätzlich schlechter als die Bank. "
        "Der kleine Branch-Vergleich misst zudem genau eine Intervention mit Bankfortsetzung; "
        "die spätere wiederholte Anwendung der PSP-Policy ist ein weiterer Unterschied zum vollen Online-Lauf.", "",
        "Priorität 1: Wahloptionen gegen den aktuellen Sitzungszustand prüfen und die Bedeutung "
        "eines erneuten `start` bei bereits offener Arbeit festlegen. Priorität 2: Zeitklassen und Abstände "
        "an aktuelles Event/Lifecycle und offenen Arbeitszustand koppeln, anschließend aktive Stunden "
        "und Kosten auf zurückgehaltenen realen Prefixen kalibrieren. E hat bereits die Eventbedingung "
        "für die Zeitklasse; es sollte genau dieselbe Arbeitszeitdiagnose bestehen, bevor sein "
        "Nutzen für das Eventkosten-Ziel beurteilt wird. Priorität 3: EOS und Lifecycle-Konsistenz "
        "über den ganzen Suffix prüfen. Priorität 4: unsichere Optionsunterschiede mit gezielt "
        "zusätzlichen Ziehungen und einer begründeten Rückfallregel behandeln. Danach einen "
        "Vergleich über mehr Fälle und mehrere Seeds durchführen.", "",
        "TF 1 und Training auf Simulator-RCT-Daten können andere Fehlerquellen adressieren; "
        "sie garantieren nicht, dass die hier nachgewiesene Zeitkopplung und Zustandslücke verschwinden.", "",
        "## Konkrete Stellen im Code", "",
        f"- [PSP: getrennte Event-/Zeitklassenziehung]({ROOT}/prescriptive/estimators/psp_model.py:891) "
        "und [Optionsabstand vor erneutem Encodieren]"
        f"({ROOT}/prescriptive/estimators/psp_model.py:1262). Die erzwungene aktuelle Option "
        "konditioniert bei D ihren eigenen Abstand nicht.",
        f"- [Simulator: Auslösen der Control-Flow-Wahl]({ROOT}/process-simulation-engine/simulation/case_engine_lifecycle_aware.py:304) "
        "prüft die vorhergesagte Option, aber nicht, ob die betreffende Sitzung schon geöffnet ist.",
        f"- [Simulator: Mitarbeiterzuordnung bei start/resume]({ROOT}/process-simulation-engine/simulation/case_engine_lifecycle_aware.py:359) "
        "überspringt eine neue Zuordnung, wenn die Aktivität schon in `case.sessions` steht. "
        "Die zuvor gezogene Zeit läuft trotzdem ab.",
        f"- [Kosten: erneuter Beginn bei offener Sitzung]({ROOT}/prescriptive/policies/ev_policy_event_costs/costs.py:178) "
        "bucht die bisher offene Arbeitszeit und eröffnet den Abschnitt erneut.",
        f"- [Policy: reine Wahl nach Mittelwert]({ROOT}/prescriptive/policies/ev_policy/psp_policy.py:46) "
        "beachtet keinen Standardfehler oder Mindestabstand zur anderen Option.", "",
        "Artefakte: `summary.json`, `prediction_vs_simulator.csv`, `psp_estimates.csv`, "
        "`original_seed_psp_estimates.csv`, `psp_draws.csv`, `simulator_branches.csv`, "
        "`choice_session_context.csv`, `case750_branches.csv`, `sampled_work_intervals.csv`, "
        "`real_lifecycle_summary.json`, `simulator_first_gap_distributions.csv`, `diagnose.png`.", ""]
    (OUT / "auswertung.md").write_text("\n".join(lines))
    sources = [ROOT / "prescriptive/estimators/psp_model.py", ROOT / "process-simulation-engine/simulation/case_engine_lifecycle_aware.py",
         ROOT / "prescriptive/policies/ev_policy_event_costs/costs.py", ROOT / "prescriptive/policies/ev_policy_event_costs/psp_policy.py",
         ROOT / "prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_klassen_option_encoder_ue.pkl", REFERENCE / "manifest.json",
         Path(__file__).resolve(), ROOT / "prescriptive/scripts/diagnose_d_sampling_20261115.py"]
    (OUT / "provenance.json").write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2)+"\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
