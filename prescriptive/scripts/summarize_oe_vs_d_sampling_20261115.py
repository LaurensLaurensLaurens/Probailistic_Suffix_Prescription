"""Add tail, session and example analysis to the paired OE/D draw probe."""
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prescriptive.scripts.diagnose_oe_vs_d_sampling_20261115 import BASE, OUT, REF, summarize, read, write
import numpy as np
import pandas as pd


def main():
    assert read(OUT / "sampling_validation.json")["status"] == "passed"
    summarize()
    d = pd.read_csv(BASE / "psp_draws.csv")
    oe_all = pd.read_csv(OUT / "draws.csv")
    oe = oe_all[oe_all.random_sample]
    table = pd.read_csv(OUT / "comparison.csv").set_index("model")
    by = pd.read_csv(OUT / "by_option.csv")
    tails = []
    for name, frame in (("D", d), ("OE", oe)):
        top_count = int(np.ceil(len(frame)*.1))
        tails.append({"model": name, "p50_cost": frame.total_eur.median(), "p90_cost": frame.total_eur.quantile(.9),
            "p95_cost": frame.total_eur.quantile(.95), "p99_cost": frame.total_eur.quantile(.99),
            "top10pct_cost_share": frame.total_eur.nlargest(top_count).sum()/frame.total_eur.sum(),
            "mean_cost_open_at_end": frame.loc[frame.open_at_end.gt(0), "total_eur"].mean(),
            "mean_cost_closed_at_end": frame.loc[frame.open_at_end.eq(0), "total_eur"].mean()})
    tail = pd.DataFrame(tails).set_index("model")
    tail.to_csv(OUT / "tail_costs.csv")
    interval_rows = []
    for name, path in (("D", BASE / "sampled_work_intervals.csv"), ("OE", OUT / "work_intervals.csv")):
        frame = pd.read_csv(path)
        if "random_sample" in frame:
            frame = frame[frame.random_sample]
        for kind, group in frame.groupby("kind"):
            interval_rows.append({"model": name, "kind": kind, "cost": group.cost.sum(), "hours": group.hours.sum(),
                                  "intervals": len(group)})
    kinds = pd.DataFrame(interval_rows)
    kinds["share_cost"] = kinds.cost/kinds.groupby("model").cost.transform("sum")
    kinds.to_csv(OUT / "work_cost_by_interval_kind.csv", index=False)
    stats = {"states": 12, "suffixes_per_model": 36000, "extra_case750_suffixes_oe": 5000,
        "cost_reduction_in_predictions_percent": (1-table.loc['OE','remaining_cost_eur']/table.loc['D','remaining_cost_eur'])*100,
        "mean_predicted_cost_d": table.loc['D','remaining_cost_eur'], "mean_predicted_cost_oe": table.loc['OE','remaining_cost_eur'],
        "simulator_reference_mean": by.sim_cost.mean(), "oe_to_simulator_cost_ratio": table.loc['OE','remaining_cost_eur']/by.sim_cost.mean(),
        "oe_cost_closer_to_simulator_in_options_of_24": int(((by.oe_cost-by.sim_cost).abs()<(by.d_cost-by.sim_cost).abs()).sum()),
        "oe_top10pct_cost_share": tail.loc['OE','top10pct_cost_share'],
        "oe_unclosed_share_of_time_cost_first_repeat": kinds.loc[kinds.model.eq('OE')&kinds.kind.eq('unclosed'), 'share_cost'].iloc[0],
        "oe_unknown_event_class_fraction": oe.unseen_event_classes.sum()/oe.events.sum()}
    write("summary.json", stats)
    # Save the concrete high-cost path used in the explanation.
    row = oe[(oe.id==6)&oe.repeat.eq(0)&oe.option.eq('W_Handle leads|start')&oe.draw.eq(279)].iloc[0]
    suffix = pickle.loads((OUT / 'suffixes_006_1.pkl').read_bytes())[279]
    pd.DataFrame(suffix, columns=['event','elapsed_seconds','gap_seconds']).to_csv(OUT / 'example_case703_draw279.csv', index_label='step')
    snapshots = read(OUT / 'manifest.json').get('training_summaries', {})
    extra = ["", "## Ergebnis und verbleibende Fehler", "",
        f"OE zieht in dieser Diagnose deutlich plausibler als das bisherige D. Die mittleren "
        f"vorhergesagten Kosten sinken um {stats['cost_reduction_in_predictions_percent']:.1f} %, "
        f"liegen aber mit {stats['mean_predicted_cost_oe']:.2f} € weiterhin etwa "
        f"{stats['oe_to_simulator_cost_ratio']:.1f}-mal über dem Mittel der Simulator-Fortsetzungen "
        f"({stats['simulator_reference_mean']:.2f} €). In "
        f"{stats['oe_cost_closer_to_simulator_in_options_of_24']} von 24 Optionszuständen liegt "
        "die OE-Kostenschätzung näher am Simulator-Mittel. Drei Simulatorfortsetzungen je Option "
        "reichen für keine präzise Aussage über die wahre individuelle Optionsrangfolge.", "",
        f"Der Median der gezogenen Kosten beträgt bei OE {tail.loc['OE','p50_cost']:.2f} € "
        f"gegenüber {tail.loc['D','p50_cost']:.2f} € bei D. "
        f"Die teuersten 10 % der OE-Suffixe verursachen {100*stats['oe_top10pct_cost_share']:.1f} % "
        f"der gesamten vorhergesagten Kosten. {100*stats['oe_unclosed_share_of_time_cost_first_repeat']:.1f} % "
        "der Arbeitszeitkosten der ersten OE-Wiederholung entfallen auf Sitzungen, die bis zum "
        "gezogenen Fallende offen bleiben. Das erklärt, warum der Median wesentlich besser "
        "aussieht als der für die Policy verwendete Mittelwert.", "",
        "### Konkretes OE-Beispiel: Application_703, Option start, Ziehung 279", "",
        "Dieser Pfad liegt ungefähr am 90. Kostenperzentil dieser Option und Wiederholung. "
        "Er enthält ausschließlich beobachtete Event-/Zeitklassenkombinationen und erlaubte "
        "aufeinanderfolgende Eventpaare. Trotzdem ist die vollständige Sitzungsfolge inkonsistent:", "",
        "- Schritt 8: `W_Complete application|resume` öffnet die Arbeit bei Tag 14,8045.",
        "- Danach werden Angebote verschickt und andere Aufgaben begonnen; "
        "`W_Complete application` erhält keinen abschließenden Lifecycle mehr.",
        "- Schritt 13: `W_Call after offers|start` folgt erneut auf einen `start` derselben bereits offenen Aufgabe.",
        "- Schritt 16: `W_Call after offers|resume` folgt nach 96,68 Stunden, obwohl seit dem letzten Start "
        "kein `suspend` gekommen ist. Diese Zeit wird als aktive Arbeit gezählt.",
        "- Am Ende bei Tag 30,1771 bleibt `W_Complete application` seit 368,94 Stunden offen, "
        "obwohl der geschäftliche Endmarker `A_Pending` vorkommt.", "",
        f"Ergebnis dieses Suffixes: {row.active_hours:.2f} aktive Arbeitsstunden und "
        f"{row.total_eur:.2f} € verbleibende Kosten. Das Problem ist hier nicht mehr ein "
        "für sich ungesehener langer `suspend`-Abstand. Es entsteht aus dem Zusammensetzen "
        "einzeln beobachteter Events und Zeitklassen bei falschem Sitzungszustand.", "",
        "### Schlussfolgerung", "",
        "OE behebt den bei D nachgewiesenen identischen ersten Optionsabstand und unterbindet "
        "unbeobachtete Event-/Zeitklassenkombinationen. Es verbessert die typischen Kostenziehungen "
        "deutlich. Für belastbare Eventkostenentscheidungen fehlen weiterhin Zustandsregeln für "
        "offene/pausierte Arbeit, konsistente Fallenden und eine Prüfung der langen Kosten-Ausläufer. "
        "Eine pauschale Kappung der Kosten wäre keine Validierung dieser Regeln. Ein besserer "
        "Online-Outcome von OE wurde mit dieser Diagnose noch nicht gemessen.", "",
        "### Einschränkung des Architekturvergleichs", ""]
    if snapshots:
        ds, os = snapshots['D']['alle Prefixe'], snapshots['OE']['alle Prefixe']
        extra += [f"Beide Modelle: {ds['Trainingsfälle']:,} Trainingsfälle, {ds['Validierungsfälle']:,} "
            f"Validierungsfälle, 15 Epochen, Lernrate 0,0001, Decoderlänge 60 und Wahlpunktgewicht 3. "
            f"D hat {ds['Proben']:,} gewichtete Trainingsproben pro Epoche, OE {os['Proben']:,}. "
            "Beim vorhandenen OE waren zusätzliche Proben am Entscheidungs-Prefix bereits im Training "
            "gewichtet; bei D wurde der Optionsmodus nach dem Training aktiviert. "
            "Der Vergleich zeigt damit die Qualität der vorhandenen Modellstände, nicht ausschließlich "
            "den isolierten Effekt der gemeinsamen Tokenarchitektur.", ""]
    path = OUT / 'auswertung.md'
    path.write_text(path.read_text()+"\n".join(extra))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for label, frame in (('D',d),('OE',oe)):
        vals = np.sort(frame.total_eur.to_numpy())
        axes[0].plot(np.maximum(vals,1e-3), 1-(np.arange(len(vals))+.5)/len(vals),label=label)
        vals = np.sort(frame.active_hours.to_numpy())
        axes[1].plot(np.maximum(vals,1e-6),1-(np.arange(len(vals))+.5)/len(vals),label=label)
    axes[0].set(xscale='log',yscale='log',xlabel='Verbleibende Kosten (€)',ylabel='Anteil höherer Ziehungen',title='OE: typischer Pfad besser, teure Ausreißer bleiben')
    axes[1].set(xscale='log',yscale='log',xlabel='Aktive Arbeitsstunden',ylabel='Anteil längerer Arbeitszeiten',title='Arbeitszeit-Ausläufer bei D und OE')
    for ax in axes:
        ax.grid(alpha=.2);ax.legend();ax.set_ylim(1e-3,1)
    fig.savefig(OUT/'vergleich.png',dpi=180)
    plt.close(fig)
    print(json.dumps(stats,indent=2))


if __name__ == '__main__':
    main()
