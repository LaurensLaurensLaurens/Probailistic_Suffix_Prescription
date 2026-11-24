"""Render the recomputed diagnostics as a standalone figure."""
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(OUT/"matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    summary = json.loads((OUT/"recomputed_summary.json").read_text())
    sim = json.loads((OUT.parent/"oe_vs_d_sampling_20261115/summary.json").read_text())["simulator_reference_mean"]
    palette = ["#46688C", "#DE8D38"]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.7), layout="constrained")
    cost_values = [sim] + [s["mean_cost"] for s in summary]
    bars = axes[0].bar(["Simulator¹", "D", "OE"], cost_values, color=["#648568"]+palette)
    axes[0].set(yscale="log", ylim=(100, 16000), ylabel="Verbleibende Kosten (€), logarithmisch",
        title="Arbeitskosten massiv überschätzt")
    for bar, value in zip(bars, cost_values):
        axes[0].text(bar.get_x()+bar.get_width()/2, value*1.12, f"{value:,.0f} €".replace(",", "."), ha="center")
    pos = np.arange(2)
    for i, s in enumerate(summary):
        axes[1].bar(pos+(i-.5)*.34,
            [s["missing_business_end_fraction"]*100, s["open_session_end_fraction"]*100],
            width=.34, color=palette[i], label=s["model"])
    axes[1].set(xticks=pos, xticklabels=["Ohne geschäftlichen\nEndmarker", "Offene Arbeit\nam Suffixende"],
        ylim=(0, 42), ylabel="Anteil gezogener Suffixe (%)", title="Lokale Regeln genügen nicht")
    axes[1].legend(frameon=False)
    bars = axes[2].bar([s["model"] for s in summary], [s["days_repeat_flip_fraction"]*100 for s in summary], color=palette)
    axes[2].set(ylim=(0, 52), ylabel="Zustände mit wechselnder Zeitoption (%)",
        title="Wahl bleibt trotz 500 Ziehungen instabil")
    for bar, count in zip(bars, ("3 von 12", "5 von 12")):
        axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, count, ha="center")
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("D und OE auf denselben zwölf Entscheidungszuständen · je 36.000 Suffixe mit Übergangssperre", fontsize=12)
    fig.supxlabel("¹ Je Option nur drei Simulatorfortsetzungen, anschließend Bankverhalten. Diagnose vorhandener Modellstände; kein neuer Online-Benchmark.", fontsize=9)
    fig.savefig(OUT/"vergleich.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
