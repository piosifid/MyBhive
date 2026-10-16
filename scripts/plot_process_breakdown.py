"""
Bar chart analogous to Foteini's thesis Fig 4.18 (right panel): total jets
per flavor class, grouped/colored by origin category.

Our version: baseline (native dijet dataset) vs flat_pT addition, per
flavor class (b, c, uds, g). flat_pT's contribution = flatpt_total -
baseline_total (per flavor, summed over all pT bins) -- no reprocessing
needed, reuses composition_results.npz from dataset_composition_v2.py.

Usage:
    python scripts/plot_process_breakdown.py
(run from the same directory where composition_results.npz lives, or
 pass its path as sys.argv[1])
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

npz_path = sys.argv[1] if len(sys.argv) > 1 else "composition_results.npz"
data = np.load(npz_path, allow_pickle=True)

groups = list(data["groups"])
counts_baseline = data["counts_baseline"]  # shape (n_pt_bins, n_groups)
counts_flatpt = data["counts_flatpt"]

baseline_totals = counts_baseline.sum(axis=0)
flatpt_totals = counts_flatpt.sum(axis=0)
flatpt_added = flatpt_totals - baseline_totals
flatpt_added = np.clip(flatpt_added, 0, None)  # guard against tiny negative noise

x = np.arange(len(groups))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5.5))
bars_base = ax.bar(x - width / 2, baseline_totals, width, label="baseline (native dijet)",
                    color="#1f4e8c")
bars_added = ax.bar(x + width / 2, flatpt_added, width, label="flat_pT addition",
                     color="#d9534f")

ax.set_yscale("log")
ax.set_ylabel("number of jets")
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_title("Jets per flavor class: baseline vs. flat_pT addition")
ax.legend()

for bars, totals_for_pct in [(bars_base, baseline_totals + flatpt_added),
                              (bars_added, baseline_totals + flatpt_added)]:
    for bar, total in zip(bars, totals_for_pct):
        height = bar.get_height()
        if height <= 0:
            continue
        pct = 100 * height / total if total > 0 else 0
        ax.annotate(f"{height:,.0f}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

fig.tight_layout()
fig.savefig("plot10_process_breakdown.png", dpi=150, bbox_inches="tight")
print("Saved plot10_process_breakdown.png")
