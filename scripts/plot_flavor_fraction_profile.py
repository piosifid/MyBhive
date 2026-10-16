"""
Two-panel jet-flavour-fraction-vs-pT plot, styled after Philipp Gadow's
mock-up (Slack, 2026-10-15): top panel = stacked-area cumulative flavour
fraction (0-100%) vs jet pT; bottom panel = log-scale per-flavour fraction
[%] for the minority flavours only, same pT axis.

Directly answers his question: "do you have the distribution of classes
in the flatpt sample without weights?" -- this is the RAW (unweighted)
per-pT-bin flavour composition, read straight from histogram.npy (the
same per-(class,pT,eta) counts used everywhere else in this investigation,
summed over eta), not from the training weights.

Flavour grouping (collapses the 12 truth classes into 4 physics groups):
    b   = isB + isBB + isGBB + isLeptonicB + isLeptonicB_C
    c   = isC + isCC + isGCC
    uds = isU + isD + isS
    g   = isG
Stack order (bottom->top): uds, g, c, b -- matches the dominance ordering
already established in Section 2 of the investigation doc ("non-b jets,
overwhelmingly uds, then g, then c").

Usage:
    python scripts/plot_flavor_fraction_profile.py <baseline_histogram.npy> <flatpt_histogram.npy>

histogram.npy shape: (22, 31, 12) = classes x pT-bins x eta-bins (confirmed
elsewhere in this investigation). Produces one figure per dataset:
    plot11_flavor_fraction_profile_baseline.png
    plot11_flavor_fraction_profile_flatpt.png
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

TRUTH_CLASSES = ["isB", "isBB", "isGBB", "isLeptonicB", "isLeptonicB_C",
                 "isC", "isCC", "isGCC", "isU", "isD", "isS", "isG"]

GROUP_MEMBERS = {
    "b": ["isB", "isBB", "isGBB", "isLeptonicB", "isLeptonicB_C"],
    "c": ["isC", "isCC", "isGCC"],
    "uds": ["isU", "isD", "isS"],
    "g": ["isG"],
}
# stack order bottom -> top, matches observed dominance (uds > g > c > b)
STACK_ORDER = ["uds", "g", "c", "b"]
COLORS = {
    "uds": "#a9d4e8",
    "g": "#6b7f8c",
    "c": "#4c9a5a",
    "b": "#d9534f",
}

BINS_PT = np.array([8, 10, 14, 19, 26, 35, 47, 63, 85, 115, 155, 209, 282, 381,
                     514, 694, 937, 1265, 1708, 2000, 2500, 3000, 3500, 4000,
                     4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000], dtype=float)
PT_CENTERS = np.sqrt(BINS_PT[:-1] * BINS_PT[1:])  # log-geometric bin centers


def group_fractions(hist):
    """hist: (n_classes, n_pt_bins, n_eta_bins) -> dict group -> fraction[%] per pT bin."""
    class_idx = {name: i for i, name in enumerate(TRUTH_CLASSES)}
    counts_per_pt = {}
    for group, members in GROUP_MEMBERS.items():
        idxs = [class_idx[m] for m in members]
        # sum over the member classes and over eta -> (n_pt_bins,)
        counts_per_pt[group] = hist[idxs].sum(axis=(0, 2))
    total = sum(counts_per_pt.values())
    fractions = {g: 100.0 * c / np.where(total > 0, total, 1) for g, c in counts_per_pt.items()}
    return fractions, total


def plot_one(hist, label, outpath):
    fractions, total = group_fractions(hist)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1.3], "hspace": 0.06},
    )

    # top: stacked area, cumulative to 100%
    stack_vals = [fractions[g] for g in STACK_ORDER]
    ax_top.stackplot(PT_CENTERS, stack_vals,
                      colors=[COLORS[g] for g in STACK_ORDER],
                      labels=STACK_ORDER, edgecolor="white", linewidth=0.3)
    ax_top.set_xscale("log")
    ax_top.set_ylim(0, 100)
    ax_top.set_ylabel("Jet flavour fraction [%]")
    ax_top.set_title(f"Jet flavour fraction vs pT -- {label} (unweighted, raw histogram counts)")
    ax_top.legend(loc="lower left", ncol=4, frameon=True, fontsize=9)

    # bottom: log-scale detail view, minority flavours only (exclude uds --
    # it's the ~85-95% dominant class already clear from the top panel)
    for g in ["c", "b", "g"]:
        ax_bot.plot(PT_CENTERS, fractions[g], marker="o", markersize=3.5,
                    linewidth=1.3, color=COLORS[g], label=g)
    ax_bot.set_yscale("log")
    ax_bot.set_xscale("log")
    ax_bot.set_xlabel("Jet pT [GeV]")
    ax_bot.set_ylabel("Fraction [%]")
    ax_bot.legend(loc="upper right", ncol=3, frameon=True, fontsize=9)
    ax_bot.grid(True, which="both", axis="y", alpha=0.25)

    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved {outpath}")
    plt.close(fig)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    hist_base = np.load(sys.argv[1])
    hist_flat = np.load(sys.argv[2])

    plot_one(hist_base, "baseline (native dijet)", "plot11_flavor_fraction_profile_baseline.png")
    plot_one(hist_flat, "flat_pT", "plot11_flavor_fraction_profile_flatpt.png")


if __name__ == "__main__":
    main()
