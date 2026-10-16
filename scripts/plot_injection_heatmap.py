"""
Plot #5 (two figures): isB raw counts over the (pT, eta) histogram grid,
baseline vs flatpt, side by side.

  Figure A: full grid (pT 8-8000 GeV, log x-axis) -- context, shows the two
            datasets look nearly identical at this zoom level.
  Figure B: zoomed to pT-bin 18-31 (pT ~1708-8000 GeV), full eta range,
            with the injection cluster (pT-bin 24-26, eta-bin 5-7, i.e.
            pT in [4500,6000) GeV, eta in [-0.5,1.0)) outlined and annotated.

Usage:
    python scripts/plot_injection_heatmap.py <baseline_histogram.npy> <flatpt_histogram.npy>

histogram.npy shape: (22, 31, 12) = classes x pT-bins x eta-bins (confirmed).
isB is class index 0.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm

ISB_IDX = 0
ISU_IDX = 8  # isU, in the 12-class ordering: isB,isBB,isGBB,isLeptonicB,isLeptonicB_C,isC,isCC,isGCC,isU,isD,isS,isG

BINS_PT = np.array([8, 10, 14, 19, 26, 35, 47, 63, 85, 115, 155, 209, 282, 381,
                     514, 694, 937, 1265, 1708, 2000, 2500, 3000, 3500, 4000,
                     4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000], dtype=float)
BINS_ETA = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5,
                      2.0, 2.5, 3.0], dtype=float)

# injection cluster, in bin-index space (pT-bin 24-26, eta-bin 5-7)
INJ_PT_LO, INJ_PT_HI = BINS_PT[24], BINS_PT[27]   # [4500, 6000)
INJ_ETA_LO, INJ_ETA_HI = BINS_ETA[5], BINS_ETA[8]  # [-0.5, 1.0)


def plot_full_grid(hist_base, hist_flat, outpath):
    isb_base = hist_base[ISB_IDX]
    isb_flat = hist_flat[ISB_IDX]
    vmax = max(isb_base.max(), isb_flat.max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, data, title in [(axes[0], isb_base, "baseline"), (axes[1], isb_flat, "flatpt")]:
        plot_data = np.ma.masked_where(data <= 0, data)
        mesh = ax.pcolormesh(BINS_PT, BINS_ETA, plot_data.T,
                              norm=LogNorm(vmin=1, vmax=vmax), cmap="Blues")
        mesh.cmap.set_bad(color="#f5f5f5")
        ax.set_xscale("log")
        ax.set_xlabel("jet pT [GeV]")
        ax.set_title(f"{title} (isB counts)")
    axes[0].set_ylabel("jet eta")
    fig.colorbar(mesh, ax=axes, label="isB jet count per bin", fraction=0.046, pad=0.04)
    fig.suptitle("isB counts over full (pT, eta) grid -- context view")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved {outpath}")
    plt.close(fig)


def plot_zoomed(hist_base, hist_flat, outpath):
    isb_base = hist_base[ISB_IDX]
    isb_flat = hist_flat[ISB_IDX]

    # crop to pT-bin 18-31 (pT ~1708-8000 GeV), full eta range
    pt_lo_idx = 18
    pt_slice = slice(pt_lo_idx, 31)
    bins_pt_zoom = BINS_PT[pt_lo_idx:]

    base_zoom = isb_base[pt_slice, :]
    flat_zoom = isb_flat[pt_slice, :]
    vmax = max(base_zoom.max(), flat_zoom.max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, data, title in [(axes[0], base_zoom, "baseline"), (axes[1], flat_zoom, "flatpt")]:
        plot_data = np.ma.masked_where(data <= 0, data)
        mesh = ax.pcolormesh(bins_pt_zoom, BINS_ETA, plot_data.T,
                              norm=LogNorm(vmin=1, vmax=vmax), cmap="Blues")
        mesh.cmap.set_bad(color="#f5f5f5")
        ax.set_xlabel("jet pT [GeV]")
        ax.set_title(f"{title} (isB counts)")

        rect = patches.Rectangle(
            (INJ_PT_LO, INJ_ETA_LO), INJ_PT_HI - INJ_PT_LO, INJ_ETA_HI - INJ_ETA_LO,
            linewidth=1.8, edgecolor="#d9534f", facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)

    axes[0].set_ylabel("jet eta")
    # annotate counts at the injection cluster
    axes[0].annotate(
        "isB count here:\nbaseline 0-8",
        xy=(INJ_PT_LO, INJ_ETA_HI), xytext=(INJ_PT_LO - 1400, INJ_ETA_HI + 0.9),
        fontsize=8, color="#d9534f",
        arrowprops=dict(arrowstyle="->", color="#d9534f", lw=1),
    )
    axes[1].annotate(
        "isB count here:\nflatpt 2,360-3,843",
        xy=(INJ_PT_LO, INJ_ETA_HI), xytext=(INJ_PT_LO - 1400, INJ_ETA_HI + 0.9),
        fontsize=8, color="#d9534f",
        arrowprops=dict(arrowstyle="->", color="#d9534f", lw=1),
    )

    fig.colorbar(mesh, ax=axes, label="isB jet count per bin", fraction=0.046, pad=0.04)
    fig.suptitle("isB counts, zoomed to high-pT tail -- injection cluster outlined")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved {outpath}")
    plt.close(fig)


def compute_raw_ratio(hist, ref_idx, class_idx):
    """Same formula as merge_datasets / histogram_weighting, BEFORE the final
    /max(weights) renormalization -- i.e. the raw ratio value whose single
    highest point becomes that divisor."""
    reference_histogram = hist[ref_idx] / hist[ref_idx].max()
    other_histogram = hist[class_idx] / hist[class_idx].max()
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_ratio = np.where(other_histogram > 0, reference_histogram / other_histogram, -10)
    return raw_ratio


def plot_ratio_zoomed(hist_base, hist_flat, outpath, class_idx=ISU_IDX, class_name="isU"):
    ratio_base = compute_raw_ratio(hist_base, ISB_IDX, class_idx)
    ratio_flat = compute_raw_ratio(hist_flat, ISB_IDX, class_idx)

    pt_lo_idx = 18
    pt_slice = slice(pt_lo_idx, 31)
    bins_pt_zoom = BINS_PT[pt_lo_idx:]

    base_zoom = ratio_base[pt_slice, :]
    flat_zoom = ratio_flat[pt_slice, :]
    vmax = max(base_zoom[base_zoom > 0].max(), flat_zoom[flat_zoom > 0].max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, data, title in [(axes[0], base_zoom, "baseline"), (axes[1], flat_zoom, "flatpt")]:
        plot_data = np.ma.masked_where(data <= 0, data)
        mesh = ax.pcolormesh(bins_pt_zoom, BINS_ETA, plot_data.T,
                              norm=LogNorm(vmin=1, vmax=vmax), cmap="Blues")
        mesh.cmap.set_bad(color="#f5f5f5")
        ax.set_xlabel("jet pT [GeV]")
        ax.set_title(f"{title} (isB/{class_name} ratio)")

        rect = patches.Rectangle(
            (INJ_PT_LO, INJ_ETA_LO), INJ_PT_HI - INJ_PT_LO, INJ_ETA_HI - INJ_ETA_LO,
            linewidth=1.8, edgecolor="#d9534f", facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)

    axes[0].set_ylabel("jet eta")
    axes[0].annotate(
        f"max ratio anywhere:\n{base_zoom.max():.1f}",
        xy=(INJ_PT_LO, INJ_ETA_HI), xytext=(INJ_PT_LO - 1400, INJ_ETA_HI + 0.9),
        fontsize=8, color="#d9534f",
        arrowprops=dict(arrowstyle="->", color="#d9534f", lw=1),
    )
    axes[1].annotate(
        f"max ratio anywhere:\n{flat_zoom.max():.1f} (here)",
        xy=(INJ_PT_LO, INJ_ETA_HI), xytext=(INJ_PT_LO - 1400, INJ_ETA_HI + 0.9),
        fontsize=8, color="#d9534f",
        arrowprops=dict(arrowstyle="->", color="#d9534f", lw=1),
    )

    fig.colorbar(mesh, ax=axes, label=f"isB / {class_name} density ratio (pre-renormalization)",
                 fraction=0.046, pad=0.04)
    fig.suptitle(f"The weighting mechanism itself: isB/{class_name} ratio, zoomed to high-pT tail")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved {outpath}")
    plt.close(fig)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    hist_base = np.load(sys.argv[1])
    hist_flat = np.load(sys.argv[2])

    plot_full_grid(hist_base, hist_flat, "plot7_isb_heatmap_full.png")
    plot_zoomed(hist_base, hist_flat, "plot8_isb_heatmap_zoomed.png")
    plot_ratio_zoomed(hist_base, hist_flat, "plot9_isu_ratio_heatmap.png")


if __name__ == "__main__":
    main()
