import os
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = os.environ["DATA_PATH"].rstrip("/")
OUT_DIR = os.path.join(DATA_PATH, "comparison_plots")
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = ["b", "bb", "leptonicB", "c", "uds", "g"]

cm_baseline = np.array([
    [  579257,    39656,     3567,   242438,   111827,     6007],
    [  186188,   938425,    11108,   252027,     5311,     1103],
    [  196573,    43057,   109661,   186574,    52656,     2111],
    [  462352,   132076,    22627,  4511090,  1204768,   128094],
    [ 1322111,    18489,    18085,  9754092, 22566773,   695766],
    [  185958,     8452,     1208,  6252543,  2311259,  1378190],
])

cm_flatpt = np.array([
    [  969878,     2219,     6762,        0,       16,        0],
    [  992530,   247576,   148597,        0,        7,        0],
    [  413395,     5853,   168941,        0,       22,        0],
    [ 6066154,    16789,   353560,        0,       41,        0],
    [34049830,      762,   194694,        0,      704,        0],
    [10061594,      152,    37871,        0,       12,        0],
])

# --- Plot: confusion matrices, row-normalized (recall), side by side ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, cm, title in [(axes[0], cm_baseline, "baseline"), (axes[1], cm_flatpt, "flatpt")]:
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = cm / row_sums
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true" if title == "baseline" else "")
    ax.set_title(f"{title} (test set, n={cm.sum():,})")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            val = norm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color="white" if val > 0.5 else "black", fontsize=8)
fig.colorbar(im, ax=axes, shrink=0.8, label="recall (row-normalized)")
fig.suptitle("Confusion matrices: baseline vs flatpt (same test set)", y=1.02)
fig.savefig(os.path.join(OUT_DIR, "plot5_confusion_matrices.png"), dpi=170, bbox_inches="tight")
print("Saved plot5_confusion_matrices.png")

# --- Plot: per-class occupancy-weighted mean weight, log scale ---
weight_classes = ["isB", "isBB", "isGBB", "isLeptonicB", "isLeptonicB_C",
                   "isC", "isCC", "isGCC", "isU", "isD", "isS", "isG"]
w_baseline = [1.0000, 0.4346, 0.0610, 0.3171, 0.1273, 0.2632, 0.4189, 0.1082, 0.0163, 0.0523, 0.2614, 0.0353]
w_flatpt   = [1.0000, 0.0415, 0.0010, 0.1133, 0.3201, 0.0001, 0.0003, 0.0002, 0.0000, 0.0000, 0.0001, 0.0000]
w_flatpt_plot = [max(v, 1e-5) for v in w_flatpt]  # floor for log scale display

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(weight_classes))
width = 0.35
ax.bar(x - width/2, w_baseline, width, label="baseline", color="#1f4e8c")
ax.bar(x + width/2, w_flatpt_plot, width, label="flatpt", color="#d98324")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(weight_classes, rotation=45, ha="right")
ax.set_ylabel("occupancy-weighted mean weight")
ax.set_title("Per-class training weight collapse: baseline vs flatpt")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
ax.legend()
ax.grid(True, axis="y", alpha=0.3, which="both")
fig.savefig(os.path.join(OUT_DIR, "plot6_weight_collapse.png"), dpi=170, bbox_inches="tight")
print("Saved plot6_weight_collapse.png")
