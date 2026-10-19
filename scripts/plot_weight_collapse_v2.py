import numpy as np
import matplotlib.pyplot as plt

TRUTH_CLASSES = ['isB','isBB','isGBB','isLeptonicB','isLeptonicB_C','isC','isCC','isGCC',
                  'isU','isD','isS','isG','isTaup1h0p_','isTaup1h1p_','isTaup1h2p_','isTaup3h0p_',
                  'isTaup3h1p_','isTaum1h0p_','isTaum1h1p_','isTaum1h2p_','isTaum3h0p_','isTaum3h1p_']
PLOTTED_CLASSES = TRUTH_CLASSES[:12]

PATHS = {
    "baseline": "/eos/user/p/piosifid/b-hive-output/DatasetConstructorTask/part_run3_lt/dijet_full_wide/histogram.npy",
    "flatpt":   "/eos/user/p/piosifid/b-hive-output/DatasetConstructorTask/part_run3_lt/dijet_full_plus_flatpt_wide/histogram.npy",
    "flatpt_below4000": "/eos/user/p/piosifid/b-hive-output/DatasetConstructorTask/part_run3_lt_below4000/dijet_full_plus_flatpt_below4000_wide/histogram.npy",
}

FLOOR = 1e-5

def occupancy_weighted_mean_weight(hist):
    ref = hist[0]
    n_classes = hist.shape[0]
    out = np.zeros(n_classes)
    for c in range(n_classes):
        cls = hist[c]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(cls > 0, ref / cls, 0.0)
        max_ratio = ratio.max()
        if max_ratio <= 0:
            out[c] = 0.0
            continue
        weight = ratio / max_ratio
        occupancy = cls
        total_occ = occupancy.sum()
        if total_occ <= 0:
            out[c] = 0.0
            continue
        out[c] = (weight * occupancy).sum() / total_occ
    return out

results = {}
for name, path in PATHS.items():
    hist = np.load(path)
    w = occupancy_weighted_mean_weight(hist)
    results[name] = w[:12]
    print(f"\n{name} (from {path}):")
    for cls_name, val in zip(PLOTTED_CLASSES, w[:12]):
        print(f"  {cls_name:16s} {val:.6g}")

x = np.arange(len(PLOTTED_CLASSES))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6.5))

colors = {"baseline": "#1f3b63", "flatpt": "#e07b1a", "flatpt_below4000": "#2ca02c"}
offsets = {"baseline": -width, "flatpt": 0, "flatpt_below4000": width}
labels = {"baseline": "baseline", "flatpt": "flatpt (full-range)", "flatpt_below4000": "flatpt (pT<4000)"}

for name in ["baseline", "flatpt", "flatpt_below4000"]:
    vals = np.clip(results[name], FLOOR, None)
    ax.bar(x + offsets[name], vals, width, label=labels[name], color=colors[name])

ax.set_yscale("log")
ax.set_ylabel("occupancy-weighted mean weight")
ax.set_title("Per-class training weight collapse: baseline vs flatpt vs flatpt (pT<4000)")
ax.set_xticks(x)
ax.set_xticklabels(PLOTTED_CLASSES, rotation=45, ha="right")
ax.legend()
ax.grid(axis="y", which="both", alpha=0.3)
ax.set_axisbelow(True)
ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)

fig.tight_layout()
fig.savefig("plot6_weight_collapse_v2.png", dpi=150)
print("\nsaved plot6_weight_collapse_v2.png")
