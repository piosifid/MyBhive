"""
Locate the (pT, eta) bin whose isB/other-class ratio spikes between baseline
and flatpt, and print a before/after numeric trace of the mechanism:
  1. raw histogram counts at that bin (isB and the affected class)
  2. the ratio value at that bin (pre-renormalization)
  3. where the GLOBAL max ratio sits in each dataset (does it move there?)
  4. the resulting renormalization divisor and what that does to a few
     "normal" bins far away from the injection

Usage:
    python scripts/locate_injection_bin.py <baseline_histogram.npy> <flatpt_histogram.npy>

histogram.npy is assumed to have shape (n_classes, n_pt_bins, n_eta_bins),
matching how `hist[c]` is indexed in utils/dataset/merging.py's merge_datasets.
If your file has a different shape, adjust N_CLASSES / axis order below.
"""
import sys
import numpy as np

TRUTH_CLASSES = ["isB", "isBB", "isGBB", "isLeptonicB", "isLeptonicB_C",
                 "isC", "isCC", "isGCC", "isU", "isD", "isS", "isG"]
REFERENCE_IDX = TRUTH_CLASSES.index("isB")


def ratio_array(hist, ref_idx, class_idx):
    ref_hist = hist[ref_idx] / hist[ref_idx].max()
    other_hist = hist[class_idx] / hist[class_idx].max()
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_ratio = np.where(other_hist > 0, ref_hist / other_hist, -10)
    return raw_ratio


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    hist_base = np.load(sys.argv[1])
    hist_flat = np.load(sys.argv[2])

    print(f"baseline histogram shape: {hist_base.shape}")
    print(f"flatpt   histogram shape: {hist_flat.shape}")
    print()

    for class_idx, class_name in enumerate(TRUTH_CLASSES):
        if class_idx == REFERENCE_IDX:
            continue

        ratio_base = ratio_array(hist_base, REFERENCE_IDX, class_idx)
        ratio_flat = ratio_array(hist_flat, REFERENCE_IDX, class_idx)

        max_base = ratio_base.max()
        max_flat = ratio_flat.max()
        bin_base = np.unravel_index(np.argmax(ratio_base), ratio_base.shape)
        bin_flat = np.unravel_index(np.argmax(ratio_flat), ratio_flat.shape)

        moved = bin_base != bin_flat
        jumped = max_flat > 5 * max_base if max_base > 0 else max_flat > 0

        if not (moved or jumped):
            continue

        print(f"=== {class_name} ===")
        print(f"  baseline: global max ratio = {max_base:.2f} at bin {bin_base}")
        print(f"  flatpt:   global max ratio = {max_flat:.2f} at bin {bin_flat}")
        print(f"  divisor jumped by {max_flat / max_base:.1f}x" if max_base > 0 else "  baseline divisor was 0/undefined")
        print()

        print(f"  -- counts at the flatpt max-ratio bin {bin_flat} --")
        print(f"     isB   count: baseline={hist_base[REFERENCE_IDX][bin_flat]:.0f}  flatpt={hist_flat[REFERENCE_IDX][bin_flat]:.0f}")
        print(f"     {class_name:14s} count: baseline={hist_base[class_idx][bin_flat]:.0f}  flatpt={hist_flat[class_idx][bin_flat]:.0f}")
        print()

        print(f"  -- effect on 3 unrelated bins (far from {bin_flat}) --")
        shape = ratio_base.shape
        sample_bins = [
            (shape[0] // 4, shape[1] // 2),
            (shape[0] // 2, shape[1] // 4),
            (3 * shape[0] // 4, 3 * shape[1] // 4),
        ]
        for b in sample_bins:
            if b == bin_flat:
                continue
            w_base = ratio_base[b] / max_base if max_base > 0 else float("nan")
            w_flat = ratio_flat[b] / max_flat if max_flat > 0 else float("nan")
            print(f"     bin {b}: weight baseline={w_base:.4f}  flatpt={w_flat:.4f}  (suppressed {w_base / w_flat:.1f}x)" if w_flat > 0 else f"     bin {b}: weight baseline={w_base:.4f}  flatpt={w_flat:.4e}")
        print()


if __name__ == "__main__":
    main()
