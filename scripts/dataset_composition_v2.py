#!/usr/bin/env python
"""
Composition vs pT for baseline (dijet_full_wide) and flatpt
(dijet_full_plus_flatpt_wide) training datasets. Reprocesses raw .lz4
chunk files once, saves results to a .npz (for fast re-plotting later),
and produces three plots:
  1. b-jet fraction vs pT (step style, baseline vs flatpt)
  2. absolute total jet counts vs pT (log scale, baseline vs flatpt)
  3. per-flavor absolute counts vs pT (log scale, b/c/uds/g, both datasets)
"""

import os
import numpy as np
import lz4.frame
import matplotlib.pyplot as plt

from utils.config.config_loader import ConfigLoader

DATA_PATH = os.environ["DATA_PATH"].rstrip("/")
CONFIG_NAME = "part_run3_lt"

DATASETS = {
    "baseline": "dijet_full_wide",
    "flatpt":   "dijet_full_plus_flatpt_wide",
}

PT_EDGES = [0, 300, 600, 1000, 1500, 2500, 8000]
GROUPS = ["b", "c", "uds", "g", "tau", "other"]

OUT_DIR = os.path.join(DATA_PATH, "comparison_plots")
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_NPZ = os.path.join(OUT_DIR, "composition_results.npz")


def build_group_map(truths_list):
    b_flags = {"isB", "isBB", "isGBB", "isLeptonicB", "isLeptonicB_C"}
    c_flags = {"isC", "isCC", "isGCC"}
    light_flags = {"isU", "isD", "isS"}
    g_flags = {"isG"}

    group_of = []
    for t in truths_list:
        if t in b_flags:
            group_of.append("b")
        elif t in c_flags:
            group_of.append("c")
        elif t in light_flags:
            group_of.append("uds")
        elif t in g_flags:
            group_of.append("g")
        elif t.startswith("isTau"):
            group_of.append("tau")
        else:
            group_of.append("other")
    return np.array(group_of)


def processed_files(dataset_version):
    path = os.path.join(DATA_PATH, "DatasetConstructorTask", CONFIG_NAME, dataset_version, "processed_files.txt")
    with open(path) as f:
        files = [l.strip() for l in f if l.strip()]
    if files and not os.path.exists(files[-1]):
        files = files[:-1]
    return files


def load_chunk(file_path, dtype, num_ele, jet_pt_idx):
    with lz4.frame.open(file_path, mode="r") as data:
        raw = data.read()
    s = np.frombuffer(raw, dtype=dtype).copy()
    s = s[2:].reshape(-1, int(s[1]), order="C")

    jet_pt = s[:, jet_pt_idx]
    labels = s[:, -(num_ele + 1):-1]
    flavour_idx = np.argmax(labels, axis=1)
    return jet_pt, flavour_idx


def compute_all():
    config = ConfigLoader.load_config(CONFIG_NAME)
    truths_list = list(config["truths"])
    global_features = list(config["global_features"])
    jet_pt_idx = global_features.index("jet_pt")
    num_ele = len(truths_list)
    dtype = config["data_precision"]

    group_of_flavidx = build_group_map(truths_list)

    results = {}
    for label, dataset_version in DATASETS.items():
        files = processed_files(dataset_version)
        print(f"\n=== {label} ({dataset_version}) : {len(files)} files ===")

        n_bins = len(PT_EDGES) - 1
        counts = np.zeros((n_bins, len(GROUPS)), dtype=np.int64)

        for i, f in enumerate(files):
            jet_pt, flavour_idx = load_chunk(f, dtype, num_ele, jet_pt_idx)
            jet_group = group_of_flavidx[flavour_idx]
            bin_idx = np.digitize(jet_pt, PT_EDGES) - 1

            for b in range(n_bins):
                in_bin = bin_idx == b
                if not in_bin.any():
                    continue
                for g_i, g in enumerate(GROUPS):
                    counts[b, g_i] += int((jet_group[in_bin] == g).sum())

            if (i + 1) % 50 == 0:
                print(f"  ...{i+1}/{len(files)} files")

        results[label] = counts
        print(f"  total jets: {counts.sum():,}")

    np.savez(RESULTS_NPZ,
             pt_edges=np.array(PT_EDGES),
             groups=np.array(GROUPS),
             **{f"counts_{label}": results[label] for label in DATASETS})
    print(f"\nSaved raw results to: {RESULTS_NPZ}")
    return results


def make_plots(results):
    centers = [(PT_EDGES[i] + PT_EDGES[i + 1]) / 2 for i in range(len(PT_EDGES) - 1)]
    edges = np.array(PT_EDGES)
    colors = {"baseline": "#1f4e8c", "flatpt": "#d98324"}
    flavor_colors = {"b": "#1f4e8c", "c": "#2e9e63", "uds": "#d98324", "g": "#8c3fa8"}

    # --- Plot 1: b-fraction vs pT, step style ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in DATASETS:
        counts = results[label]
        totals = counts.sum(axis=1)
        b_frac = np.where(totals > 0, counts[:, GROUPS.index("b")] / totals, 0) * 100
        ax.stairs(b_frac, edges, baseline=None, label=label, color=colors[label], linewidth=2)
        ax.scatter(centers, b_frac, color=colors[label], s=30, zorder=5)

    # annotate the near-empty first bin
    first_total = results["baseline"][0].sum()
    ax.annotate(f"n={first_total:,} (negligible)", xy=(centers[0], 0), xycoords=("data", "axes fraction"),
                xytext=(centers[0], 0.03), textcoords=("data", "axes fraction"),
                fontsize=8, color="gray", ha="center")

    ax.set_xlabel("jet pT [GeV]")
    ax.set_ylabel("b-jet fraction [%]")
    ax.set_title("Training data: b-jet fraction vs pT, baseline vs flatpt")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, "plot1_b_fraction.png"), dpi=170, bbox_inches="tight")
    print("Saved plot1_b_fraction.png")

    # --- Plot 2: absolute total counts vs pT, log scale, step style ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in DATASETS:
        totals = results[label].sum(axis=1)
        ax.stairs(totals, edges, baseline=None, label=label, color=colors[label], linewidth=2)
        ax.scatter(centers, totals, color=colors[label], s=30, zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("jet pT [GeV]")
    ax.set_ylabel("total jets")
    ax.set_title("Training data: total jet count vs pT, baseline vs flatpt")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.savefig(os.path.join(OUT_DIR, "plot2_absolute_counts.png"), dpi=170, bbox_inches="tight")
    print("Saved plot2_absolute_counts.png")

    # --- Plot 3: per-flavor counts vs pT, log scale, both datasets ---
    fig, ax = plt.subplots(figsize=(9, 6))
    for flavor in ["b", "c", "uds", "g"]:
        g_i = GROUPS.index(flavor)
        for label, ls in [("baseline", "-"), ("flatpt", "--")]:
            counts = results[label][:, g_i]
            ax.stairs(counts, edges, baseline=None, color=flavor_colors[flavor],
                      linestyle=ls, linewidth=2,
                      label=f"{flavor} ({label})")
    ax.set_yscale("log")
    ax.set_xlabel("jet pT [GeV]")
    ax.set_ylabel("jet count per flavor")
    ax.set_title("Training data: per-flavor jet counts vs pT (solid=baseline, dashed=flatpt)")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.savefig(os.path.join(OUT_DIR, "plot3_per_flavor.png"), dpi=170, bbox_inches="tight")
    print("Saved plot3_per_flavor.png")


if __name__ == "__main__":
    if os.path.exists(RESULTS_NPZ):
        print(f"Found existing results at {RESULTS_NPZ}, loading instead of reprocessing.")
        data = np.load(RESULTS_NPZ)
        results = {label: data[f"counts_{label}"] for label in DATASETS}
    else:
        results = compute_all()
    make_plots(results)
