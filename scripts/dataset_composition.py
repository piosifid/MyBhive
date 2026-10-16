#!/usr/bin/env python
"""
Flavor composition vs pT for the baseline (dijet_full_wide) and flatpt
(dijet_full_plus_flatpt_wide) TRAINING datasets -- raw, unweighted jet
counts read directly from the DatasetConstructorTask .lz4 chunk files,
using the same decompression/column layout as utils/torch/LZ4Dataset.py.

Answers: (1) do non-b jets exist at high pT in the training data, and
(2) how b-jet-poor is baseline's pool relative to flatpt's.

Run inside the b_hive conda env, from the repo root, with DATA_PATH set:

    source /eos/user/u/uwillems/miniforge3/etc/profile.d/conda.sh
    conda activate b_hive
    export DATA_PATH="/eos/user/p/piosifid/b-hive-output/"
    python scripts/dataset_composition.py
"""

import os
import numpy as np
import lz4.frame

from utils.config.config_loader import ConfigLoader

DATA_PATH = os.environ["DATA_PATH"].rstrip("/")
CONFIG_NAME = "part_run3_lt"

DATASETS = {
    "baseline": "dijet_full_wide",
    "flatpt":   "dijet_full_plus_flatpt_wide",
}

PT_EDGES = [0, 300, 600, 1000, 1500, 2500, 8000]


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


def main():
    config = ConfigLoader.load_config(CONFIG_NAME)
    truths_list = list(config["truths"])
    global_features = list(config["global_features"])
    jet_pt_idx = global_features.index("jet_pt")
    num_ele = len(truths_list)
    dtype = config["data_precision"]

    group_of_flavidx = build_group_map(truths_list)
    groups = ["b", "c", "uds", "g", "tau", "other"]

    results = {}
    for label, dataset_version in DATASETS.items():
        files = processed_files(dataset_version)
        print(f"\n=== {label} ({dataset_version}) : {len(files)} files ===")

        n_bins = len(PT_EDGES) - 1
        counts = np.zeros((n_bins, len(groups)), dtype=np.int64)
        total_jets = 0

        for i, f in enumerate(files):
            jet_pt, flavour_idx = load_chunk(f, dtype, num_ele, jet_pt_idx)
            jet_group = group_of_flavidx[flavour_idx]
            bin_idx = np.digitize(jet_pt, PT_EDGES) - 1

            for b in range(n_bins):
                in_bin = bin_idx == b
                if not in_bin.any():
                    continue
                for g_i, g in enumerate(groups):
                    counts[b, g_i] += int((jet_group[in_bin] == g).sum())

            total_jets += jet_pt.shape[0]
            if (i + 1) % 50 == 0:
                print(f"  ...{i+1}/{len(files)} files, {total_jets:,} jets so far")

        print(f"  total jets: {total_jets:,}")
        results[label] = counts

        header = f"{'pT bin':>16}" + "".join(f"{g:>12}" for g in groups) + f"{'total':>14}{'%b':>8}"
        print(header)
        for b in range(n_bins):
            row_total = counts[b].sum()
            pct_b = 100 * counts[b][groups.index("b")] / row_total if row_total > 0 else 0
            row_label = f"[{PT_EDGES[b]},{PT_EDGES[b+1]})"
            row = f"{row_label:>16}" + "".join(f"{counts[b,g_i]:>12,}" for g_i in range(len(groups)))
            row += f"{row_total:>14,}{pct_b:>7.2f}%"
            print(row)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    centers = [(PT_EDGES[i] + PT_EDGES[i+1]) / 2 for i in range(len(PT_EDGES) - 1)]
    for label, color in [("baseline", "#1f4e8c"), ("flatpt", "#d98324")]:
        counts = results[label]
        totals = counts.sum(axis=1)
        b_frac = np.where(totals > 0, counts[:, groups.index("b")] / totals, 0) * 100
        ax.plot(centers, b_frac, marker="o", label=label, color=color)
    ax.set_xlabel("jet pT [GeV]")
    ax.set_ylabel("b-jet fraction [%]")
    ax.set_title("Training data: b-jet fraction vs pT, baseline vs flatpt")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_dir = os.path.join(DATA_PATH, "comparison_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dataset_composition_b_fraction.png")
    fig.savefig(out_path, dpi=170)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
