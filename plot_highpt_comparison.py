"""
Baseline vs flat-pT-augmented comparison, binned in jet pT, up to 6 TeV.

For each model, loads the raw InferenceTask outputs (prediction.npy,
truth.npy, kinematics.npy) for both the baseline (dijet_full) and augmented
(dijet_full_plus_flatpt) training/inference runs, computes the identical
inclusive BvsL discriminant that WorkingPointTask uses, derives a
fixed-mistag operating threshold with the repo's own working-point
utilities, and then applies that single threshold per pT bin to get
b-efficiency and light-mistag vs jet pT.

Also PRINTS the per-bin (N_b, N_light) jet counts for every curve, so you
can judge directly whether a wiggle in the plot is real or just low
statistics in that bin -- no need to guess from error bars alone.

Run from the MyBhive repo root, inside the b_hive conda env:

    cd /eos/user/p/piosifid/MyBhive
    source /eos/user/u/uwillems/miniforge3/etc/profile.d/conda.sh
    conda activate b_hive
    export DATA_PATH="/eos/user/p/piosifid/b-hive-output/"
    python plot_highpt_comparison.py

Models whose InferenceTask outputs aren't present yet are skipped -- rerun
this same script later once they're done and that column appears.
"""

import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
from scipy.special import softmax

from utils.evaluation.working_point import (
    calculate_efficiency_curve,
    calculate_working_point,
)

hep.style.use("CMS")

DATA_PATH = os.environ["DATA_PATH"]
CONFIG = "part_run3_lt"
TEST_DATASET_VERSION = "dijet_test"

TARGET_MISTAG = 0.01
PT_MAX = 6000.0

MODELS = [
    ("UParT", "UParT_v0", "uparT_dijet", "uparT_dijet_flatpt"),
    ("ParT2LT", "ParticleTransformer2_LT", "parT2LT_dijet", "parT2LT_dijet_flatpt"),
    ("ParticleNet", "ParticleNet_InPro", "particleNet_dijet", "particleNet_dijet_flatpt"),
]

PT_EDGES = np.array(
    [400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800,
     2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000],
    dtype=float,
)
PT_EDGES = PT_EDGES[PT_EDGES <= PT_MAX]
PT_CENTERS = 0.5 * (PT_EDGES[:-1] + PT_EDGES[1:])


def find_inference_files(dataset_version, training_version, model_name):
    pattern = os.path.join(
        DATA_PATH, "InferenceTask", CONFIG, dataset_version, TEST_DATASET_VERSION,
        training_version, model_name, "**", "prediction.npy",
    )
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    pred_path = matches[0]
    base = os.path.dirname(pred_path)
    return {
        "prediction": os.path.join(base, "prediction.npy"),
        "truth": os.path.join(base, "truth.npy"),
        "kinematics": os.path.join(base, "kinematics.npy"),
    }


def load_run(dataset_version, training_version, model_name):
    paths = find_inference_files(dataset_version, training_version, model_name)
    if paths is None:
        return None
    predictions = np.load(paths["prediction"], allow_pickle=True)
    truth = np.load(paths["truth"], allow_pickle=True)
    kinematics = np.load(paths["kinematics"], allow_pickle=True)
    jet_pt = kinematics[..., 0]

    predictions = softmax(predictions, axis=-1)
    b_jets = (truth == 0) | (truth == 1) | (truth == 2)
    c_jets = truth == 3
    l_jets = (truth == 4) | (truth == 5)

    b_pred = predictions[:, :3].sum(axis=1)
    l_pred = predictions[:, -2:].sum(axis=1)
    bvsl = np.where((b_pred + l_pred) > 0, b_pred / (b_pred + l_pred), -1)

    threshold_curve, eff_curve, mistag_curve = calculate_efficiency_curve(
        bvsl[~c_jets], b_jets[~c_jets], n_points=200
    )
    wp_threshold, wp_eff, wp_mistag = calculate_working_point(
        threshold_curve, eff_curve, mistag_curve, TARGET_MISTAG
    )

    return {
        "jet_pt": jet_pt,
        "bvsl": bvsl,
        "b_jets": b_jets,
        "l_jets": l_jets,
        "c_jets": c_jets,
        "threshold": wp_threshold,
        "global_eff": wp_eff,
        "global_mistag": wp_mistag,
    }


def binned_rate_and_counts(jet_pt, passed, selected_mask, edges):
    rate = np.full(len(edges) - 1, np.nan)
    err = np.full(len(edges) - 1, np.nan)
    counts = np.zeros(len(edges) - 1, dtype=int)
    for i in range(len(edges) - 1):
        in_bin = selected_mask & (jet_pt >= edges[i]) & (jet_pt < edges[i + 1])
        n = in_bin.sum()
        counts[i] = n
        if n == 0:
            continue
        k = (passed & in_bin).sum()
        p = k / n
        rate[i] = p
        err[i] = np.sqrt(p * (1 - p) / n)
    return rate, err, counts


def compute_curves(run):
    pass_sel = run["bvsl"] >= run["threshold"]
    eff, eff_err, n_b = binned_rate_and_counts(run["jet_pt"], pass_sel, run["b_jets"], PT_EDGES)
    mistag, mistag_err, n_l = binned_rate_and_counts(run["jet_pt"], pass_sel, run["l_jets"], PT_EDGES)
    return eff, eff_err, mistag, mistag_err, n_b, n_l


def print_counts(name, label, n_b, n_l):
    print(f"\n{name} -- {label}: per-bin jet counts")
    print(f"{'pT bin [GeV]':>20} {'N_b':>10} {'N_light':>10}")
    for i in range(len(PT_EDGES) - 1):
        lo, hi = PT_EDGES[i], PT_EDGES[i + 1]
        print(f"{lo:>9.0f}-{hi:<9.0f} {n_b[i]:>10d} {n_l[i]:>10d}")


def main():
    results = {}
    for display_name, model_name, baseline_tv, augmented_tv in MODELS:
        baseline = load_run("dijet_full", baseline_tv, model_name)
        augmented = load_run("dijet_full_plus_flatpt", augmented_tv, model_name)
        if baseline is None and augmented is None:
            continue
        results[display_name] = (baseline, augmented)

    if not results:
        print("No InferenceTask outputs found yet for any model.")
        return

    ordered_names = [name for name, *_ in MODELS if name in results]
    ncols = len(ordered_names)
    fig, axes = plt.subplots(
        2, ncols, figsize=(5.5 * ncols, 9), sharex="col", sharey="row", squeeze=False
    )

    for col, name in enumerate(ordered_names):
        baseline, augmented = results[name]
        ax_eff, ax_mistag = axes[0][col], axes[1][col]

        for run, label, style in (
            (baseline, "baseline (dijet_full)", dict(color="tab:blue", linestyle="-", marker="o")),
            (augmented, "+ flat-pT (dijet_full_plus_flatpt)", dict(color="tab:red", linestyle="--", marker="s")),
        ):
            if run is None:
                continue
            eff, eff_err, mistag, mistag_err, n_b, n_l = compute_curves(run)
            print_counts(name, label, n_b, n_l)
            ax_eff.errorbar(
                PT_CENTERS, eff, yerr=eff_err, label=label, markersize=4,
                capsize=2, **style,
            )
            ax_mistag.errorbar(
                PT_CENTERS, mistag, yerr=mistag_err, label=label, markersize=4,
                capsize=2, **style,
            )

        ax_mistag.set_yscale("log")
        ax_eff.set_title(name, fontsize=18, pad=10)
        ax_eff.set_ylim(0, 1.05)
        ax_eff.set_xlim(PT_EDGES[0], PT_EDGES[-1])
        ax_eff.xaxis.set_major_locator(mticker.MultipleLocator(1000))
        ax_mistag.set_xlabel("Jet $p_{T}$ [GeV]")
        if col == 0:
            ax_eff.set_ylabel(f"b-jet efficiency @ {TARGET_MISTAG:.0%} incl. mistag")
            ax_mistag.set_ylabel("Light-jet mistag rate")
        ax_eff.legend(fontsize=9, loc="lower left")
        ax_eff.grid(alpha=0.3)
        ax_mistag.grid(alpha=0.3)

    fig.subplots_adjust(top=0.90)
    fig.text(0.01, 0.985, "CMS", fontsize=20, fontweight="bold", ha="left", va="top")
    fig.text(0.01, 0.965, "Simulation Preliminary", fontsize=14, style="italic", ha="left", va="top")
    fig.text(0.99, 0.985, "(13.6 TeV)", fontsize=14, ha="right", va="top")

    out_path = "highpt_comparison_baseline_vs_flatpt.png"
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.replace(".png", ".pdf"))
    print(f"\nSaved {out_path} (and .pdf)")
    print(f"Models included: {ordered_names}")


if __name__ == "__main__":
    main()
