"""
Baseline vs flat-pT-augmented ROC comparison (pT-inclusive), styled after
Foteini's slides 16-17: baseline = dashed, augmented ("custom") = solid,
one panel per model.

Uses the same BvsL discriminant as WorkingPointTask (c-jets vetoed), so
it's consistent with the pT-binned comparison script rather than Foteini's
possibly-different combined b-vs-(c+light) discriminant.

NOTE: this is pT-inclusive over the test set as currently built, which
(per config/part_run3_lt.yml's bins_pt) only contains jets in
937-2000 GeV. It is not a signal-only (Z'-only) comparison -- that needs a
separate Z'-only test filelist and a fresh InferenceTask run per model,
which we haven't built yet.

Run from the MyBhive repo root, inside the b_hive conda env:

    cd /eos/user/p/piosifid/MyBhive
    source /eos/user/u/uwillems/miniforge3/etc/profile.d/conda.sh
    conda activate b_hive
    export DATA_PATH="/eos/user/p/piosifid/b-hive-output/"
    python plot_roc_comparison.py

Models whose InferenceTask outputs aren't present yet are skipped -- rerun
this same script later once they're done and that panel appears.
"""

import glob
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.special import softmax

from utils.evaluation.working_point import calculate_efficiency_curve

hep.style.use("CMS")

DATA_PATH = os.environ["DATA_PATH"]
CONFIG = "part_run3_lt"
TEST_DATASET_VERSION = "dijet_test"

MODELS = [
    ("UParT", "UParT_v0", "uparT_dijet", "uparT_dijet_flatpt"),
    ("ParT2LT", "ParticleTransformer2_LT", "parT2LT_dijet", "parT2LT_dijet_flatpt"),
    ("ParticleNet", "ParticleNet_InPro", "particleNet_dijet", "particleNet_dijet_flatpt"),
]


def find_inference_files(dataset_version, training_version, model_name):
    pattern = os.path.join(
        DATA_PATH, "InferenceTask", CONFIG, dataset_version, TEST_DATASET_VERSION,
        training_version, model_name, "**", "prediction.npy",
    )
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    base = os.path.dirname(matches[0])
    return {
        "prediction": os.path.join(base, "prediction.npy"),
        "truth": os.path.join(base, "truth.npy"),
    }


def load_roc_curve(dataset_version, training_version, model_name):
    paths = find_inference_files(dataset_version, training_version, model_name)
    if paths is None:
        return None
    predictions = np.load(paths["prediction"], allow_pickle=True)
    truth = np.load(paths["truth"], allow_pickle=True)

    predictions = softmax(predictions, axis=-1)
    b_jets = (truth == 0) | (truth == 1) | (truth == 2)
    c_jets = truth == 3

    b_pred = predictions[:, :3].sum(axis=1)
    l_pred = predictions[:, -2:].sum(axis=1)
    bvsl = np.where((b_pred + l_pred) > 0, b_pred / (b_pred + l_pred), -1)

    _, eff, mistag = calculate_efficiency_curve(
        bvsl[~c_jets], b_jets[~c_jets], n_points=200
    )
    return eff, mistag


def main():
    results = {}
    for display_name, model_name, baseline_tv, augmented_tv in MODELS:
        baseline = load_roc_curve("dijet_full", baseline_tv, model_name)
        augmented = load_roc_curve("dijet_full_plus_flatpt", augmented_tv, model_name)
        if baseline is None and augmented is None:
            continue
        results[display_name] = (baseline, augmented)

    if not results:
        print("No InferenceTask outputs found yet for any model.")
        return

    ordered_names = [name for name, *_ in MODELS if name in results]
    ncols = len(ordered_names)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5.5), squeeze=False)
    axes = axes[0]

    for col, name in enumerate(ordered_names):
        baseline, augmented = results[name]
        ax = axes[col]

        if baseline is not None:
            eff, mistag = baseline
            ax.plot(eff, mistag, linestyle="--", color="tab:blue", label="baseline (dijet_full)")
        if augmented is not None:
            eff, mistag = augmented
            ax.plot(eff, mistag, linestyle="-", color="tab:red", label="+ flat-pT (custom)")

        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_ylim(1e-4, 1)
        ax.set_title(name, fontsize=18, pad=10)
        ax.set_xlabel("b-jet tag efficiency")
        if col == 0:
            ax.set_ylabel("Light-jet mis-identification")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.3)

    fig.subplots_adjust(top=0.85)
    fig.text(0.01, 0.97, "CMS", fontsize=20, fontweight="bold", ha="left", va="top")
    fig.text(0.01, 0.945, "Simulation Preliminary", fontsize=14, style="italic", ha="left", va="top")
    fig.text(0.99, 0.97, "(13.6 TeV)", fontsize=14, ha="right", va="top")

    out_path = "roc_comparison_baseline_vs_flatpt.png"
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.replace(".png", ".pdf"))
    print(f"Saved {out_path} (and .pdf)")
    print(f"Models included: {ordered_names}")


if __name__ == "__main__":
    main()
