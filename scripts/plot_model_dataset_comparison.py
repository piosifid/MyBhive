#!/usr/bin/env python
"""
Combined b-vs-(c+udsg) ROC comparison across the 3 model architectures and the
2 dataset variants (baseline vs flatpt-augmented). Pure post-hoc analysis on
the already-saved InferenceTask outputs -- no GPU, no rerunning any task.
"""

import gc
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

DATA_PATH = os.environ["DATA_PATH"].rstrip("/")
TEST_DATASET_VERSION = "dijet_test_wide"
EPOCHS = 40

ARCH_COLOR = {
    "UParT": "black",
    "ParT2LT": "magenta",
    "ParticleNet": "cyan",
}

DATASET_STYLE = {
    "baseline": ("--", "baseline"),
    "flatpt": ("-", "custom"),
}

MODELS = {
    ("UParT", "baseline"):       ("dijet_full_wide",            "uparT_dijet_wide",             "UParT_v0"),
    ("UParT", "flatpt"):         ("dijet_full_plus_flatpt_wide", "uparT_dijet_flatpt_wide",      "UParT_v0"),
    ("ParT2LT", "baseline"):     ("dijet_full_wide",            "parT2LT_dijet_wide",           "ParticleTransformer2_LT"),
    ("ParT2LT", "flatpt"):       ("dijet_full_plus_flatpt_wide", "parT2LT_dijet_flatpt_wide",    "ParticleTransformer2_LT"),
    ("ParticleNet", "baseline"): ("dijet_full_wide",            "particleNet_dijet_wide",       "ParticleNet_InPro"),
    ("ParticleNet", "flatpt"):   ("dijet_full_plus_flatpt_wide", "particleNet_dijet_flatpt_wide","ParticleNet_InPro"),
}


def inference_dir(dataset_version, training_version, model_name):
    return os.path.join(
        DATA_PATH,
        "InferenceTask",
        "part_run3_lt",
        dataset_version,
        TEST_DATASET_VERSION,
        training_version,
        model_name,
        f"epochs_{EPOCHS}",
        "nominal",
        "test_attack_nominal",
    )


def bvsall_score_and_label(predictions, truth):
    if np.abs(np.mean(np.sum(predictions, axis=-1)) - 1) > 1e-3:
        from scipy.special import softmax
        predictions = softmax(predictions, axis=-1)

    b_jets = (truth == 0) | (truth == 1) | (truth == 2)
    c_jets = truth == 3
    uds_jets = truth == 4
    g_jets = truth == 5
    l_jets = uds_jets | g_jets

    b_pred = predictions[:, :3].sum(axis=1)
    c_pred = predictions[:, 3]
    l_pred = predictions[:, -2:].sum(axis=1)

    denom = b_pred + l_pred + c_pred
    valid = denom > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        bvsall = np.where(valid, b_pred / denom, -1)

    keep = valid & (b_jets | c_jets | l_jets)
    return bvsall[keep], b_jets[keep]


def main():
    fig, ax = plt.subplots(figsize=(7, 6))

    for (arch, dataset), (dataset_version, training_version, model_name) in MODELS.items():
        d = inference_dir(dataset_version, training_version, model_name)
        print(f"Loading {arch} / {dataset} from {d}")

        predictions = np.load(os.path.join(d, "prediction.npy"), allow_pickle=True)
        truth = np.load(os.path.join(d, "truth.npy"), allow_pickle=True)

        score, label = bvsall_score_and_label(predictions, truth)
        del predictions, truth
        gc.collect()

        fpr, tpr, _ = roc_curve(label, score)
        del score, label
        gc.collect()

        color = ARCH_COLOR[arch]
        linestyle, prefix = DATASET_STYLE[dataset]
        ax.plot(tpr, fpr, color=color, linestyle=linestyle, linewidth=1.8,
                label=f"{prefix} {arch}")

        del fpr, tpr
        gc.collect()

    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1e-4, 1.0)
    ax.set_xlabel("b-jet tag efficiency")
    ax.set_ylabel("c/udsg-jet misidentification efficiency")
    ax.set_title("dijet_test_wide", loc="left", fontsize=10)
    ax.text(0.98, 0.04, "Z' signal + QCD background",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_dir = os.path.join(DATA_PATH, "comparison_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path_png = os.path.join(out_dir, "model_dataset_comparison_bvsall.png")
    out_path_pdf = os.path.join(out_dir, "model_dataset_comparison_bvsall.pdf")
    fig.savefig(out_path_png, dpi=200)
    fig.savefig(out_path_pdf)
    print(f"Saved: {out_path_png}")
    print(f"Saved: {out_path_pdf}")


if __name__ == "__main__":
    main()
