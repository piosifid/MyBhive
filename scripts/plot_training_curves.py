#!/usr/bin/env python
"""
Train/val loss & accuracy curves for all 6 model/dataset combinations, laid
out as a 3x2 grid (rows = architecture, cols = baseline/flatpt), to check for
over/undertraining. Marks the best (min val loss) epoch per model.

Pure post-hoc read of the already-saved training_metrics.npz /
validation_metrics.npz -- no GPU, no rerunning any b-Hive task.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = os.environ["DATA_PATH"].rstrip("/")
EPOCHS = 40

ROWS = ["UParT", "ParT2LT", "ParticleNet"]
COLS = ["baseline", "flatpt"]

MODELS = {
    ("UParT", "baseline"):       ("dijet_full_wide",             "uparT_dijet_wide",              "UParT_v0"),
    ("UParT", "flatpt"):         ("dijet_full_plus_flatpt_wide", "uparT_dijet_flatpt_wide",       "UParT_v0"),
    ("ParT2LT", "baseline"):     ("dijet_full_wide",             "parT2LT_dijet_wide",            "ParticleTransformer2_LT"),
    ("ParT2LT", "flatpt"):       ("dijet_full_plus_flatpt_wide", "parT2LT_dijet_flatpt_wide",     "ParticleTransformer2_LT"),
    ("ParticleNet", "baseline"): ("dijet_full_wide",             "particleNet_dijet_wide",        "ParticleNet_InPro"),
    ("ParticleNet", "flatpt"):   ("dijet_full_plus_flatpt_wide", "particleNet_dijet_flatpt_wide", "ParticleNet_InPro"),
}


def training_dir(dataset_version, training_version, model_name):
    return os.path.join(
        DATA_PATH, "TrainingTask", "part_run3_lt", dataset_version,
        training_version, model_name, f"epochs_{EPOCHS}", "nominal",
    )


def load_metrics(arch, dataset):
    dataset_version, training_version, model_name = MODELS[(arch, dataset)]
    d = training_dir(dataset_version, training_version, model_name)
    train = np.load(os.path.join(d, "training_metrics.npz"))
    val = np.load(os.path.join(d, "validation_metrics.npz"))
    return train["loss"], train["acc"], val["loss"], val["acc"]


def make_grid(key, ylabel, out_name):
    fig, axes = plt.subplots(3, 2, figsize=(13, 13), sharex=True)
    epochs = np.arange(1, EPOCHS + 1)

    for i, arch in enumerate(ROWS):
        for j, dataset in enumerate(COLS):
            ax = axes[i, j]
            train_loss, train_acc, val_loss, val_acc = load_metrics(arch, dataset)
            train_y = train_loss if key == "loss" else train_acc
            val_y = val_loss if key == "loss" else val_acc

            ax.plot(epochs, train_y, color="#1f4e8c", linewidth=1.6, label="train")
            ax.plot(epochs, val_y, color="#d98324", linewidth=1.6, label="val")

            best_epoch = int(np.argmin(val_loss)) + 1
            ax.axvline(best_epoch, color="#888888", linestyle="--", linewidth=1.0)

            ax.set_title(f"{arch} - {dataset}  (best val-loss epoch: {best_epoch})", fontsize=11)
            ax.grid(True, alpha=0.3)
            if i == 2:
                ax.set_xlabel("epoch")
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0 and j == 0:
                ax.legend(loc="best", fontsize=9, frameon=False)

    fig.tight_layout()
    out_dir = os.path.join(DATA_PATH, "comparison_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, out_name + ".png")
    out_pdf = os.path.join(out_dir, out_name + ".pdf")
    fig.savefig(out_png, dpi=170)
    fig.savefig(out_pdf)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    plt.close(fig)


def print_summary():
    print("\n===== Training summary (per model) =====")
    header = f"{'model':<24}{'final train loss':>18}{'final val loss':>16}{'best val loss':>16}{'best epoch':>12}{'final train acc':>18}{'final val acc':>16}"
    print(header)
    for arch in ROWS:
        for dataset in COLS:
            train_loss, train_acc, val_loss, val_acc = load_metrics(arch, dataset)
            best_epoch = int(np.argmin(val_loss)) + 1
            label = f"{arch} {dataset}"
            print(
                f"{label:<24}{train_loss[-1]:>18.4f}{val_loss[-1]:>16.4f}"
                f"{val_loss.min():>16.4f}{best_epoch:>12d}{train_acc[-1]:>18.4f}{val_acc[-1]:>16.4f}"
            )


def main():
    print_summary()
    make_grid("loss", "loss", "training_loss_comparison")
    make_grid("acc", "accuracy", "training_acc_comparison")


if __name__ == "__main__":
    main()
