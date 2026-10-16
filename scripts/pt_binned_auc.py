#!/usr/bin/env python
"""
Compute b-vs-all AUC in pT bins, per model, from already-saved InferenceTask
outputs (prediction.npy, truth.npy, kinematics.npy). Does not touch the GPU
and does not rerun any b-Hive task -- pure post-hoc numpy/sklearn analysis.
"""

import gc
import os

import numpy as np
from sklearn.metrics import roc_auc_score

DATA_PATH = os.environ["DATA_PATH"].rstrip("/")

MODELS = [
    ("UParT baseline",      "dijet_full_wide",            "uparT_dijet_wide",            "UParT_v0"),
    ("UParT flatpt",        "dijet_full_plus_flatpt_wide", "uparT_dijet_flatpt_wide",     "UParT_v0"),
    ("ParT2LT baseline",    "dijet_full_wide",            "parT2LT_dijet_wide",          "ParticleTransformer2_LT"),
    ("ParT2LT flatpt",      "dijet_full_plus_flatpt_wide", "parT2LT_dijet_flatpt_wide",   "ParticleTransformer2_LT"),
    ("ParticleNet baseline","dijet_full_wide",            "particleNet_dijet_wide",      "ParticleNet_InPro"),
    ("ParticleNet flatpt",  "dijet_full_plus_flatpt_wide", "particleNet_dijet_flatpt_wide","ParticleNet_InPro"),
]

TEST_DATASET_VERSION = "dijet_test_wide"
EPOCHS = 40

PT_EDGES = [0, 300, 600, 1000, 1500, 2500, 8000]


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

    return bvsall, b_jets, valid


def auc_per_pt_bin(bvsall, label, pts, valid):
    rows = []
    for lo, hi in zip(PT_EDGES[:-1], PT_EDGES[1:]):
        m = valid & (pts >= lo) & (pts < hi)
        n = int(m.sum())
        n_pos = int(label[m].sum())
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            rows.append((lo, hi, n, n_pos, None))
            continue
        auc = roc_auc_score(label[m], bvsall[m])
        rows.append((lo, hi, n, n_pos, auc))
    return rows


def main():
    all_results = {}

    for label, dataset_version, training_version, model_name in MODELS:
        d = inference_dir(dataset_version, training_version, model_name)
        print(f"\n=== {label} ===")
        print(f"  reading from: {d}")

        predictions = np.load(os.path.join(d, "prediction.npy"), allow_pickle=True)
        truth = np.load(os.path.join(d, "truth.npy"), allow_pickle=True)
        kinematics = np.load(os.path.join(d, "kinematics.npy"), allow_pickle=True)
        pts = kinematics[..., 0]

        bvsall, b_jets, valid = bvsall_score_and_label(predictions, truth)
        rows = auc_per_pt_bin(bvsall, b_jets, pts, valid)
        all_results[label] = rows

        for lo, hi, n, n_pos, auc in rows:
            auc_str = f"{auc:.4f}" if auc is not None else "n/a (single class)"
            print(f"  pT [{lo:5d},{hi:5d}) GeV : N={n:>10d}  n_b={n_pos:>10d}  AUC={auc_str}")

        del predictions, truth, kinematics, pts, bvsall, b_jets, valid
        gc.collect()

    print("\n\n===== SUMMARY: bvsall AUC per pT bin =====")
    header = "model".ljust(22) + "".join(f"[{lo},{hi})".rjust(14) for lo, hi in zip(PT_EDGES[:-1], PT_EDGES[1:]))
    print(header)
    for label, rows in all_results.items():
        line = label.ljust(22)
        for lo, hi, n, n_pos, auc in rows:
            line += (f"{auc:.4f}".rjust(14) if auc is not None else "n/a".rjust(14))
        print(line)


if __name__ == "__main__":
    main()
