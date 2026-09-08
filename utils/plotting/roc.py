import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.cm import get_cmap
from sklearn.metrics import auc, roc_curve
from scipy.special import softmax

from utils.plotting.termplot import terminal_roc
from matplotlib.cm import get_cmap

import boost_histogram as bh
from boost_histogram import Histogram, storage

color_set_name = "Dark2"
cmap = get_cmap(color_set_name)  # type: matplotlib.colors.ListedColormap
color_set_list = cmap.colors  # type: list

plt.style.use(hep.cms.style.CMS)

def pt_binned_eff(truth, disc, pts):
    pt_axis = bh.axis.Regular(20, 20, 7500)

    b_h_total = Histogram(pt_axis, storage=storage.Weight())
    nonb_h_total = Histogram(pt_axis, storage=storage.Weight())
    
    # tight, medium, loose,
    WP_list = [0.9816, 0.7954, 0.2815]
    # WP_list = [0.9536, 0.7251, 0.2537]
    TP_h_tagged_list = [Histogram(pt_axis, storage=storage.Weight()), Histogram(pt_axis, storage=storage.Weight()), Histogram(pt_axis, storage=storage.Weight())]
    FP_h_tagged_list = [Histogram(pt_axis, storage=storage.Weight()), Histogram(pt_axis, storage=storage.Weight()), Histogram(pt_axis, storage=storage.Weight())]

    print(f'pts.shape={pts.shape}')
    b = pts[truth]
    nonb = pts[~truth]
    print(f'b.shape={b.shape}')
    print(f'nonb.shape={nonb.shape}')

    b_h_total.fill(b)
    nonb_h_total.fill(nonb)

    i = 0
    for working_point in WP_list:
        b_tagged = (disc > working_point)
        b_tagged_TP = pts[truth & b_tagged]
        b_tagged_FP = pts[~truth & b_tagged]
        print(f'WP={working_point}, b_tagged_TP.shape={b_tagged_TP.shape}')
        print(f'WP={working_point}, b_tagged_FP.shape={b_tagged_FP.shape}')
        # Numerator: tagged b-jets
        TP_h_tagged_list[i].fill(b_tagged_TP)
        FP_h_tagged_list[i].fill(b_tagged_FP)
        i += 1

    b_total_view = b_h_total.view()
    nonb_total_view = nonb_h_total.view()
    b_total_vals = b_total_view.value
    b_total_vars = b_total_view.variance
    nonb_total_vals = nonb_total_view.value
    nonb_total_vars = nonb_total_view.variance
    print(f'b_total_vals.shape={b_total_vals.shape}')
    print(f'b_total_vars.shape={b_total_vars.shape}')
    print(f'nonb_total_vals.shape={nonb_total_vals.shape}')
    print(f'nonb_total_vars.shape={nonb_total_vars.shape}')

    i = 0
    eff_list = []
    eff_err_list = []
    eff_centers = []
    mis_list = []
    mis_err_list = []
    mis_centers = []
    for working_point in WP_list:
        TP_tagged_view = TP_h_tagged_list[i].view()
        TP_tagged_vals = TP_tagged_view.value
        TP_tagged_vars = TP_tagged_view.variance

        FP_tagged_view = FP_h_tagged_list[i].view()
        FP_tagged_vals = FP_tagged_view.value
        FP_tagged_vars = FP_tagged_view.variance

        print(f'WP={working_point}, TP_tagged_vals.shape={TP_tagged_vals.shape}')
        print(f'WP={working_point}, TP_tagged_vars.shape={TP_tagged_vars.shape}')
        print(f'WP={working_point}, FP_tagged_vals.shape={FP_tagged_vals.shape}')
        print(f'WP={working_point}, FP_tagged_vars.shape={FP_tagged_vars.shape}')

        # Efficiency and uncertainty
        eff = np.zeros_like(TP_tagged_vals)
        eff_err = np.zeros_like(TP_tagged_vals)
        mis = np.zeros_like(FP_tagged_vals)
        mis_err = np.zeros_like(FP_tagged_vals)

        print(f'WP={working_point}, eff.shape={eff.shape}')
        print(f'WP={working_point}, eff_err.shape={eff_err.shape}')
        print(f'WP={working_point}, mis.shape={mis.shape}')
        print(f'WP={working_point}, mis_err.shape={mis_err.shape}')

        eff[b_total_vals>0] = TP_tagged_vals[b_total_vals>0] / b_total_vals[b_total_vals>0]
        eff_err[b_total_vals>0] = eff[b_total_vals>0] * np.sqrt(
            (TP_tagged_vars[b_total_vals>0] / TP_tagged_vals[b_total_vals>0]**2) +
            (b_total_vars[b_total_vals>0] / b_total_vals[b_total_vals>0]**2)
        )
        mis[nonb_total_vals>0] = FP_tagged_vals[nonb_total_vals>0] / nonb_total_vals[nonb_total_vals>0]
        mis_err[nonb_total_vals>0] = mis[nonb_total_vals>0] * np.sqrt(
            (FP_tagged_vars[nonb_total_vals>0] / FP_tagged_vals[nonb_total_vals>0]**2) +
            (nonb_total_vars[nonb_total_vals>0] / nonb_total_vals[nonb_total_vals>0]**2)
        )

        print(f'WP={working_point}, eff.shape={eff.shape}')
        print(f'WP={working_point}, eff_err.shape={eff_err.shape}')
        print(f'WP={working_point}, mis.shape={mis.shape}')
        print(f'WP={working_point}, mis_err.shape={mis_err.shape}')

        TP_centers = TP_h_tagged_list[i].axes[0].centers
        FP_centers = FP_h_tagged_list[i].axes[0].centers

        eff_list.append(eff)
        eff_err_list.append(eff_err)
        eff_centers.append(TP_centers)
        mis_list.append(mis)
        mis_err_list.append(mis_err)
        mis_centers.append(FP_centers)

        # ax.errorbar(centers, eff, yerr=eff_err, fmt=marker, color=color, capsize=2, markersize=s, label=f'{tagger_label} {WP_label}', alpha=alpha)
        i += 1

    return (eff_centers, eff_list, eff_err_list, mis_centers, mis_list, mis_err_list)



def plot_roc_list(
    discs,
    truths,
    vetos,
    labels,
    xlabels,
    ylabels,
    output_directory,
    pt_min,
    pt_max,
    name,
    xmin=0.0,ymin=1e-5,
    energy="13.6 TeV",
    save_numpy=True,
):
    color = color_set_list[0]
    AUC_arr = {}
    for disc, truth, veto, roc_label, xlabel, ylabel in zip(
        discs,
        truths,
        vetos,
        labels,
        xlabels,
        ylabels,
    ):
        try:
            # fpr, tpr, _ = roc_curve(truth[veto], disc[veto])
            fpr, tpr, thresh = roc_curve(truth[veto], disc[veto])
        except ValueError as e:
            print(e)
            print(
                "Your ROC could not be plotted. Please check if this is not a debug set"
            )
            continue
        area = auc(fpr, tpr)
        AUC_arr[roc_label] = area
        if save_numpy:
            np.save(
                os.path.join(output_directory, f"roc_{name}_{roc_label}.npy"),
                np.array((fpr, tpr, thresh)),
            )
        for ext in ["png", "pdf"]:
            plot_name = os.path.join(output_directory, f"roc_{name}_{roc_label}.{ext}")
            plot_roc(
                [(fpr, tpr, area)],
                [roc_label],
                name,
                pt_min=pt_min,
                pt_max=pt_max,
                x_label=xlabel,
                y_label=ylabel,
                output_path=plot_name,
                colors=color,
                r_label=energy,
                xmin=xmin,ymin=ymin
            )
            
    np.save(
        os.path.join(output_directory, f"AUC_{name}_all.npy"),
        np.array(AUC_arr),
    )


def plot_roc_eff_list(
    pts,
    discs,
    truths,
    vetos,
    labels,
    xlabels,
    ylabels,
    output_directory,
    pt_min,
    pt_max,
    name,
    xmin=0.0,ymin=1e-5,
    energy="13.6 TeV",
    save_numpy=True,
):
    color = color_set_list[0]
    AUC_arr = {}
    for disc, truth, veto, roc_label, xlabel, ylabel in zip(
        discs,
        truths,
        vetos,
        labels,
        xlabels,
        ylabels,
    ):
        try:
            # fpr, tpr, _ = roc_curve(truth[veto], disc[veto])
            fpr, tpr, thresh = roc_curve(truth[veto], disc[veto])
        except ValueError as e:
            print(e)
            print(
                "Your ROC could not be plotted. Please check if this is not a debug set"
            )
            continue
        area = auc(fpr, tpr)
        AUC_arr[roc_label] = area
        if save_numpy:
            np.save(
                os.path.join(output_directory, f"roc_{name}_{roc_label}.npy"),
                np.array((fpr, tpr, thresh)),
            )
        for ext in ["png", "pdf"]:
            plot_name = os.path.join(output_directory, f"roc_{name}_{roc_label}.{ext}")
            plot_roc(
                [(fpr, tpr, area)],
                [roc_label],
                name,
                pt_min=pt_min,
                pt_max=pt_max,
                x_label=xlabel,
                y_label=ylabel,
                output_path=plot_name,
                colors=color,
                r_label=energy,
                xmin=xmin,ymin=ymin
            )

        if roc_label=='bvsall':
            try:
                eff_centers, eff, eff_err, mis_centers, mis, mis_err = pt_binned_eff(truth[veto], disc[veto], pts[veto])
            except ValueError as e:
                print(e)
                print(
                    "Your pt_binned_eff could not be plotted. Please check if this is not a debug set"
                )
                continue
            if save_numpy:
                i = 0
                for l in ['tight', 'medium', 'loose']:
                    np.save(
                        os.path.join(output_directory, f"eff_{l}.npy"),
                        np.array((eff_centers[i], eff[i], eff_err[i])),
                    )
                    np.save(
                        os.path.join(output_directory, f"mis_{l}.npy"),
                        np.array((mis_centers[i], mis[i], mis_err[i])),
                    )
                    print(f'WP={l}, eff_centers.shape={eff_centers[i].shape}, eff.shape={eff[i].shape}, eff_err.shape={eff_err[i].shape}')
                    print(f'WP={l}, mis_centers.shape={mis_centers[i].shape}, mis.shape={mis[i].shape}, mis_err.shape={mis_err[i].shape}')
                    i += 1
            
    np.save(
        os.path.join(output_directory, f"AUC_{name}_all.npy"),
        np.array(AUC_arr),
    )


# adapted from https://github.com/AlexDeMoor/DeepJet/blob/ParticleTransformer/scripts/plot_roc.py and https://github.com/AlexDeMoor/DeepJet/blob/ParticleTransformer/scripts/plot_roc.ipynb
def calculate_roc(truth, discriminator, veto, output_directory, dataset_key, name):
    fpr, tpr, _ = roc_curve(truth[veto], discriminator[veto])

    index = np.unique(fpr, return_index=True)[1]
    fpr = np.asarray([fpr[i] for i in sorted(index)])
    tpr = np.asarray([tpr[i] for i in sorted(index)])
    area = auc(fpr, tpr)
    return fpr, tpr, area


def plot_losses(train_loss, test_loss, output_dir, ran_epochs, nepochs):
    fig, ax = plt.subplots()
    ax.set_title("Losses")
    if train_loss is not None:
        ax.plot(
            np.arange(1, nepochs + 1),
            train_loss,
            '-d',
            label="Train",
            color="blue",
        )
    if test_loss is not None:
        ax.plot(
            np.arange(1, nepochs + 1),
            test_loss,
            '-s',
            label="Validation",
            color="orange",
        )
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid()
    fig.savefig(os.path.join(output_dir, "loss.pdf"))
    fig.savefig(os.path.join(output_dir, "loss.png"))


def plot_accuracy(train_acc, test_acc, output_dir, ran_epochs, nepochs):
    fig, ax = plt.subplots()
    ax.set_title("Accuracy")
    if train_acc is not None:
        ax.plot(
            np.arange(1, nepochs + 1),
            train_acc,
            '-d',
            label="Train",
            color="blue",
        )
    if test_acc is not None:
        ax.plot(
            np.arange(1, nepochs + 1),
            test_acc,
            '-s',
            label="Validation",
            color="orange",
        )
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid()
    fig.savefig(os.path.join(output_dir, "acc.pdf"))
    fig.savefig(os.path.join(output_dir, "acc.png"))


def plot_roc(
    roc_list,
    label_list,
    dataset_label=None,
    pt_min=None,
    pt_max=None,
    x_label="Tagging Efficiency",
    y_label="Mistagging rate",
    r_label=None,
    l_label="Preliminary",
    output_path="roc.pdf",
    colors=None,
    xmin=None,ymin=None,
    writeout_auc=True,
):
    if not (isinstance(roc_list, list)):
        roc_list = [roc_list]
    if not (isinstance(label_list, list)):
        label_list = [label_list]
    if colors is None:
        colors = color_set_list[: len(roc_list)]
    if not (isinstance(colors, list)):
        colors = [colors]
    if len(colors) < len(roc_list):
        colors *= len(roc_list)

    pt_text = rf"${pt_min} \leq p_T \leq {pt_max}\,GeV$"
    eta_text = rf"$|\eta| \leq 2.5$"

    plt.figure()
    for roc, label, color in zip(roc_list, label_list, colors):
        try:
            fpr, tpr, area = roc
        except ValueError as e:
            from sklearn.metrics import (
                auc,
            )  # for some reason it could not find auc without another import, but I do not see why.

            fpr, tpr = roc
            index = np.unique(fpr, return_index=True)[1]
            fpr = np.asarray([fpr[i] for i in sorted(index)])
            tpr = np.asarray([tpr[i] for i in sorted(index)])
            area = auc(fpr, tpr)
        plt.plot(
            tpr,
            fpr,
            label=(
                f"{label}" + rf"(AUC${{\approx}}${np.round(area, 3)})"
                if writeout_auc
                else f"{label}"
            ),
            color=color,
        )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.yscale("log")
    plt.xlim(xmin, 1)
    # plt.ylim(2 * 1e-4, 1)
    plt.ylim(1e-5, 1)
    plt.grid(which="minor", alpha=0.85)
    plt.grid(which="major", alpha=0.95, color="black")
    title = ""
    if dataset_label:
        title += f"{dataset_label} jets \n"
    if pt_min and pt_max:
        title += f"{pt_text}, {eta_text}"
    plt.legend(
        title=title,
        loc="best",
        alignment="left",
    )
    hep.cms.label(l_label, rlabel=r_label, com=13)

    print("saving to:\t", output_path)
    plt.savefig(output_path)
    plt.close()
