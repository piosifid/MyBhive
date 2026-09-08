import numpy as np
import termplotlib as tpl
from sklearn.metrics import roc_curve, auc
from scipy.special import softmax


def terminal_roc(
    predictions, truth, title=None, truth_index=0, veto_index=None, xlabel="b-id"
):
    # check if sum of logits == 1.
    if np.abs(np.mean(np.sum(predictions, axis=-1)) - 1) > 1e-3:
        predictions = softmax(predictions, axis=-1)
    if len(predictions.shape) == 1:
        bvsl = predictions
    else:
        b_pred = predictions[:, :1].sum(axis=-1)
        l_pred = predictions[:, 1:].sum(axis=-1)
        bvsl = np.where((b_pred + l_pred) > 0, (b_pred) / (b_pred + l_pred), -1)
    if np.sum(np.unique(truth)) > 1:
        b_jets = truth == 0
        if not (veto_index is None):
            veto = truth != veto_index
        veto = np.ones(len(b_jets), dtype=bool)
    else:
        b_jets = truth
        if not (veto_index is None):
            veto = np.ones(truth.shape, dtype=bool)
        else:
            veto = np.ones(len(b_jets), dtype=bool)

    try:
        fig = tpl.figure()
        label = "b vs l"
        fpr, tpr, _ = roc_curve(b_jets[veto], bvsl[veto])
        fig.plot(
            tpr,
            fpr,
            width=90,
            height=30,
            xlim=(0.1, 1),
            ylim=(0.0001, 1),
            label=label,
            xlabel=xlabel,
            title=title,
            extra_gnuplot_arguments=[
                "set ylabel 'mis-id'",  # Label for the y-axis
                "set logscale y",       # Logarithmic scale for y-axis
                "set key at graph 0.3, 0.9",  # Positioning the legend at the top-left corner
            ],
        )
        fig.show()
    except FileNotFoundError as e:
        print(e)
        print("Is gnuplot installed on your machine?")


def _term_roc(disc, truth, label, xlabel, title):
    fig = tpl.figure()
    fpr, tpr, _ = roc_curve(truth, disc)
    fig.plot(
        tpr,
        fpr,
        width=90,
        height=30,
        xlim=(0.1, 1),
        ylim=(0.0001, 1),
        label=label,
        xlabel=xlabel,
        title=title,
        extra_gnuplot_arguments=["set ylabel mis-id", "set logscale y"],
    )
    fig.show()
