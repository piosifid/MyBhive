import matplotlib
import matplotlib.pyplot as plt
import mplhep

from typing import List, Tuple

matplotlib.pyplot.style.use(mplhep.style.CMS)


def plot_working_points(
    curves: list,
    working_points: list[tuple],
    mistag_rates: list[float],
    labels: list[str],
    out_path="working_points.jpg",
    x_label: str = "B-id threshold",
    y_label: str = "Misidentification",
    l_label: str = "Preliminary",
    r_label: str = None,
    xlim: tuple = (0.0, 1.0),
    ylim: tuple = (4e-4, 1.0),
    color="C0",
):
    """_summary_

    Args:
        curves (list): _description_
        working_points (list[tuple): Tuple consisting of (threshold, efficiency, mistag_rate)
        mistag_rates (list[float]): _description_
        labels (list[str]): _description_
        out_path (str, optional): _description_. Defaults to "working_points.jpg".
        x_label (str, optional): _description_. Defaults to "B efficiency".
        y_label (str, optional): _description_. Defaults to "Misidentification".
        l_label (str, optional): _description_. Defaults to "Preliminary".
        r_label (str, optional): _description_. Defaults to None.
        xlim (tuple, optional): _description_. Defaults to (0.0, 1.0).
        ylim (tuple, optional): _description_. Defaults to (4e-4, 1.0).
    """
    if not isinstance(curves, (list)):
        curves = [curves]
    if not isinstance(working_points, (list, tuple)):
        working_points = [working_points]
    if not isinstance(labels, (list, tuple)):
        labels = [labels]

    fig = plt.figure()
    plot = fig.add_subplot(111)

    for curve, label in zip(curves, labels):
        x, y = curve
        plot.plot(x, y, "-", label=label, color=color)

    plot.set_xlabel(x_label, x=1.0, ha="right")
    plot.set_ylabel(y_label, y=1.0, va="bottom", ha="right")
    plot.set_yscale("log")
    plot.set_xlim(xlim)
    plot.set_ylim(ylim)
    plot.legend()

    if mistag_rates is not None:
        for mistag_rate in mistag_rates:
            plot.axhline(y=mistag_rate, lw=0.5, c="k")
            plot.text(
                0.01,
                mistag_rate * 1.06,
                "{0:1.3f}".format(mistag_rate),
                c="k",
                fontsize=14,
                va="bottom",
                ha="left",
            )
    if working_points is not None:
        for wp in working_points:
            thresh, eff, mistag = wp
            plot.text(
                0.01,
                mistag * 0.96,
                "thresh:{0:1.3f}/eff:{1:1.3f}".format(thresh, eff),
                c=color,
                fontsize=14,
                va="top",
                ha="left",
            )
    mplhep.cms.label(l_label, rlabel=r_label)

    fig.tight_layout()
    fig.savefig(out_path)
