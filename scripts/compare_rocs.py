import argparse

import numpy as np

from utils.plotting.roc import plot_roc


def main(rocs, labels, output, dataset_label, pt_min, pt_max, colors, auc, xmin):
    roc_list = [np.load(roc) for roc in rocs]
    plot_roc(
        roc_list,
        labels,
        dataset_label=dataset_label,
        pt_min=pt_min,
        pt_max=pt_max,
        output_path=output,
        x_label="B-tagging efficiency",
        y_label="Light flavour misidentification",
        r_label="(13.6 TeV)",
        l_label="Preliminary",
        colors=colors,
        xmin=xmin,
        writeout_auc=auc
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rocs", "-r", type=str, nargs="+", help="Numpy files that should be read in"
    )
    parser.add_argument("--labels", "-l", type=str, nargs="+", help="Labels that should be read in")
    parser.add_argument("--colors", "-c", type=str, nargs="+", help="Colors to use for plots")
    parser.add_argument("--output", "-o", type=str, help="Output path")
    parser.add_argument("--dataset", "-d", type=str, help="dataset to use")
    parser.add_argument("--pt-min", type=int, help="pt_min")
    parser.add_argument("--pt-max", type=int, help="pt_max")
    parser.add_argument("--x-min", type=float, help="x-min", default=0.0)
    parser.add_argument("--auc", action="store_true", help="Write AUC into plot")
    args = parser.parse_args()

    main(args.rocs, args.labels, args.output, args.dataset, args.pt_min, args.pt_max, args.colors, args.auc, args.x_min)
