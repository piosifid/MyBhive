import numpy as np
import scipy
import tqdm


def calculate_efficiency_curve(discriminator, truth, n_points=201):
    efficiencies, mistags = [], []
    thresholds = np.linspace(0, 1, n_points)

    pbar = tqdm.tqdm(thresholds)
    for threshold in pbar:
        tagged = discriminator > threshold
        fake = truth == 0
        true = truth == 1

        efficiency = np.sum(true[tagged]) / np.sum(truth)

        mistag = np.sum(np.logical_and(tagged,fake)) / np.sum(fake)
        print("mistagged: {}".format(np.sum(np.logical_and(tagged,fake))))

        efficiencies.append(efficiency)
        mistags.append(mistag)

    return thresholds, np.array(efficiencies), np.array(mistags)


def calculate_working_point(threshold, efficiency, mistag, wp):
    wp_lin = np.linspace(0, 1, 100001)
    f_beff = scipy.interpolate.splrep(threshold, efficiency, s=0, k=1)
    f_mistag = scipy.interpolate.splrep(threshold, mistag, s=0, k=1)

    def shifted_mistag(wp, shift=0):
        return (scipy.interpolate.splev(wp, f_mistag) - shift) ** 2

    rates = shifted_mistag(wp_lin, shift=wp)
    thresh = wp_lin[rates.argmin()]

    return thresh, scipy.interpolate.splev(thresh, f_beff), wp
