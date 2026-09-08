import numpy as np
from rich.progress import track

from typing import Dict, List


def weight_all_files_histrogram_weighting(
    files,
    histograms: Dict = None,
    reference_key="isB",
    bins_pt=None,
    bins_eta=None,
    pt_key=None,
    eta_key=None,
) -> List[str]:
    # for file in track(files, "Evaluating and saving the weights..."):
    for file in files:
        samples = np.load(file, allow_pickle=True)
        jet_pt = samples["global_features"][pt_key]
        jet_eta = samples["global_features"][eta_key]
        pt_coordinate = np.digitize(jet_pt, bins_pt) - 1
        eta_coordinate = np.digitize(jet_eta, bins_eta) - 1
        weights = histogram_weighting(histograms, reference_key=reference_key)

        """
        print('### Weighting Debug ###')
        print('bins pt: ',bins_pt)
        print('bins eta: ',bins_eta)
        print('pt coordinate: ',pt_coordinate)
        print('len pt coordinate: ',len(pt_coordinate))
        print('len eta coordinate: ',len(eta_coordinate))
        print('eta coordinate: ',eta_coordinate)
        """
        
        # values need to be casted to list first!
        # otherwise it will be an object array
        flavour_idx = np.lib.recfunctions.apply_along_fields(
            np.argmax, samples["truth"]
        )
        
        weights = np.array(list(weights.values()))
        """
        print('flavour index; ', flavour_idx)
        print(len(flavour_idx))
        print('flavour_idx:', flavour_idx)
        print('pt_coordinate:', pt_coordinate)
        print('eta_coordinate:', eta_coordinate)
        print('weights: ', weights)
        print(len(weights))
        """
        
        w = weights[flavour_idx, pt_coordinate, eta_coordinate]

        np.savez(file, **samples, weight=w)
    return files


def histogram_weighting(
    histograms: Dict,
    reference_key="isB",
) -> Dict:
    # Get the weights
    reference_histogram = histograms[reference_key].view() / np.max(
        histograms[reference_key].view()
    )
    weights_list = {}
    for key, hist in histograms.items():
        other_histogram = hist.view() / np.max(hist.view())
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(
                other_histogram > 0, reference_histogram / other_histogram, -10
            )
        weights = weights / np.max(weights)

        weights[weights < 0] = 1
        weights[weights == np.nan] = 1

        weights_list[key] = weights
    """
    print('+++ Histogram Weighting Debug +++')
    print('weights list: ',weights_list)
    """
    return weights_list
