import awkward as ak
import numpy as np
from functools import reduce

from utils.coffea_processors.pf_candidate_and_vertex import PFCandidateAndVertexProcessing
from utils.dataset.structured_arrays import (
    structured_array_from_tree,
    structured_array_from_tree_truth_from_dict,
)

class L1PFCandidateAndVertexProcessing(PFCandidateAndVertexProcessing):
    def callColumnAccumulator(self, output, events, flag, **kwargs):
        # slicing based on p_T and eta
        pt_slice = np.logical_and(
            ak.to_numpy(ak.flatten(events["jet_pt"], axis=0)) >= min(self.bins_pt),
            ak.to_numpy(ak.flatten(events["jet_pt"], axis=0)) <= max(self.bins_pt),
        )
        eta_slice = np.logical_and(
            ak.to_numpy(ak.flatten(events["jet_eta"], axis=0)) >= min(self.bins_eta),
            ak.to_numpy(ak.flatten(events["jet_eta"], axis=0)) <= max(self.bins_eta),
        )

        genCut = events['jet_genmatch_pt'] > -1

        if isinstance(self.truths, dict):
            truth_arr = structured_array_from_tree_truth_from_dict(
                events=events,
                truth_dict=self.truths,
                precision=bool,
                feature_length=1,
            )
        else:
            truth_arr = structured_array_from_tree(
                events=events,
                keys=self.truths,
                precision=bool,
                feature_length=1,
            )

        data_slice = np.array(
            (pt_slice & eta_slice & genCut)
            & reduce(
                np.logical_or,
                [
                    truth_arr[truth]
                    for truth in (
                        self.truths.keys()
                        if isinstance(self.truths, dict)
                        else self.truths
                    )
                ],
            ),
            dtype=bool,
        )
        truth_arr = truth_arr[data_slice]

        global_arr = structured_array_from_tree(
            events=events[data_slice],
            keys=self.global_features,
            precision=self.precision,
            feature_length=1,
        )

        cpf_arr = structured_array_from_tree(
            events=events[data_slice],
            keys=self.cpf,
            precision=self.precision,
            feature_length=self.n_cpf,
        )
        npf_arr = structured_array_from_tree(
            events=events[data_slice],
            keys=self.npf,
            precision=self.precision,
            feature_length=self.n_npf,
        )
        vtx_arr = structured_array_from_tree(
            events=events[data_slice],
            keys=self.vtx,
            precision=self.precision,
            feature_length=self.n_vtx,
        )

        # create an array with the process value
        process = np.ones(len(global_arr)) * flag

        glob_mask = reduce(
            np.logical_and,
            [~np.any(np.isnan(global_arr[key])) for key in global_arr.dtype.names],
        )
        if cpf_arr.dtype.names:
            cpf_mask = reduce(
                np.logical_and,
                [
                    ~np.any(np.isnan(cpf_arr[key]), axis=1)
                    for key in cpf_arr.dtype.names
                ],
            )
        else:
            cpf_mask = np.ones(len(global_arr))
        if npf_arr.dtype.names:
            npf_mask = reduce(
                np.logical_and,
                [
                    ~np.any(np.isnan(npf_arr[key]), axis=1)
                    for key in npf_arr.dtype.names
                ],
            )
        else:
            npf_mask = np.ones(len(global_arr))
        if vtx_arr.dtype.names:
            vtx_mask = reduce(
                np.logical_and,
                [~np.any(np.isnan(vtx_arr[key]), axis=1) for key in vtx_arr.dtype.names],
            )
        else:
            vtx_mask = np.ones(len(global_arr))

        nan_mask = reduce(np.logical_and, [glob_mask, cpf_mask, npf_mask, vtx_mask])

        return (
            global_arr[nan_mask],
            cpf_arr[nan_mask],
            npf_arr[nan_mask],
            vtx_arr[nan_mask],
            truth_arr[nan_mask],
            process[nan_mask],
        )
