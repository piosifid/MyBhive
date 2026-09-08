import awkward as ak
import numpy as np
from functools import reduce

from utils.coffea_processors.base import DataPreprocessing_BaseClass
from utils.dataset.structured_arrays import (
    structured_array_from_tree,
    structured_array_from_tree_truth_from_dict,
    structured_custom_array_from_tree,
)
from utils.coffea_processors.pf_candidate_and_vertex import PFCandidateAndVertexProcessing



class LZ4Processing(PFCandidateAndVertexProcessing):

    def process_array(self, arr, n_jet, n_Cand=None):
        """
        Safely process an array by checking if it's valid and non-empty.
        Returns a reshaped version or a placeholder array if the array is empty.
        """
        if len(arr) == 0 or not len(arr.dtype.names):
            # Return a placeholder array with the correct shape
            if n_Cand == None:
                return np.zeros((n_jet, 0), dtype=self.precision)
            else:
                return np.zeros((n_jet, 0, n_Cand), dtype=self.precision) 
        return arr.view((arr.dtype[0], len(arr.dtype.names))).astype(self.precision)
        
    def saveOutput(
        self, output_location, global_arr, cpf_arr, npf_arr, vtx_arr, lt_arr, truth, process
    ):
        n_jet = truth.shape[0]
        
        # Process individual arrays, replacing missing arrays with placeholders
        if (len(global_arr) == 0):
            assert(len(cpf_arr) == 0)
            assert(len(npf_arr) == 0)
            assert(len(vtx_arr) == 0)
            assert(len(truth) == 0)
            assert(len(process) == 0)
            return
        global_part = self.process_array(global_arr, n_jet).reshape(n_jet, -1)
        cpf_part = np.swapaxes(self.process_array(cpf_arr, n_jet, n_Cand=self.n_cpf), 1, 2).reshape(n_jet, -1)
        npf_part = np.swapaxes(self.process_array(npf_arr, n_jet, n_Cand=self.n_npf), 1, 2).reshape(n_jet, -1)
        vtx_part = np.swapaxes(self.process_array(vtx_arr, n_jet, n_Cand=self.n_vtx), 1, 2).reshape(n_jet, -1)
        lt_part  = np.swapaxes(self.process_array(lt_arr,  n_jet, n_Cand=self.n_lt),  1, 2).reshape(n_jet, -1)
        process_part = process.astype(self.precision).reshape(n_jet, -1)
        truth_part = truth.view((truth.dtype[0], len(truth.dtype.names))).astype(self.precision).reshape(n_jet, -1)
        
        arr = np.concatenate([global_part, cpf_part, npf_part, vtx_part, lt_part, process_part, truth_part], axis=1)
        
        arr = arr[~np.any(np.isnan(arr), axis=-1)]
        arr = arr[~np.any(np.isinf(arr), axis=-1)]

        np.save(
            output_location[:-4],
            arr,
        )
