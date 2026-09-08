import os
from typing import List

import awkward as ak
import hist
import numpy as np
from coffea import processor


class DataPreprocessing_BaseClass(processor.ProcessorABC):
    n_cpf = 128
    n_npf = 128
    n_vtx = 10
    n_lt  = 5

    def __init__(
        self,
        output_directory=None,
        bins_pt: List = None,
        bins_eta: List = None,
        prefix="",
        precision="float32",
        global_features: List[str] = None,
        global_custom_features: dict = None,
        cpf_candidates: List[str] = None,
        cpf_custom_features: dict = None,
        npf_candidates: List[str] = None,
        npf_custom_features: dict = None,
        vtx_features: List[str] = None,
        vtx_custom_features: dict = None,
        lt_candidates: List[str] = None,
        lt_custom_features: dict = None,
        truths: List[str] = None,
        processes: List[str] = None,
        n_cpf_candidates=128,
        n_npf_candidates=0,
        n_vtx_features=10,
        n_lt_candidates=0,
        pt_key: str = None,
        eta_key: str = None,
    ):
        self._accumulator = processor.dict_accumulator({})
        self.bins_eta = bins_eta
        self.bins_pt = bins_pt
        self.output_dir = output_directory
        self.precision = np.dtype(precision).type
        self.prefix = prefix
        self.processes = processes
        self.cpf = cpf_candidates
        self.cpf_custom = cpf_custom_features
        self.npf = npf_candidates
        self.npf_custom = npf_custom_features
        self.vtx = vtx_features
        self.vtx_custom = vtx_custom_features
        self.lt = lt_candidates
        self.lt_custom = lt_custom_features
        self.global_features = global_features
        self.global_custom_features = global_custom_features
        self.truths = truths
        self.n_cpf = n_cpf_candidates
        self.n_npf = n_npf_candidates
        self.n_vtx = n_vtx_features
        self.n_lt  = n_lt_candidates
        self.pt_key = pt_key
        self.eta_key = eta_key
        if self.processes is None:
            self.processes = []

        self.truth_hists = {}

        for truth in self.truths:
            self.truth_hists[truth] = (
                hist.Hist.new.Variable(self.bins_pt, name="pt")
                .Variable(self.bins_eta, name="eta")
                .Int64()
            )
        self.setFeatureNamesAndEdges()

    def setFeatureNamesAndEdges(self):
        feature_edges = []
        feature_names = []

        feature_names.append(self.global_features)
        feature_names.append(self.global_custom_features)
        feature_edges.append(len(feature_names))
        feature_edges.append(
            feature_edges[-1] + (len(self.cpf) + len(self.cpf_custom)) * self.n_cpf
        )
        feature_names.extend(self.cpf)
        feature_names.extend(self.cpf_custom)
        feature_edges.append(
            feature_edges[-1] + (len(self.npf) + len(self.npf_custom)) * self.n_npf
        )
        feature_names.extend(self.npf)
        feature_names.extend(self.npf_custom)
        feature_edges.append(
            feature_edges[-1] + (len(self.vtx) + len(self.vtx_custom)) * self.n_vtx
        )
        feature_names.extend(self.vtx)
        feature_names.extend(self.vtx_custom)
        feature_edges.append(
            feature_edges[-1] + (len(self.lt) + len(self.lt_custom)) * self.n_lt
        )
        feature_names.extend(self.lt)
        feature_names.extend(self.lt_custom)
        
        feature_names.append("truths")
        feature_names.extend(self.truths)
        feature_names.append("process")

        self.feature_edges = feature_edges
        self.features = feature_names

    def saveOutput(self, output_location, output):
        pass

    @property
    def accumulator(self):
        return self._accumulator

    def callColumnAccumulator(self, output, events, **kwargs):
        pass

    def process(self, events):
        dataset = events.metadata["dataset"]

        start = events.metadata["entrystart"]
        stop = events.metadata["entrystop"]
        filename = "_".join(events.metadata["filename"].split("/")[1:]).split(".")[0]

        # assign process number
        if self.processes == ["default"]:
            proc_flag = 0
        else:
            proc_flag = -1
            for i, proc in enumerate(self.processes):
                if proc in dataset:
                    proc_flag = i
        
        output = self.accumulator
        output_location_list = []

        (
            global_arr,
            cpf_arr,
            npf_arr,
            vtx_arr,
            lt_arr,
            truth,
            process,
        ) = self.callColumnAccumulator(
            output, events, proc_flag, filename=events.metadata["filename"]
        )
        
        for truth_label in self.truths:
            self.truth_hists[truth_label].fill(
                global_arr[self.pt_key][truth[truth_label]],
                global_arr[self.eta_key][truth[truth_label]],
            )

        output_location = os.path.join(
            self.output_dir, f"{self.prefix}{dataset}_{filename}_{start}_{stop}.npz"
        )

        output_location_list.append(output_location)

        self.saveOutput(
            output_location, global_arr, cpf_arr, npf_arr, vtx_arr, lt_arr, truth, process
        )
        
        ret = {}
        ret["output_location"] = output_location_list
        for label, hist in self.truth_hists.items():
            ret[label] = hist
        return ret

    def postprocess(self, accumulator):
        pass