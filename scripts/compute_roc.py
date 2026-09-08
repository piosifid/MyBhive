import argparse
import os
from typing import List

import awkward as ak
import numpy as np
import pandas
from coffea import processor
from coffea.nanoevents import BaseSchema
from coffea.processor.accumulator import column_accumulator
from sklearn.metrics import auc, roc_curve

from utils.evaluation.working_point import (
    calculate_efficiency_curve,
    calculate_working_point,
)
from utils.plotting.roc import plot_roc
from utils.plotting.termplot import terminal_roc
from utils.plotting.working_point import plot_working_points


def setup_fileset(input_paths, labels, maxFiles=None):
    if not (isinstance(input_paths, list)):
        input_paths = [input_paths]
        labels = [labels]
    fileset = {}
    for path, label in zip(input_paths, labels):
        if path.endswith(".txt"):
            with open(path, "r") as txt_file_list:
                file_list = list(map(lambda x: x.rstrip("\n"), txt_file_list.readlines()))
        elif path.endswith(".root"):
            file_list = [path]
        else:
            raise NotImplementedError
        fileset[label] = file_list[0 : None if not maxFiles else min(maxFiles, len(file_list))]
    return fileset


def setup_prediction(flat_jets, phase2=False):
    jets = ak.zip(
        {
            "jet_pt": flat_jets.jet_pt,
            "jet_phi": flat_jets.jet_phi,
            "jet_mass": flat_jets.jet_mass,
            "jet_eta": flat_jets.jet_eta,
            "isB": flat_jets.isB,
            "isBB": flat_jets.isBB,
            "isGBB": flat_jets.isGBB,
            "isLeptonicB": flat_jets.isLeptonicB,
            "isLeptonicB_C": flat_jets.isLeptonicB_C,
            "isC": flat_jets.isC,
            "isGCC": flat_jets.isGCC,
            "isCC": flat_jets.isCC,
            "isUD": flat_jets.isUD,
            "isS": flat_jets.isS,
            "isG": flat_jets.isG,
            "isTau": flat_jets.isTau,
            "isUndefined": flat_jets.isUndefined,
            "hltPFDeepFlavourJetTags_probb": flat_jets.hltPFDeepFlavourJetTags_probb
            if not phase2
            else flat_jets.hltPfDeepFlavourJetTags_probb,
            "hltPFDeepFlavourJetTags_probbb": flat_jets.hltPFDeepFlavourJetTags_probbb
            if not phase2
            else flat_jets.hltPfDeepFlavourJetTags_probbb,
            "hltPFDeepFlavourJetTags_problepb": flat_jets.hltPFDeepFlavourJetTags_problepb
            if not phase2
            else flat_jets.hltPfDeepFlavourJetTags_problepb,
            "hltPFDeepFlavourJetTags_probc": flat_jets.hltPFDeepFlavourJetTags_probc
            if not phase2
            else flat_jets.hltPfDeepFlavourJetTags_probc,
            "hltPFDeepFlavourJetTags_probuds": flat_jets.hltPFDeepFlavourJetTags_probuds
            if not phase2
            else flat_jets.hltPfDeepFlavourJetTags_probuds,
            "hltPFDeepFlavourJetTags_probg": flat_jets.hltPFDeepFlavourJetTags_probg
            if not phase2
            else flat_jets.hltPfDeepFlavourJetTags_probg,
            "category": ak.zip(
                {
                    "Jet_category_all": flat_jets.jet_pt > -1,
                    "Jet_category_light": flat_jets.isUD + flat_jets.isS + flat_jets.isG,
                    "Jet_category_C": flat_jets.isC + flat_jets.isGCC + flat_jets.isCC,
                    "Jet_category_B": flat_jets.isB
                    + flat_jets.isBB
                    + flat_jets.isGBB
                    + flat_jets.isLeptonicB
                    + flat_jets.isLeptonicB_C,
                }
            ),
        }
    )
    return jets


class PredictionExporter(processor.ProcessorABC):
    def __init__(self, phase2) -> None:
        super().__init__()
        self.phase2 = phase2

    def process(self, events):
        dataset = events.metadata["dataset"]

        jets = setup_prediction(events, phase2=self.phase2)

        jet_mask = (jets.jet_pt > 30.0) & (jets.jet_pt < 1000.0) & (np.abs(jets.jet_eta) <= 2.5)

        jets = jets[jet_mask]

        ret = {
            dataset: {
                "flavour_b": column_accumulator(jets.category.Jet_category_B.to_numpy()),
                "flavour_c": column_accumulator(jets.category.Jet_category_C.to_numpy()),
                "flavour_light": column_accumulator(jets.category.Jet_category_light.to_numpy()),
                "probb": column_accumulator(jets.hltPFDeepFlavourJetTags_probb.to_numpy()),
                "probbb": column_accumulator(jets.hltPFDeepFlavourJetTags_probbb.to_numpy()),
                "problepb": column_accumulator(jets.hltPFDeepFlavourJetTags_problepb.to_numpy()),
                "probc": column_accumulator(jets.hltPFDeepFlavourJetTags_probc.to_numpy()),
                "probuds": column_accumulator(jets.hltPFDeepFlavourJetTags_probuds.to_numpy()),
                "probg": column_accumulator(jets.hltPFDeepFlavourJetTags_probg.to_numpy()),
            }
        }

        return ret

    def postprocess(self, accumulator):
        pass


def main(
    file_list: List,
    output: str,
    labels: List,
    phase2: bool,
    color: str,
    debug: bool,
    workers: int = 64,
):
    file_set = setup_fileset(file_list, labels, maxFiles=None)

    iterative_run = processor.Runner(
        executor=processor.FuturesExecutor(compression=None, workers=workers),
        schema=BaseSchema,
        maxchunks=None if not (debug) else 100,
    )

    out = iterative_run(
        file_set,
        treename="DeepJetNTupler/DeepJetvars",
        processor_instance=PredictionExporter(phase2=phase2),
    )

    for label, proc in zip(["BvsL - TT", "BvsL - QCD"], file_set.keys()):
        b_pred = out[proc]["probb"].value + out[proc]["probbb"].value + out[proc]["problepb"].value
        c_pred = out[proc]["probc"].value
        l_pred = out[proc]["probuds"].value + out[proc]["probg"].value

        bvsl = np.where((b_pred + l_pred) > 0, (b_pred) / (b_pred + l_pred), -1)
        bvsc = np.where((b_pred + c_pred) > 0, (b_pred) / (b_pred + c_pred), -1)
        cvsb = np.where((b_pred + c_pred) > 0, (c_pred) / (b_pred + c_pred), -1)
        cvsl = np.where((l_pred + c_pred) > 0, (c_pred) / (l_pred + c_pred), -1)

        b_jets = out[proc]["flavour_b"].value
        c_jets = out[proc]["flavour_c"].value == 1

        b_veto = out[proc]["flavour_b"].value == 0
        c_veto = out[proc]["flavour_c"].value == 0
        l_veto = out[proc]["flavour_light"].value == 0

        os.makedirs(output, exist_ok=True)
        terminal_roc(bvsl, out[proc]["flavour_b"].value)
        """
        WPs 
        """
        working_points = [0.1, 0.05, 0.01, 0.005, 0.001]
        threshold, eff, mistag = calculate_efficiency_curve(
            bvsl[~c_jets], b_jets[~c_jets], n_points=200
        )
        wps = [calculate_working_point(threshold, eff, mistag, wp) for wp in working_points]
        df = pandas.DataFrame()
        df = pandas.DataFrame()
        df["mistag rate"] = mistag
        df["b jet efficiency"] = eff
        df["thresholds"] = threshold
        print("saving wp to: ", os.path.join(output, "wps_{}.csv".format(proc)))
        df.to_csv(os.path.join(output, "wps_{}.csv".format(proc)), index=False)

        plot_working_points(
            (threshold, mistag),
            wps,
            working_points,
            label,
            out_path=os.path.join(output, "working_point_{}.jpg".format(proc)),
            color=color,
        )

        """
        Rocs
        """
        for roc_label, disc, veto, truth in zip(
            ["bvsl", "bvsc", "cvsb", "cvsl"],
            [bvsl, bvsc, cvsb, cvsl],
            [c_veto, l_veto, b_veto, b_veto],
            [b_jets, b_jets, c_jets, c_jets],
        ):
            fpr, tpr, _ = roc_curve(truth[veto], disc[veto])
            area = auc(fpr, tpr)
            outpath = os.path.join(output, f"roc_{proc}_{roc_label}.npy")
            np.save(outpath, np.array((fpr, tpr)))

            outpath = os.path.join(output, f"roc_{proc}_{roc_label}.png")
            plot_roc([(fpr, tpr, area)], [roc_label], dataset_label=proc, pt_min=30, pt_max=1000, output_path=outpath, colors=color, xmin=0.4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filelist", "-f", type=str, nargs="+", help="Filelist that should be read in."
    )
    parser.add_argument(
        "--labels", "-l", type=str, nargs="+", help="Labels that should be read in."
    )
    parser.add_argument("--output", "-o", type=str, help="Output path.")
    parser.add_argument("--debug", "-d", action="store_true", help="Output path.")
    parser.add_argument("--phase2", "-p2", action="store_true", help="Phase2")
    parser.add_argument("--color", "-c", type=str, help="Colors to use for plots")
    args = parser.parse_args()

    file_list = args.filelist
    output = args.output
    labels = args.labels
    phase2 = args.phase2

    main(file_list, output, labels, phase2, args.color, args.debug)
