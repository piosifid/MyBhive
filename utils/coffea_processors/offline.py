from typing import List

import awkward as ak
import hist
import numpy as np
from coffea import processor

from utils.coffea_processors.base import DataPreprocessing_BaseClass


class OfflineDataPreprocessing(DataPreprocessing_BaseClass):
    """
    Extracts features from ROOT files needed for a DeepJet training using a coffea processor. Furthermore, it generates histograms in p_T/eta space for each flavor (b, bb, leptonic b, c, uds, g).

    Parameters
    ----------
    self.output_dir : string
                      Defines the directory, where the output will be saved.
    self._accumulator : array-like
                        Coffea accumulator used to store extracted values in a dictionary. For more infos look at https://github.com/CoffeaTeam/coffea.
    self.lower_pt : float
                    Lower p_T cut applied while extracting features.
    self.upper_pt : float
                    Upper p_T cut applied while extracting features.
    self.lower_eta : float
                     Lower eta cut applied while extracting features.
    self.upper_eta : float
                     Upper eta cut applied while extracting features.
    self.bins_pt : list
                   Binning used for p_T. Due to the behaviour of the hist package, the right most bin had to be modified to ensure compatibility with numpy's binning.
    self.bins_eta : list
                    Binning used for eta. Due to the behaviour of the hist package, the right most bin had to be modified to ensure compatibility with numpy's binning.
    self.b_hist : histogram
                  Initialises the histogram for the flavor b using the binning defined by self.bins_pt and self.bins_eta. For more information look at https://github.com/scikit-hep/hist.
    self.bb_hist : histogram
                   Initialises the histogram for the flavor bb using the binning defined by self.bins_pt and self.bins_eta. For more information look at https://github.com/scikit-hep/hist.
    self.lepb_hist : histogram
                    Initialises the histogram for the flavor leptonic b using the binning defined by self.bins_pt and self.bins_eta. For more information look at https://github.com/scikit-hep/hist.
    self.c_hist : histogram
                Initialises the histogram for the flavor c using the binning defined by self.bins_pt and self.bins_eta. For more information look at https://github.com/scikit-hep/hist.
    self.uds_hist : histogram
                    Initialises the histogram for the flavor uds using the binning defined by self.bins_pt and self.bins_eta. For more information look at https://github.com/scikit-hep/hist.
    self.g_hist : histogram
                  Initialises the histogram for the flavor g using the binning defined by self.bins_pt and self.bins_eta. For more information look at https://github.com/scikit-hep/hist.
    self.setFeatureNamesAndEdges() :
                                     Function to define the feature names to extract and position in the finale dataset.
    """

    def setFeatureNamesAndEdges(self):
        feature_edges = []
        feature_names = [
            "pt",
            "eta",
            "DeepJet_nCpfcand",
            "DeepJet_nNpfcand",
            "DeepJet_nsv",
            "DeepJet_npv",
            "DeepCSV_trackSumJetEtRatio",
            "DeepCSV_trackSumJetDeltaR",
            "DeepCSV_vertexCategory",
            "DeepCSV_trackSip2dValAboveCharm",
            "DeepCSV_trackSip2dSigAboveCharm",
            "DeepCSV_trackSip3dValAboveCharm",
            "DeepCSV_trackSip3dSigAboveCharm",
            "DeepCSV_jetNSelectedTracks",
            "DeepCSV_jetNTracksEtaRel",
        ]
        feature_edges.append(len(feature_names))
        cpf = [
            [
                f"DeepJet_Cpfcan_BtagPf_trackEtaRel_{i}",
                f"DeepJet_Cpfcan_BtagPf_trackPtRel_{i}",
                f"DeepJet_Cpfcan_BtagPf_trackPPar_{i}",
                f"DeepJet_Cpfcan_BtagPf_trackDeltaR_{i}",
                f"DeepJet_Cpfcan_BtagPf_trackPParRatio_{i}",
                f"DeepJet_Cpfcan_BtagPf_trackSip2dVal_{i}",
                f"DeepJet_Cpfcan_BtagPf_trackSip2dSig_{i}",
                f"DeepJet_Cpfcan_BtagPf_trackSip3dVal_{i}",
                f"DeepJet_Cpfcan_BtagPf_trackSip3dSig_{i}",
                f"DeepJet_Cpfcan_BtagPf_trackJetDistVal_{i}",
                f"DeepJet_Cpfcan_ptrel_{i}",
                f"DeepJet_Cpfcan_drminsv_{i}",
                f"DeepJet_Cpfcan_VTX_ass_{i}",
                f"DeepJet_Cpfcan_puppiw_{i}",
                f"DeepJet_Cpfcan_chi2_{i}",
                f"DeepJet_Cpfcan_quality_{i}",
            ]
            for i in range(25)
        ]
        feature_names.extend([item for sublist in cpf for item in sublist])
        feature_edges.append(len(feature_names))
        npf = [
            [
                f"DeepJet_Npfcan_ptrel_{i}",
                f"DeepJet_Npfcan_deltaR_{i}",
                f"DeepJet_Npfcan_isGamma_{i}",
                f"DeepJet_Npfcan_HadFrac_{i}",
                f"DeepJet_Npfcan_drminsv_{i}",
                f"DeepJet_Npfcan_puppiw_{i}",
            ]
            for i in range(25)
        ]
        feature_names.extend([item for sublist in npf for item in sublist])
        feature_edges.append(len(feature_names))
        vtx = [
            [
                f"DeepJet_sv_pt_{i}",
                f"DeepJet_sv_deltaR_{i}",
                f"DeepJet_sv_mass_{i}",
                f"DeepJet_sv_ntracks_{i}",
                f"DeepJet_sv_chi2_{i}",
                f"DeepJet_sv_normchi2_{i}",
                f"DeepJet_sv_dxy_{i}",
                f"DeepJet_sv_dxysig_{i}",
                f"DeepJet_sv_d3d_{i}",
                f"DeepJet_sv_d3dsig_{i}",
                f"DeepJet_sv_costhetasvpv_{i}",
                f"DeepJet_sv_enratio_{i}",
            ]
            for i in range(4)
        ]
        feature_names.extend([item for sublist in vtx for item in sublist])
        feature_edges.append(len(feature_names))
        feature_names.append("truth")
        feature_names.append("process")
        self.features = feature_names
        self.feature_edges = feature_edges

    def callColumnAccumulator(self, output, events, flag):
        pt_slice = np.logical_and(
            ak.to_numpy(ak.flatten(events["Jet"]["pt"], axis=1)) >= min(self.bins_pt),
            ak.to_numpy(ak.flatten(events["Jet"]["pt"], axis=1)) <= max(self.bins_pt),
        )
        eta_slice = np.logical_and(
            ak.to_numpy(ak.flatten(events["Jet"]["eta"], axis=1)) >= min(self.bins_eta),
            ak.to_numpy(ak.flatten(events["Jet"]["eta"], axis=1)) <= max(self.bins_eta),
        )
        data_slice = np.logical_and(pt_slice, eta_slice)

        for f in self.features[:-1]:
            output[f"Jet_{f}"] = processor.column_accumulator(
                ak.to_numpy(
                    ak.values_astype(ak.flatten(events["Jet"][f"{f}"], axis=1), np.float32)
                )[data_slice]
            )
        flavsplit = ak.to_numpy(
            ak.values_astype(ak.flatten(events["Jet"]["FlavSplit"], axis=1), np.float32)
        )[data_slice]
        target_class = np.full_like(flavsplit, 1)
        target_class = np.where(flavsplit == 500, 0, target_class)  # b
        target_class = np.where(
            np.bitwise_or(flavsplit == 510, flavsplit == 511), 1, target_class
        )  # bb
        target_class = np.where(
            np.bitwise_or(flavsplit == 520, flavsplit == 521), 2, target_class
        )  # leptonicb
        target_class = np.where(
            np.bitwise_or(flavsplit == 400, flavsplit == 410, flavsplit == 411),
            3,
            target_class,
        )  # c
        target_class = np.where(
            np.bitwise_or(flavsplit == 1, flavsplit == 2), 4, target_class
        )  # uds
        target_class = np.where(flavsplit == 0, 5, target_class)  # g

        output[f"Jet_truth"] = processor.column_accumulator(target_class)
        output["Jet_process"] = processor.column_accumulator(
            np.full_like(target_class[data_slice], flag)
        )
        return output

    def saveOutput(self, output_location, output):
        arr = np.stack(
            [np.concatenate([output[f"Jet_{feature}"].value]) for feature in self.features],
            axis=1,
        )

        arr = arr[~np.any(np.isnan(arr), axis=-1)]
        np.save(
            output_location,
            arr,
        )
