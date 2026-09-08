import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax

from utils.models.base_model import Classifier_base
from utils.models.helpers import (
    DenseClassifier, 
    InputProcess, 
    InputProcess_HLT,
    MoDDenseClassifier,
    MoDInputProcess,
)
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm import tqdm
from utils.plotting.termplot import terminal_roc

torch.multiprocessing.set_sharing_strategy("file_system")

class DeepJet(Classifier_base):
    
    classes = {
        "b": ["isB"],
        "bb": ["isBB", "isGBB"],
        "leptonicB": ["isLeptonicB", "isLeptonicB_C"],
        "c": ["isC", "isCC", "isGCC"],
        "uds": ["isUD", "isS"],
        "g": ["isG"],
    }

    integer_features = {
        "global_features": ["n_Cpfcand", "nCpfcan", "n_Npfcand", "nNpfcan", "nsv", "npv",
                            "TagVarCSV_vertexCategory", "TagVarCSV_jetNSelectedTracks", "TagVarCSV_jetNTracksEtaRel"],
        "cpf_candidates": ["Cpfcan_VTX_ass", "Cpfcan_puppiw", "Cpfcan_chi2", "Cpfcan_quality"],
        "npf_candidates": ["Npfcan_isGamma", "Npfcan_HadFrac", "Npfcan_puppiw"],
        "vtx_features": ["sv_ntracks"]
    }

    def __init__(self,
                 config,
                 cpf_conv = [64, 32, 32, 8],
                 npf_conv = [32, 16, 4],
                 vtx_conv = [64, 32, 32, 8],
                 n_layers_lstm = 1,
                 lstm_dim = [150, 50, 50],
                 dense_clas_dim = [200, 100, 100, 100, 100, 100, 100, 100, 100],
                 **kwargs
        ):
        
        super().__init__(config, **kwargs)

        global_dim, cpf_dim, npf_dim, vtx_dim, _ = self.all_model_features_length()  
        cpf_conv = [cpf_dim] + cpf_conv
        npf_conv = [npf_dim] + npf_conv
        vtx_conv = [vtx_dim] + vtx_conv
        
        self.InputProcess = InputProcess(cpf_conv, npf_conv, vtx_conv)

        dense_clas_dim_full = [sum(lstm_dim) + global_dim] + dense_clas_dim
        self.DenseClassifier = DenseClassifier(dense_clas_dim_full)

        self.global_bn = torch.nn.BatchNorm1d(global_dim, eps=0.001, momentum=0.6)
        self.cpf_lstm = torch.nn.LSTM(
            input_size=cpf_conv[-1], hidden_size=lstm_dim[0], num_layers=n_layers_lstm, batch_first=True
        )
        self.npf_lstm = torch.nn.LSTM(
            input_size=npf_conv[-1], hidden_size=lstm_dim[1],  num_layers=n_layers_lstm, batch_first=True
        )
        self.vtx_lstm = torch.nn.LSTM(
            input_size=vtx_conv[-1], hidden_size=lstm_dim[2],  num_layers=n_layers_lstm, batch_first=True
        )

        self.cpf_bn = torch.nn.BatchNorm1d(lstm_dim[0], eps=0.001, momentum=0.6)
        self.npf_bn = torch.nn.BatchNorm1d(lstm_dim[1], eps=0.001, momentum=0.6)
        self.vtx_bn = torch.nn.BatchNorm1d(lstm_dim[2], eps=0.001, momentum=0.6)

        self.cpf_dropout = torch.nn.Dropout(0.1)
        self.npf_dropout = torch.nn.Dropout(0.1)
        self.vtx_dropout = torch.nn.Dropout(0.1)

        self.Linear = nn.Linear(100, len(self.classes))
        
    def forward(self, inpt):
         
        global_features, cpf_features, npf_features, vtx_features = inpt[0], inpt[1], inpt[2], inpt[3]
        
        global_features = self.global_bn(global_features)
        
        cpf, npf, vtx = self.InputProcess(cpf_features, npf_features, vtx_features)
        
        cpf = self.cpf_lstm(torch.flip(cpf, dims=[1]))[0][:, -1]
        cpf = self.cpf_dropout(self.cpf_bn(cpf))
            
        npf = self.npf_lstm(torch.flip(npf, dims=[1]))[0][:, -1]
        npf = self.npf_dropout(self.npf_bn(npf))

        vtx = self.vtx_lstm(torch.flip(vtx, dims=[1]))[0][:, -1]
        vtx = self.vtx_dropout(self.vtx_bn(vtx))

        fts = torch.cat((global_features, cpf, npf, vtx), dim=1)
        fts = self.DenseClassifier(fts)

        output = self.Linear(fts)

        return output


class DeepJetHLT(Classifier_base):

    # Define classes used by model and connect it to the config definitions
    classes = {
        "b": ["label_b", "label_gbb", "label_bb", "label_leptonicb", "label_leptonicbc"],
        "c": ["label_c", "label_gcc", "label_cc"],
        "uds": ["label_ud", "label_s"],
        "g": ["label_g"],
        "taup": ["label_taup"],
        "taum": ["label_taum"],
    }
    cpf_candidates = [
        "jet_pfcand_deta",
        "jet_pfcand_dphi",
        "jet_pfcand_pt_log",
        "jet_pfcand_energy_log",
        "jet_pfcand_eta",
        "jet_pfcand_charge",
        "jet_pfcand_frompv",
        "jet_pfcand_nlostinnerhits",
        "jet_pfcand_track_chi2",
        "jet_pfcand_track_qual",
        "jet_pfcand_dz",
        "jet_pfcand_dzsig",
        "jet_pfcand_dxy",
        "jet_pfcand_dxysig",
        "jet_pfcand_etarel",
        "jet_pfcand_pperp_ratio",
        "jet_pfcand_ppara_ratio",
        "jet_pfcand_trackjet_d3d",
        "jet_pfcand_trackjet_d3dsig",
        "jet_pfcand_trackjet_dist",
        "jet_pfcand_trackjet_decayL",
        "jet_pfcand_npixhits",
        "jet_pfcand_nstriphits",
    ]
    vtx_features = [
        "jet_sv_deta",
        "jet_sv_dphi",
        "jet_sv_pt_log",
        "jet_sv_mass",
        "jet_sv_eta",
        "jet_sv_ntrack",
        "jet_sv_chi2",
        "jet_sv_dxy",
        "jet_sv_dxysig",
        "jet_sv_d3d",
        "jet_sv_d3dsig"
    ]
    
    # Features that have integer dtype and should not be changed during adv attacks
    integer_features = {
        "global_features": ["jet_elf", "jet_muf", "jet_ncand", "jet_nbhad", "jet_nchad"],
        "cpf_candidates": ["jet_pfcand_calofraction", "jet_pfcand_hcalfraction", "jet_pfcand_frompv", "jet_pfcand_id",
                             "jet_pfcand_charge", "jet_pfcand_track_qual", "jet_pfcand_track_chi2", "jet_pfcand_npixhits", 
                             "jet_pfcand_nstriphits", "jet_pfcand_nlostinnerhits"],
        "npf_candidates": [],
        "vtx_features": ["jet_sv_ntrack"],
    }

    def __init__(self,
                 config,
                 cpf_conv = [64, 32, 32, 8],
                 vtx_conv = [64, 32, 32, 8],
                 n_layers_lstm = 1,
                 lstm_dim = [150, 50],
                 dense_clas_dim = [200, 100, 100, 100, 100, 100, 100, 100, 100],
                 **kwargs
        ):
        
        super().__init__(config, **kwargs)

        _, cpf_dim, _, vtx_dim, _ = self.all_model_features_length()  
        
        cpf_conv = [cpf_dim] + cpf_conv
        vtx_conv = [vtx_dim] + vtx_conv
        
        self.InputProcess = InputProcess_HLT(cpf_conv, vtx_conv)

        dense_clas_dim_full = [sum(lstm_dim)] + dense_clas_dim
        self.DenseClassifier = DenseClassifier(dense_clas_dim_full)

        self.cpf_lstm = torch.nn.LSTM(
            input_size=cpf_conv[-1], hidden_size=lstm_dim[0], num_layers=n_layers_lstm, batch_first=True
        )
        self.vtx_lstm = torch.nn.LSTM(
            input_size=vtx_conv[-1], hidden_size=lstm_dim[1],  num_layers=n_layers_lstm, batch_first=True
        )

        self.cpf_bn = torch.nn.BatchNorm1d(lstm_dim[0], eps=0.001, momentum=0.6)
        self.vtx_bn = torch.nn.BatchNorm1d(lstm_dim[1], eps=0.001, momentum=0.6)

        self.cpf_dropout = nn.Dropout(0.1)
        self.vtx_dropout = nn.Dropout(0.1)

        self.Linear = nn.Linear(100, len(self.classes))
        
    def forward(self, inpt):
         
        _, cpf_features, _, vtx_features = inpt[0], inpt[1], inpt[2], inpt[3]
                
        cpf, vtx = self.InputProcess(cpf_features, vtx_features)
        
        cpf = self.cpf_lstm(torch.flip(cpf, dims=[1]))[0][:, -1]
        cpf = self.cpf_dropout(self.cpf_bn(cpf))

        vtx = self.vtx_lstm(torch.flip(vtx, dims=[1]))[0][:, -1]
        vtx = self.vtx_dropout(self.vtx_bn(vtx))

        fts = torch.cat((cpf, vtx), dim=1)
        fts = self.DenseClassifier(fts)

        output = self.Linear(fts)

        return output
        
    def calculate_roc_list(
        self,
        predictions,
        truth,
    ):
        if np.abs(np.mean(np.sum(predictions, axis=-1)) - 1) > 1e-3:
            predictions = softmax(predictions, axis=-1)

        b_index   = list(self.classes).index('b')
        c_index   = list(self.classes).index('c')
        uds_index = list(self.classes).index('uds')
        g_index   = list(self.classes).index('g')
        taup_index = list(self.classes).index('taup')
        taum_index = list(self.classes).index('taum')
        
        b_jets   = (truth == b_index)
        tau_jets = (truth == taup_index) | (truth == taum_index)
        c_jets   = (truth == c_index)
        uds_jets = (truth == uds_index)
        g_jets   = (truth == g_index)
        l_jets   = uds_jets | g_jets
        summed_jets = b_jets + c_jets + l_jets + tau_jets

        b_pred = predictions[:, b_index]
        taup_pred = predictions[:, taup_index]
        taum_pred = predictions[:, taum_index]
        c_pred = predictions[:, c_index]
        uds_pred = predictions[:, uds_index]
        g_pred = predictions[:, g_index]
        l_pred = predictions[:, [uds_index, g_index]].sum(axis=1)

        bvsl = np.where((b_pred + l_pred) > 0, (b_pred) / (b_pred + l_pred), -1)
        bvsc = np.where((b_pred + c_pred) > 0, (b_pred) / (b_pred + c_pred), -1)
        cvsb = np.where((b_pred + c_pred) > 0, (c_pred) / (b_pred + c_pred), -1)
        cvsl = np.where((l_pred + c_pred) > 0, (c_pred) / (l_pred + c_pred), -1)
        bvsall = np.where(
            (b_pred + l_pred + c_pred + taup_pred + taum_pred) > 0, (b_pred) / (b_pred + l_pred + c_pred + taup_pred + taum_pred), -1
        )
        uds_vs_g = np.where((uds_pred + g_pred) > 0, (uds_pred) / (uds_pred + g_pred), -1)

        tau_veto = (truth != taup_index) & (truth != taum_index) & (summed_jets != 0)
        b_veto = (truth != b_index) & tau_veto & (summed_jets != 0)
        c_veto = (truth != c_index) & tau_veto & (summed_jets != 0)
        bc_veto = b_veto & c_veto
        l_veto = (truth != uds_index) & (truth != g_index) & tau_veto & (summed_jets != 0)
        no_veto = np.ones(b_veto.shape, dtype=bool)

        labels = ["bvsl", "bvsc", "cvsb", "cvsl", "bvsall", "uds_vs_g"]
        discs = [bvsl, bvsc, cvsb, cvsl, bvsall, uds_vs_g]
        vetos = [c_veto, l_veto, l_veto, b_veto, no_veto, bc_veto]
        truths = [b_jets, b_jets, c_jets, c_jets, b_jets, uds_jets]
        xlabels = [
            "b-identification",
            "b-identification",
            "c-identification",
            "c-identification",
            "b-identification",
            "uds-identification",
        ]
        ylabels = ["light mis-id.", "c mis-id", "b mis-id.", "light mis-id.", "mis-id.", "gluons mis-id."]

        return discs, truths, vetos, labels, xlabels, ylabels
        

class MoDJet(DeepJet):

    classes = {
        "b": ["isB"],
        "bb": ["isBB", "isGBB"],
        "leptonicB": ["isLeptonicB", "isLeptonicB_C"],
        "c": ["isC", "isCC", "isGCC"],
        "uds": ["isU", "isD", "isS"],
        "g": ["isG"],
    }

    global_features = [
        "jet_px",
        "jet_py",
        "jet_pz",
        "jet_energy",
        "jet_mass",
        "n_Cpfcand",
        "n_Npfcand",
        "nsv",
        "npv",
        "TagVarCSV_trackSumJetEtRatio",
        "TagVarCSV_trackSumJetDeltaR",
        "TagVarCSV_vertexCategory",
        "TagVarCSV_trackSip2dValAboveCharm",
        "TagVarCSV_trackSip2dSigAboveCharm",
        "TagVarCSV_trackSip3dValAboveCharm",
        "TagVarCSV_trackSip3dSigAboveCharm",
        "TagVarCSV_jetNSelectedTracks",
        "TagVarCSV_jetNTracksEtaRel",
    ]

    cpf_candidates = [
        "Cpfcan_px",
        "Cpfcan_py",
        "Cpfcan_pz",
        "Cpfcan_e",
        "Cpfcan_mass",
        "Cpfcan_BtagPf_trackEtaRel",
        "Cpfcan_BtagPf_trackPtRel",
        "Cpfcan_BtagPf_trackPPar",
        "Cpfcan_BtagPf_trackDeltaR",
        "Cpfcan_BtagPf_trackPParRatio",
        "Cpfcan_BtagPf_trackSip2dVal",
        "Cpfcan_BtagPf_trackSip2dSig",
        "Cpfcan_BtagPf_trackSip3dVal",
        "Cpfcan_BtagPf_trackSip3dSig",
        "Cpfcan_BtagPf_trackJetDistVal",
        "Cpfcan_ptrel",
        "Cpfcan_drminsv",
        "Cpfcan_VTX_ass",
        "Cpfcan_puppiw",
        "Cpfcan_chi2",
        "Cpfcan_quality",
    ]

    npf_candidates = [
        "Npfcan_px",
        "Npfcan_py",
        "Npfcan_pz",
        "Npfcan_e",
        "Npfcan_mass",
        "Npfcan_ptrel",
        "Npfcan_deltaR",
        "Npfcan_isGamma",
        "Npfcan_HadFrac",
        "Npfcan_drminsv",
        "Npfcan_puppiw",
    ]

    vtx_features = [
        "sv_px",
        "sv_py",
        "sv_pz",
        "sv_e",
        "sv_mass",
        "sv_deltaR",
        "sv_ntracks",
        "sv_chi2",
        "sv_normchi2",
        "sv_dxy",
        "sv_dxysig",
        "sv_d3d",
        "sv_d3dsig",
        "sv_costhetasvpv",
        "sv_enratio",
    ]