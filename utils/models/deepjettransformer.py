import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from functools import partial
import numpy as np
import os
from pathlib import Path
from typing import List
from scipy.special import softmax
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from utils.models.base_model import Classifier_base, CustomTimeElapsedColumn


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.SiLU()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * self.act(x2)

class FFNBlock(nn.Module):
    def __init__(self, dim, mult, swiglu = True, dropout=0.1):
        super(FFNBlock, self).__init__()
        self.hid_dim = dim * 4 * 2 if swiglu else dim * 4
        self.dense_1 = nn.Linear(dim, self.hid_dim)
        
        self.activation = SwiGLU() if swiglu else nn.SiLU()
        self.dense_2 = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_norm = RMSNorm(dim*4)

    def forward(self, x):
        x_proj = self.dense_1(x)

        x = self.attn_norm(self.activation(x_proj))

        x = self.dense_2(x)
        x = self.dropout(x)

        return x

class InputConv(nn.Module):

    def __init__(self, in_chn, out_chn, dropout_rate = 0.1, **kwargs):
        super(InputConv, self).__init__(**kwargs)
        
        self.lin = torch.nn.Conv1d(in_chn, out_chn, kernel_size=1)
        self.bn1 = torch.nn.BatchNorm1d(out_chn, eps = 0.001, momentum = 0.1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sc, skip: bool = True):
        
        x2 = self.dropout(self.bn1(self.act(self.lin(x))))
        if skip:
            x = sc + x2
        else:
            x = x2
        return x
    
class LinLayer(nn.Module):

    def __init__(self, in_chn, out_chn, dropout_rate = 0.1, **kwargs):
        super(LinLayer, self).__init__(**kwargs)
        
        self.lin = torch.nn.Linear(in_chn, out_chn)
        self.bn1 = torch.nn.BatchNorm1d(out_chn, eps = 0.001, momentum = 0.1)
        self.bn2 = torch.nn.BatchNorm1d(out_chn, eps = 0.001, momentum = 0.1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sc, skip = True):
        
        x2 = self.dropout(self.bn1(self.act(self.lin(x))))
        if skip:
            x = self.bn2(sc + x2)
        else:
            x = self.bn2(x2)
        return x

class LinLayer2(nn.Module):

    def __init__(self, in_chn, out_chn, dropout_rate = 0.1, **kwargs):
        super(LinLayer2, self).__init__(**kwargs)

        self.lin = torch.nn.Linear(in_chn, out_chn)
        self.ln = torch.nn.LayerNorm(out_chn, eps = 0.001)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        x = self.dropout(self.ln(self.act(self.lin(x))))
        return x

class InputProcess(nn.Module):

    def __init__(self, cpf_dim, npf_dim, vtx_dim, embed_dim, **kwargs):
        super().__init__(**kwargs)
        
        self.cpf_bn0 = torch.nn.BatchNorm1d(cpf_dim, eps = 0.001, momentum = 0.1)
        self.cpf_conv1 = InputConv(cpf_dim,embed_dim)
        self.cpf_conv3 = InputConv(embed_dim*1,embed_dim)

        self.npf_bn0 = torch.nn.BatchNorm1d(npf_dim, eps = 0.001, momentum = 0.1)
        self.npf_conv1 = InputConv(npf_dim,embed_dim)
        self.npf_conv3 = InputConv(embed_dim*1,embed_dim)

        self.vtx_bn0 = torch.nn.BatchNorm1d(vtx_dim, eps = 0.001, momentum = 0.1)
        self.vtx_conv1 = InputConv(vtx_dim,embed_dim)
        self.vtx_conv3 = InputConv(embed_dim*1,embed_dim)


    def forward(self, cpf, npf, vtx):
                
        cpf = self.cpf_bn0(torch.transpose(cpf, 1, 2))
        cpf = self.cpf_conv1(cpf, cpf, skip = False)
        cpf = self.cpf_conv3(cpf, cpf, skip = False)

        npf = self.npf_bn0(torch.transpose(npf, 1, 2))
        npf = self.npf_conv1(npf, npf, skip = False)
        npf = self.npf_conv3(npf, npf, skip = False)

        vtx = self.vtx_bn0(torch.transpose(vtx, 1, 2))
        vtx = self.vtx_conv1(vtx, vtx, skip = False)
        vtx = self.vtx_conv3(vtx, vtx, skip = False)

        out = torch.cat((cpf,npf,vtx), dim = 2)
        out = torch.transpose(out, 1, 2)
        
        return out

class InputProcess_HLT_old(nn.Module):

    def __init__(self, cpf_dim, vtx_dim, embed_dim, **kwargs):
        super().__init__(**kwargs)

        self.cpf_bn0 = torch.nn.BatchNorm1d(cpf_dim, eps = 0.001, momentum = 0.1)
        self.cpf_conv1 = InputConv(cpf_dim,embed_dim)
        self.cpf_conv3 = InputConv(embed_dim*1,embed_dim)

        self.vtx_bn0 = torch.nn.BatchNorm1d(vtx_dim, eps = 0.001, momentum = 0.1)
        self.vtx_conv1 = InputConv(vtx_dim,embed_dim)
        self.vtx_conv3 = InputConv(embed_dim*1,embed_dim)

    def forward(self, cpf, vtx):

        cpf = self.cpf_bn0(torch.transpose(cpf, 1, 2))
        cpf = self.cpf_conv1(cpf, cpf, skip = False)
        cpf = self.cpf_conv3(cpf, cpf, skip = False)

        vtx = self.vtx_bn0(torch.transpose(vtx, 1, 2))
        vtx = self.vtx_conv1(vtx, vtx, skip = False)
        vtx = self.vtx_conv3(vtx, vtx, skip = False)

        out = torch.cat((cpf,vtx), dim = 2)
        out = torch.transpose(out, 1, 2)
        
        return out

class InputConv_HLT(nn.Module):

    def __init__(self, in_chn, out_chn, dropout_rate = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.lin = torch.nn.Linear(in_chn, out_chn)
        self.bn1 = RMSNorm(in_chn)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sc, skip = True):
        
        x2 = self.dropout(self.act(self.lin(self.bn1(x))))
        if skip:
            x = sc + x2
        else:
            x = x2
        return x

class InputProcess_HLT(nn.Module):
    def __init__(self, cpf_dim, vtx_dim, embed_dim, **kwargs):
        super().__init__(**kwargs)

        self.cpf_conv1 = InputConv_HLT(cpf_dim, embed_dim)
        self.cpf_conv3 = InputConv_HLT(embed_dim * 1, embed_dim)

        self.vtx_conv1 = InputConv_HLT(vtx_dim, embed_dim)
        self.vtx_conv3 = InputConv_HLT(embed_dim * 1, embed_dim)

    def forward(self, cpf, vtx):
    
        cpf = self.cpf_conv1(cpf, cpf, skip=False)
        cpf = self.cpf_conv3(cpf, cpf, skip=True)

        vtx = self.vtx_conv1(vtx, vtx, skip=False)
        vtx = self.vtx_conv3(vtx, vtx, skip=True)

        out = torch.cat((cpf, vtx), dim=1)

        return out
        
class DenseClassifier(nn.Module):

    def __init__(self, embed_dim, **kwargs):
        super(DenseClassifier, self).__init__(**kwargs)
             
        self.LinLayer1 = LinLayer(embed_dim, embed_dim)

    def forward(self, x):
        
        x = self.LinLayer1(x, x, skip = True)
        
        return x
    
class AttentionPooling(nn.Module):

    def __init__(self, dim = 128, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)

        self.ConvLayer = torch.nn.Conv1d(dim, 1, kernel_size=1)
        self.Softmax = nn.Softmax(dim=-1)
        self.bn = torch.nn.BatchNorm1d(dim, eps = 0.001, momentum = 0.1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, pool_mask):
        
        a = self.ConvLayer(torch.transpose(x, 1, 2))
        a = self.Softmax(a + pool_mask)
        
        y = torch.matmul(a,x)
        y = torch.squeeze(y, dim = 1)
        y = self.dropout(self.bn(self.act(y)))
        
        return y
    
class HF_TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
       >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dropout=0.1, activation="relu", swiglu=True):
        super().__init__()
        # MultiheadAttention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Implementation of Feedforward model
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FFNBlock(d_model,4,swiglu = swiglu, dropout=dropout)

        self.activation = nn.SiLU()  # _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = nn.SiLU()
        super(HF_TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, mask):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        residual = src 
        src = self.attn_norm(src) #Starting the Norm->Att->LS->Dr series
        src = self.self_attn(
            src,
            src,
            src,
            key_padding_mask=mask,
        )[0]

        src = residual + src
        residual = src

        src = self.ffn_norm(src) #Starting the Norm->FFN->LS->Dr series
        src = self.ffn(src)
        src = residual + src
        
        return src


class HF_TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers):
        super(HF_TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, mask):
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, mask)

        return output
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.ReLU()

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class DeepJetTransformer(Classifier_base):

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

    cpf_candidates = [
        "Cpfcan_ptrel",
        "Cpfcan_drminsv",
        "Cpfcan_VTX_ass",
        "Cpfcan_quality",
        "Cpfcan_pt",
        "Cpfcan_eta",
        "Cpfcan_puppiw",
        "Cpfcan_chi2",
        "Cpfcan_phi",
        "Cpfcan_e"
    ]
    
    vtx_features = [
        "sv_deltaR",
        "sv_normchi2",
        "sv_dxy",
        "sv_eta",
        "sv_phi",
        "sv_mass",
        "sv_ntracks",
        "sv_chi2",
        "sv_e"
    ]

    def __init__(
        self,
        config,
        num_classes=6,
        num_enc=3,
        num_head=8,
        embed_dim=128,
        for_inference=False,
        build_4v=True,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        
        self.for_inference = for_inference
        self.num_enc_layers = num_enc
        self.num_head = num_head
        self.dtype = torch.float16

        _, cpf_dim, npf_dim, vtx_dim, _ = self.all_model_features_length()  
        
        self.InputProcess = InputProcess(cpf_dim, npf_dim, vtx_dim, embed_dim)
        self.Linear = nn.Linear(embed_dim, num_classes)
        self.DenseClassifier = DenseClassifier(embed_dim)
        self.Pooling = AttentionPooling()

        self.EncoderLayer = HF_TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_head, dropout=0.1
        )
        self.Encoder = HF_TransformerEncoder(self.EncoderLayer, num_layers=num_enc)
    
    def forward(self, inpt):

        global_features, cpf_features, npf_features, vtx_features = inpt[0], inpt[1], inpt[2], inpt[3]
        cpf, npf, vtx = cpf_features, npf_features, vtx_features

        padding_mask = torch.cat((cpf.abs().sum(dim=-1), npf.abs().sum(dim=-1), vtx.abs().sum(dim=-1)), dim = 1)
        padding_mask = torch.eq(padding_mask, 0.0)
        pool_mask = padding_mask.unsqueeze(dim=1) * torch.finfo(self.dtype).min
        
        enc = self.InputProcess(cpf, npf, vtx)
        enc = self.Encoder(enc, padding_mask)
        enc = self.Pooling(enc, pool_mask)
        
        x = self.DenseClassifier(enc)
        output = self.Linear(x)
        
        if self.for_inference:
            output = torch.softmax(output, dim=1)

        return output


class DeepJetTransformerHLT(Classifier_base):

    # Define classes used by model and connect it to the config definitions
    classes = {
        "b": ["label_b", "label_gbb", "label_bb", "label_leptonicb", "label_leptonicbc"],
        "c": ["label_c", "label_gcc", "label_cc"],
        "uds": ["label_ud", "label_s"],
        "g": ["label_g"],
        "taup": ["label_taup"],
        "taum": ["label_taum"],
    }
    
    n_cpf_candidates = 20
    n_lt_candidates = 3
    n_npf_candidates = 15
    n_vtx_candidates = 2
    
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
        "jet_sv_d3dsig",
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

    def __init__(
        self,
        config,
        num_classes=6,
        num_enc=3,
        num_head=8,
        embed_dim=128,
        for_inference=False,
        build_4v=True,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        
        self.for_inference = for_inference
        self.num_enc_layers = num_enc
        self.num_head = num_head
        self.dtype = torch.float16

        _, cpf_dim, _, vtx_dim, _ = self.all_model_features_length()  
        
        self.InputProcess = InputProcess_HLT(cpf_dim, vtx_dim, embed_dim)
        self.Linear = nn.Linear(embed_dim, num_classes)
        self.DenseClassifier = DenseClassifier(embed_dim)
        self.Pooling = AttentionPooling()

        self.EncoderLayer = HF_TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_head, dropout=0.1
        )
        self.Encoder = HF_TransformerEncoder(self.EncoderLayer, num_layers=num_enc)
    
    def forward(self, inpt):

        _, cpf_features, _, vtx_features = inpt[0], inpt[1], inpt[2], inpt[3]
        cpf, vtx = cpf_features, vtx_features

        padding_mask = torch.cat((cpf.abs().sum(dim=-1), vtx.abs().sum(dim=-1)), dim = 1)
        padding_mask = torch.eq(padding_mask, 0.0)
        pool_mask = padding_mask.unsqueeze(dim=1) * torch.finfo(self.dtype).min #float('inf')#

        enc = self.InputProcess(cpf, vtx)
        enc = self.Encoder(enc, padding_mask)
        enc = self.Pooling(enc, pool_mask)
        
        x = self.DenseClassifier(enc)
        output = self.Linear(x)
        
        if self.for_inference:
            output = torch.softmax(output, dim=1)

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
        tau_pred = taup_pred + taum_pred
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
        tauvsl = np.where((tau_pred + l_pred) > 0, (tau_pred) / (tau_pred + l_pred), -1)

        tau_veto = (truth != taup_index) & (truth != taum_index) & (summed_jets != 0)
        b_veto = (truth != b_index) & tau_veto & (summed_jets != 0)
        c_veto = (truth != c_index) & tau_veto & (summed_jets != 0)
        bc_veto_tau = (truth != c_index) & (truth != b_index) & (summed_jets != 0)
        bc_veto = b_veto & c_veto
        l_veto = (truth != uds_index) & (truth != g_index) & tau_veto & (summed_jets != 0)
        no_veto = np.ones(b_veto.shape, dtype=bool)

        labels = ["bvsl", "bvsc", "cvsb", "cvsl", "bvsall", "uds_vs_g", "tauvsl"]
        discs = [bvsl, bvsc, cvsb, cvsl, bvsall, uds_vs_g, tauvsl]
        vetos = [c_veto, l_veto, l_veto, b_veto, no_veto, bc_veto, bc_veto_tau]
        truths = [b_jets, b_jets, c_jets, c_jets, b_jets, uds_jets, tau_jets]
        xlabels = [
            "b-identification",
            "b-identification",
            "c-identification",
            "c-identification",
            "b-identification",
            "uds-identification",
            "tau-identification",
        ]
        ylabels = [
            "light mis-id.", 
            "c mis-id", 
            "b mis-id.", 
            "light mis-id.", 
            "mis-id.", 
            "gluons mis-id.", 
            "light mis-id."
        ]

        return discs, truths, vetos, labels, xlabels, ylabels



class GlobalDeepJetTransformerHLT(DeepJetTransformerHLT):
        
    def __init__(
        self,
        config,
        num_classes=6,
        num_enc=3,
        num_head=8,
        embed_dim=128,
        for_inference=False,
        build_4v=True,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        
        self.for_inference = for_inference
        self.num_enc_layers = num_enc
        self.num_head = num_head
        self.dtype = torch.float16

        global_dim, cpf_dim, _, vtx_dim, _ = self.all_model_features_length()
        
        self.global_bn = torch.nn.BatchNorm1d(global_dim, eps=0.001, momentum=0.6)
        self.InputProcess = InputProcess_HLT(cpf_dim, vtx_dim, embed_dim)
        self.Linear = nn.Linear(embed_dim + global_dim, num_classes)
        self.DenseClassifier = DenseClassifier(embed_dim + global_dim)
        self.Pooling = AttentionPooling()

        self.EncoderLayer = HF_TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_head, dropout=0.1
        )
        self.Encoder = HF_TransformerEncoder(self.EncoderLayer, num_layers=num_enc)
    
    def forward(self, inpt):

        global_features, cpf_features, _, vtx_features = inpt[0], inpt[1], inpt[2], inpt[3]
        cpf, vtx = cpf_features, vtx_features

        global_features = self.global_bn(global_features)

        padding_mask = torch.cat((cpf.abs().sum(dim=-1), vtx.abs().sum(dim=-1)), dim = 1)
        padding_mask = torch.eq(padding_mask, 0.0)
        mask = (padding_mask.unsqueeze(1) + padding_mask.unsqueeze(2)).to(cpf.dtype).repeat((self.num_head,1,1)) * torch.finfo(self.dtype).min
        pool_mask = padding_mask.unsqueeze(dim=1) * torch.finfo(self.dtype).min
        
        enc = self.InputProcess(cpf, vtx)
        enc = self.Encoder(enc, mask)
        enc = self.Pooling(enc, pool_mask)

        fts = torch.cat((global_features, enc), dim=1)
        
        x = self.DenseClassifier(fts)
        output = self.Linear(x)
        
        if self.for_inference:
            output = torch.softmax(output, dim=1)

        return output

class UDeepJetTransformerHLT(DeepJetTransformerHLT):
        
    def __init__(
        self,
        config,
        num_classes=6,
        num_enc=3,
        num_head=8,
        embed_dim=128,
        for_inference=False,
        build_4v=True,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        
        self.for_inference = for_inference
        self.num_enc_layers = num_enc
        self.num_head = num_head

        self.jet_pt_index = config.get('global_features',[]).index('jet_pt')
        self.jet_eta_index = config.get('global_features',[]).index('jet_eta')
        self.gen_lep_pt_index = config.get('global_features',[]).index('jet_genmatch_lep_vis_pt')
        self.gen_pt = config.get('global_features',[]).index('jet_genmatch_pt')

        self.taup_index = list(self.classes).index('taup')
        self.taum_index = list(self.classes).index('taum')
        self.register_buffer(
            "tau_indices",
            torch.tensor(
                [i for i, k in enumerate(self.classes) if "tau" in k],
                dtype=torch.long
            )
        )
        _, cpf_dim, _, vtx_dim, _ = self.all_model_features_length()
        
        self.InputProcess = InputProcess_HLT(cpf_dim, vtx_dim, embed_dim)
        self.Linear = nn.Linear(embed_dim, num_classes + 1)
        self.DenseClassifier = DenseClassifier(embed_dim)
        self.Pooling = AttentionPooling(dim = embed_dim)

        self.EncoderLayer = HF_TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_head, dropout=0.1
        )
        self.Encoder = HF_TransformerEncoder(self.EncoderLayer, num_layers=num_enc)

    def forward(self, inpt):

        _, cpf_features, _, vtx_features = inpt[0], inpt[1], inpt[2], inpt[3]
        cpf, vtx = cpf_features, vtx_features

        padding_mask = torch.cat((cpf.abs().sum(dim=-1), vtx.abs().sum(dim=-1)), dim = 1)
        padding_mask = torch.eq(padding_mask, 0.0)
        mask = (padding_mask.unsqueeze(1) + padding_mask.unsqueeze(2)).to(cpf.dtype).repeat((self.num_head,1,1)) * torch.finfo(self.dtype).min
        pool_mask = padding_mask.unsqueeze(dim=1) * torch.finfo(self.dtype).min
        
        enc = self.InputProcess(cpf, vtx)
        enc = self.Encoder(enc, mask)
        enc = self.Pooling(enc, pool_mask)
        
        x = self.DenseClassifier(enc)
        output = self.Linear(x)
        
        if self.for_inference:
            output1 = torch.softmax(output[:,:-1], dim=1)
        else:
            output1 = output[:,:-1]
        
        output2 = output[:,-1:]

        return output1, output2

    def train_model(
        self,
        training_data,
        validation_data,
        directory,
        loss_fn,
        attack=None,
        optimizer=None,
        scheduler=None,
        batch_lr=False,
        device=None,
        nepochs=0,
        best_loss_val = np.inf,
        resume_epochs=0,
        train_metrics=None,
        validation_metrics=None,
        class_weights=None,
        terminal_plot=False,
        torch_compile_mode=None,
        **kwargs,
    ):
        
        if self.use_torch_compile:
            self.compile_step = torch.compile(self.step, mode=torch_compile_mode)
            
        scaler = torch.amp.GradScaler(device)

        if os.path.isfile(f'{directory}/training_metrics.npz'): 
            train_metrics_file = np.load(f'{directory}/training_metrics.npz')
            validation_metrics_file = np.load(f'{directory}/validation_metrics.npz')
            train_metrics["loss_cat"] = list(train_metrics_file["loss_cat"])
            train_metrics["loss_reg"] = list(train_metrics_file["loss_reg"])
            validation_metrics["loss_cat"] = list(validation_metrics_file["loss_cat"])
            validation_metrics["loss_reg"] = list(validation_metrics_file["loss_reg"])
        else:  
            train_metrics["loss_cat"], train_metrics["loss_reg"] = [], []
            validation_metrics["loss_cat"], validation_metrics["loss_reg"] = [], []
        
        if os.path.isfile(f'{directory}/train_time.npy') and os.path.isfile(f'{directory}/val_time.npy'):
            train_time = np.load(f'{directory}/train_time.npy')
            val_time   = np.load(f'{directory}/val_time.npy')
        else:
            train_time, val_time = np.zeros(nepochs-resume_epochs), np.zeros(nepochs-resume_epochs)
            
        for t in range(resume_epochs, nepochs):
            print("Epoch", t + 1, "of", nepochs)
            training_data.dataset.shuffleFileList()  # Shuffle the file list as mini-batch training requires it for regularisation of a non-convex problem
            
            loss_training, loss_cat, loss_reg, acc_training, train_time[t] = self.update(
                training_data,
                loss_fn,
                optimizer,
                scheduler=scheduler,
                batch_lr=batch_lr,
                attack=attack,
                scaler=scaler,
                device=device,
            )
            train_metrics["loss"].append(loss_training)
            train_metrics["loss_cat"].append(loss_cat)
            train_metrics["loss_reg"].append(loss_reg)
            train_metrics["acc"].append(acc_training)

            loss_validation, loss_cat, loss_reg, acc_validation, val_time[t] = self.validate_model(
                validation_data, 
                loss_fn, 
                device,
                terminal_plot=terminal_plot
            )  
            
            validation_metrics["loss"].append(loss_validation)
            validation_metrics["loss_cat"].append(loss_validation)
            validation_metrics["loss_reg"].append(loss_validation)
            validation_metrics["acc"].append(acc_validation)

            # Save the model state and other details
            checkpoint = {
                "epoch": t,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss_train": loss_training,
                "acc_train": acc_training,
                "loss_val": loss_validation,
                "acc_val": acc_validation,
            }
            
            # Save the current model
            torch.save(checkpoint, f"{directory}/model_{t}.pt")
            
            # Save the best model if current validation loss is lower
            if loss_validation < best_loss_val:
                best_loss_val = loss_validation
                torch.save(checkpoint, f"{directory}/best_model.pt")

            # Save time taken for training and validation
            np.save(f'{directory}/train_time.npy', train_time)
            np.save(f'{directory}/val_time.npy', val_time)

            np.savez(
                f'{directory}/training_metrics',
                loss=train_metrics["loss"],
                loss_cat=train_metrics["loss_cat"],
                loss_reg=train_metrics["loss_reg"],
                acc=train_metrics["acc"],
                allow_pickle=True,
            )
            np.savez(
                f'{directory}/validation_metrics',
                loss=validation_metrics["loss"],
                loss_cat=validation_metrics["loss_cat"],
                loss_reg=validation_metrics["loss_reg"],
                acc=validation_metrics["acc"],
                allow_pickle=True,
            )
        
        return train_metrics, validation_metrics
        
    def step(self, inpt, truth, loss_fn, target_pt, attack=None, device="cpu", mixed_precision=True):
        with torch.autocast(device, enabled=mixed_precision):
            pred_cat, pred_reg = self.forward(inpt)
        loss, loss_cat, loss_reg = loss_fn(pred_cat, truth, pred_reg, target_pt, device)  
        return pred_cat, loss, loss_cat, loss_reg

    def update(
        self,
        dataloader,
        loss_fn,
        optimizer,
        scheduler=None,
        batch_lr=False,
        attack=None,
        scaler=None,
        device="cpu",
        verbose=True,
    ):
        losses, losses_cat, losses_reg = [], [], []
        accuracy = 0.0
        self.train()

        elapsed_column = CustomTimeElapsedColumn()

        with Progress(
            TextColumn("{task.description}"),
            elapsed_column,
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn(f"0/~{dataloader.nits_expected} its"),
            expand=True,
        ) as progress:
            N = 0
            task = progress.add_task("Training...", total=dataloader.nits_expected)
            print("entering traing loop")
            for (x, truth, w, p) in dataloader:
                x = x.float().to(device, non_blocking=True)
                truth = truth.type(torch.LongTensor).to(device, non_blocking=True)
                w = w.float().to(device, non_blocking=True)

                inpt, truth = self.get_inpt(x, truth=truth, loss_fn=loss_fn, attack=attack, device=device)
                jet_pt = inpt[0][:, self.jet_pt_index]
                tau_mask = torch.isin(truth, self.tau_indices)
                gen_pt = torch.where(
                    tau_mask,
                    inpt[0][:, self.gen_lep_pt_index],  # values where tau_mask is True
                    inpt[0][:, self.gen_pt]             # values where tau_mask is False
                )
        
                target_pt = torch.clip(torch.nan_to_num(gen_pt / jet_pt, nan=0, posinf=0, neginf=0), min=0.3, max=2.0).reshape(-1, 1)
                
                if self.use_torch_compile:
                    pred, loss, loss_cat, loss_reg = self.compile_step(
                        inpt, truth, loss_fn, 
                        target_pt=target_pt,
                        device=device, mixed_precision=self.mixed_precision
                    )
                else:
                    pred, loss, loss_cat, loss_reg = self.step(
                        inpt, truth, loss_fn, 
                        target_pt=target_pt,
                        device=device, mixed_precision=self.mixed_precision
                    )
                if torch.isnan(loss).any():    
                    raise ValueError("Loss contains NaN values! Something's wrong with calculation, please check.")
        
                optimizer.zero_grad(set_to_none=True)
                
                if scaler is not None:
                    # Mixed-precision training
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard precision training
                    loss.backward()
                    optimizer.step()

                # Step the learning rate scheduler if applicable
                if batch_lr and (scheduler is not None):
                    scheduler.step()
      
                losses.append(loss.item())
                losses_cat.append(loss_cat.item())
                losses_reg.append(loss_reg.item())
                accuracy += (
                    (pred.argmax(1) == truth.to(device)).type(torch.float).sum().item()
                )

                N += len(pred)
                curr_lr = optimizer.param_groups[0]['lr']
                progress.update(
                    task, advance=1, description=f"Training...   | Loss: {loss:.4f}, lr: {curr_lr:.5f}"
                )
                progress.columns[-1].text_format = "{}/{} its".format(
                    N // dataloader.dataset.batch_size, f"~{dataloader.nits_expected}"
                )
            progress.update(task, completed=dataloader.nits_expected)

            if (not batch_lr) and (scheduler is not None):
                scheduler.step()
        #dataloader.nits_expected = N // dataloader.dataset.batch_size
        accuracy /= N
        print("  ", f"Average loss: {np.array(losses).mean():.4f}")
        print("  ", f"Average accuracy: {float(100*accuracy):.4f}")

        return np.array(losses).mean(), np.array(losses_cat).mean(), np.array(losses_reg).mean(), float(accuracy), elapsed_column.elapsed_time

    def validate_model(self, dataloader, loss_fn, device="cpu", verbose=True, terminal_plot=False):
        losses, losses_cat, losses_reg = 0.0, 0.0, 0.0
        accuracy = 0.0
        self.eval()

        predictions = np.empty((0, len(self.classes)))
        truths = np.empty((0))
        processes = np.empty((0))

        elapsed_column = CustomTimeElapsedColumn()

        with Progress(
            TextColumn("{task.description}"),
            elapsed_column,
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn(f"0/~{dataloader.nits_expected} its"),
            expand=True,
        ) as progress:
            N = 1
            task = progress.add_task("Validation...", total=dataloader.nits_expected)
            for b, (x, truth, w, process) in enumerate(dataloader):
                
                x = x.float().to(device, non_blocking=True)
                truth = truth.type(torch.LongTensor).to(device, non_blocking=True)
                w = w.float().to(device, non_blocking=True)

                with torch.no_grad():
                    inpt, truth = self.get_inpt(x, truth=truth, loss_fn=loss_fn, attack=None, device=device)

                    jet_pt = inpt[0][:, self.jet_pt_index]
                    tau_mask = torch.isin(truth, self.tau_indices)
                    gen_pt = torch.where(
                        tau_mask,
                        inpt[0][:, self.gen_lep_pt_index],  # values where tau_mask is True
                        inpt[0][:, self.gen_pt]             # values where tau_mask is False
                    )
            
                    target_pt = torch.clip(torch.nan_to_num(gen_pt / jet_pt, nan=0, posinf=0, neginf=0), min=0.3, max=2.0).reshape(-1, 1)
                    
                    if self.use_torch_compile:
                        pred, loss, loss_cat, loss_reg = self.compile_step(
                            inpt, truth, loss_fn, 
                            target_pt=target_pt,
                            device=device, mixed_precision=self.mixed_precision
                        )
                    else:
                        pred, loss, loss_cat, loss_reg = self.step(
                            inpt, truth, loss_fn, 
                            target_pt=target_pt,
                            device=device, mixed_precision=self.mixed_precision
                        )
                        
                    losses += loss.item()
                    losses_cat += loss_cat.item()
                    losses_reg += loss_reg.item()
                    accuracy += (
                        (pred.argmax(1) == truth.to(device))
                        .type(torch.float)
                        .sum()
                        .item()
                    )
                    if(terminal_plot):
                        predictions = np.append(predictions, pred.to("cpu").numpy(), axis=0)
                        truths = np.append(truths, truth.to("cpu").numpy(), axis=0)
                        processes = np.append(processes, process.to("cpu").numpy(), axis=0)
                    
                N += len(pred)
                progress.update(
                    task, advance=1, description=f"Validation... | Loss: {loss:.2f}"
                )
                progress.columns[-1].text_format = "{}/{} its".format(
                    N // dataloader.dataset.batch_size, f"~{dataloader.nits_expected}"
                )
            progress.update(task, completed=dataloader.nits_expected)
        dataloader.nits_expected = N // dataloader.dataset.batch_size
        accuracy /= N
        print("  ", f"Validation loss: {losses/(b+1):.4f}")
        print("  ", f"Validation loss (cat): {losses_cat/(b+1):.4f}")
        print("  ", f"Validation loss (reg): {losses_reg/(b+1):.4f}")
        print("  ", f"Validation accuracy: {float(100*accuracy):.4f}")

        if verbose and terminal_plot:
            print("Printing terminal ROC")
            terminal_roc(predictions, truths, title="Validation ROC")

        return losses/(b+1), losses_cat/(b+1), losses_reg/(b+1), float(accuracy), elapsed_column.elapsed_time
           
    def predict_model(
        self, 
        dataloader, 
        output,
        loss_fn,
        device, 
        attack=None
    ):
        losses, losses_cat, losses_reg = 0.0, 0.0, 0.0
        accuracy = 0.0
        self.eval()
        
        kinematics = []
        truths = []
        processes = []
        predictions = []
        predictions_reg = []

        elapsed_column = CustomTimeElapsedColumn()
        
        with Progress(
            TextColumn("{task.description}"),
            elapsed_column,
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn(f"0/{dataloader.nits_expected} its"),
            expand=True,
        ) as progress:
            N = 1
            task = progress.add_task("Inference...", total=dataloader.nits_expected)
            
            for b, (x, truth, w, process) in enumerate(dataloader):
                
                x = x.float().to(device, non_blocking=True)
                truth = truth.type(torch.LongTensor).to(device, non_blocking=True)
                w = w.float().to(device, non_blocking=True)

                torch.backends.cudnn.enabled = False
                inpt, _ = self.get_inpt(x, truth=truth, loss_fn=loss_fn, attack=attack, device=device)
                torch.backends.cudnn.enabled = True

                jet_pt = inpt[0][:, self.jet_pt_index]
                tau_mask = torch.isin(truth, self.tau_indices)
                gen_pt = torch.where(
                    tau_mask,
                    inpt[0][:, self.gen_lep_pt_index],  # values where tau_mask is True
                    inpt[0][:, self.gen_pt]             # values where tau_mask is False
                )
        
                target_pt = torch.clip(torch.nan_to_num(gen_pt / jet_pt, nan=0, posinf=0, neginf=0), min=0.3, max=2.0).reshape(-1, 1)

                with torch.no_grad():
                    pred_cat, pred_reg = self.forward(inpt)
                    loss, loss_cat, loss_reg = loss_fn(pred_cat, truth, pred_reg, target_pt, device)
        
                kinematics.append(inpt[0][..., [self.jet_pt_index, self.jet_eta_index, self.gen_lep_pt_index, self.gen_pt]].cpu().numpy())
                truths.append(truth.cpu().numpy().astype(int))
                processes.append(process.cpu().numpy())
                predictions.append(pred_cat.cpu().numpy())
                predictions_reg.append(pred_reg.cpu().numpy())

                losses += loss.item()
                losses_cat += loss_cat.item()
                losses_reg += loss_reg.item()
                accuracy += (
                    (pred_cat.argmax(1) == truth.to(device))
                    .type(torch.float)
                    .sum()
                    .item()
                )
                N += len(pred_cat)
                progress.update(
                    task, advance=1, description=f"Inference...   | Loss: {loss:.2f}"
                )
                progress.columns[-1].text_format = "{}/{} its".format(
                    N // dataloader.dataset.batch_size, dataloader.nits_expected
                )
            progress.update(task, completed=dataloader.nits_expected)
            
        accuracy /= N
        print("  ", f"Average loss: {losses/(b+1):.4f}")
        print("  ", f"Average loss (cat): {losses_cat/(b+1):.4f}")
        print("  ", f"Average loss (reg): {losses_reg/(b+1):.4f}")
        print("  ", f"Average accuracy: {float(100*accuracy):.4f}")
        
        predictions = np.concatenate(predictions)
        predictions_reg = np.concatenate(predictions_reg)
        kinematics = np.concatenate(kinematics)
        truths = np.concatenate(truths)
        processes = np.concatenate(processes)

        np.save(output["prediction"].path, predictions)
        np.save(Path(output["prediction"].path).parent / 'prediction_reg.npy', predictions_reg)
        np.save(output["truth"].path, truths)
        np.save(output["kinematics"].path, kinematics)
        np.save(output["process"].path, processes)
        np.save(output["inference_time"].path, elapsed_column.elapsed_time)
        
        return predictions, truths, kinematics, processes, elapsed_column.elapsed_time