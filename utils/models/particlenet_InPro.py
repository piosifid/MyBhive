import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
import math
import torch
from pathlib import Path
import torch.nn as nn
import os
from typing import List
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from utils.models.base_model import Classifier_base, CustomTimeElapsedColumn

torch.set_float32_matmul_precision('high')

"""Taken from https://github.com/hqucms/weaver/blob/master/utils/nn/model/ParticleNet.py"""
"""Based on https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py."""

class InputConv(nn.Module):

    def __init__(self, in_chn, out_chn, dropout_rate = 0.1, **kwargs):
        super(InputConv, self).__init__(**kwargs)
        
        self.lin = torch.nn.Conv1d(in_chn, out_chn, kernel_size=1)
        self.bn1 = torch.nn.BatchNorm1d(out_chn, eps = 0.001, momentum = 0.1)
        #self.bn2 = torch.nn.BatchNorm1d(out_chn, eps = 0.001, momentum = 0.1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sc, skip: bool = True):
        
        x2 = self.dropout(self.bn1(self.act(self.lin(x))))
        if skip:
            x = sc + x2
        else:
            x = x2
        return x

class InputProcess(nn.Module):

    def __init__(self, cpf_dim, npf_dim, vtx_dim, embed_dim, **kwargs):
        super(InputProcess, self).__init__(**kwargs)
        
        self.cpf_bn0 = torch.nn.BatchNorm1d(cpf_dim, eps = 0.001, momentum = 0.1)
        self.cpf_conv1 = InputConv(cpf_dim,embed_dim)
        self.cpf_conv2 = InputConv(embed_dim,embed_dim*4)
        self.cpf_conv3 = InputConv(embed_dim*4,embed_dim)

        self.npf_bn0 = torch.nn.BatchNorm1d(npf_dim, eps = 0.001, momentum = 0.1)
        self.npf_conv1 = InputConv(npf_dim,embed_dim)
        self.npf_conv2 = InputConv(embed_dim,embed_dim*4)
        self.npf_conv3 = InputConv(embed_dim*4,embed_dim)

        self.vtx_bn0 = torch.nn.BatchNorm1d(vtx_dim, eps = 0.001, momentum = 0.1)
        self.vtx_conv1 = InputConv(vtx_dim,embed_dim)
        self.vtx_conv2 = InputConv(embed_dim,embed_dim*4)
        self.vtx_conv3 = InputConv(embed_dim*4,embed_dim)

#        self.meta_conv = InputConv(8*16,8*16)

    def forward(self, cpf, npf, vtx):
                
        cpf = self.cpf_bn0(torch.transpose(cpf, 1, 2))
        cpf = self.cpf_conv1(cpf, cpf, skip = False)
        cpf = self.cpf_conv2(cpf, cpf, skip = False)
        cpf = self.cpf_conv3(cpf, cpf, skip = False)

        npf = self.npf_bn0(torch.transpose(npf, 1, 2))
        npf = self.npf_conv1(npf, npf, skip = False)
        npf = self.npf_conv2(npf, npf, skip = False)
        npf = self.npf_conv3(npf, npf, skip = False)

        vtx = self.vtx_bn0(torch.transpose(vtx, 1, 2))
        vtx = self.vtx_conv1(vtx, vtx, skip = False)
        vtx = self.vtx_conv2(vtx, vtx, skip = False)
        vtx = self.vtx_conv3(vtx, vtx, skip = False)

        out = torch.cat((cpf,npf,vtx), dim = 2)
        out = torch.transpose(out, 1, 2)
        
        return out

class InputProcess_HLT(nn.Module):

    def __init__(self, cpf_dim, vtx_dim, embed_dim, **kwargs):
        super(InputProcess_HLT, self).__init__(**kwargs)
        
        self.cpf_bn0 = torch.nn.BatchNorm1d(cpf_dim, eps = 0.001, momentum = 0.1)
        self.cpf_conv1 = InputConv(cpf_dim,embed_dim)
        self.cpf_conv2 = InputConv(embed_dim,embed_dim*4)
        self.cpf_conv3 = InputConv(embed_dim*4,embed_dim)

        self.vtx_bn0 = torch.nn.BatchNorm1d(vtx_dim, eps = 0.001, momentum = 0.1)
        self.vtx_conv1 = InputConv(vtx_dim,embed_dim)
        self.vtx_conv2 = InputConv(embed_dim,embed_dim*4)
        self.vtx_conv3 = InputConv(embed_dim*4,embed_dim)

    def forward(self, cpf, vtx):
                
        cpf = self.cpf_bn0(torch.transpose(cpf, 1, 2))
        cpf = self.cpf_conv1(cpf, cpf, skip = False)
        cpf = self.cpf_conv2(cpf, cpf, skip = False)
        cpf = self.cpf_conv3(cpf, cpf, skip = False)

        vtx = self.vtx_bn0(torch.transpose(vtx, 1, 2))
        vtx = self.vtx_conv1(vtx, vtx, skip = False)
        vtx = self.vtx_conv2(vtx, vtx, skip = False)
        vtx = self.vtx_conv3(vtx, vtx, skip = False)

        out = torch.cat((cpf,vtx), dim = 2)
        out = torch.transpose(out, 1, 2)
        
        return out
        
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    # print(f"x.shape={x.shape}")
    # print(f"inner.shape={inner.shape}")
    # print(f"xx.shape={xx.shape}")
    # print(f"pairwise_distance.shape={pairwise_distance.shape}")
    # print(f"k={k}")
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][
        :, :, 1:
    ]  # (batch_size, num_points, k)
    return idx


# v1 is faster on GPU
def get_graph_feature_v1(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(2, 1).reshape(
        -1, num_dims
    )  # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    fts = fts[idx, :].view(
        batch_size, num_points, k, num_dims
    )  # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
    fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_points, k)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
    return fts


# v2 is faster on CPU
def get_graph_feature_v2(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(
        num_dims, -1
    )  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(
        num_dims, batch_size, num_points, k
    )  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)

    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)

    return fts


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(
        self, k, in_feat, out_feats, batch_norm=True, activation=True, cpu_mode=False
    ):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.get_graph_feature = (
            get_graph_feature_v2 if cpu_mode else get_graph_feature_v1
        )

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(
                nn.Conv2d(
                    2 * in_feat if i == 0 else out_feats[i - 1],
                    out_feats[i],
                    kernel_size=1,
                    bias=False if self.batch_norm else True,
                )
            )

        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm2d(out_feats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def forward(self, points, features):
        topk_indices = knn(points, self.k)
        x = self.get_graph_feature(features, self.k, topk_indices)

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            if bn:
                x = bn(x)
            if act:
                x = act(x)

        fts = x.mean(dim=-1)  # (N, C, P)

        # shortcut
        if self.sc:
            sc = self.sc(features)  # (N, C_out, P)
            sc = self.sc_bn(sc)
        else:
            sc = features

        return self.sc_act(sc + fts)  # (N, C_out, P)


class ParticleNet(nn.Module):
    def __init__(
        self,
        input_dims,
        num_classes,
        conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
        fc_params=[(128, 0.1)],
        use_fusion=True,
        use_fts_bn=True,
        use_counts=True,
        for_inference=False,
        for_segmentation=False,
        pT_regression=False,
        **kwargs,
    ):
        super(ParticleNet, self).__init__(**kwargs)

        self.use_fts_bn = use_fts_bn
        self.pT_regression = pT_regression
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)

        self.use_counts = use_counts

        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(
                EdgeConvBlock(
                    k=k, in_feat=in_feat, out_feats=channels, cpu_mode=for_inference
                )
            )

        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(
                nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_chn),
                nn.ReLU(),
            )

        self.for_segmentation = for_segmentation

        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            if self.for_segmentation:
                fcs.append(
                    nn.Sequential(
                        nn.Conv1d(in_chn, channels, kernel_size=1, bias=False),
                        nn.BatchNorm1d(channels),
                        nn.ReLU(),
                        nn.Dropout(drop_rate),
                    )
                )
            else:
                fcs.append(
                    nn.Sequential(
                        nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)
                    )
                )
        if self.for_segmentation:
            if self.pT_regression:
                fcs.append(nn.Conv1d(fc_params[-1][0], num_classes + 1, kernel_size=1))
            else:
                fcs.append(nn.Conv1d(fc_params[-1][0], num_classes, kernel_size=1))
        else:
            if self.pT_regression:
                fcs.append(nn.Linear(fc_params[-1][0], num_classes + 1))
            else:
                fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference

    def forward(self, points, features, mask=None):

        # print(f"points.shape={points.shape}")
        
        if mask is None:
            mask = features.abs().sum(dim=1, keepdim=True) != 0  # (N, 1, P)
        points *= mask
        features *= mask
        coord_shift = (mask == 0) * 1e9
        if self.use_counts:
            counts = mask.float().sum(dim=-1)
            counts = torch.max(counts, torch.ones_like(counts))  # >=1

        if self.use_fts_bn:
            fts = self.bn_fts(features) * mask
        else:
            fts = features
            
        outputs = []
        
        for idx, conv in enumerate(self.edge_convs):
            pts = (points if idx == 0 else fts) + coord_shift
            fts = conv(pts, fts) * mask
            if self.use_fusion:
                outputs.append(fts)
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask

        #         assert(((fts.abs().sum(dim=1, keepdim=True) != 0).float() - mask.float()).abs().sum().item() == 0)

        if self.for_segmentation:
            x = fts
        else:
            if self.use_counts:
                x = fts.sum(dim=-1) / counts  # divide by the real counts
            else:
                x = fts.mean(dim=-1)

        output = self.fc(x)
        if self.for_inference:
            if self.pT_regression:
                output1 = torch.softmax(output[:, :-1], dim=1)
                output2 = output[:, -1:]
                output = torch.cat((output1, output2), dim = 1)
            else:  
                output = torch.softmax(output, dim=1)
        return output


class FeatureConv(nn.Module):
    def __init__(self, in_chn, out_chn, **kwargs):
        super(FeatureConv, self).__init__(**kwargs)
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_chn),
            nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_chn),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class ParticleNetTagger(Classifier_base):

    classes = {
        "b": ["isB"],
        "bb": ["isBB", "isGBB"],
        "leptonicB": ["isLeptonicB", "isLeptonicB_C"],
        "c": ["isC", "isCC", "isGCC"],
        "uds": ["isU", "isD", "isS"],
        "g": ["isG"],
    }
    
    # classes = {
    #     "b": ["isB"],
    #     "bb": ["isBB", "isGBB"],
    #     "leptonicB": ["isLeptonicB", "isLeptonicB_C"],
    #     "c": ["isC", "isCC", "isGCC"],
    #     "uds": ["isUD", "isS"],
    #     "g": ["isG"],
    # }

    integer_features = {
            "global_features": ["n_Cpfcand", "nCpfcan", "n_Npfcand", "nNpfcan", "nsv", "npv",
                                "TagVarCSV_vertexCategory", "TagVarCSV_jetNSelectedTracks", "TagVarCSV_jetNTracksEtaRel"],
            "cpf_candidates": ["Cpfcan_VTX_ass", "Cpfcan_puppiw", "Cpfcan_chi2", "Cpfcan_quality"],
            "npf_candidates": ["Npfcan_isGamma", "Npfcan_HadFrac", "Npfcan_puppiw"],
            "vtx_features": ["sv_ntracks"]
    }

    def __init__(
        self,
        config,
        conv_params=[(50, (128, 128, 128)), (50, (256, 256, 256))],
        # conv_params=[(25, (128, 128, 128)), (25, (256, 256, 256))],
        fc_params=[(256, 0.1), (128, 0.1)],
        embed_dim=128,
        use_fusion=True,
        use_fts_bn=True,
        use_counts=True,
        cpf_input_dropout=None,
        npf_input_dropout=None,
        vtx_input_dropout=None,
        for_segmentation = False,
        for_inference=False,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        
        self.for_inference = for_inference

        _, cpf_dim, npf_dim, vtx_dim, _ = self.all_model_features_length()  
        
        self.InputProcess = InputProcess(cpf_dim, npf_dim, vtx_dim, embed_dim)
        
        self.pn = ParticleNet(
            input_dims=embed_dim,
            num_classes=len(self.classes),
            conv_params=conv_params,
            fc_params=fc_params,
            use_fusion=use_fusion,
            use_fts_bn=use_fts_bn,
            use_counts=use_counts,
            for_inference=for_inference,
            pT_regression=False,
        )
    
    def forward(self, inpt):

        global_fts, cpf_fts, npf_fts, vtx_fts = inpt[0], inpt[1], inpt[2], inpt[3]

        enc = self.InputProcess(cpf_fts, npf_fts, vtx_fts)
        
        pn = self.pn(enc.transpose(1, 2), enc.transpose(1, 2))
        
        return pn


class ParticleNetTaggerHLT(Classifier_base):
    
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
        #"pfcand_mask"
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
        #"sv_mask"
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
        conv_params=[(50, (128, 128, 128)), (50, (256, 256, 256))],
        fc_params=[(256, 0.1), (128, 0.1)],
        embed_dim=128,
        use_fusion=True,
        use_fts_bn=True,
        use_counts=True,
        for_segmentation = False,
        for_inference=False,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        
        self.for_inference = for_inference

        _, cpf_dim, _, vtx_dim, _ = self.all_model_features_length()  
        
        self.InputProcess = InputProcess_HLT(cpf_dim, vtx_dim, embed_dim)
        
        self.pn = ParticleNet(
            input_dims=embed_dim,
            num_classes=len(self.classes),
            conv_params=conv_params,
            fc_params=fc_params,
            use_fusion=use_fusion,
            use_fts_bn=use_fts_bn,
            use_counts=use_counts,
            for_inference=for_inference,
            pT_regression=False,
        )
    
    def forward(self, inpt):

        global_fts, cpf_fts, _, vtx_fts = inpt[0], inpt[1], inpt[2], inpt[3]

        enc = self.InputProcess(cpf_fts, vtx_fts)
        
        pn = self.pn(enc.transpose(1, 2), enc.transpose(1, 2))
        
        return pn

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


class UParticleNetTaggerHLT(ParticleNetTaggerHLT):
        
    def __init__(
        self,
        config,
        conv_params=[(50, (128, 128, 128)), (50, (256, 256, 256))],
        fc_params=[(256, 0.1), (128, 0.1)],
        embed_dim=128,
        use_fusion=True,
        use_fts_bn=True,
        use_counts=True,
        for_segmentation = False,
        for_inference=False,
        loss_lambda=1,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        
        self.for_inference = for_inference
        self.loss_lambda = loss_lambda

        self.jet_pt_index = config.get('global_features',[]).index('jet_pt')
        self.jet_eta_index = config.get('global_features',[]).index('jet_eta')
        self.gen_lep_pt_index = config.get('global_features',[]).index('jet_genmatch_lep_vis_pt')
        self.gen_pt = config.get('global_features',[]).index('jet_genmatch_pt')
        self.taup_index = list(self.classes).index('taup')
        self.taum_index = list(self.classes).index('taum')

        _, cpf_dim, _, vtx_dim, _ = self.all_model_features_length()  
        
        self.InputProcess = InputProcess_HLT(cpf_dim, vtx_dim, embed_dim)
        
        self.pn = ParticleNet(
            input_dims=embed_dim,
            num_classes=len(self.classes),
            conv_params=conv_params,
            fc_params=fc_params,
            use_fusion=use_fusion,
            use_fts_bn=use_fts_bn,
            use_counts=use_counts,
            for_inference=for_inference,
            pT_regression=True,
        )
    
    def forward(self, inpt):

        global_fts, cpf_fts, _, vtx_fts = inpt[0], inpt[1], inpt[2], inpt[3]

        enc = self.InputProcess(cpf_fts, vtx_fts)
        
        pn = self.pn(enc.transpose(1, 2), enc.transpose(1, 2))

        return pn[:, :-1], pn[:, -1:]
        

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

            if (not batch_lr) and (scheduler is not None):
                scheduler.step()

            loss_validation, loss_cat, loss_reg, acc_validation, val_time[t] = self.validate_model(validation_data, loss_fn, device,terminal_plot=terminal_plot)  
            
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
        losses, losses_cat, losses_reg = 0.0, 0.0, 0.0
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
            for b, (x, truth, w, process) in enumerate(dataloader):

                x = x.float().to(device, non_blocking=True)
                truth = truth.type(torch.LongTensor).to(device, non_blocking=True)
                w = w.float().to(device, non_blocking=True)

                inpt, truth = self.get_inpt(x, truth=truth, loss_fn=loss_fn, attack=attack, device=device)

                jet_pt = inpt[0][:, self.jet_pt_index]
                tau_mask = (truth == self.taup_index) | (truth == self.taum_index)
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
      
                losses += loss.item()
                losses_cat += loss_cat.item()
                losses_reg += loss_reg.item()
                accuracy += (
                    (pred.argmax(1) == truth.to(device))
                    .type(torch.float)
                    .sum()
                    .item()
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
        #dataloader.nits_expected = N // dataloader.dataset.batch_size
        accuracy /= N
        print("  ", f"Average loss: {losses/(b+1):.4f}")
        print("  ", f"Average loss (cat): {losses_cat/(b+1):.4f}")
        print("  ", f"Average loss (reg): {losses_reg/(b+1):.4f}")
        print("  ", f"Average accuracy: {float(100*accuracy):.4f}")

        return losses/(b+1), losses_cat/(b+1), losses_reg/(b+1), float(accuracy), elapsed_column.elapsed_time

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
            TextColumn("0/? its"),
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
                    tau_mask = (truth == self.taup_index) | (truth == self.taum_index)
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
                tau_mask = (truth == self.taup_index) | (truth == self.taum_index)
                gen_pt = torch.where(
                    tau_mask,
                    inpt[0][:, self.gen_lep_pt_index],  # values where tau_mask is True
                    inpt[0][:, self.gen_pt]             # values where tau_mask is False
                )
        
                target_pt = torch.clip(torch.nan_to_num(gen_pt / jet_pt, nan=0, posinf=0, neginf=0), min=0.3, max=2.0).reshape(-1, 1)

                with torch.no_grad():
                    #print(f'inpt.shape = {inpt[0].shape}')
                    pred_cat, pred_reg = self.forward(inpt)
                    loss, loss_cat, loss_reg = loss_fn(pred_cat, truth, pred_reg, target_pt, device)
                    #print(f'loss = {loss}')
        
                kinematics.append(inpt[0][..., [self.jet_pt_index, self.jet_eta_index, self.gen_lep_pt_index, self.gen_pt]].cpu().numpy())
                truths.append(truth.cpu().numpy().astype(int))
                processes.append(process.cpu().numpy())
                predictions.append(pred_cat.cpu().numpy())
                predictions_reg.append(pred_reg.cpu().numpy())
                #print(f'len(kinematics) = {len(kinematics)}')

                losses += loss.item()
                losses_cat += loss_cat.item()
                losses_reg += loss_reg.item()
                accuracy += (
                    (pred_cat.argmax(1) == truth.to(device))
                    .type(torch.float)
                    .sum()
                    .item()
                )
                #print(f'losses = {losses}')
                N += len(pred_cat)
                #print(f'N = {N}')
                progress.update(
                    task, advance=1, description=f"Inference...   | Loss: {loss:.2f}"
                )
                progress.columns[-1].text_format = "{}/{} its".format(
                    N // dataloader.dataset.batch_size, dataloader.nits_expected
                )
                #print('MMmmm')
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