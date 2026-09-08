import numpy as np
import torch
import torch.nn as nn
from utils.models.abstract_base_models import Classifier
from utils.plotting.termplot import terminal_roc
from utils.torch import PNetDataset
from scipy.special import softmax

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

"""Taken from https://github.com/hqucms/weaver/blob/master/utils/nn/model/ParticleNet.py"""
"""Based on https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py."""


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
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
        **kwargs,
    ):
        super(ParticleNet, self).__init__(**kwargs)

        self.use_fts_bn = use_fts_bn
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
            fcs.append(nn.Conv1d(fc_params[-1][0], num_classes, kernel_size=1))
        else:
            fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference

    def forward(self, points, features, mask=None):
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



class ParticleNetTagger(Classifier, nn.Module):
    classes = {
        "b": ["label_b"],
        "c": ["label_c"],
        "light_gluon": ["label_uds", "label_g"],
        "tau": ["label_taup", "label_taum"],
    }

    n_cpf = 50
    n_vtx = 5
    datasetClass = PNetDataset

    cpf_points = [
        "jet_pfcand_deta",
        "jet_pfcand_dphi",
    ]
    vtx_points = [
        "jet_sv_deta",
        "jet_sv_dphi",
    ]

    global_features = [
        "jet_pt",
        "jet_pt_raw",
        "jet_eta",
        "jet_phi",
        "jet_mass",
        "jet_mass_raw",
        "jet_energy",
        "jet_chf",
        "jet_nhf",
        "jet_phf",
        "jet_elf",
        "jet_muf",
        "jet_ncand",
        "jet_nbhad",
        "jet_nchad",
    ]

    cpf_candidates = [
        "jet_pfcand_pt",
        "jet_pfcand_eta",
        "jet_pfcand_phi",
        "jet_pfcand_mass",
        "jet_pfcand_energy",
        "jet_pfcand_pt_log",
        "jet_pfcand_energy_log",
        "jet_pfcand_calofraction",
        "jet_pfcand_hcalfraction",
        "jet_pfcand_dxy",
        "jet_pfcand_dxysig",
        "jet_pfcand_dz",
        "jet_pfcand_dzsig",
        "jet_pfcand_pperp_ratio",
        "jet_pfcand_ppara_ratio",
        "jet_pfcand_deta",
        "jet_pfcand_dphi",
        "jet_pfcand_etarel",
        "jet_pfcand_frompv",
        "jet_pfcand_id",
        "jet_pfcand_charge",
        "jet_pfcand_track_qual",
        "jet_pfcand_track_chi2",
        "jet_pfcand_npixhits",
        "jet_pfcand_nstriphits",
        "jet_pfcand_nlostinnerhits",
        "jet_pfcand_trackjet_d3d",
        "jet_pfcand_trackjet_d3dsig",
        "jet_pfcand_trackjet_dist",
        "jet_pfcand_trackjet_decayL",
    ]

    vtx_features = [
        "jet_sv_ntracks",
        "jet_sv_pt",
        "jet_sv_pt_log",
        "jet_sv_eta",
        "jet_sv_phi",
        "jet_sv_mass",
        "jet_sv_energy",
        "jet_sv_energy_log",
        "jet_sv_deta",
        "jet_sv_dphi",
        "jet_sv_chi2",
        "jet_sv_dxy",
        "jet_sv_dxysig",
        "jet_sv_d3d",
        "jet_sv_d3dsig",
    ]

    def __init__(
        self,
        conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
        fc_params=[(128, 0.1)],
        use_fusion=True,
        use_fts_bn=True,
        use_counts=True,
        pf_input_dropout=None,
        sv_input_dropout=None,
        for_inference=False,
        **kwargs,
    ):
        super(ParticleNetTagger, self).__init__(**kwargs)
        self.pf_input_dropout = (
            nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        )
        self.sv_input_dropout = (
            nn.Dropout(sv_input_dropout) if sv_input_dropout else None
        )
        """
        these feature convs might be wrong...
        it could be that it wludl be (len(features), cutoff)
        """
        self.pf_conv = FeatureConv(len(self.cpf_candidates), 32)  # this was 32
        self.sv_conv = FeatureConv(len(self.vtx_features), 32)  # this was 32
        self.pn = ParticleNet(
            input_dims=32,
            num_classes=len(self.classes),
            conv_params=conv_params,
            fc_params=fc_params,
            use_fusion=use_fusion,
            use_fts_bn=use_fts_bn,
            use_counts=use_counts,
            for_inference=for_inference,
        )

    def forward(
        self, pf_points, cpf_features, pf_mask, sv_points, vtx_features, sv_mask
    ):
        if self.pf_input_dropout:
            pf_mask = (self.pf_input_dropout(pf_mask) != 0).float()
            pf_points *= pf_mask
            cpf_features *= pf_mask
        if self.sv_input_dropout:
            sv_mask = (self.sv_input_dropout(sv_mask) != 0).float()
            sv_points *= sv_mask
            vtx_features *= sv_mask

        points = torch.cat((pf_points, sv_points), dim=1)
        a = self.pf_conv(
            cpf_features.transpose(1, 2) * pf_mask.transpose(1, 2)
        ) * pf_mask.transpose(1, 2)
        b = self.sv_conv(
            vtx_features.transpose(1, 2) * sv_mask.transpose(1, 2)
        ) * sv_mask.transpose(1, 2)
        features = torch.cat(
            (a, b),
            dim=2,
        )
        mask = torch.cat((pf_mask, sv_mask), dim=1)
        return self.pn(points.transpose(1, 2), features, mask.transpose(1, 2))

    def train_model(
        self,
        training_data,
        validation_data,
        directory,
        optimizer=None,
        device=None,
        nepochs=0,
        resume_epochs=0,
        **kwargs,
    ):
        best_loss_val = np.inf
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        train_metrics = []
        validation_metrics = []
        print("Initial ROC")

        if validation_data:
            _, _ = self.validate_model(validation_data, loss_fn, device)
        for t in range(resume_epochs, nepochs):
            print("Epoch", t + 1, "of", nepochs)
            training_data.dataset.shuffleFileList()  # Shuffle the file list as mini-batch training requires it for regularisation of a non-convex problem
            loss_train, acc_train = self.update(
                training_data,
                loss_fn,
                optimizer,
                device,
            )
            train_metrics = np.zeros((nepochs, 2))

            if validation_data:
                loss_val, acc_val = self.validate_model(
                    validation_data, loss_fn, device
                )
                validation_metrics.append([loss_val, acc_val])
            else:
                validation_metrics.append([0, 0])

            torch.save(
                {
                    "epoch": t,
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_train": loss_train,
                    "acc_train": acc_train,
                    "loss_val": loss_val,
                    "acc_val": acc_val,
                },
                "{}/model_{}.pt".format(directory, t),
            )

            if loss_val < best_loss_val:
                best_loss_val = loss_val
                torch.save(
                    {
                        "epoch": t,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss_train": loss_train,
                        "acc_train": acc_train,
                        "loss_val": loss_val,
                        "acc_val": acc_val,
                    },
                    "{}/best_model.pt".format(directory),
                )

        return train_metrics, validation_metrics

    def predict_model(self, dataloader, device=None):
        self.eval()
        kinematics = []
        truths = []
        processes = []
        predictions = []
        # append jet_pt and jet_eta
        # this should be done differenlty in the future... avoid array slicing with magic numbers!
        for (
            global_features,
            cpf_features,
            vtx_features,
            cpf_points,
            vtx_points,
            truth,
            weight,
            process,
        ) in dataloader:
            pred = self.forward(
                *[
                    feature.float().to(device)
                    for feature in [
                        cpf_points,
                        cpf_features,
                        cpf_features.abs().sum(dim=2, keepdim=True) != 0,  # (N, 1, P)
                        vtx_points,
                        vtx_features,
                        vtx_features.abs().sum(dim=2, keepdim=True) != 0,  # (N, 1, P),
                    ]
                ]
            )
            kinematics.append(global_features[..., :2].cpu().numpy())
            truths.append(truth.cpu().numpy())
            processes.append(process.cpu().numpy())
            predictions.append(pred.detach().cpu().numpy())

        predictions = np.concatenate(predictions)
        kinematics = np.concatenate(kinematics)
        truths = np.concatenate(truths).astype(dtype=np.int)
        processes = np.concatenate(processes).astype(dtype=np.int)
        return predictions, truths, kinematics, processes

    def update(
        self,
        dataloader,
        loss_fn,
        optimizer,
        device="cpu",
        verbose=True,
    ):
        losses = []
        accuracy = 0.0
        self.train()

        with Progress(
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("0/? its"),
            expand=True,
        ) as progress:
            N = 0
            task = progress.add_task("Training...", total=dataloader.nits_expected)
            for (
                global_features,
                cpf_features,
                vtx_features,
                cpf_points,
                vtx_points,
                truth,
                weight,
                process,
            ) in dataloader:#self, pf_points, cpf_features, pf_mask, sv_points, vtx_features, sv_mask
                pred = self.forward(
                    *[
                        feature.float().to(device)
                        for feature in [
                            cpf_points,
                            cpf_features,
                            cpf_features.abs().sum(dim=2, keepdim=True)
                            != 0,  # (N, 1, P)
                            vtx_points,
                            vtx_features,
                            vtx_features.abs().sum(dim=2, keepdim=True)
                            != 0,  # (N, 1, P),
                        ]
                    ]
                )

                loss = loss_fn(pred, truth.type(torch.LongTensor).to(device)).mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                accuracy += (
                    (pred.argmax(1) == truth.to(device)).type(torch.float).sum().item()
                )
                N += len(pred)
                progress.update(
                    task, advance=1, description=f"Training...   | Loss: {loss:.2f}"
                )
                progress.columns[-1].text_format = "{}/{} its".format(
                    N // dataloader.batch_size,
                    "?"
                    if dataloader.nits_expected == len(dataloader)
                    else f"~{dataloader.nits_expected}",
                )
            progress.update(task, completed=dataloader.nits_expected)
        dataloader.nits_expected = N // dataloader.batch_size
        accuracy /= N
        print("  ", f"Average loss: {np.array(losses).mean():.4f}")
        print("  ", f"Average accuracy: {float(100*accuracy):.4f}")
        return np.array(losses).mean(), float(accuracy)

    def validate_model(self, dataloader, loss_fn, device="cpu", verbose=True):
        losses = []
        accuracy = 0.0
        self.eval()

        predictions = np.empty((0, len(self.classes)))
        truths = np.empty((0))
        processes = np.empty((0))

        with Progress(
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("0/? its"),
            expand=True,
        ) as progress:
            N = 0
            task = progress.add_task("Validation...", total=dataloader.nits_expected)
            for (
                global_features,
                cpf_features,
                vtx_features,
                cpf_points,
                vtx_points,
                truth,
                weight,
                process,
            ) in dataloader:
                with torch.no_grad():
                    pred = self.forward(
                        *[
                            feature.float().to(device)
                            for feature in [
                                cpf_points,
                                cpf_features,
                                cpf_features.abs().sum(dim=2, keepdim=True)
                                != 0,  # (N, 1, P)
                                vtx_points,
                                vtx_features,
                                vtx_features.abs().sum(dim=2, keepdim=True)
                                != 0,  # (N, 1, P),
                            ]
                        ]
                    )
                    loss = loss_fn(pred, truth.type(torch.LongTensor).to(device)).mean()
                    losses.append(loss.item())

                    accuracy += (
                        (pred.argmax(1) == truth.to(device))
                        .type(torch.float)
                        .sum()
                        .item()
                    )
                    predictions = np.append(predictions, pred.to("cpu").numpy(), axis=0)
                    truths = np.append(truths, truth.to("cpu").numpy(), axis=0)
                    processes = np.append(processes, process.to("cpu").numpy(), axis=0)
                N += global_features.size(dim=0)
                progress.update(
                    task, advance=1, description=f"Validation... | Loss: {loss:.2f}"
                )
                progress.columns[-1].text_format = "{}/{} its".format(
                    N // dataloader.batch_size,
                    "?"
                    if dataloader.nits_expected == len(dataloader)
                    else f"~{dataloader.nits_expected}",
                )
            progress.update(task, completed=dataloader.nits_expected)
        dataloader.nits_expected = N // dataloader.batch_size
        accuracy /= N
        if verbose:
            terminal_roc(predictions, truths, title="Validation ROC")

        print("  ", f"Average loss: {np.array(losses).mean():.4f}")
        print("  ", f"Average accuracy: {float(accuracy):.4f}")
        return np.array(losses).mean(), float(accuracy)

    def calculate_roc_list(
        self,
        predictions,
        truth,
    ):
        if np.abs(np.mean(np.sum(predictions, axis=-1)) - 1) > 1e-3:
            predictions = softmax(predictions, axis=-1)

        b_jets = (truth == 0) | (truth == 1) | (truth == 2)
        others = truth != 0
        summed_jets = b_jets + others

        b_pred = predictions[:, 0]
        other_pred = predictions[:, 1:].sum(axis=1)

        bvsall = np.where(
            (b_pred + other_pred) > 0, (b_pred) / (b_pred + other_pred), -1
        )

        no_veto = np.ones(b_pred.shape, dtype=bool)

        labels = ["bvsall"]
        discs = [bvsall]
        vetos = [no_veto]
        truths = [b_jets]
        xlabels = [
            "b-identification",
        ]
        ylabels = ["mis-id."]

        return discs, truths, vetos, labels, xlabels, ylabels
