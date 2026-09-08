import numpy as np
import torch
import torch.nn as nn
import time

from utils.models.abstract_base_models import Classifier
from utils.torch import LZ4FP16Dataset
from utils.plotting.termplot import terminal_roc
from utils.models.helpers import DenseClassifier, InputProcess
from scipy.special import softmax

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class LZ16Speed(Classifier, nn.Module):
    n_cpf = 26
    n_npf = 25
    n_vtx = 5
    datasetClass = LZ4FP16Dataset
    optimizerClass = torch.optim.Adam

    classes = {
        "b": ["isB"],
        "bb": ["isBB", "isGBB"],
        "leptonicB": ["isLeptonicB", "isLeptonicB_C"],
        "c": ["isC", "isCC", "isGCC"],
        "uds": ["isUD", "isS"],
        "g": ["isG"],
    }

    cpf_candidates = [
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
        "Npfcan_ptrel",
        "Npfcan_deltaR",
        "Npfcan_isGamma",
        "Npfcan_HadFrac",
        "Npfcan_drminsv",
        "Npfcan_puppiw",
    ]

    vtx_features = [
        "sv_pt",
        "sv_deltaR",
        "sv_mass",
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

    global_features = [
        "jet_pt",
        "jet_eta",
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

    def __init__(self, feature_edges=[15, 415, 565, 613], **kwargs):
        super(LZ16Speed, self).__init__(**kwargs)

        self.feature_edges = np.array(feature_edges)

        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.lin = nn.Linear(1,1)
        
        # integer positions and default values still have to be checked
        self.glob_integers = torch.tensor([2, 3, 4, 5, 8, 13, 14])
        self.cpf_integers = torch.tensor([12, 13, 14, 15])
        self.npf_integers = torch.tensor([2])
        self.vtx_integers = torch.tensor([3])
        self.integers = [
            self.glob_integers,
            self.cpf_integers,
            self.npf_integers,
            self.vtx_integers,
        ]
        self.glob_defaults = torch.tensor([0])
        self.cpf_defaults = torch.tensor([0])
        self.npf_defaults = torch.tensor([0])
        self.vtx_defaults = torch.tensor([0])
        self.defaults = [
            self.glob_defaults,
            self.cpf_defaults,
            self.npf_defaults,
            self.vtx_defaults,
        ]

    def forward(self, global_features, cpf_features, npf_features, vtx_features):

        return (global_features, cpf_features, npf_features, vtx_features)

    def train_model(
        self,
        training_data,
        validation_data,
        directory,
        attack=None,
        optimizer=None,
        device=None,
        nepochs=0,
        resume_epochs=0,
        **kwargs,
    ):

        loss_train = []
        acc_train = []
        loss_val = []
        acc_val = []

        scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
        best_loss_val = np.inf

        for t in range(resume_epochs, nepochs):
            print("Epoch", t + 1, "of", nepochs)
            training_data.dataset.shuffleFileList()  # Shuffle the file list as mini-batch training requires it for regularisation of a non-convex problem
            timer = 0
            start = time.time()

            loss_trainining, acc_training = self.update(
                training_data,
                attack=attack,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
            )

            start2 = time.time()
            print("Time elapsed : ", start2-start)
            
            loss_train += loss_trainining 
            acc_train.append(acc_training)

            loss_validation, acc_validation = self.validate_model(validation_data, device)
            loss_val += loss_validation
            acc_val.append(acc_validation)

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

            if np.mean(loss_validation) < best_loss_val:
                best_loss_val = np.mean(loss_validation)
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

        return loss_train, loss_val, acc_train, acc_val,

    def predict_model(self, dataloader, device, attack=None):
        self.eval()
        kinematics = []
        truths = []
        processes = []
        predictions = []
        for (x, truth, w,) in dataloader:

            pred = torch.zeros((truth.shape[0],6)).float()
            kinematics.append(global_features[..., :2].cpu().numpy())
            truths.append(truth.cpu().numpy().astype(int))
            processes.append(truth.cpu().numpy())
            predictions.append(pred.cpu().numpy())

        predictions = np.concatenate(predictions)
        kinematics = np.concatenate(kinematics)
        truths = np.concatenate(truths)
        processes = np.concatenate(processes)
        return predictions, truths, kinematics, processes

    def update(
        self,
        dataloader,
        optimizer,
        attack=None,
        scaler=None,
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
            print("entering traing loop")
            for (x, truth, w,) in dataloader:
                # We select either cuda float16 mixed precision or cpu float32 as LSTMs does not accept bfloat16
                with torch.autocast(device_type=device, enabled=True if device == "cuda" else False):
                    loss = torch.tensor([0]) #self.loss_fn(pred, truth.type(torch.LongTensor).to(device)).mean()

                    losses.append(loss.item())
                    accuracy += 0
                    N += 1
                    progress.update(task, advance=1, description=f"Training...   | Loss: {loss.item():.2f}")
                    progress.columns[-1].text_format = "{}/{} its".format(N // dataloader.batch_size, ("?" if dataloader.nits_expected == len(dataloader) else f"~{dataloader.nits_expected}"),)
                progress.update(task, completed=dataloader.nits_expected)
        dataloader.nits_expected = N // dataloader.batch_size
        accuracy /= N
        print(N)
        print("  ", f"Average loss: {np.array(losses).mean():.4f}")
        print("  ", f"Average accuracy: {float(100*accuracy):.4f}")
        return losses, accuracy

    def validate_model(self, dataloader, device="cpu", verbose=True):
        losses = []
        accuracy = 0.0
        self.eval()

        predictions = np.empty((0, 6))
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
            for (x, truth, w,) in dataloader:
                with torch.no_grad():
                    loss = torch.tensor([0]) #self.loss_fn(pred, truth.type(torch.LongTensor).to(device)).mean()
                    losses.append(loss.item())

                    accuracy += 0
                    predictions = 0# np.append(predictions, pred.to("cpu").numpy(), axis=0)
                    truths = 0 #np.append(truths, truth.to("cpu").numpy(), axis=0)
                    processes = 0 #np.append(processes, process.to("cpu").numpy(), axis=0)
                N += 1 #global_features.size(dim=0)
                progress.update(task, advance=1, description=f"Validation... | Loss: {loss.item():.2f}")
                progress.columns[-1].text_format = "{}/{} its".format(N // dataloader.batch_size, ("?" if dataloader.nits_expected == len(dataloader) else f"~{dataloader.nits_expected}"),)
            progress.update(task, completed=dataloader.nits_expected)
        dataloader.nits_expected = N // dataloader.batch_size
        accuracy /= N
        print("  ", f"Validation loss: {np.array(losses).mean():.4f}")
        print("  ", f"Validation accuracy: {float(accuracy):.4f}")

        return losses, float(accuracy)

    def calculate_roc_list(
        self,
        predictions,
        truth,
    ):
        if np.abs(np.mean(np.sum(predictions, axis=-1)) - 1) > 1e-3:
            predictions = softmax(predictions, axis=-1)

        b_jets = (truth == 0) | (truth == 1) | (truth == 2)
        c_jets = truth == 3
        l_jets = (truth == 4) | (truth == 5)
        summed_jets = b_jets + c_jets + l_jets

        b_pred = predictions[:, :3].sum(axis=1)
        c_pred = predictions[:, 3]
        l_pred = predictions[:, -2:].sum(axis=1)

        bvsl = np.where((b_pred + l_pred) > 0, (b_pred) / (b_pred + l_pred), -1)
        bvsc = np.where((b_pred + c_pred) > 0, (b_pred) / (b_pred + c_pred), -1)
        cvsb = np.where((b_pred + c_pred) > 0, (c_pred) / (b_pred + c_pred), -1)
        cvsl = np.where((l_pred + c_pred) > 0, (c_pred) / (l_pred + c_pred), -1)
        bvsall = np.where(
            (b_pred + l_pred + c_pred) > 0, (b_pred) / (b_pred + l_pred + c_pred), -1
        )

        b_veto = (truth != 0) & (truth != 1) & (truth != 2) & (summed_jets != 0)
        c_veto = (truth != 3) & (summed_jets != 0)
        l_veto = (truth != 4) & (truth != 5) & (summed_jets != 0)
        no_veto = np.ones(b_veto.shape, dtype=np.bool)

        labels = ["bvsl", "bvsc", "cvsb", "cvsl", "bvsall"]
        discs = [bvsl, bvsc, cvsb, cvsl, bvsall]
        vetos = [c_veto, l_veto, l_veto, b_veto, no_veto]
        truths = [b_jets, b_jets, c_jets, c_jets, b_jets]
        xlabels = [
            "b-identification",
            "b-identification",
            "c-identification",
            "c-identification",
            "b-identification",
        ]
        ylabels = ["light mis-id.", "c mis-id", "b mis-id.", "light mis-id.", "mis-id."]

        return discs, truths, vetos, labels, xlabels, ylabels
