import numpy as np
import torch
import torch.nn as nn
from utils.plotting.termplot import terminal_roc
from utils.torch import L1TDataset
from scipy.special import softmax
from utils.models.abstract_base_models import Classifier

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

class L1TTorchBase(Classifier, nn.Module):
    classes = {
        "b": ["label_b"],
        "uds": ["label_uds"],
        "g": ["label_g"],
        # "tau": ["label_tau"],
    }

    n_cpf = 16
    datasetClass = L1TDataset

    global_features = [
        "jet_pt",
        "jet_eta",
        "jet_phi",
        "jet_mass",
        "jet_energy",
        "jet_npfcand",
    ]

    cpf_candidates = [
        "jet_pfcand_pt_rel",
        "jet_pfcand_deta",
        "jet_pfcand_dphi",
        "jet_pfcand_charge",
        "jet_pfcand_id",
        "jet_pfcand_track_vx",
        "jet_pfcand_track_vy",
        "jet_pfcand_track_vz",
    ]

    def __init__(
        self,
        for_inference=False,
        **kwargs,
    ):
        super(L1TTorchBase, self).__init__(**kwargs)

        self.layers1 = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 8))
        )
        self.layers2 = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(
        self, cpf_features,
    ):
            
        c = self.layers1(cpf_features)
        output = torch.mean(c, -1)
        output = self.layers2(output)
        return output

    def train_model(
        self,
        training_data,
        validation_data,
        directory,
        device,
        nepochs=0,
        learning_rate=0.001,
        **kwargs,
    ):
        best_loss_val = np.inf
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-7, weight_decay = 0.0001)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        train_metrics = np.zeros((nepochs, 2))
        validation_metrics = np.zeros((nepochs, 2))
        print("Initial ROC")

        if validation_data:
            _, _ = self.validate_model(validation_data, loss_fn, device)
        for t in range(nepochs):
            print("Epoch", t + 1, "of", nepochs)
            training_data.dataset.shuffleFileList()  # Shuffle the file list as mini-batch training requires it for regularisation of a non-convex problem
            loss_train, acc_train = self.update(
                training_data,
                loss_fn,
                optimizer,
                device,
            )
            train_metrics[t, :] = np.array([loss_train, acc_train])

            if validation_data:
                _, _ = self.validate_model(validation_data, loss_fn, device)
                loss_val, acc_val = self.validate_model(
                    validation_data, loss_fn, device
                )
                validation_metrics[t, :] = np.array([loss_val, acc_val])
            else:
                validation_metrics[t, :] = np.array([0.0, 0.0])

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

    def predict_model(self, dataloader, device):
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
            truth,
            weight,
            process,
        ) in dataloader:
            pred = self.forward(
                *[
                    feature.float().to(device)
                    for feature in [
                        cpf_features,
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
                truth,
                weight,
                process,
            ) in dataloader:
                pred = self.forward(
                    *[
                        feature.float().to(device)
                        for feature in [
                            cpf_features,
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
        print("  ", f"Average accuracy: {float(accuracy):.4f}")
        return np.array(losses).mean(), float(accuracy)

    def validate_model(self, dataloader, loss_fn, device="cpu", verbose=True):
        losses = []
        accuracy = 0.0
        self.eval()

        predictions = np.empty((0, len(self.classes.items())))
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
                truth,
                weight,
                process,
            ) in dataloader:
                with torch.no_grad():
                    pred = self.forward(
                        *[
                            feature.float().to(device)
                            for feature in [
                                cpf_features,
                                # cpf_features.abs().sum(dim=2, keepdim=True)
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

        b_jets = (truth == 0)
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
