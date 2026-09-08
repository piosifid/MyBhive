import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
import os
import numpy as np

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from utils.plotting.termplot import terminal_roc
from scipy.special import softmax

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group['lr'])
                    p.data.copy_(p_data_fp32)

        return loss


class CustomTimeElapsedColumn(TimeElapsedColumn):
    def __init__(self):
        super().__init__()
        self.elapsed_time = 0

    def render(self, task):
        self.elapsed_time = task.elapsed
        return super().render(task)
        
class Classifier_base(nn.Module):
    
    n_cpf = 26
    n_npf = 25
    n_vtx = 5
    optimizerClass = RAdam
    input_dims = [(1,15), (26, 20), (25, 10), (5, 15)]
    #input_dim_torchinfo = [[(1,15), (26, 20), (25, 10), (5, 15)]]

    feature_edges = []
    v = 0
    for dim in input_dims:
        v += dim[0]*dim[1]
        feature_edges.append(v)

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
        "Cpfcan_pt",
        "Cpfcan_eta",
        "Cpfcan_phi",
        "Cpfcan_e",
    ]

    npf_candidates = [
        "Npfcan_ptrel",
        "Npfcan_deltaR",
        "Npfcan_isGamma",
        "Npfcan_HadFrac",
        "Npfcan_drminsv",
        "Npfcan_puppiw",
        "Npfcan_pt",
        "Npfcan_eta",
        "Npfcan_phi",
        "Npfcan_e",
    ]

    vtx_features = [
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
        "sv_pt",
        "sv_eta",
        "sv_phi",
        "sv_e",
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

    # integer positions and default values still have to be checked
    glob_integers = torch.tensor([2, 3, 4, 5, 8, 13, 14])
    cpf_integers = torch.tensor([12, 13, 14, 15])
    npf_integers = torch.tensor([2])
    vtx_integers = torch.tensor([3])
    integers = [
        glob_integers,
        cpf_integers,
        npf_integers,
        vtx_integers,
    ]
    glob_defaults = torch.tensor([0])
    cpf_defaults = torch.tensor([0])
    npf_defaults = torch.tensor([0])
    vtx_defaults = torch.tensor([0])
    defaults = [
        glob_defaults,
        cpf_defaults,
        npf_defaults,
        vtx_defaults,
    ]

    def train_model(
        self,
        training_data,
        validation_data,
        directory,
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
        **kwargs,
    ):
        
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

        if os.path.isfile(f'{directory}/train_time.npy') and os.path.isfile(f'{directory}/val_time.npy'):
            train_time = np.load(f'{directory}/train_time.npy')
            val_time   = np.load(f'{directory}/val_time.npy')
        else:
            train_time, val_time = np.zeros(nepochs-resume_epochs), np.zeros(nepochs-resume_epochs)
            
        for t in range(resume_epochs, nepochs):
            print("Epoch", t + 1, "of", nepochs)
            training_data.dataset.shuffleFileList()  # Shuffle the file list as mini-batch training requires it for regularisation of a non-convex problem
            loss_training, acc_training, train_time[t] = self.update(
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
            train_metrics["acc"].append(acc_training)

            if (not batch_lr) and (scheduler is not None):
                scheduler.step()

            loss_validation, acc_validation, val_time[t] = self.validate_model(validation_data, loss_fn, device)
            
            validation_metrics["loss"].append(loss_validation)
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
                acc=train_metrics["acc"],
                allow_pickle=True,
            )
            np.savez(
                f'{directory}/validation_metrics',
                loss=validation_metrics["loss"],
                acc=validation_metrics["acc"],
                allow_pickle=True,
            )
        
        return train_metrics, validation_metrics

    def predict_model(self, dataloader, device, attack=None):
        self.eval()
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        
        kinematics = []
        truths = []
        processes = []
        predictions = []

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
            task = progress.add_task("Inference...", total=dataloader.nits_expected)
            
            for (x, truth, w, process) in dataloader:

                x = x.float().to(device)
                truth = truth.float().to(device)
                w = w.float().to(device)

                inpt, _ = self.get_inpt(x, truth=truth, loss_fn=loss_fn, attack=attack, device=device)
                
                with torch.no_grad():
                    pred = self(inpt)
                    loss = loss_fn(pred, truth.type(torch.LongTensor).to(device)).mean()

                kinematics.append(inpt[0][..., :2].cpu().numpy())
                truths.append(truth.cpu().numpy().astype(int))
                processes.append(process.cpu().numpy())
                predictions.append(pred.cpu().numpy())
                
                N += len(pred)
                progress.update(
                    task, advance=1, description=f"Inference...   | Loss: {loss:.2f}"
                )
                progress.columns[-1].text_format = "{}/{} its".format(
                    N // dataloader.batch_size,
                    (
                        "?"
                        if dataloader.nits_expected == len(dataloader)
                        else f"~{dataloader.nits_expected}"
                    ),
                )
            progress.update(task, completed=dataloader.nits_expected)

        predictions = np.concatenate(predictions)
        kinematics = np.concatenate(kinematics)
        truths = np.concatenate(truths)
        processes = np.concatenate(processes)
        
        return predictions, truths, kinematics, processes, elapsed_column.elapsed_time
    
    #@torch.compile(mode='max-autotune')
    def step(self, x, truth, loss_fn, attack=None, device="cpu", mixed_precision=True):

        inpt, truth = self.get_inpt(x, truth=truth, loss_fn=loss_fn, attack=attack, device=device)

        if mixed_precision:
            with torch.cuda.amp.autocast():
                pred = self.forward(inpt)
                loss = loss_fn(pred, truth).mean()
        else:
            pred = self.forward(inpt)
            loss = loss_fn(pred, truth).mean()
            
        return pred, loss
    
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
        losses = []
        accuracy = 0.0
        self.train()

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
            task = progress.add_task("Training...", total=dataloader.nits_expected)
            print("entering traing loop")
            for (x, truth, w, p) in dataloader:

                x = x.float().to(device)
                truth = truth.type(torch.LongTensor).to(device)
                w = w.float().to(device)

                if self.use_torch_compile:
                    pred, loss = self.compile_step(x, truth, loss_fn, attack=attack, device=device, mixed_precision=self.mixed_precision)
                else:
                    pred, loss = self.step(x, truth, loss_fn, attack=attack, device=device, mixed_precision=self.mixed_precision)

                if scaler != None:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                if batch_lr and (scheduler is not None):
                    scheduler.step()
      
                losses.append(loss.item())
                accuracy += (
                    (pred.argmax(1) == truth.to(device)).type(torch.float).sum().item()
                )

                N += len(pred)
                curr_lr = optimizer.param_groups[0]['lr']
                progress.update(
                    task, advance=1, description=f"Training...   | Loss: {loss:.4f}, lr: {curr_lr:.5f}"
                )
                progress.columns[-1].text_format = "{}/{} its".format(
                    N // dataloader.batch_size,
                    (
                        "?"
                        if dataloader.nits_expected == len(dataloader)
                        else f"~{dataloader.nits_expected}"
                    ),
                )
            progress.update(task, completed=dataloader.nits_expected)
        dataloader.nits_expected = N // dataloader.batch_size
        accuracy /= N
        print("  ", f"Average loss: {np.array(losses).mean():.4f}")
        print("  ", f"Average accuracy: {float(100*accuracy):.4f}")

        return np.array(losses).mean(), float(accuracy), elapsed_column.elapsed_time

    def validate_model(self, dataloader, loss_fn, device="cpu", verbose=True):
        losses = []
        accuracy = 0.0
        self.eval()

        predictions = np.empty((0, 6))
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
            for (x, truth, w, process) in dataloader:
                
                x = x.float().to(device)
                truth = truth.type(torch.LongTensor).to(device)
                w = w.float().to(device)

                with torch.no_grad():
                    if self.use_torch_compile:
                        pred, loss = self.compile_step(x, truth, loss_fn, attack=None, device=device, mixed_precision=self.mixed_precision)
                    else:
                        pred, loss = self.step(x, truth, loss_fn, attack=None, device=device, mixed_precision=self.mixed_precision)
                        
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
                    
                N += len(pred)
                progress.update(
                    task, advance=1, description=f"Validation... | Loss: {loss:.2f}"
                )
                progress.columns[-1].text_format = "{}/{} its".format(
                    N // dataloader.batch_size,
                    (
                        "?"
                        if dataloader.nits_expected == len(dataloader)
                        else f"~{dataloader.nits_expected}"
                    ),
                )
            progress.update(task, completed=dataloader.nits_expected)
        dataloader.nits_expected = N // dataloader.batch_size
        accuracy /= N
        print("  ", f"Validation loss: {np.array(losses).mean():.4f}")
        print("  ", f"Validation accuracy: {float(100*accuracy):.4f}")

        if verbose:
            print("Printing terminal ROC")
            terminal_roc(predictions, truths, title="Validation ROC")

        return np.array(losses).mean(), float(accuracy), elapsed_column.elapsed_time

    def get_inpt(self, x, truth=None, loss_fn=None, attack=None, device='cpu'):

        feature_edges = torch.Tensor(self.feature_edges).int()
        
        feature_lengths = feature_edges[1:] - feature_edges[:-1]
        feature_lengths = torch.cat((feature_edges[:1], feature_lengths))
        glob, cpf, npf, vtx = x.split(feature_lengths.tolist(), dim=1)
        
        glob = glob.reshape(glob.shape[0], -1)
        cpf = cpf.reshape(cpf.shape[0], self.input_dims[1][0], -1)
        npf = npf.reshape(npf.shape[0], self.input_dims[2][0], -1)
        vtx = vtx.reshape(vtx.shape[0], self.input_dims[3][0], -1)

        if attack is not None:
            (
                glob,
                cpf,
                npf,
                vtx,
                truth,
            ) = attack(
                [
                    feature.float().to(device)
                    for feature in [
                        glob,
                        cpf,
                        npf,
                        vtx,
                        ]
                    ],
                truth.type(torch.LongTensor).to(device),
                loss_fn,
                self,
            ) 
        
        return (glob.detach(), cpf.detach(), npf.detach(), vtx.detach()), truth

    def calculate_roc_list(
        self,
        predictions,
        truth,
    ):
        if np.abs(np.mean(np.sum(predictions, axis=-1)) - 1) > 1e-3:
            predictions = softmax(predictions, axis=-1)

        b_jets = (truth == 0) | (truth == 1) | (truth == 2)
        c_jets = truth == 3
        uds_jets = truth == 4
        g_jets = truth == 5
        l_jets = uds_jets | g_jets
        summed_jets = b_jets + c_jets + l_jets

        b_pred = predictions[:, :3].sum(axis=1)
        c_pred = predictions[:, 3]
        uds_pred = predictions[:, 4]
        g_pred = predictions[:, 5]
        l_pred = predictions[:, -2:].sum(axis=1)

        bvsl = np.where((b_pred + l_pred) > 0, (b_pred) / (b_pred + l_pred), -1)
        bvsc = np.where((b_pred + c_pred) > 0, (b_pred) / (b_pred + c_pred), -1)
        cvsb = np.where((b_pred + c_pred) > 0, (c_pred) / (b_pred + c_pred), -1)
        cvsl = np.where((l_pred + c_pred) > 0, (c_pred) / (l_pred + c_pred), -1)
        bvsall = np.where(
            (b_pred + l_pred + c_pred) > 0, (b_pred) / (b_pred + l_pred + c_pred), -1
        )
        uds_vs_g = np.where((uds_pred + g_pred) > 0, (uds_pred) / (uds_pred + g_pred), -1)

        b_veto = (truth != 0) & (truth != 1) & (truth != 2) & (summed_jets != 0)
        c_veto = (truth != 3) & (summed_jets != 0)
        bc_veto = (truth != 0) & (truth != 1) & (truth != 2) & (truth != 3) & (summed_jets != 0)
        l_veto = (truth != 4) & (truth != 5) & (summed_jets != 0)
        no_veto = np.ones(b_veto.shape, dtype=np.bool)

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