import math
import torch
import torch.nn as nn
import os
import numpy as np
from functools import partial

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from utils.plotting.termplot import terminal_roc
from utils.config.config_loader import ConfigLoader
from scipy.special import softmax

from utils.loss.LossFunctionLoader import LossFunctionLoader


class CustomTimeElapsedColumn(TimeElapsedColumn):
    def __init__(self):
        super().__init__()
        self.elapsed_time = 0

    def render(self, task):
        self.elapsed_time = task.elapsed
        return super().render(task)
        
class Classifier_base(nn.Module):

    integer_features = {
            "global_features": [],
            "cpf_candidates": [],
            "npf_candidates": [],
            "vtx_features": [],
            "lt_candidates": [],
    }

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.create_integers_defaults()
        self.create_feature_shapes()
        for key in ['n_cpf_candidates', 'n_npf_candidates', 'n_vtx_candidates', 'n_lt_candidates']:
            setattr(self, key, getattr(self, key, config[key]))     

    def _create_feature_indices(self, feature_list, feature_key):
        """Helper function to create feature indices from model attributes or config."""
        source = getattr(self, feature_key, self.config.get(feature_key, []))
        return torch.tensor([source.index(item) for item in feature_list if item in source], dtype=torch.int64)

    def create_integers_defaults(self):

        # Create indexes for each integer feature group
        self.integers = [self._create_feature_indices(features, key) for key, features in self.integer_features.items()]
        # Defaults for all feature groups
        self.defaults = [torch.tensor([0]) for _ in self.integer_features]
        
    def model_feature_length(self, attr_name, attr_name_2):
        """
        Returns len(getattr(self, attr_name)) if it exists,
        otherwise calls the provided calculate_feature_length.
        """
        if hasattr(self, attr_name):
            return len(getattr(self, attr_name))
        return self.calculate_feature_length(attr_name, attr_name_2)

    def all_model_features_length(self):
        global_dim = self.model_feature_length('global_features', 'global_custom_features')
        cpf_dim = self.model_feature_length('cpf_candidates', 'cpf_custom_features')
        npf_dim = self.model_feature_length('npf_candidates', 'npf_custom_features')
        vtx_dim = self.model_feature_length('vtx_features', 'vtx_custom_features')
        lt_dim  = self.model_feature_length('lt_candidates', 'lt_custom_features')
        return global_dim, cpf_dim, npf_dim, vtx_dim, lt_dim
        
    def calculate_feature_length(self, base_key, custom_key=None):
        base_length = 0
        if base_key in self.config:
            base_length += len(self.config[base_key])
        if custom_key and custom_key in self.config:
            base_length += len(self.config[custom_key])
        return base_length
    
    def create_feature_shapes(self):
        # Constructions of input shape from config.yaml file
        len_glob_fts_full = self.calculate_feature_length('global_features', 'global_custom_features')
        len_cpf_fts_full = self.calculate_feature_length('cpf_candidates', 'cpf_custom_features')
        len_npf_fts_full = self.calculate_feature_length('npf_candidates', 'npf_custom_features')
        len_vtx_fts_full = self.calculate_feature_length('vtx_features', 'vtx_custom_features')
        len_lt_fts_full  = self.calculate_feature_length('lt_candidates', 'lt_custom_features')
        
        self.input_dims = [
            (1,                               len_glob_fts_full),
            (self.config['n_cpf_candidates'], len_cpf_fts_full),
            (self.config['n_npf_candidates'], len_npf_fts_full),
            (self.config['n_vtx_candidates'], len_vtx_fts_full),
            (self.config['n_lt_candidates'],  len_lt_fts_full),
        ]

        feature_edges = []
        v = 0
        for dim in self.input_dims:
            v += dim[0]*dim[1]
            feature_edges.append(v)
    
        self.feature_edges = torch.Tensor(feature_edges).int()    
        feature_lengths = self.feature_edges[1:] - self.feature_edges[:-1]
        self.feature_lengths = torch.cat((self.feature_edges[:1], feature_lengths))

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

            loss_validation, acc_validation, val_time[t] = self.validate_model(
                validation_data, 
                loss_fn, 
                device,
                terminal_plot=terminal_plot
            )  
            
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

    def predict_model(
        self, 
        dataloader, 
        output,
        loss_fn,
        device, 
        attack=None
    ):
        losses = 0.0
        accuracy = 0.0
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
            TextColumn(f"0/{dataloader.nits_expected} its"),
            expand=True,
        ) as progress:
            N = 0
            task = progress.add_task("Inference...", total=dataloader.nits_expected)
            
            for (x, truth, w, process) in dataloader:

                x = x.float().to(device, non_blocking=True)
                truth = truth.float().to(device, non_blocking=True)
                w = w.float().to(device, non_blocking=True)

                torch.backends.cudnn.enabled = False
                inpt, _ = self.get_inpt(x, truth=truth, loss_fn=loss_fn, attack=attack, device=device)
                torch.backends.cudnn.enabled = True
                
                with torch.no_grad():
                    pred = self(inpt)
                    loss = loss_fn(pred, truth.type(torch.LongTensor).to(device)).mean()

                kinematics.append(inpt[0][..., :2].cpu().numpy())
                truths.append(truth.cpu().numpy().astype(int))
                processes.append(process.cpu().numpy())
                predictions.append(pred.cpu().numpy())

                N += len(pred)
                losses += loss.item() * len(pred)
                accuracy += (
                    (pred.argmax(1) == truth.to(device)).type(torch.float).sum().item()
                )
                
                progress.update(
                    task, advance=1, description=f"Inference...   | Average loss: {losses/N:.4f} | Batch loss: {loss.item():.4f}"
                )
                progress.columns[-1].text_format = (
                    f"{(N // dataloader.dataset.batch_size) / elapsed_column.elapsed_time:.2f} b/s | "
                    f"{N // dataloader.dataset.batch_size}/{dataloader.nits_expected} its"
                )
            progress.update(task, completed=dataloader.nits_expected)
       
        accuracy /= N
        losses /= N
        print("  ", f"Average loss: {losses:.4f}")
        print("  ", f"Average accuracy: {float(100*accuracy):.4f}")
        
        predictions = np.concatenate(predictions)
        kinematics = np.concatenate(kinematics)
        truths = np.concatenate(truths)
        processes = np.concatenate(processes)

        np.save(output["prediction"].path, predictions)
        np.save(output["truth"].path, truths)
        np.save(output["kinematics"].path, kinematics)
        np.save(output["process"].path, processes)
        np.save(output["inference_time"].path, elapsed_column.elapsed_time)
        
        return predictions, truths, kinematics, processes, elapsed_column.elapsed_time
    
    #@torch.compile(mode='max-autotune')
    def step(self, inpt, truth, loss_fn, attack=None, device="cpu", mixed_precision=True):
        
        with torch.autocast(device, enabled=mixed_precision):
            pred = self.forward(inpt)         
            loss = loss_fn(pred.float(), truth).mean()
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
        losses = 0.0
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
            for b, (x, truth, w, p) in enumerate(dataloader):

                x = x.float().to(device, non_blocking=True)
                truth = truth.type(torch.LongTensor).to(device, non_blocking=True)
                w = w.float().to(device, non_blocking=True)

                inpt, truth = self.get_inpt(x, truth=truth, loss_fn=loss_fn, attack=attack, device=device)
                # print(f"Input [{inpt.shape}]: {inpt}")
                # print(f"Input [{inpt[0].shape}, {inpt[1].shape}, {inpt[2].shape}, {inpt[3].shape}, {inpt[4].shape}]")
                # print(f"Input [{inpt[0].shape}, {inpt[1].shape}, {inpt[2].shape}, {inpt[3].shape}, {inpt[4].shape}]: {inpt}")
                # print(f"Truth [{truth.shape}]: {truth}")
                
                if self.use_torch_compile:
                    pred, loss = self.compile_step(inpt, truth, loss_fn, device=device, mixed_precision=self.mixed_precision)
                else:
                    pred, loss = self.step(inpt, truth, loss_fn, device=device, mixed_precision=self.mixed_precision)

                # print(f"Pred [{pred.shape}]: {pred}")
                # print(f"Loss [{loss.shape}]: {loss}")
                if verbose:
                    if torch.isnan(loss).any():
                        raise ValueError("Loss contains NaN values! Something's wrong with calculation, please check.")

                optimizer.zero_grad(set_to_none=True)
                
                if self.mixed_precision:
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
                accuracy += (
                    (pred.argmax(1) == truth.to(device)).type(torch.float).mean().item()
                )

                curr_lr = optimizer.param_groups[0]['lr']
                progress.update(
                    task, advance=1, description=f"Training...   | Average loss: {losses/(b+1):.4f} | Batch loss: {loss.item():.4f} | lr: {curr_lr:.5f}"
                )
                progress.columns[-1].text_format = f"Speed: {(b + 1)/elapsed_column.elapsed_time:.1f} b/s | {b + 1}/~{dataloader.nits_expected} its"
            progress.update(task, completed=dataloader.nits_expected)

            if (not batch_lr) and (scheduler is not None):
                scheduler.step()

        accuracy /= (b+1)
        losses /= (b+1)
        print("  ", f"Average loss: {losses:.4f}")
        print("  ", f"Average accuracy: {float(100*accuracy):.4f}")

        return losses, float(accuracy), elapsed_column.elapsed_time

    def validate_model(
        self, 
        dataloader, 
        loss_fn, 
        device="cpu", 
        verbose=True, 
        terminal_plot=False
    ):
        losses = 0.0
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
            N = 0
            task = progress.add_task("Validation...", total=dataloader.nits_expected)
            for (x, truth, w, process) in dataloader:

                x = x.float().to(device, non_blocking=True)
                truth = truth.type(torch.LongTensor).to(device, non_blocking=True)
                w = w.float().to(device, non_blocking=True)

                with torch.no_grad():
                    inpt, truth = self.get_inpt(x, truth=truth, loss_fn=loss_fn, attack=None, device=device)
                    
                    if self.use_torch_compile:
                        pred, loss = self.compile_step(inpt, truth, loss_fn, device=device, mixed_precision=self.mixed_precision)
                    else:
                        pred, loss = self.step(inpt, truth, loss_fn, device=device, mixed_precision=self.mixed_precision)

                    if verbose:
                        if torch.isnan(loss).any():
                            raise ValueError("Loss contains NaN values! Something's wrong with calculation, please check.")
                        
                    losses += loss.item() * len(pred)

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
                    task, advance=1, description=f"Validation... | Average loss: {losses/N:.4f} | Batch loss: {loss.item():.4f}"
                )
                progress.columns[-1].text_format = (
                    f"{(N // dataloader.dataset.batch_size) / elapsed_column.elapsed_time:.2f} b/s | "
                    f"{N // dataloader.dataset.batch_size}/~{dataloader.nits_expected} its"
                )
            progress.update(task, completed=dataloader.nits_expected)
        accuracy /= N
        losses /= N
        print("  ", f"Validation loss: {losses:.4f}")
        print("  ", f"Validation accuracy: {float(100*accuracy):.4f}")

        if verbose and terminal_plot:
            print("Printing terminal ROC")
            terminal_roc(predictions, truths, title="Validation ROC")

        return losses, float(accuracy), elapsed_column.elapsed_time


    def _subselect_features(self, tensor, name_fts, custom_name_fts, length):
        """
        Subselects features from a tensor based on the model's feature list or the config.

        Args:
            tensor (torch.Tensor): The input tensor to subselect from.
            name_fts (str): The name of features group to subselect.
            custom_name_fts (str): The name of custom features group to add.
            length (int): The length of the list of candidates to restrict to.

        Returns:
            torch.Tensor: The subselection of the input tensor.
        """
        if hasattr(self, name_fts):
            model_features = getattr(self, name_fts)
            all_fts = self.config.get(name_fts,[]) + [list(d.keys())[0] for d in self.config.get(custom_name_fts,[])]
            indices = [all_fts.index(f) for f in model_features]
        else:
            indices = list(range(tensor.size(-1)))  # Default to all features if not specified
        return tensor.index_select(dim=-1, index=torch.tensor(indices, device=tensor.device, dtype=torch.long))[:, :length]

    
    def get_inpt(self, x, truth=None, loss_fn=None, attack=None, device='cpu'):
        """
        Processes the input tensor `x` by splitting, subselecting features and applying attacks.

        Args:
            x (torch.Tensor): Input tensor.
            truth (Optional): Ground truth labels.
            loss_fn (Optional): Loss function.
            attack (Optional): Adversarial attack method.
            device (str): Device to use for tensors.

        Returns:
            Tuple[torch.Tensor]: Processed tensors for glob, cpf, npf, and vtx.
        """
        # Split the input tensor
        glob, cpf, npf, vtx, lt = x.split(self.feature_lengths.tolist(), dim=1)

        # Reshape tensors as per input dimensions
        glob = glob.reshape(glob.shape[0], -1)
        cpf = cpf.reshape(cpf.shape[0], *self.input_dims[1])
        npf = npf.reshape(npf.shape[0], *self.input_dims[2])
        vtx = vtx.reshape(vtx.shape[0], *self.input_dims[3])
        lt  = lt.reshape(lt.shape[0],   *self.input_dims[4])

        # Subselect features
        glob = self._subselect_features(glob, "global_features", "global_custom_features", None)
        cpf = self._subselect_features(cpf, "cpf_candidates", "cpf_custom_features", self.n_cpf_candidates)
        npf = self._subselect_features(npf, "npf_candidates", "npf_custom_features", self.n_npf_candidates)
        vtx = self._subselect_features(vtx, "vtx_features", "vtx_custom_features", self.n_vtx_candidates)
        lt  = self._subselect_features(lt,  "lt_features", "lt_custom_features", self.n_lt_candidates)
        
        if attack is not None:
            if (isinstance(truth, tuple)):
                jet_pt        = glob[:, truth[1]]
                gen_pt_WithNu = glob[:, truth[3]]
                gen_pt        = glob[:, truth[2]]
        
                target_pt_WithNu = torch.clip(torch.nan_to_num(gen_pt_WithNu / jet_pt, nan=0, posinf=0, neginf=0), min=0.3, max=2.0).reshape(-1)
                target_pt        = torch.clip(torch.nan_to_num(gen_pt        / jet_pt, nan=0, posinf=0, neginf=0), min=0.3, max=2.0).reshape(-1, 1)
                truth = (truth[0].type(torch.LongTensor).to(device), target_pt.type(torch.LongTensor).to(device), target_pt_WithNu.type(torch.LongTensor).to(device))
            else:
                truth = truth.type(torch.LongTensor).to(device)
            # print(f"truth={truth}")
            (
                glob,
                cpf,
                npf,
                vtx,
                lt,
                truth,
            ) = attack(
                [
                    feature.float().to(device)
                    for feature in [
                        glob,
                        cpf,
                        npf,
                        vtx,
                        lt,
                        ]
                    ],
                # truth.type(torch.LongTensor).to(device),
                truth,
                self,
                loss_fn,
                # nn.CrossEntropyLoss(reduction="none"),
                # LossFunctionLoader("CrossEntropyLoss", class_weights=None),
                device,
            ) 
        
        return (glob.detach(), cpf.detach(), npf.detach(), vtx.detach(), lt.detach()), truth
        

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

        with np.errstate(divide='ignore', invalid='ignore'):
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