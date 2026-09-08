import law
import luigi
import os
import json
import torch
import lz4.frame
from torch.utils.data import DataLoader
from pathlib import Path

#debug
import psutil
import time

from tasks.base import BaseTask
from tasks.dataset import DatasetConstructorTask
from tasks.parameter_mixins import (
    AttackDependency,
    DatasetDependency,
    TrainingDependency,
)
from utils.config.config_loader import ConfigLoader
from utils.models.models import BTaggingModels
from utils.loss.LossFunctionLoader import LossFunctionLoader
from utils.plotting.roc import plot_roc_list, plot_losses
from utils.weighting.batches import compute_class_weights, expected_batches
from torch.nn import Module

def import_libraries():
    from torch.utils.data import DataLoader
    import torch
    import numpy as np

    from utils.adversarial_attacks.pick_attack import pick_attack
    from utils.plotting.roc import plot_roc_list, plot_losses, plot_accuracy
    from utils.optimizing.SchedulerLoader import SchedulerLoader
    from utils.optimizing.OptimizerLoader import OptimizerLoader
    from utils.torch.DatasetLoader import DatasetLoader
    from torch.nn import Module
    
    return DataLoader, torch, np, pick_attack, plot_roc_list, plot_losses, plot_accuracy, SchedulerLoader, OptimizerLoader, DatasetLoader, Module

law.contrib.load("numpy")

def check_memory_usage():
    memory_usage = psutil.virtual_memory().used / (1024.0**3)
    return memory_usage

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def check_resume(base_path, model_prefix="model_", model_suffix=".pt", load_epoch=None):
    models = {}
    for p in Path(base_path).glob(f"{model_prefix}[0-9]*{model_suffix}"):
        path_name = str(p)
        name = path_name.split("/")[-1]
        epoch = int(name.replace(model_prefix, "").replace(model_suffix, ""))
        models[epoch] = path_name
    if len(models.values()) == 0:
        raise FileNotFoundError
    else:
        if not load_epoch:
            max_epoch = max(models)
            return models[max_epoch], max_epoch + 1, models
        else:
            return models[load_epoch - 1], load_epoch, models


def load_resume_training(model, path, device, output, optimizer=None, scheduler=None, epoch=None):
    try:
        model_path, ran_epochs, models = check_resume(path, load_epoch=epoch)
        _model = torch.load(
            model_path,
            map_location=torch.device(device),
        )
        model.load_state_dict(_model["model_state_dict"])
        if not (optimizer is None):
            optimizer.load_state_dict(_model["optimizer_state_dict"])
        if not (scheduler is None):
            scheduler.load_state_dict(_model["scheduler_state_dict"])
        print(f"Resuming on epoch {ran_epochs}:\n{model_path}")
    except FileNotFoundError:
        print("No training to resume found. Starting a new one")
        ran_epochs = 0
        best_loss_val = float('inf')
    
    train_metrics = {"loss": [], "acc": []}
    validation_metrics = {"loss": [], "acc": []}
    
    if ran_epochs != 0:
        try:
            train_metrics = output["training_metrics"].load()
            train_metrics = {key: value.tolist() for key, value in train_metrics.items()}
            validation_metrics = output["validation_metrics"].load()
            validation_metrics = {key: value.tolist() for key, value in validation_metrics.items()}
        except FileNotFoundError:
            print("Training and validation metrics were not saved correctly during previous run. Recovering metrics from .pt files.")
            for epoch in range(ran_epochs):
                checkpoint = torch.load(models[epoch])
                train_metrics['loss'].append(checkpoint["loss_train"])
                train_metrics['acc'].append(checkpoint["acc_train"])
                validation_metrics['loss'].append(checkpoint["loss_val"])
                validation_metrics['acc'].append(checkpoint["acc_val"])
        best_loss_val = min(validation_metrics["loss"])        
                
    return model, optimizer, scheduler, ran_epochs, train_metrics, validation_metrics, best_loss_val


class TrainingTask(AttackDependency, TrainingDependency, DatasetDependency, BaseTask):
    loss_weighting = luigi.BoolParameter(
        False,
        description="Whether to weight the loss or use weighted sampling from the dataset (default).",
    )

    resume_training = luigi.BoolParameter(
        False,
        description="Whether to resume the training if it already ran partially and failed. Set this to true if you want to resume.",
    )
    
    resume_epoch = luigi.IntParameter(
        False,
        description="Whether to resume the training from a specific epoch.",
    )

    extend_training = luigi.IntParameter(
        0,
        description="Number of epochs to extend a training.",
    )

    train_val_split = luigi.FloatParameter(
        default=0.85,
        description="The ratio to divide dataset for train/validation. Default: 0.85"
    )

    n_train_files = luigi.IntParameter(
        default=-1,
        description="The number of train files to use. Default: -1 (all)"
    )
    
    def requires(self):
        return DatasetConstructorTask.req(self)

    def output(self):
        return {
            "training_metrics": self.local_target("training_metrics.npz"),
            "validation_metrics": self.local_target("validation_metrics.npz"),
            # "model": (
            #     self.local_target(f"model_{self.epochs-1 + self.extend_training}.pt")
            #     if issubclass(
            #         type(
            #             BTaggingModels(
            #                 self.model_name, ConfigLoader.load_config(self.config)
            #             )
            #         ),
            #         Module,
            #     )
            #     else self.local_target(f"model_{self.epochs-1}.keras")
            # ),
            "best_model": (
                self.local_target("best_model.pt")
                if issubclass(
                    type(
                        BTaggingModels(
                            self.model_name, ConfigLoader.load_config(self.config)
                        )
                    ),
                    Module,
                )
                else self.local_target(f"best_model.keras")
            ),
        }

    def run(self):
        DataLoader, torch, np, pick_attack, plot_roc_list, plot_losses, plot_accuracy, SchedulerLoader, OptimizerLoader, DatasetLoader, Module = import_libraries()
        
        self.set_device()
        
        # Load config
        config = ConfigLoader.load_config(self.config)

        # Load files
        os.makedirs(self.local_path(), exist_ok=True)
        print("Loading Dataset")
        files = self.input()["file_list"].load().split("\n")
        if not os.path.exists(files[-1]):
            files = files[:-1]

        # Split into train/val sets
        assert (self.train_val_split <= 1) and (self.train_val_split >= 0), "train_val_split should be between 0 and 1!"
        n_train = max((1, int(len(files) * self.train_val_split)))  # has at least one training file
        n_val   = len(files) - n_train
        validation_files = files[-n_val:]
        if self.n_train_files != -1:
            n_train = self.n_train_files
            print(f'Restricted to {self.n_train_files} from {n_train} training files manualy!')
        training_files = files[:n_train]
        
        if n_train == len(files):
            print("\nWARNING!")
            print("No validation files found. Please check your dataset. Most likely you only have one file!")
            print("Using the training file for validation\n")
        if not (isinstance(training_files, list)):
            training_files = [training_files]
        if not (isinstance(validation_files, list)):
            validation_files = [validation_files]
        print(f"#Train files: {len(training_files)}")
        print(f"#Val files: {len(validation_files)}")

        # Model Defintion
        if issubclass(type(model := BTaggingModels(self.model_name, config)), torch.nn.Module):
            model = model.to(self.device)
            model.mixed_precision = self.mixed_precision
            model.use_torch_compile = self.use_torch_compile
            optimizer = OptimizerLoader(
                self.optimizer, 
                self.learning_rate, 
                model.parameters(),
                betas = self.betas, 
                eps=self.eps,
            )
        else:
            optimizer = model.optimizer

        # Reweighing
        with open(self.input()["weights"].path, "r") as f:
            metadata = json.load(f)
        chunk_sizes  = np.array(metadata["chunk_size"])
        sum_weights  = np.array(metadata["sum_weight"])
        
        # back-compatibility
        if len(chunk_sizes) == 1:
            chunk_sizes = np.ones(len(training_files))*chunk_sizes
        print(f"chunk_sizes: {chunk_sizes}")
        
        if self.loss_weighting:
            print("Using loss weighting")
            class_weights = compute_class_weights(
                classes = model.classes,
                truths  = config["truths"],
                histogram_path = self.input()["histogram"].path,
                dtype = np.float32,#self.training_precision,
                device = self.device,
            )
            expected_train_number_of_batches, N_train_events = expected_batches(chunk_sizes[:n_train], self.batch_size, self.n_threads, mode='training')
            expected_val_number_of_batches, N_val_events     = expected_batches(chunk_sizes[-n_val:],  self.batch_size, self.n_threads, mode='validation')
            rw = ""
    
        else:
            print("Using weighted sampling")
            class_weights = None
            expected_train_number_of_batches, N_train_events = expected_batches(sum_weights[:n_train], self.batch_size, self.n_threads, mode='training')
            expected_val_number_of_batches, N_val_events     = expected_batches(sum_weights[-n_val:],  self.batch_size, self.n_threads, mode='validation')
            rw = ' (after reweighing)'
        print(f'Estimated number of jets/events in training{rw}:   {N_train_events:,}')
        print(f'Estimated number of jets/events in validation{rw}: {N_val_events:,}')

        if expected_train_number_of_batches == 0:
            raise ValueError("The training DataLoader is empty. Ensure that you have enough data to form at least one batch.")
        
        # Pick attack
        print(rf"Applying {self.attack} attack (attack_magnitude={self.attack_magnitude}, n_iterations={self.attack_iterations}, attack_uncertainty={self.attack_uncertainty}).")
        epsilon_dir = self.input()["file_list"].path.strip("processed_files.txt") + "epsilons/"

        feature_keys = []
        for name_fts in ['global_features', 'cpf_candidates', 'npf_candidates', 'vtx_features']:
            if hasattr(model, name_fts):
                feature_keys.append(getattr(model, name_fts))
            else:
                feature_keys.append(config[name_fts] if name_fts in config.keys() else [])
             
        attack = pick_attack(
            attack=self.attack,
            device=self.device,
            input_keys=feature_keys,
            integer_positions=model.integers,
            default_values=model.defaults,
            epsilon=self.attack_magnitude,
            epsilon_factors=self.attack_individual_factors,
            iterations=self.attack_iterations,
            reduce=self.attack_reduce,
            restrict_impact=self.attack_restrict_impact,
            number_classes=len(model.classes),
            overshoot=self.attack_overshoot,
            epsilon_dir=epsilon_dir,
            store_path=self.local_path(""),
            uncertainty=self.attack_uncertainty,
        )
        #attack = torch.compile(attack, mode="max-autotune")
        
        print("Dataset construction")
        datasetClass = DatasetLoader(config["dataset"])
        
        # Define the training and validation datasets
        training_data = datasetClass(
            files=training_files,
            model=model,
            config=config,
            data_type="training",
            weighted_sampling=not (self.loss_weighting),
            device=self.device,
            verbose=self.verbose,
            batch_size=self.batch_size,
            out_precision=self.training_precision,
        )
        
        validation_data = datasetClass(
            files=validation_files,
            model=model,
            config=config,
            data_type="validation",
            weighted_sampling=not (self.loss_weighting),
            device=self.device,
            verbose=self.verbose,
            batch_size=self.batch_size,
            out_precision=self.training_precision,
        )

        # Define the corresponding dataloaders
        training_dataloader = DataLoader(
            training_data,
            batch_size = None,
            drop_last=False,
            pin_memory=True,  # Pin Memory for faster CPU/GPU memory load
            num_workers=self.n_threads,
        )
        # Expected number of iterations
        training_dataloader.nits_expected = expected_train_number_of_batches

        validation_dataloader = DataLoader(
            validation_data,
            batch_size = None,
            drop_last=False,
            pin_memory=True,
            num_workers=self.n_threads,
        )
        validation_dataloader.nits_expected = expected_val_number_of_batches
        
        # The learning rate scheduler
        scheduler, batch_lr =  SchedulerLoader(
            self.lr_scheduler, 
            self.learning_rate,
            self.lr_decay_factor,
            optimizer, 
            self.epochs, 
            dataloader = training_dataloader
        )
        loss_function = LossFunctionLoader(self.loss_function, class_weights=class_weights)
        
        print(f"Model:     {self.model_name}")
        print(f'Optimizer: {self.optimizer}')
        print(f'Scheduler: {self.lr_scheduler}')
        
        if self.resume_training or self.resume_epoch or self.extend_training:
            model, optimizer, scheduler, ran_epochs, train_metrics, validation_metrics, best_loss_val = load_resume_training(
                model,
                self.local_path(),
                self.device,
                self.output(),
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=self.resume_epoch,
            )
        else:
            ran_epochs = 0
            train_metrics = {"loss": [], "acc": []}
            validation_metrics = {"loss": [], "acc": []}
            best_loss_val = np.inf

        
        # TO BE IMPLEMENTED
        #print(summary(model, input_size=model.input_dim_torchinfo, device = self.device))
        
        print(f'Total number of parameters: {count_parameters(model):,}')
        
        # Training
        print("Start training on " + self.device)
        train_metrics, validation_metrics = model.train_model(
            training_dataloader,
            validation_dataloader,
            self.local_path(),
            loss_function,
            device=self.device,
            attack=attack,
            optimizer=optimizer,
            scheduler=scheduler,
            batch_lr=batch_lr,
            best_loss_val=best_loss_val,
            nepochs=self.epochs,
            resume_epochs=ran_epochs,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            terminal_plot = self.terminal_plot,
            torch_compile_mode = self.torch_compile_mode,
            #attack_magnitude=self.attack_magnitude,
            #attack_iterations=self.attack_iterations,
        )
        
        if 'loss' in train_metrics.keys():
            plot_losses(
                train_metrics['loss'], 
                validation_metrics['loss'], 
                self.local_path(), 
                ran_epochs, 
                self.epochs
            )
        if 'acc' in train_metrics.keys():
            plot_accuracy(
                train_metrics['acc'], 
                validation_metrics['acc'], 
                self.local_path(), 
                ran_epochs, 
                self.epochs
            )
        print("Training finished.")