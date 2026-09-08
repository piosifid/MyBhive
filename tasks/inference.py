from tasks.base import BaseTask
from tasks.dataset import DatasetConstructorTask
from tasks.training import TrainingTask
from tasks.parameter_mixins import (
    AttackDependency,
    DatasetDependency,
    TestAttackDependency,
    TestDatasetDependency,
    TrainingDependency,
)

from utils.adversarial_attacks.pick_attack import pick_attack
from utils.config.config_loader import ConfigLoader
from utils.models.models import BTaggingModels
from utils.plotting.termplot import terminal_roc
from utils.torch.DatasetLoader import DatasetLoader
from utils.loss.LossFunctionLoader import LossFunctionLoader
from utils.weighting.batches import expected_batches

import os
import lz4.frame
import law
import json

def import_libraries():
    from torch.utils.data import DataLoader
    import torch
    import numpy as np

    return DataLoader, torch, np 


import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="You are using `torch.load` with `weights_only=False`.*",
)

# to make formatters work
law.contrib.load("numpy")

class InferenceTask(
    TestAttackDependency,
    AttackDependency,
    TrainingDependency,
    TestDatasetDependency,
    DatasetDependency,
    BaseTask,
):
    def requires(self):
        return {
            "training": TrainingTask.req(
                self
            ),  # this is to make cli-steering with different attack possible
            "test_dataset": DatasetConstructorTask.req(
                self,
                dataset_version=self.test_dataset_version,
                filelist=self.test_filelist,
            ),
        }

    def output(self):
        return {
            "prediction": self.local_target("prediction.npy"),
            "process": self.local_target("process.npy"),
            "truth": self.local_target("truth.npy"),
            "kinematics": self.local_target("kinematics.npy"),
            "inference_time": self.local_target("inference_time.npy"),
        }

    def run(self):
        DataLoader, torch, np = import_libraries()
        self.set_device()
        # create directory
        self.output()["prediction"].parent.touch()
        config = ConfigLoader.load_config(self.config)

        # Model Defintion
        print("Build Model")
        print(self.model_name)
        if issubclass(type(model := BTaggingModels(self.model_name, config)), torch.nn.Module):
            model = model.to(self.device)
            model.create_integers_defaults()
            model.create_feature_shapes()
            model.mixed_precision = self.mixed_precision
            model.use_torch_compile = self.use_torch_compile
            best_model = torch.load(
                self.input()["training"]["best_model"].path,
                map_location=torch.device(self.device),
                weights_only=False,
            )
            model.load_state_dict(best_model["model_state_dict"])
        else:
            model.model = keras.models.load_model(
                self.input()["training"]["best_model"].path,
                custom_objects=model.custom_objects,
            )

        # Picking attack
        print(
            rf"Applying {self.test_attack} attack (attack_magnitude={self.test_attack_magnitude}, n_iterations={self.test_attack_iterations}, attack_uncertainty={self.test_attack_uncertainty})."
        )
        epsilon_dir = (
            self.input()["test_dataset"]["file_list"].path.strip("processed_files.txt")
            + "epsilons/"
        )

        feature_keys = [
            config[name] if name in config.keys() else [] 
            for name in ['global_features', 'cpf_candidates', 'npf_candidates', 'vtx_features']
        ]
        attack = pick_attack(
            attack=self.test_attack,
            device=self.device,
            input_keys=feature_keys,
            integer_positions=model.integers,
            default_values=model.defaults,
            epsilon=self.test_attack_magnitude,
            epsilon_factors=self.test_attack_individual_factors,
            iterations=self.test_attack_iterations,
            reduce=self.test_attack_reduce,
            restrict_impact=self.test_attack_restrict_impact,
            number_classes=len(model.classes),
            overshoot=self.test_attack_overshoot,
            epsilon_dir=epsilon_dir,
            store_path=self.local_path(""),
            uncertainty=self.test_attack_uncertainty,
        )
        #attack = torch.compile(attack, mode="max-autotune")

        print("Loading Dataset")
        files = self.input()["test_dataset"]["file_list"].load().split("\n")

        # protection if the last line in processed_files.txt is empty
        if not os.path.exists(files[-1]):
            files = files[:-1]

        print("Initialize datasets")
        datasetClass = DatasetLoader(config["dataset"])
        weights_file = self.input()["test_dataset"]["weights"].path
            
        with open(weights_file, "r") as f:
            metadata = json.load(f)
        
        chunk_sizes = np.array(metadata["chunk_size"])
        if len(chunk_sizes) == 1:
            chunk_sizes = np.ones(len(files))*chunk_sizes
        
        expected_test_number_of_batches, N_inf_events = expected_batches(chunk_sizes, self.batch_size, self.n_threads, mode='test')
        test_data = datasetClass(
            files=files,
            model=model,
            config=config,
            data_type="inference",
            verbose=self.verbose,
            batch_size=self.batch_size,
            out_precision=self.testing_precision,
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size = None, #self.batch_size if config["dataset"] not in {"LZ4FP16Dataset", "LZ4Dataset", "LZ4PAIReDDataset"} else None,
            num_workers=self.n_threads,
            pin_memory=True,  # Pin Memory for faster CPU/GPU memory load
        )
        test_dataloader.nits_expected = expected_test_number_of_batches

        loss_function = LossFunctionLoader(self.loss_function)
        
        print("Start inference on", self.device)

        predictions, truths, kinematics, processes, inference_time = model.predict_model(
            test_dataloader, self.output(), loss_function, self.device, attack=attack
        )
        
        if(self.terminal_plot):
            terminal_roc(predictions, truths, title="Inference ROC")