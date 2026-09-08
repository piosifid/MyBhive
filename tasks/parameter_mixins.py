import luigi
import law
from rich.console import Console

from utils.config.config_loader import ConfigLoader

class DatasetDependency(object):
    dataset_version = luigi.Parameter(
        default="dataset_version_01",
        description="Version Tag for dataset to save file with",
    )
    filelist = luigi.Parameter(
        description="txt file with input root files",
        significant=False,
        default="",
    )
    training_precision = luigi.Parameter(
        default="float32",
        description="The precision of data for training. Default: float32",
    )

    def store_parts(self):
        parts = super().store_parts()
        # append dataset-version to path
        parts += (self.dataset_version,)
        return parts


class TestDatasetDependency(object):
    test_dataset_version = luigi.Parameter(
        default="dataset_version_01",
        description="Version Tag for dataset to save file with",
    )
    test_filelist = luigi.Parameter(
        description="txt file with input root files",
        significant=False,
        default="",
    )
    testing_precision = luigi.Parameter(
        default="float32",
        description="The precision of data for testing. Default: float32",
    )

    def store_parts(self):
        parts = super().store_parts()
        # append dataset-version to path
        parts += (self.test_dataset_version,)
        return parts


class TrainingDependency(object):
    training_version = luigi.Parameter(
        default="training_version_01",
        description="Version Tag for training to save file with",
    )
    epochs = luigi.IntParameter(default=1)
    model_name = luigi.Parameter()
    n_threads = luigi.IntParameter(
        default=4, description="Number of threads to use for dataloader. Default: 4"
    )
    batch_size = luigi.IntParameter(default=1024)
    learning_rate = luigi.FloatParameter(default=1e-3)
    optimizer = luigi.Parameter(
        default="AdamW",
        description="The optimizer to minimize loss. Default: AdamW",
    )
    loss_function = luigi.Parameter(
        default="CrossEntropyLoss",
        description="The loss function to be used in training NN. Default: CrossEntropyLogCosh",
    )
    betas = law.CSVParameter(
        cls=luigi.FloatParameter,
        default=(0.95, 0.999),
        description="The comma-separated list of coefficients betas for optimizer (if applicable). Default: (0.95, 0.999)",
    )
    eps = luigi.FloatParameter(
        default=1e-6,
        description="The epsilon to use in optimizer. Default: 1e-6"
    )
    lr_scheduler = luigi.Parameter(
        default="epoch_lin_decay",
        description="The learning rate scheduler. Default: epoch_lin_decay",
    )
    lr_decay_factor = luigi.FloatParameter(
        default=1e-2,
        description="The factor to decrease the learning rate using scheduler. Default: 1e-2"
    )
    mixed_precision = luigi.BoolParameter(
        default=False,
        description="Decides whether to use Automatic Mixed Precision for training PyTorch models. Default: False",
    )
    use_torch_compile = luigi.BoolParameter(
        default=False,
        description="Decides whether to use torch.compile for acceleration of PyTorch training. Default: False",
    )
    torch_compile_mode = luigi.Parameter(
        default="default",
        description="The mode for torch.compile if used. Default: default",
    )
    
    def store_parts(self):
        parts = super().store_parts()

        parts += (self.training_version,)
        parts += (self.model_name,)
        parts += ("epochs_{0:d}".format(self.epochs),)

        return parts

    def set_device(self):
        c = Console()
        from utils.models.models import BTaggingModels
        from torch.nn import Module
        if issubclass(type(BTaggingModels(self.model_name, ConfigLoader.load_config(self.config))), Module):
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                c.print(
                    "[black on yellow]Warning:", "CUDA device available. Running on cuda!"
                )
            else:
                self.device = "cpu"
        else:
            self.device = "cpu"
            c.print(
                "[black on yellow]Warning:", "No CUDA device available. Running on cpu..."
            )


class AttackDependency(object):

    attack = luigi.Parameter(
        default="nominal", description="Specify adversarial attack to use."
    )
    attack_magnitude = luigi.FloatParameter(
        default=0.0,
        description="Only use in combination with attack!=nominal. Set the magnitude for choosen attack.",
    )
    attack_iterations = luigi.IntParameter(
        default=1,
        description="Only use in combination with attack!=None and attack_magnitude!=0. Set the number of interations for choosen attack, if applicable.",
    )
    attack_individual_factors = luigi.BoolParameter(
        default=False,
        description="Decides whether individual attack magnitudes should be used per feature or not.",
    )
    attack_reduce = luigi.BoolParameter(
        default=True,
        description="Decides whether default values and integer values should be changed or not.",
    )
    attack_restrict_impact = luigi.FloatParameter(
        default=-1.0,
        description="Sets a maximal l-inf distance that each feature can be changed as a fraction of the nominal one. -1.0 means no restriction.",
    )
    attack_overshoot = luigi.FloatParameter(
        default=0.02,
        description="Only use in combination with attack==jetfool. Used to prevent vanishing updates.",
    )
    attack_uncertainty = luigi.FloatParameter(
        default=1.0,
        description="Only use in combination with attack==optimizer. Sets the allowed and scaled global uncertainty for input features.",
    )

    def store_parts(self):
        parts = super().store_parts()

        parts += (self.attack,)
        if self.attack_magnitude > 0.0:
            parts += ("epsilon_{}".format(self.attack_magnitude),)
            parts += ("iterations_{}".format(self.attack_iterations),)
        if self.attack == "jetfool":
            parts += ("overshoot_{}".format(self.attack_overshoot),)
        if self.attack == "optimizer":
            parts += ("uncertainty_{}".format(self.attack_uncertainty),)

        return parts


class TestAttackDependency(object):

    test_attack = luigi.Parameter(
        default="nominal", description="Specify adversarial attack to use for testing."
    )
    test_attack_magnitude = luigi.FloatParameter(
        default=0.0,
        description="Only use in combination with attack!=nominal. Set the magnitude for choosen attack for testing.",
    )
    test_attack_iterations = luigi.IntParameter(
        default=1,
        description="Only use in combination with attack!=None and attack_magnitude!=0. Set the number of interations for choosen attack, if applicable, for testing.",
    )
    test_attack_individual_factors = luigi.BoolParameter(
        default=False,
        description="Decides whether individual attack magnitudes should be used per feature or not, for testing.",
    )
    test_attack_reduce = luigi.BoolParameter(
        default=True,
        description="Decides whether default values and integer values should be changed or not, for testing.",
    )
    test_attack_restrict_impact = luigi.FloatParameter(
        default=-1.0,
        description="Sets a maximal l-inf distance that each feature can be changed as a fraction of the nominal one. -1.0 means no restriction, for testing.",
    )
    test_attack_overshoot = luigi.FloatParameter(
        default=0.02,
        description="Only use in combination with attack==jetfool. Used to prevent vanishing updates.",
    )
    test_attack_uncertainty = luigi.FloatParameter(
        default=1.0,
        description="Only use in combination with attack==optimizer. Sets the allowed and scaled global uncertainty for input features.",
    )

    def store_parts(self):
        parts = super().store_parts()

        parts += (f"test_attack_{self.test_attack}",)
        if self.test_attack_magnitude > 0.0:
            parts += ("test_epsilon_{}".format(self.test_attack_magnitude),)
            parts += ("test_iterations_{}".format(self.test_attack_iterations),)
        if self.test_attack == "jetfool":
            parts += ("test_overshoot_{}".format(self.test_attack_overshoot),)
        if self.test_attack == "optimizer":
            parts += ("uncertainty_{}".format(self.test_attack_uncertainty),)

        return parts
