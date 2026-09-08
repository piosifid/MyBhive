from rich.console import Console
import luigi
import torch
import law
import os


c = Console()


class BaseTask(law.Task):
    # now done with self.set_device()
    """
    if torch.cuda.is_available():
        device = "cuda"
        c.print(
            "[black on yellow]Warning:", "CUDA device available. Running on cuda!"
        )
        # device = 'cpu'
    else:
        device = "cpu"
        c.print(
            "[black on yellow]Warning:", "No CUDA device available. Running on cpu..."
        )
    """

    debug = luigi.BoolParameter(
        default=False,
        description="Debug Flag to test things. Functionality needs to be implemented for each task.",
    )
    terminal_plot = luigi.BoolParameter(
            default=False,
            description="Display terminal ROC curve. True or False",
    )
    config = luigi.Parameter(
        default="default",
        description="Config to use. These are sepcified in the config directory as .yml files.",
        significant=True,
    )
    verbose = luigi.BoolParameter(
        default=False, description="Verbosity, True or False."
    )
    seed = luigi.IntParameter(default=123456, description="Random Seed to use.")

    def local_path(self, *path):
        parts = [str(p) for p in self.store_parts() + path]
        # DATA_PATH is defined in setup.sh
        return os.path.join(os.environ["DATA_PATH"], *parts)

    def local_target(self, *path, **kwargs):
        return law.LocalFileTarget(self.local_path(*path), **kwargs)

    def store_parts(self):
        """
        This function parses arguments into a path
        """
        parts = (self.__class__.__name__,)
        parts += (self.config,)
        if self.debug:
            parts += ("debug",)
        return parts
