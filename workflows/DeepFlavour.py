import law
import luigi

from tasks.plotting import ROCCurveTask
from tasks.working_point import WorkingPointTask
from rich.console import Console

c = Console()


class DeepJetRunHLT(law.WrapperTask):
    """
    This runs a full DeepJet Training on the specified files
    with a specified version
    """

    version = luigi.Parameter()
    # Training
    filelist = luigi.Parameter(
        description="txt file with input root files for training."
    )
    test_filelist = luigi.Parameter(
        description="txt file with input root files for testing."
    )

    def requires(self):
        kwargs = {
            "training_version": self.version,
            "dataset_version": self.version,
            "test_dataset_version": f"testfiles_{self.version}",
            "model_name": "DeepJetHLT",
            "epochs": 20,
            "config": "hlt_run3",
        }
        return [
            ROCCurveTask.req(self, **kwargs),
            WorkingPointTask.req(self, **kwargs),
        ]


class DeepJetRun(DeepJetRunHLT):
    """
    This runs a full DeepJet Training on the specified files
    with a specified version offline
    """

    def requires(self):
        kwargs = {
            "training_version": self.version,
            "dataset_version": self.version,
            "test_dataset_version": f"testfiles_{self.version}",
            "model_name": "DeepJet",
            "epochs": 3,
            "config": "offline_run3",
        }
        return [
            ROCCurveTask.req(self, **kwargs),
            WorkingPointTask.req(self, **kwargs),
        ]

class DeepJetRunHLTPhase2(DeepJetRunHLT):
    """
    This runs a full DeepJet Training on the specified files
    with a specified version offline
    """

    def requires(self):
        kwargs = {
            "training_version": self.version,
            "dataset_version": self.version,
            "test_dataset_version": f"testfiles_{self.version}",
            "model_name": "DeepJetHLT",
            "epochs": 300,
            "config": "hlt_phase2",
        }
        return [
            ROCCurveTask.req(self, **kwargs),
            WorkingPointTask.req(self, **kwargs),
        ]


class ParticleNetRunHLT(DeepJetRunHLT):
    """
    This runs a full ParticleNet Training on the specified files
    with a specified version online
    """

    def requires(self):
        kwargs = {
            "training_version": self.version,
            "dataset_version": self.version,
            "test_dataset_version": f"testfiles_{self.version}",
            "model_name": "ParticleNet",
            "epochs": 3,
            "config": "hlt_run3_pnet",
        }
        return [
            ROCCurveTask.req(self, **kwargs),
        ]
