import numpy as np
import os
import pandas

from scipy.special import softmax

from tasks.base import BaseTask
from tasks.dataset import DatasetConstructorTask
from tasks.parameter_mixins import (
    AttackDependency,
    DatasetDependency,
    TrainingDependency,
    TestAttackDependency,
    TestDatasetDependency,
)
from tasks.inference import InferenceTask

from utils.plotting.working_point import plot_working_points
from utils.evaluation.working_point import (
    calculate_working_point,
    calculate_efficiency_curve,
)


class WorkingPointTask(
    TestAttackDependency, AttackDependency, TrainingDependency, TestDatasetDependency, DatasetDependency, BaseTask
):
    def requires(self):
        return {
            "inference": InferenceTask.req(self),
            "test_dataset": DatasetConstructorTask.req(
                self,
                dataset_version=self.test_dataset_version,
                filelist=self.test_filelist,
            ),
        }

    def output(self):
        return {
            "working_points": self.local_target("working_point.pdf"),
        }

    def run(self):
        os.makedirs(self.local_path(), exist_ok=True)

        print("loading files")
        predictions = np.load(
            self.input()["inference"]["prediction"].path, allow_pickle=True
        )
        kinematics = np.load(
            self.input()["inference"]["kinematics"].path, allow_pickle=True
        )
        truth = np.load(self.input()["inference"]["truth"].path, allow_pickle=True)
        process = np.load(self.input()["inference"]["process"].path, allow_pickle=True)
        jet_pt = kinematics[..., 0]

        print("starting wp calculation")
        working_points = [0.1, 0.05, 0.01, 0.005, 0.001]
        label = "BvsL - Combined"
        predictions = softmax(predictions, axis=-1)

        # Use entire dataset without process separation or pT filtering
        truth_ = truth.copy()
        output_data_ = predictions.copy()

        b_jets = (truth_ == 0) | (truth_ == 1) | (truth_ == 2)
        c_jets = truth_ == 3
        l_jets = (truth_ == 4) | (truth_ == 5)

        b_pred = output_data_[:, :3].sum(axis=1)
        c_pred = output_data_[:, 3]
        l_pred = output_data_[:, -2:].sum(axis=1)

        bvsl = np.where((b_pred + l_pred) > 0, (b_pred) / (b_pred + l_pred), -1)

        threshold, eff, mistag = calculate_efficiency_curve(
            bvsl[~c_jets], b_jets[~c_jets], n_points=200
        )
        wps = [
            calculate_working_point(threshold, eff, mistag, wp)
            for wp in working_points
        ]

        df = pandas.DataFrame()
        df["mistag rate"] = mistag
        df["b jet efficiency"] = eff
        df["thresholds"] = threshold

        plot_working_points(
            (threshold, mistag),
            wps,
            working_points,
            label,
            out_path=self.output()["working_points"].path,
            x_label="Selection threshold on b-tagger score",
            y_label="Light-flavour misidentification",
            r_label="(13.6 TeV)",
            color="darkorange",
        )

        df.to_csv(
            os.path.join(self.local_path(), "wps_combined.csv"), index=False
        )
