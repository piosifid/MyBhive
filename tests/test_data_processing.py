import os
import subprocess
import unittest
import unittest.result

import numpy as np
from rich.progress import track


class TestDatasetConstructor(unittest.TestCase):
    """
    This test module creates a dataset of chunk_size 100, 1000 and 10000 using the DatasetConstructorTask.
    It then compares the number of samples in the created datasets, the sum, mean and standard deviation of the stored
    jet_pt values,
    If the values all match, the test is passed; it fails otherwise

    The test can be used to check changes in the datasets, if someone tweaks with their construction.
    It is advised to use a subset of samples, to have the three datasets created in a reasonable timeframe.
    The dataset .txt files and config file should all be adjusted to the use-case (see run_dataset_constructor_for_different_chunksizes) function.
    """

    def run_dataset_constructor_for_different_chunksizes(
        self, chunk_sizes, output_prefix
    ):
        for chunk_size in chunk_sizes:
            arguments = [
                "law",
                "run",
                "DatasetConstructorTask",
                "--test-dataset-path",
                "minitest.txt",  # Should be adjusted to use-case
                "--training-dataset-path",
                "minitraining.txt",  # Should be adjusted to use-case
                "--dataset-version",
                f"{output_prefix}_{chunk_size}",
                "--config",
                "hlt_phase2",  # Should be adjusted to use-case
                "--chunk-size",
                str(chunk_size),
            ]
            subprocess.run(arguments)

    def test_data_processing(self):
        chunk_sizes = [100, 1000, 10000]
        output_prefix = "test_miniphase2_chunk_size"
        self.run_dataset_constructor_for_different_chunksizes(
            chunk_sizes=chunk_sizes, output_prefix=output_prefix
        )
        results = np.zeros((len(chunk_sizes), 4))
        for i, chunk_size in enumerate(chunk_sizes):
            output = os.environ["DATA_PATH"]
            path = (
                output
                + f"/DatasetConstructorTask/{output_prefix}_{chunk_size}/processed_files.txt"
            )
            with open(path, "r") as file_list:
                files = file_list.read().split("\n")
                array = np.array([])
                for file in track(
                    files, f"Reading in jet_pt for chunk size {chunk_size}..."
                ):
                    data = np.load(file)
                    array = np.append(array, data["global_features"]["jet_pt"])
                res = np.array(
                    [
                        array.shape[0],
                        np.round(array.sum(), 5),
                        np.round(array.mean(), 5),
                        np.round(array.std(), 5),
                    ]
                )
                results[i] = res
        matching_outputs = np.equal(results[0], results)

        self.assertEqual(
            True,
            matching_outputs.all(),
            f"Error with the dataset constructor. The created 3 datasets from the files have differing shapes, sum of jet_pt, means and standard deviations. The obtained results are: {matching_outputs}\n with arrays {results}",
        )


if __name__ == "__main__":
    unittest.main()
