import numpy as np
import torch
import lz4.frame

from functools import reduce
from numpy.lib import recfunctions
from rich.progress import track
from torch.utils.data import IterableDataset

class LZ4Dataset(IterableDataset):
    def __init__(
        self,
        files,
        model,
        config,
        data_type="training",
        weighted_sampling=False,
        device="cpu",
        process_weights=None,
        verbose=0,
        batch_size = 512,
        out_precision='float32',
        **kwargs
    ):
        self.verbose = verbose
        self.files = files
        self.Nedges = [0]
        self.data_type = data_type
        self.batch_size = batch_size
        self.data_precision = config["data_precision"]
        self.out_precision  = out_precision
        
        if data_type == "validation" or data_type == "inference":
            self.data_type = "test"
        
        print(f'Precision of {data_type} dataset is set to {self.out_precision}')
        #self.all_number_of_samples = all_number_of_samples
        self.weighted_sampling = weighted_sampling
        self.process_weights = process_weights
        self.device = device
        self.model = model
        self.config_truths = list(config['truths'])
        self.num_truth = len(model.classes.keys())
        self.num_ele = len(self.config_truths)
        self.process_weights = (
            [config["process_weights"][p] for p in config["processes"]]
            if "process_weights" in config
            else None
        )

    #def __len__(self):
    #    return int(self.all_number_of_samples)

    def __getitem__(self, index):
        raise NotImplementedError

    def shuffleFileList(self):
        np.random.shuffle(self.files)

    def __iter__(self):
        # Multi-worker support:
        worker_info = torch.utils.data.get_worker_info()
        files_to_read = self.files
        if worker_info is not None:
            files_to_read = np.array_split(files_to_read, worker_info.num_workers)[worker_info.id]

        leftover_data = None
        for file in files_to_read:
            if self.verbose:
                print(f"Loading {file}")

            #Add LZ4 loading
            with lz4.frame.open(file, mode='r') as data:
                output_data = data.read()
            s = np.frombuffer(output_data, dtype=self.data_precision).copy()
            s = s[2:].reshape(-1, int(s[1]), order="C")

            if self.out_precision=='float16':
                max_f16 = np.finfo(np.float16).max  # ≈ 65504.0
                s = np.clip(s, -max_f16, max_f16)
                s = s.astype(np.float16)

            process = s[:, -(self.num_ele+2)]
            if self.weighted_sampling:
                random_number = np.random.rand(s.shape[0])
                if not (self.process_weights is None):
                    for proc, proc_w in enumerate(self.process_weights):
                        random_number[process == proc] *= proc_w
                mask = random_number < s[:, -1]
                s = s[mask]

            s = np.random.permutation(s)

            truths = np.zeros(s.shape[0])
            labels = s[:,-(self.num_ele+1):-1]
            for index, (name, flavours) in enumerate(self.model.classes.items()):
                for flav in flavours:
                    idx = self.config_truths.index(flav)
                    truths[labels[:,idx] == 1] = index
            weights = s[:, -1]
            process = s[:, -(self.num_ele+2)]
            s = s[:,:-(self.num_ele+2)]

            if leftover_data is not None:
                s = np.concatenate((leftover_data[0], s))
                truths = np.concatenate((leftover_data[1], truths))
                weights = np.concatenate((leftover_data[2], weights))
                process = np.concatenate((leftover_data[3], process))
                leftover_data = None

            num_samples = s.shape[0]

            for batch_start in range(0, num_samples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_samples)
                if (batch_end - batch_start) == self.batch_size:
                    yield (
                        s[batch_start:batch_end],
                        truths[batch_start:batch_end],
                        weights[batch_start:batch_end],
                        process[batch_start:batch_end],
                    )
                else:
                    leftover_data = [
                        s[batch_start:batch_end],
                        truths[batch_start:batch_end],
                        weights[batch_start:batch_end],
                        process[batch_start:batch_end],
                    ]
            # last not full batch in case of validation/inference
            if (leftover_data is not None) and (self.data_type == "test"):
                yield (
                    leftover_data[0],
                    leftover_data[1],
                    leftover_data[2],
                    leftover_data[3],
                )
                
        return None

    def get_all_weights(self):
        weights = np.empty((self.Nedges[-1]))
        N = 0
        for file in track(
            self.files, "Reading in the weights for the " + self.data_type + " data"
        ):
            with open(file, "rb") as np_file:
                data = np.load(np_file)
            n_elements = int(data.shape[0])
            weights[N : N + n_elements] = data[:, -3]
            N += n_elements
        return weights
