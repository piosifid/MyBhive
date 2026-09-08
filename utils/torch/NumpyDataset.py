from functools import reduce

import numpy as np
import torch
from numpy.lib import recfunctions
from rich.progress import track
from torch.utils.data import IterableDataset

from utils.dataset.structured_arrays import join_struct_arrays


class NumpyDataset(IterableDataset):
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
        self.data_type = data_type
        self.batch_size = batch_size
        self.out_precision = out_precision
        
        if data_type == "validation" or data_type == "inference":
            self.data_type = "test"
            
        self.weighted_sampling = weighted_sampling

        self.process_weights = (
            [config["process_weights"][p] for p in config["processes"]]
            if "process_weights" in config
            else None
        )

        self.device = device
        self.model = model

        self.rec_funs = {}
        for feat in ['global_features', 'cpf_candidates', 'npf_candidates', 'vtx_features', 'lt_candidates']:
            if config.get(feat) != None:
                self.rec_funs[feat] = self._safe_structured_to_3d
            else:
                self.rec_funs[feat] = self._empty_to_3d
            

    def __len__(self):
        return int(self.all_number_of_samples)

    def _safe_structured_to_3d(self, arr, n_rows):
        return (
            recfunctions.structured_to_unstructured(arr)
            .reshape(n_rows, len(arr.dtype.names), -1)
            .transpose(0, 2, 1)
            .reshape(n_rows, -1)
            #.reshape(n_rows, len(arr.dtype.names), -1)
            #.transpose(0, 2, 1)
        )
    
    def _empty_to_3d(self, arr, n_rows):
        """
        Return an empty (n_rows, 0, 0) array when the dtype has no fields.
        """
        return np.empty((n_rows, 0), dtype=self.out_precision)

    def __getitem__(self, index):
        raise NotImplementedError

    def shuffleFileList(self):
        np.random.shuffle(self.files)

    def __iter__(self):
        # Multi-worker support: each worker gets a separate set of files
        # to iterate over to avoid double iterations
        worker_info = torch.utils.data.get_worker_info()
        files_to_read = list(filter(None, self.files))
        if worker_info is not None:
            files_to_read = np.array_split(files_to_read, worker_info.num_workers)[
                worker_info.id
            ]
        leftover_data = None
        
        for file in files_to_read:
            if self.verbose:
                print(f"Loading {file}")
            with np.load(file) as data:
                global_arrs = data["global_features"]
                _truths = data["truth"]
                cpf_arrs = data["cpf_arr"]
                npf_arrs = data["npf_arr"]
                vtx_arrs = data["vtx_arr"]
                lt_arrs = data["lt_arr"]
                weight = data["weight"]
                process = data["process"]
                
                if self.weighted_sampling:
                    random_number = np.random.rand(len(global_arrs))
                    if not (self.process_weights is None):
                        for proc, proc_w in enumerate(self.process_weights):
                            random_number[process == proc] *= proc_w
                    mask = random_number < weight
                else:
                    mask = np.ones(global_arrs.shape, dtype=np.bool8)

                if self.verbose:
                    print(f"Keeping {np.sum(mask)}/{len(mask)} events")

                # truth from all truths to classes
                truths = np.ones(len(_truths))
                # this is not nice at all but here we are...
                for index, (name, flavours) in enumerate(self.model.classes.items()):
                    for flav in flavours:
                        truths[_truths[flav]] = index
                truths = truths[mask]
                processes = process[mask]
                weights = weight[mask]
               
                global_arrs = global_arrs[mask]#[self.model.global_features]
                cpf_arrs = cpf_arrs[mask]#[self.model.cpf_candidates]
                npf_arrs = npf_arrs[mask]#[self.model.npf_candidates]
                vtx_arrs = vtx_arrs[mask]#[self.model.vtx_features]
                lt_arrs = lt_arrs[mask]#[self.model.lt_candidates]
                
                N = len(global_arrs)
                global_arrs = recfunctions.structured_to_unstructured(global_arrs)
                cpf_arrs = self.rec_funs['cpf_candidates'](cpf_arrs, N)
                npf_arrs = self.rec_funs['npf_candidates'](npf_arrs, N)
                vtx_arrs = self.rec_funs['vtx_features'](vtx_arrs, N)
                lt_arrs  = self.rec_funs['lt_candidates'](lt_arrs, N)
                
                arrs = np.concatenate([global_arrs, cpf_arrs, npf_arrs, vtx_arrs, lt_arrs], axis = 1)
                if self.out_precision=='float16':
                    max_f16 = np.finfo(np.float16).max  # ≈ 65504.0
                    arrs = np.clip(arrs, -max_f16, max_f16)
                    arrs = arrs.astype(np.float16)
                    #truths = truths.astype(np.float16)

                if leftover_data is not None:
                    arrs = np.concatenate((leftover_data[0], arrs))
                    truths = np.concatenate((leftover_data[1], truths))
                    weights = np.concatenate((leftover_data[2], weights))
                    processes = np.concatenate((leftover_data[3], processes))
                    leftover_data = None
    
                num_samples = arrs.shape[0]
    
                for batch_start in range(0, num_samples, self.batch_size):
                    batch_end = min(batch_start + self.batch_size, num_samples)
                    if (batch_end - batch_start) == self.batch_size:
                        #print(f"{batch_start}: {arrs[batch_start:batch_end].shape}")
                        yield (
                            arrs[batch_start:batch_end],
                            truths[batch_start:batch_end],
                            weights[batch_start:batch_end],
                            processes[batch_start:batch_end],
                        )
                    else:
                        leftover_data = [
                            arrs[batch_start:batch_end],
                            truths[batch_start:batch_end],
                            weights[batch_start:batch_end],
                            processes[batch_start:batch_end],
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
