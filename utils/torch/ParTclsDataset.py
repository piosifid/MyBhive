import numpy as np
import torch

from functools import reduce
from numpy.lib import recfunctions
from rich.progress import track
from torch.utils.data import IterableDataset

from utils.dataset.structured_arrays import join_struct_arrays


class ParTclsDataset(IterableDataset):
    def __init__(
        self,
        files,
        model,
        data_type="training",
        weighted_sampling=False,
        device="cpu",
        histogram_training=None,
        max_length=1,
        process_weights=None,
        bins_pt=None,
        bins_eta=None,
        verbose=0,
    ):
        self.verbose = verbose
        self.files = files
        if (bins_pt is None) or (bins_eta is None):
            raise ValueError("You need to specify bins!")
        self.bins_pt = bins_pt
        self.bins_eta = bins_eta
        self.Nedges = [0]
        self.data_type = data_type
        if data_type == "validation":
            self.data_type = "test"
        if histogram_training is not None:
            self.all_number_of_samples = histogram_training.sum()
        else:
            if len(files):
                f = np.load(files[0])
                self.all_number_of_samples = len(f[f.files[0]])
            else:
                self.all_number_of_samples = 0
        self.weighted_sampling = weighted_sampling

        self.process_weights = process_weights

        self.device = device
        self.model = model

    def __len__(self):
        return int(self.all_number_of_samples)

    def __getitem__(self, index):
        raise NotImplementedError

    def shuffleFileList(self):
        np.random.shuffle(self.files)

    def __iter__(self):
        # Multi-worker support: each worker gets a separate set of files
        # to iterate over to avoid double iterations
        worker_info = torch.utils.data.get_worker_info()
        files_to_read = self.files
        if worker_info is not None:
            files_to_read = np.array_split(files_to_read, worker_info.num_workers)[
                worker_info.id
            ]

        for file in files_to_read:
            if self.verbose:
                print(f"Loading {file}")
            with np.load(file) as data:
                if self.weighted_sampling:
                    random_number = np.random.rand(len(data["global_features"]))
                    if not (self.process_weights is None):
                        for proc, proc_w in enumerate(self.process_weights):
                            random_number[data["process"] == proc] *= proc_w
                    mask = random_number < data["weight"]
                else:
                    mask = np.ones(data["global_features"].shape, dtype=bool)

                if self.verbose:
                    print(f"Keeping {np.sum(mask)}/{len(mask)} events")

                # truth from all truths to classes
                truths = np.ones(len(data["truth"]))
                # this is not nice at all but here we are...
                for index, (name, flavours) in enumerate(self.model.classes.items()):
                    for flav in flavours:
                        truths[data["truth"][flav]] = index
                truths = truths[mask]
                processes = data["process"][mask]
                weights = data["weight"][mask]
                
                """
                Create vector arrays
                """
                cpf_vectors = np.array(
                    [data["cpf_arr"][coordinate][mask] for coordinate in self.model.cpf_vectors]
                )
                npf_vectors = np.array(
                    [data["npf_arr"][coordinate][mask] for coordinate in self.model.npf_vectors]
                )
                vtx_vectors = np.array(
                    [data["vtx_arr"][coordinate][mask] for coordinate in self.model.vtx_vectors]
                )
                
                #swap axes to get (length, candidates, 4 (px, py, pz, energy))
                cpf_vectors = np.swapaxes(cpf_vectors, 0, 1)
                cpf_vectors = np.swapaxes(cpf_vectors, 1, 2)
                npf_vectors = np.swapaxes(npf_vectors, 0, 1)
                npf_vectors = np.swapaxes(npf_vectors, 1, 2)
                vtx_vectors = np.swapaxes(vtx_vectors, 0, 1)
                vtx_vectors = np.swapaxes(vtx_vectors, 1, 2)

                # print(' +++ Shape of the vectors +++')
                # print("cpf_vectors", cpf_vectors.shape)
                # print("npf_vectors", npf_vectors.shape)
                # print("vtx_vectors", vtx_vectors.shape)
                # print(' +++ Shape of the global features +++')
                # print("global_features", data["global_features"][mask][self.model.global_features][0].shape)
                # print('############################')
                
                """
                select only necessary branches
                """
                global_arrs = data["global_features"][mask][self.model.global_features]
                cpf_arrs = data["cpf_arr"][mask][self.model.cpf_candidates]
                npf_arrs = data["npf_arr"][mask][self.model.npf_candidates]
                vtx_arrs = data["vtx_arr"][mask][self.model.vtx_features]
                
                N = len(global_arrs)
                global_arrs = recfunctions.structured_to_unstructured(global_arrs)
                # reshape arrays in (length, candidates, features)
                cpf_arrs = (
                    recfunctions.structured_to_unstructured(cpf_arrs)
                    .reshape(N, len(cpf_arrs.dtype.names), -1)
                    .transpose(0, 2, 1)
                )
                npf_arrs = (
                    recfunctions.structured_to_unstructured(npf_arrs)
                    .reshape(N, len(npf_arrs.dtype.names), -1)
                    .transpose(0, 2, 1)
                )
                vtx_arrs = (
                    recfunctions.structured_to_unstructured(vtx_arrs)
                    .reshape(N, len(vtx_arrs.dtype.names), -1)
                    .transpose(0, 2, 1)
                )
                for (
                    global_arr,
                    cpf_arr,
                    npf_arr,
                    vtx_arr,
                    cpf_vector,
                    npf_vector,
                    vtx_vector,
                    truth,
                    weight,
                    process,
                ) in zip(
                    global_arrs,
                    cpf_arrs,
                    npf_arrs,
                    vtx_arrs,
                    cpf_vectors,
                    npf_vectors,
                    vtx_vectors,
                    truths,
                    weights,
                    processes,
                ):
                    # trim down to number of candidates
                    cpf_arr = cpf_arr[: self.model.n_cpf]
                    npf_arr = npf_arr[: self.model.n_npf]
                    vtx_arr = vtx_arr[: self.model.n_vtx]
                    cpf_vector = cpf_vector[: self.model.n_cpf]
                    npf_vector = npf_vector[: self.model.n_npf]
                    vtx_vector = vtx_vector[: self.model.n_vtx]
                    
                    yield global_arr, cpf_arr, npf_arr, vtx_arr, cpf_vector, npf_vector, vtx_vector, truth, weight, process
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
