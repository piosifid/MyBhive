import numpy as np
import torch
from functools import reduce
from numpy.lib import recfunctions
from rich.progress import track
from torch.utils.data import IterableDataset
from utils.dataset.structured_arrays import join_struct_arrays


class PAIReDDataset(IterableDataset):
    #@profile
    def __init__(
        self,
        files,
        model,
        data_type="training",
        weighted_sampling=False,
        device="cpu",
        histogram_training=None,
        max_length=1,
        bins_pt=None,
        bins_eta=None,
        verbose=0,
        batch_size=512,
        **kwargs
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
        self.batch_size = batch_size

        self.device = device
        self.model = model

    #@profile
    def __len__(self):
        return int(self.all_number_of_samples)
    
    #@profile
    def __getitem__(self, index):
        raise NotImplementedError

    #@profile
    def shuffleFileList(self):
        np.random.shuffle(self.files)

    #@profile
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
                    print('Use Weighted Sampling')
                    random_number = np.random.rand(len(data["global_features"]))
                    mask = random_number < data["weight"]
                else:
                    print('### NO weighted sampling ###')
                    mask = np.ones(data["global_features"].shape, dtype=bool) #This is just a workaround to not cut everything away. This needs to be fixed! Normally it works with the if-else statement above and this line does not exist.

                if self.verbose:
                    print('----------- MASK DEBUG --------------')
                    print('Random number: ', random_number)
                    print('Weights: ', data["weight"])
                    print('Mask: ', mask.shape)
                    print('Number of events before weighted sampling: ', len(data["global_features"]))
                    print('Number of events after weighted sampling: ', len(data["global_features"][mask]))
                    print(f"Keeping {np.sum(mask)}/{len(mask)} events")
                    print('-------------------------------------')

                # truth from all truths to classes
                truths = np.ones(len(data["truth"]))
                # truth_un = recfunctions.structured_to_unstructured(data["truth"])
                # flav_count = 0
                # count up all flavours and assign value
                # this is not nice at all but here we are...

                for index, (name, flavours) in enumerate(self.model.classes.items()):
                    for flav in flavours:
                        truths[data["truth"][flav]] = index
                
                truths = truths[mask]
                processes = data["process"][mask]
                weights = data["weight"][mask]

                
                '''
                Create new input features for PAIReDTagger
                '''
                # cpf features
                custom_cpf_features = ["part_pt_log", "part_e_log", "part_logptrel", "part_logerel", "part_deltaR1", "part_deltaR2", "part_d0", "part_dz"]
                dt_cpf = np.dtype([(name, np.float32, (data['cpf_arr']['part_pt']).shape[-1]) for name in custom_cpf_features])
                cpf_features_to_add = np.empty(len(data["cpf_arr"][mask]), dtype=dt_cpf)

                cpf_features_to_add["part_pt_log"] = np.array(
                    np.where(np.isnan(np.log(data['cpf_arr']['part_pt'][mask])), -1e3, np.log(data['cpf_arr']['part_pt'][mask])),
                    dtype=[("part_pt_log", np.float32)],
                )                
                cpf_features_to_add["part_pt_log"] = np.where(cpf_features_to_add["part_pt_log"] == -np.inf, 0, cpf_features_to_add["part_pt_log"])
                
                cpf_features_to_add["part_e_log"] = np.array(
                    np.where(np.isnan(np.log(data['cpf_arr']['part_energy'][mask])), -1e3, np.log(data['cpf_arr']['part_energy'][mask])),
                    dtype=[("part_e_log", np.float32)],
                )
                cpf_features_to_add["part_e_log"] = np.where(cpf_features_to_add["part_e_log"] == -np.inf, 0, cpf_features_to_add["part_e_log"]) #NOTE there are unequal many infs and zeros, thus there are zeros that are not caused by the lengths of 128, that need to be considered

                # jet_pt only has shape (N,) --> create it of shape (N,1) to be able to divide part_pt by it
                jet1_pt = data['global_features']['jet1_pt'][mask].reshape(-1, 1)
                jet2_pt = data['global_features']['jet2_pt'][mask].reshape(-1, 1)
                jet1_energy = data['global_features']['jet1_energy'][mask].reshape(-1, 1)
                jet2_energy = data['global_features']['jet2_energy'][mask].reshape(-1, 1)

                cpf_features_to_add["part_logptrel"] = np.array(
                    np.where(np.isnan(np.log(data['cpf_arr']['part_pt'][mask]/(jet1_pt+jet2_pt))), -1e3, np.log(data['cpf_arr']['part_pt'][mask]/(jet1_pt+jet2_pt))),
                    dtype=[("part_logptrel", np.float32)],
                )
                cpf_features_to_add["part_logerel"] = np.array(
                    np.where(np.isnan(np.log(data['cpf_arr']['part_energy'][mask]/(jet1_energy+jet2_energy))), -1e3, np.log(data['cpf_arr']['part_energy'][mask]/(jet1_energy+jet2_energy))),
                    dtype=[("part_logerel", np.float32)],
                )
                cpf_features_to_add["part_deltaR1"] = np.array(
                    np.hypot(data['cpf_arr']['part_deta1'][mask], data['cpf_arr']['part_dphi1'][mask]), #new variables need to be created here like this!
                    dtype=[("part_deltaR1", np.float32)],
                )
                cpf_features_to_add["part_deltaR2"] = np.array(
                    np.hypot(data['cpf_arr']['part_deta2'][mask], data['cpf_arr']['part_dphi2'][mask]),
                    dtype=[("part_deltaR2", np.float32)],
                )
                cpf_features_to_add["part_d0"] = np.array(
                    np.where(data["cpf_arr"]["part_d0val"][mask] == -1, 0, np.tanh(data["cpf_arr"]["part_d0val"][mask]).astype(np.float32)),
                    dtype=[("part_d0", np.float32)],
                )

                cpf_features_to_add["part_dz"] = np.array(
                    np.where(data["cpf_arr"]["part_dzval"][mask] == -1, 0, np.tanh(data["cpf_arr"]["part_dzval"][mask]).astype(np.float32)),
                    dtype=[("part_dz", np.float32)],
                )
                

                # sv features
                custom_sv_features = ["sv_px", "sv_py", "sv_pz", "sv_energy"]
                dt_sv = np.dtype([(name, np.float32, (data['vtx_arr']['sv_pt']).shape[-1]) for name in custom_sv_features])
                sv_features_to_add = np.empty(len(data["vtx_arr"][mask]), dtype=dt_sv)
                
                sv_features_to_add["sv_px"] = np.array(
                    np.cos(data['vtx_arr']['sv_phi'][mask]) * data['vtx_arr']['sv_pt'][mask],
                    dtype=[("sv_px", np.float32)],
                )
                sv_features_to_add["sv_py"] = np.array(
                    np.sin(data['vtx_arr']['sv_phi'][mask]) * data['vtx_arr']['sv_pt'][mask],
                    dtype=[("sv_py", np.float32)],
                )
                sv_features_to_add["sv_pz"] = np.array(
                    np.sinh(data['vtx_arr']['sv_eta'][mask]) * data['vtx_arr']['sv_pt'][mask],
                    dtype=[("sv_pz", np.float32)],
                )
                sv_features_to_add["sv_energy"] = np.array(
                    np.sqrt(data['vtx_arr']['sv_mass'][mask]**2 + (data['vtx_arr']['sv_pt'][mask]*np.cosh(data['vtx_arr']['sv_eta'][mask]))**2),
                    dtype=[("sv_energy", np.float32)],
                )


                # create vectors
                cpf_vectors = np.array(
                    [data["cpf_arr"][coordinate][mask] for coordinate in self.model.cpf_vectors]
                )
                sv_vectors = np.array(
                    [sv_features_to_add[coordinate] for coordinate in self.model.sv_vectors]  # need to use vtx_arrs, since in data["vtx_arr"] the new variables are not included
                )
                
                #swap axes to get (length, candidates, 4 (px, py, pz, energy))
                cpf_vectors = np.swapaxes(cpf_vectors, 0, 1)
                cpf_vectors = np.swapaxes(cpf_vectors, 1, 2)
                sv_vectors = np.swapaxes(sv_vectors, 0, 1)
                sv_vectors = np.swapaxes(sv_vectors, 1, 2)
                
                
                '''
                Create dedicated structured arrays for each input type
                '''
                global_arrs = data["global_features"][mask][self.model.global_features]
                cpf_arrs = data["cpf_arr"][mask]
                vtx_arrs = data["vtx_arr"][mask]

                
                '''
                Add new created features to the structured arrays
                '''
                cpf_arrs = join_struct_arrays(cpf_arrs, cpf_features_to_add)
                #vtx_arrs = join_struct_arrays(vtx_arrs, sv_features_to_add) # must not join this, otherwise sv vector features will be in features

                """
                only keep fields that are part of the model; sort it according to model input
                """
                cpf_arrs = cpf_arrs[self.model.cpf_candidates]
                vtx_arrs = vtx_arrs[self.model.sv_features]

                '''
                Transformations on part_pt_log, part_e_log, part_logptrel, part_logerel, part_deltaR1, part_deltaR2, part_d0err, part_dzerr
                '''
                # default: clip between -5, 5
                #cpf_arrs['part_pt_log'] = np.clip(cpf_arrs['part_pt_log'], -10, 10)
                #cpf_arrs['part_e_log'] = np.clip(cpf_arrs['part_e_log'], -10, 10)
                cpf_arrs['part_logptrel'] = np.clip(cpf_arrs['part_logptrel'], -10, 10) #should not affect anything different than -inf
                cpf_arrs['part_logerel'] = np.clip(cpf_arrs['part_logerel'], -10, 10)
                cpf_arrs['part_dzerr'] = np.clip(cpf_arrs['part_dzerr'], 0, 1) #is not normalized in weaver either before clipping
                cpf_arrs['part_d0err'] = np.clip(cpf_arrs['part_d0err'], 0, 1)
                                
                
                
                N = len(cpf_arrs)
                #ab hier: keine structured arrays mehr, also zuvor variablen über branch namen erstellen
                #global_arrs = recfunctions.structured_to_unstructured(global_arrs)
                # reshape arrays in (length, candidates, features)
                
                cpf_arrs = recfunctions.structured_to_unstructured(cpf_arrs).reshape(
                    N, len(cpf_arrs.dtype.names), -1 #hier iwie -1 und len tauschen und dann transposen?
                ).transpose(0, 2, 1)
                
                vtx_arrs = recfunctions.structured_to_unstructured(vtx_arrs).reshape(
                    N, len(vtx_arrs.dtype.names), -1
                ).transpose(0, 2, 1)
                
                
                #here the features are given batchwise to the model
                for (
                    #global_arr,
                    cpf_arr,
                    vtx_arr,
                    cpf_vector,
                    sv_vector,
                    truth,
                    weight,
                    process,
                ) in zip(
                    #global_arrs,
                    cpf_arrs,
                    vtx_arrs,
                    cpf_vectors,
                    sv_vectors,
                    truths,
                    weights,
                    processes,
                ):
                    # trim down to number of candidates
                    cpf_arr = cpf_arr[: self.model.n_cpf]
                    vtx_arr = vtx_arr[: self.model.n_sv]
                    cpf_vector = cpf_vector[: self.model.n_cpf]
                    sv_vector = sv_vector[: self.model.n_sv]
                    
                    yield cpf_arr, vtx_arr, cpf_vector, sv_vector, truth, weight, process
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
