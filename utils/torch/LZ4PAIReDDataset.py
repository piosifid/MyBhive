import numpy as np
import torch
from functools import reduce
from numpy.lib import recfunctions
from rich.progress import track
from torch.utils.data import IterableDataset
from utils.dataset.structured_arrays import join_struct_arrays

import lz4.frame


class LZ4PAIReDDataset(IterableDataset):
    def __init__(
        self,
        files,
        model,
        config,
        data_type="training",
        weighted_sampling=False,
        device="cpu",
        verbose=0,
        batch_size=512,
        **kwargs
    ):
        self.verbose = verbose
        self.files = files
        self.Nedges = [0]
        self.data_type = data_type
        
        if data_type == "validation" or data_type == "inference":
            self.data_type = "test"
                
        self.weighted_sampling = weighted_sampling
        self.process_weights = (
            [config["process_weights"][p] for p in config["processes"]]
            if "process_weights" in config
            else None
        )
        self.batch_size = batch_size
        self.num_ele = 0

        for list_truth in model.classes:
            self.num_ele += len(model.classes[list_truth])

        self.device = device
        self.model = model
        self.config = config

        self.input_dims = [
            (1,                          len(config['global_features'])), 
            (config['n_cpf_candidates'], len(config['cpf_candidates'])), 
            (config['n_npf_candidates'], 0), 
            (config['n_vtx_candidates'], len(config['vtx_features']))
        ]
        
        feature_edges = []
        v = 0
        for dim in self.input_dims:
            v += dim[0]*dim[1]
            feature_edges.append(v)
    
        feature_edges = torch.Tensor(feature_edges).int()    
        feature_lengths = feature_edges[1:] - feature_edges[:-1]
        self.feature_lengths = torch.cat((feature_edges[:1], feature_lengths))

    #@profile
    #def __len__(self):
    #    return int(self.all_number_of_samples)
    
    #@profile
    def __getitem__(self, index):
        raise NotImplementedError

    #@profile
    def shuffleFileList(self):
        np.random.shuffle(self.files)

    def dict_to_struct_array(data):
        """
        Converts a dictionary to a structured NumPy array.
        Keys are used as field names, and values as field data.
        """
        names = list(data.keys())
        arrays = [data[name] for name in names]
        dtype = [(name, arr.dtype, arr.shape[1:]) for name, arr in zip(names, arrays)]
        return np.core.records.fromarrays(arrays, dtype=dtype)
    
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

        leftover_data = None
        
        for file in files_to_read:
            
            if self.verbose:
                print(f"Loading {file}")

            # Loading data from .lz4 file
            with lz4.frame.open(file, mode='r') as data:
                output_data = data.read()
            s = np.frombuffer(output_data, dtype='float32')
            s = s[2:].reshape(-1, int(s[1]))

            # Masking NaN and inf values
            s1 = ~np.isnan(s).any(axis = 1)
            s2 = ~np.isinf(s).any(axis = 1)
            s = s[s1*s2]

            # Applying mask based on weights
            if self.weighted_sampling:
                random_number = np.random.rand(s.shape[0])
                s = s[random_number < s[:, -1]]
            
            s = np.random.permutation(s)
            # Extracting truths, weights, and processes
            truths = np.zeros(s.shape[0])
            labels = s[:,-(self.num_ele+1):-1]
            idx = 0
            for index, (name, flavours) in enumerate(self.model.classes.items()):
                for flav in flavours:
                    truths[labels[:,idx] == 1] = index
                    idx += 1
            weights = s[:, -1]
            processes = s[:, -(self.num_ele+2)]
            s = s[:,:-(self.num_ele+2)]

            # Extracting glocal, cpf, and vtx features
            indices = np.cumsum(self.feature_lengths.tolist())[:-1]
            glob, cpf, _, vtx = np.split(s, indices, axis=1)

            # Reshape the arrays as needed
            glob = glob.reshape(glob.shape[0], -1)
            cpf = cpf.reshape(cpf.shape[0], self.input_dims[1][0], -1)
            vtx = vtx.reshape(vtx.shape[0], self.input_dims[3][0], -1)
            
            # Create dictionaries of features
            glob_arr = {name: glob[:, i] for i, name in enumerate(self.config['global_features'])}
            cpf_arr  = {name: cpf[:, :, i] for i, name in enumerate(self.config['cpf_candidates'])}
            vtx_arr  = {name: vtx[:, :, i] for i, name in enumerate(self.config['vtx_features'])}

            '''
            Create new input features for PAIReDTagger
            '''
            ################
            # cpf features #
            ################
            
            custom_cpf_features = ["part_pt_log", "part_e_log", "part_logptrel", "part_logerel", "part_deltaR1", "part_deltaR2", "part_d0", "part_dz"]
            cpf_features_to_add = {}

            # part_pt_log
            mask_pt = cpf_arr['part_pt'] > 0
            cpf_features_to_add["part_pt_log"] = np.full_like(cpf_arr['part_pt'], 0, dtype=np.float32)
            cpf_features_to_add["part_pt_log"][mask_pt] = np.log(cpf_arr['part_pt'][mask_pt])
                           
            mask_energy = cpf_arr['part_energy'] > 0
            cpf_features_to_add["part_e_log"] = np.full_like(cpf_arr['part_energy'], 0, dtype=np.float32)
            cpf_features_to_add["part_e_log"][mask_energy] = np.log(cpf_arr['part_energy'][mask_energy])
            #NOTE there are unequal many infs and zeros, thus there are zeros that are not caused by the lengths of 128, that need to be considered

            # jet_pt only has shape (N,) --> create it of shape (N,1) to be able to divide part_pt by it
            jet1_pt = glob_arr['jet1_pt'].reshape(-1, 1)
            jet2_pt = glob_arr['jet2_pt'].reshape(-1, 1)
            jet1_energy = glob_arr['jet1_energy'].reshape(-1, 1)
            jet2_energy = glob_arr['jet2_energy'].reshape(-1, 1)

            # Compute the sums of jet pt and energy
            jet_pt_sum = jet1_pt + jet2_pt
            jet_energy_sum = jet1_energy + jet2_energy
            
            # Create masks for valid ratios
            valid_pt_mask     = (cpf_arr['part_pt'] > 0)     & (jet_pt_sum > 0)
            valid_energy_mask = (cpf_arr['part_energy'] > 0) & (jet_energy_sum > 0)

            # Initialize result arrays with default value 0
            part_logptrel = np.full_like(cpf_arr['part_pt'], 1e-3, dtype=np.float32)
            part_logerel  = np.full_like(cpf_arr['part_energy'], 1e-3, dtype=np.float32)

            # Compute ratios where valid
            part_logptrel[valid_pt_mask]    = (cpf_arr['part_pt'] / jet_pt_sum)[valid_pt_mask]
            part_logerel[valid_energy_mask] = (cpf_arr['part_energy'] / jet_energy_sum)[valid_energy_mask]
            
            # Assign to the dictionary (and also apply transformations on part_pt_log, part_e_log)
            cpf_features_to_add["part_logptrel"] = np.clip(part_logptrel, -10, 10)
            cpf_features_to_add["part_logerel"] = np.clip(part_logerel, -10, 10)

            cpf_features_to_add["part_deltaR1"] = np.array(
                np.hypot(cpf_arr['part_deta1'], cpf_arr['part_dphi1']), #new variables need to be created here like this!
            )
            cpf_features_to_add["part_deltaR2"] = np.array(
                np.hypot(cpf_arr['part_deta2'], cpf_arr['part_dphi2']),
            )
            cpf_features_to_add["part_d0"] = np.array(
                np.where(cpf_arr["part_d0val"] == -1, 0, np.tanh(cpf_arr["part_d0val"]).astype(np.float32)),
            )
            cpf_features_to_add["part_dz"] = np.array(
                np.where(cpf_arr["part_dzval"] == -1, 0, np.tanh(cpf_arr["part_dzval"]).astype(np.float32)),
            )

            # Add new features to cpf
            cpf_features_to_add_array = np.stack(
                [cpf_features_to_add[key] for key in cpf_features_to_add if key in self.model.cpf_candidates], axis=2
            )
            cpf = np.concatenate([cpf, cpf_features_to_add_array], axis=2)
        
            ################
            # sv features  #
            ################
            
            custom_sv_features = ["sv_px", "sv_py", "sv_pz", "sv_energy"]
            sv_features_to_add = {}
            
            sv_features_to_add["sv_px"] = np.array(
                np.cos(vtx_arr['sv_phi']) * vtx_arr['sv_pt'],
                dtype=np.float32,
            )
            sv_features_to_add["sv_py"] = np.array(
                np.sin(vtx_arr['sv_phi']) * vtx_arr['sv_pt'],
                dtype=np.float32,
            )
            sv_features_to_add["sv_pz"] = np.array(
                np.sinh(vtx_arr['sv_eta']) * vtx_arr['sv_pt'],
                dtype=np.float32,
            )
            sv_features_to_add["sv_energy"] = np.array(
                np.sqrt(vtx_arr['sv_mass']**2 + (vtx_arr['sv_pt']*np.cosh(vtx_arr['sv_eta']))**2),
                dtype=np.float32,
            )

            # create vectors with axes (length, candidates, 4 (px, py, pz, energy))
            cpf_vectors = np.stack(
                [cpf_arr[coordinate] for coordinate in self.model.cpf_vectors], axis=2
            )
            sv_vectors = np.stack(
                [sv_features_to_add[coordinate] for coordinate in self.model.sv_vectors] , axis=2
            )
            cpf_features_full = self.config['cpf_candidates'].copy()
            cpf_features_full.extend(custom_cpf_features)
            vtx_feature_full = self.config['vtx_features'].copy()
            vtx_feature_full.extend(custom_sv_features)
            
            # Map feature names to indices for cpf and vtx
            cpf_feature_indices = {name: idx for idx, name in enumerate(cpf_features_full)}
            vtx_feature_indices = {name: idx for idx, name in enumerate(vtx_feature_full)}
            
            # Get indices of desired features based on the model
            desired_cpf_indices = [cpf_feature_indices[name] for name in self.model.cpf_candidates]
            desired_vtx_indices = [vtx_feature_indices[name] for name in self.model.sv_features]
            
            # Select desired features from cpf and vtx arrays
            cpf_selected = cpf[:, :, desired_cpf_indices]
            vtx_selected = vtx[:, :, desired_vtx_indices]
                
            # Apply transformations directly on cpf_selected
            transformation_dict = {
                'part_dzerr': (0, 1),
                'part_d0err': (0, 1)
            }
            
            for feature_name, (min_val, max_val) in transformation_dict.items():
                idx = cpf_features_full.index(feature_name)
                cpf_selected[:, :, idx] = np.clip(cpf_selected[:, :, idx], min_val, max_val)  
                
            # Reshape cpf_selected and vtx_selected for concatenation
            cpf_arrs = cpf_selected.reshape(cpf_selected.shape[0], -1).astype(np.float32)
            vtx_arrs = vtx_selected.reshape(vtx_selected.shape[0], -1).astype(np.float32)
            
            # Reshape cpf_vectors and sv_vectors if necessary
            cpf_vectors_flat = cpf_vectors.reshape(cpf_vectors.shape[0], -1)
            sv_vectors_flat = sv_vectors.reshape(sv_vectors.shape[0], -1)

            united = np.concatenate([
                cpf_arrs, 
                cpf_vectors_flat, 
                vtx_arrs, 
                sv_vectors_flat
            ], axis=1)

            if leftover_data is not None:
                united = np.concatenate((leftover_data[0], united))
                truths = np.concatenate((leftover_data[1], truths))
                weights = np.concatenate((leftover_data[2], weights))
                processes = np.concatenate((leftover_data[3], processes))
                leftover_data = None

            num_samples = united.shape[0]

            for batch_start in range(0, num_samples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_samples)
                if(batch_end - batch_start) == self.batch_size:
                    #print(f"united[batch_start:batch_end].shape = {united[batch_start:batch_end].shape}")
                    yield (
                        united[batch_start:batch_end],
                        truths[batch_start:batch_end],
                        weights[batch_start:batch_end],
                        processes[batch_start:batch_end],
                    )
                else:
                    leftover_data = [
                        united[batch_start:batch_end],
                        truths[batch_start:batch_end],
                        weights[batch_start:batch_end],
                        processes[batch_start:batch_end],
                    ]
                    
        return None