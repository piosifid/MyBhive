import numpy as np
from pathlib import Path
from tqdm import tqdm


"""
Usage:
    1) Define, which input categories will be loaded. They have to match the keys in the .npz files created by the DatasetConstructorTask.
    2) Set the absolute path the processed_files.txt file the directory containing the .npz files.
    3) Run the script.
"""


# setup by user
categories = []  # for example: ["global_features", "cpf_arr", "npf_arr", "vtx_arr"]
filelist = ""  # for example: "/path/to/b-hive/dir/DatasetConstructorTask/.../processed_files.txt"
verbose = True

# creating directory, if necessary
output_path = filelist.strip("processed_files.txt") + "epsilons"

if not Path(output_path).exists():
    Path(output_path).mkdir(exist_ok=True)

# loading all data
with open(filelist, "r") as file:
    filelist = [line.strip() for line in file]
data_list = [np.load(f"{file}") for file in filelist]

# extracting features, calculating feature specific epsilons and saving them to disk
for i, c in enumerate(categories):
    if verbose:
        print("\n")
        print("#################")
        print(f"#{c}#")
        print("#################")
    keys = data_list[0][c].dtype.fields

    dtype = np.dtype([(key, np.float32) for key in keys])
    arr = np.empty(1, dtype=dtype)

    mean_epsilons = []
    for key in tqdm(keys):
        epsilons = []
        for data in data_list:
            x = data[c][key]
            x = x[x != 0]
            p = np.percentile(x, [20, 80])
            epsilon = np.std(x[np.logical_and(x >= p[0], x <= p[1])])
            epsilons.append(epsilon)
        mean_epsilons.append(np.mean(np.array(epsilons)))
        arr[key] = mean_epsilons[-1]
        if verbose:
            print(f"epsilon for {key} = {mean_epsilons[-1]}")

    np.save(f"{output_path}/input_category_{i}.npy", arr)
