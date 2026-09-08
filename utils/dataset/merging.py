import os
from collections import defaultdict
from math import inf
import lz4.frame
from concurrent.futures import ThreadPoolExecutor
import threading
import json

import numpy as np
import numpy.lib.recfunctions as rfn
from utils.weighting.histogram import histogram_weighting

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# debug
import psutil


def merge_structured_arrays(array_list: list, delta: int = None, shuffle: bool = True):
    merged = {}
    rest = {}
    # Start merging into traget array:
    for key in array_list[0].keys():
        # merge all arrays up the very last one, if there are multiple files
        if len(array_list) > 1:
            merged[key] = np.concatenate([a[key] for a in array_list[:-1]])
            # merge the last one partially:
            merged[key] = np.concatenate([merged[key], array_list[-1][key][:delta]])
        # if there is only one file, slice it by delta
        else:
            merged[key] = array_list[-1][key][:delta]
        # keep the last chunk (overflow)
        rest[key] = array_list[-1][key][delta:]
    
    if shuffle:
        indices = np.arange(len(merged[key]))
        np.random.shuffle(indices)
        for key in merged.keys():
            merged[key] = merged[key][indices]
    
    return merged, rest


def check_memory_usage():
    memory_usage = psutil.virtual_memory().used / (1024.0**3)
    return memory_usage


def merge_datasets(
    files, 
    path, 
    label="", 
    chunk_size=100000, 
    verbose=0,
    workers=8,
    processor = "", 
    shuffle=True, 
    debug=False, 
    histograms = None, 
    reference_key= None, 
    bins_pt=None, 
    bins_eta=None,
    pt_key=None,
    eta_key=None,
    global_features=None,
    precision='float32'
):
    files = [f for f in files if os.path.isfile(f[:-4]+'.npy')]
    if shuffle:
        np.random.shuffle(files)
    n_chunk = 0
    file_list = []
    dtype = np.dtype(precision).type

    reference_histogram = histograms[reference_key] #reference_key is an index in the context of LZ4 datasets
    reference_histogram = reference_histogram / np.max(reference_histogram)
    weights_list = []
    weights_list_sum = []
    chuck_size_list = []
    for c in range(histograms.shape[0]):
        other_histogram = histograms[c]
        max_hist = np.max(other_histogram)
        if max_hist != 0:
            other_histogram = other_histogram / max_hist
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(other_histogram > 0, reference_histogram / other_histogram, -10)
            weights = weights / np.max(weights)
        
        weights[weights < 0] = 1
        weights[np.isnan(weights)] = 1
        
        weights_list.append(weights)
        
    weights_list = np.array(weights_list)
    weight_filename = os.path.join(path, "weights.json")

    with Progress(
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn(f"0/{len(files)} files merged"),
    ) as progress:
        task = progress.add_task("Merging...", total=len(files))
        
        if "LZ4" in processor:

            pt_key_index = global_features.index(pt_key)
            eta_key_index = global_features.index(eta_key)
        
            dim = np.load(files[0][:-4]+'.npy', allow_pickle=True).shape[-1]
            chunk = np.empty((chunk_size, dim), dtype=dtype)
            any_file_created = False
            chunk_lock = threading.Lock()
            i = 0
            num_files = len(files)
            
            def process_file(file):
                nonlocal chunk, n_chunk, any_file_created, weights_list, weights_list_sum, file_list, i, num_files

                data = np.load(file[:-4]+'.npy', allow_pickle=True).astype(dtype)
                os.remove(file[:-4]+'.npy')
                n_samples = len(data)

                # First locked section: calculate indices and update chunk
                with chunk_lock:
                    i += 1
                    if n_chunk + n_samples > chunk_size:
                        index_range = chunk_size - n_chunk
                    else:
                        index_range = n_samples

                    chunk[n_chunk : n_chunk + index_range] = data[:index_range]
                    n_chunk += index_range
        
                    should_process = (n_chunk >= chunk_size) or (i == num_files)
                    if should_process:
                        # Make a copy of the chunk to process outside the lock
                        chunk_to_process = chunk.copy()
                        
                        # Reset chunk for next iteration
                        if i != num_files:
                            chunk = np.zeros((chunk_size, dim), dtype=dtype)
                            chunk[: n_samples - index_range] = data[index_range:]
                            n_chunk = n_samples - index_range
                            
                if should_process:
                    if i == num_files:
                        chunk_to_process = chunk_to_process[: n_chunk]
            
                    s1 = ~np.isnan(chunk_to_process).any(axis = 1)
                    s2 = ~np.isinf(chunk_to_process).any(axis = 1)
                    chunk_to_process = chunk_to_process[s1*s2]

                    pt_coordinate  = np.digitize(chunk_to_process[:, pt_key_index],  bins_pt)  - 1
                    eta_coordinate = np.digitize(chunk_to_process[:, eta_key_index], bins_eta) - 1
        
                    flavour_idx = np.argmax(chunk_to_process[:,-histograms.shape[0]:], axis=-1)
                    
                    w = weights_list[flavour_idx, pt_coordinate, eta_coordinate].astype(dtype)
                    chunk_to_process = np.concatenate((chunk_to_process,w.reshape(-1,1)), axis=1)
                    size = np.array(chunk_to_process.shape).astype(dtype)
        
                    arr = np.concatenate((size, chunk_to_process.astype(dtype).flatten()))
                    arr = arr.tobytes()

                    # Lock only when updating shared resources
                    with chunk_lock:
                        filename = os.path.join(path, f"{label}_{len(file_list)}.lz4")
                        file_list.append(filename)
                        weights_list_sum.append(float(w.sum()))
                        chuck_size_list.append(len(w))

                    print(filename)
                    with lz4.frame.open(filename, mode='wb') as fp:
                        bytes_written = fp.write(arr)
                        if not any_file_created:
                            any_file_created = True

                progress.update(task, advance=1, description=f"Merging...")
                progress.columns[-1].text_format = "{}/{} its".format(i, len(files))

            # Parallel file processing
            with ThreadPoolExecutor(max_workers=workers) as executor:  # Adjust workers as needed
                rests = list(executor.map(process_file, files))
                progress.update(task, completed=len(files))
    
            if not any_file_created:
                raise ValueError("Expected at least one file to be created, but none were. This is probably to the small amount of data and/or large chunk size.")
                
        else:
            merge_arrays = []
            
            fields = np.load(files[0], allow_pickle=True, mmap_mode="r").files
            for i, file in enumerate(files):
                with np.load(file, allow_pickle=True) as data:
                    n_samples = len(data[data.files[0]])
                    merge_arrays.append(dict(data))
                os.remove(file)
                    
                # Check memory usage
                if debug:
                    if i % 100 == 0:
                        memory_usage = check_memory_usage()
                        print(
                            "Current memory usage: {0:1.2f} GB; arrays: {1:4d}; size of one element: {2:1.2f}MB;n_samples: {3:8d}/{4:8d} - {5:2.1f}%".format(
                                memory_usage,
                                len(merge_arrays),
                                list(merge_arrays[0].values())[0].size
                                * list(merge_arrays[0].values())[0].itemsize
                                / 1e6,
                                n_chunk,
                                chunk_size,
                                n_chunk / chunk_size * 100,
                            )
                        )
                # If samples overflow chunk-size, write out new file
                while n_chunk + n_samples >= chunk_size:
                    if verbose:
                        print("Merging remaining arrays")
                    merged, rest = merge_structured_arrays(
                        merge_arrays,
                        delta=chunk_size - n_chunk,
                        shuffle=True,
                    )
                    pt_coordinate  = np.digitize(merged['global_features'][pt_key],  bins_pt)  - 1
                    eta_coordinate = np.digitize(merged['global_features'][eta_key], bins_eta) - 1
                    flavour_idx = np.lib.recfunctions.apply_along_fields(
                        np.argmax, merged["truth"]
                    )
                    w = weights_list[flavour_idx, pt_coordinate, eta_coordinate].astype(dtype)
                    merged['weight'] = w
                    weights_list_sum.append(float(w.sum()))
                    chuck_size_list.append(chunk_size)

                    filename = os.path.join(path, f"{label}_{len(file_list)}.npz")
                    file_list.append(filename)
                    np.savez(filename, **merged)
                    merge_arrays.clear()
                    merge_arrays = [rest]
                    del merged, rest
                    n_chunk = 0
                    n_samples = len(merge_arrays[0][fields[0]])
                n_chunk += n_samples

                progress.update(task, advance=1)
                progress.columns[-1].text_format = f"{i+1}/{len(files)} files merged"
                # writeout remaining arrays
            if len(merge_arrays) > 0:
                if verbose:
                    print("Merging remaining arrays")
                merged, _ = merge_structured_arrays(merge_arrays,shuffle=True)
                pt_coordinate  = np.digitize(merged['global_features'][pt_key],  bins_pt)  - 1
                eta_coordinate = np.digitize(merged['global_features'][eta_key], bins_eta) - 1
                flavour_idx = np.lib.recfunctions.apply_along_fields(
                    np.argmax, merged["truth"]
                )
                w = weights_list[flavour_idx, pt_coordinate, eta_coordinate].astype(dtype)
                merged['weight'] = w
                weights_list_sum.append(float(w.sum()))
                chuck_size_list.append(len(w))
                
                filename = os.path.join(path, f"{label}_{len(file_list)}.npz")
                file_list.append(filename)
                np.savez(filename, **merged)
                
        metadata = {
            "chunk_size": chuck_size_list,
            "sum_weight": weights_list_sum
        }
        
        with open(os.path.join(weight_filename), "w") as out:
            json.dump(metadata, out, indent=2)
            
    return file_list
