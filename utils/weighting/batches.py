import torch
import numpy as np
from typing import Tuple

def compute_class_weights(
    classes: dict,
    truths: list,
    histogram_path: str,
    dtype: str,
    device: torch.device
) -> torch.Tensor:
    """
    Compute per‐class loss weights based on numbers of samples.
    Returns the torch.Tensor of class weights.
    """
    hist = np.load(histogram_path, allow_pickle=True)
    abundance = np.array([
        hist[[truths.index(flav) for flav in flavours]].sum()
        for flavours in classes.values()
    ])
    raw_weights = abundance.sum() / abundance
    normalized = raw_weights / raw_weights.max()
    
    print(f"Total number of members: {abundance.astype(int).sum():,}")
    print(f"Number of class members: {list(abundance.astype(int))}")
    print(f"Class weights:           {[round(w,4) for w in normalized]}")
    dt = np.dtype(dtype).type
    return torch.tensor(normalized.astype(dt), device=device)


def expected_batches(
    num_samples: np.ndarray,
    batch_size: int,
    n_threads: int,
    mode: str = "training"
) -> Tuple[int, int]:
    """
    Estimate the total number of batches given a batch size, accounting for thread-wise splitting
    and optionally including the final partial batch in validation mode.

    Parameters
    ----------
    num_samples : np.ndarray
        Array containing either number of samples or sum of weights per file.
    batch_size : int
        Number of samples per batch.
    n_threads : int
        Number of parallel threads for dataloader.
    mode : str, optional
        Either "training" or "validation"/"test". In validation mode, the final partial batch is included.
        Default is "training".

    Returns
    -------
    int
        Estimated number of batches.
    """
    total_batches = 0
    total_events  = 0

    for chunk in np.array_split(num_samples, n_threads):
        total = int(chunk.sum())
        full, rem = divmod(total, batch_size)

        if mode=='training':
            # training: drop the last partial
            total_batches += full
            total_events  += full * batch_size
        else:
            # validation/test: keep the last partial if it exists
            total_batches += full + (1 if rem > 0 else 0)
            total_events  += full * batch_size + rem

    return int(total_batches), int(total_events)
