import awkward as ak
import numpy as np
from functools import reduce
from typing import List
import operator


def join_struct_arrays(*arrs):
    dtype = [(name, d[0]) for arr in arrs for name, d in arr.dtype.fields.items()]
    r = np.empty(arrs[0].shape, dtype=dtype)
    for a in arrs:
        for name in a.dtype.names:
            r[name] = a[name]
    return r


def structured_array_from_tree(
    events=None,
    keys: list[str] = None,
    feature_length: int = None,
    precision=np.float32,
) -> np.ndarray:
    dtype = np.dtype(
        [
            (
                (name, precision, feature_length)
                if feature_length > 1
                else (name, precision)
            )
            for name in keys
        ]
    )
    arr = np.empty((len(events),), dtype=dtype)
    #print('KEYS:', keys)
    for key, dtype_name in zip(keys, dtype.fields):
        if feature_length == 1:
            arr[key] = np.array(events[key], dtype=[(dtype_name, precision)])
        else:
            arr[key] = ak.to_numpy(
                ak.values_astype(
                    ak.fill_none(
                        ak.pad_none(events[key], feature_length)[:, :feature_length], 0
                    ),
                    precision,
                )
            )
    return arr


def structured_custom_array_from_tree(
    events=None,
    keys: list[str] = None,
    custom_keys: List[str] = None,
    custom_formulas: List[str] = None,
    feature_length: int = None,
    precision=np.float32,
) -> np.ndarray:
    dtype = np.dtype(
        [
            (
                (name, precision, feature_length)
                if feature_length > 1
                else (name, precision)
            )
            for name in (keys + custom_keys)
        ]
    )

    eval_dict = {key: events[key] for key in events.fields}
    eval_dict.update({"np": np, "numpy": np, "ak": ak, "awkward": ak})

    arr = np.empty((len(events),), dtype=dtype)

    dtype = np.dtype(
        [
            (
                (name, precision, feature_length)
                if feature_length > 1
                else (name, precision)
            )
            for name in keys
        ]
    )
    for key, dtype_name in zip(keys, dtype.fields):
        if (len(events[key]) == 0):
            continue
        elif feature_length == 1:
            arr[key] = np.array(events[key], dtype=[(dtype_name, precision)])
        else:
            arr[key] = ak.to_numpy(
                ak.values_astype(
                    ak.fill_none(
                        ak.pad_none(events[key], feature_length)[:, :feature_length], 0
                    ),
                    np.float32,
                )
            ).astype(precision)

    dtype = np.dtype(
        [
            (
                (name, precision, feature_length)
                if feature_length > 1
                else (name, precision)
            )
            for name in custom_keys
        ]
    )
    with np.errstate(all="ignore"):
        for key, formula, dtype_name in zip(custom_keys, custom_formulas, dtype.fields):
            if feature_length == 1:
                arr[key] = np.array(
                    eval(formula, eval_dict), dtype=[(dtype_name, precision)]
                )
            else:
                arr[key] = ak.to_numpy(
                    ak.values_astype(
                        ak.fill_none(
                            ak.pad_none(eval(formula, eval_dict), feature_length)[
                                :, :feature_length
                            ],
                            0,
                        ),
                        np.float32,
                    )
                ).astype(precision)
    return arr

# Helper function: interpret a condition (raw scalar or dict like {">": 0})
def matches_condition(branch_values, condition):
    """
    Returns a boolean array specifying which events match the condition.
    """
    # Simple operator lookup
    OPERATORS = {
        "==": operator.eq,
        "!=": operator.ne,
        ">":  operator.gt,
        "<":  operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
    }
    # If condition is just a scalar, do equality check
    if not isinstance(condition, dict):
        return branch_values == condition

    # Otherwise parse operator and threshold
    (op_key, threshold), = condition.items()  # e.g. {">": 0} -> op_key=">", threshold=0
    return OPERATORS[op_key](branch_values, threshold)

def structured_array_from_tree_truth_from_dict(
    events=None,
    truth_dict: dict[str] = None,
    feature_length: int = None,
    precision=np.float32,
) -> np.ndarray:
    dtype = np.dtype([(name, precision) for name in truth_dict.keys()])
    arr = np.empty((len(events),), dtype=dtype)
    """
    this stitches flavour branches together example:

    label_b = (hflav_== 5 && tau_flav == 0)
    label_ud = (tauflav == 0 && hflav == 0 && ( pflav == 0 || pflav == 1 || ... )) 

    """
    for (key, truth_config), dtype_name in zip(truth_dict.items(), dtype.fields):
        if isinstance(truth_config, list):
            # in case of nested structure
            arr[key] = np.array(
                reduce(
                    np.logical_and,
                    (
                        reduce(
                            np.logical_or,
                            (
                                matches_condition(events[flav_branch], value)
                                for value in values
                            ),
                        )
                        for flav_branches in truth_config
                        for flav_branch, values in flav_branches.items()
                    ),
                ),
                dtype=[(dtype_name, precision)],
            )
            # print(
            #     f"flavour select: {key}\n",
            #     " and\n".join(
            #         " or ".join(f"{flav_branch} == {value}" for value in values)
            #         for flav_branches in truth_config
            #         for flav_branch, values in flav_branches.items()
            #     ),
            # )
            # print("#" * 10)
        elif isinstance(truth_config, str):
            # in case for flat array
            arr[key] = np.array(events[key], dtype=[(dtype_name, precision)])
    return arr
