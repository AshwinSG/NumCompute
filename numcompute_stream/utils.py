"""Shared input checks and errors for NumCompute.

Shape rules used across the package:
    X is a 2-D array with shape (n_samples, n_features). Rows are samples and
    columns are features. Every per-feature statistic is worked out along
    axis=0.
    y is a 1-D array with shape (n_samples,).
    A 1-D X is not reshaped for you. An error is raised instead, so a mix-up
    between (n,) and (1, n) is never missed.
"""
from __future__ import annotations

import numpy as np


class NotFittedError(RuntimeError):
    """Raised when a method that needs a trained model is called too early."""


def check_array(X, name: str = "X", ndim: int = 2, dtype=float,
                allow_nan: bool = True, allow_empty: bool = False) -> np.ndarray:
    """Turn X into a NumPy array and check its shape.

    Parameters:
        X: array-like input.
        name: name used in error messages.
        ndim: number of dimensions X must have (1 or 2).
        dtype: dtype to convert to. None keeps the dtype NumPy picks (for
            example object arrays that hold category strings).
        allow_nan: if False, NaN values raise ValueError.
        allow_empty: if False, arrays with 0 rows raise ValueError.

    Returns:
        np.ndarray
    """
    if X is None:
        raise TypeError(f"{name} must be array-like, got None")
    try:
        arr = np.asarray(X, dtype=dtype) if dtype is not None else np.asarray(X)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} could not be converted to a numeric array of dtype "
            f"{dtype}: {exc}") from exc
    if arr.ndim != ndim:
        hint = ""
        if ndim == 2 and arr.ndim == 1:
            hint = (" Reshape with X.reshape(-1, 1) for a single feature or "
                    "X.reshape(1, -1) for a single sample.")
        raise ValueError(
            f"{name} must be {ndim}-D, got {arr.ndim}-D array with shape "
            f"{arr.shape}.{hint}")
    if not allow_empty and arr.shape[0] == 0:
        raise ValueError(f"{name} has 0 samples; a non-empty chunk is required")
    if arr.dtype.kind == "f":
        if np.isinf(arr).any():
            raise ValueError(f"{name} contains infinite values")
        if not allow_nan and np.isnan(arr).any():
            raise ValueError(f"{name} contains NaN values")
    return arr


def check_X_y(X, y, dtype_X=float, allow_nan: bool = True):
    """Check a feature matrix and a label vector of the same length.

    Returns:
        X with shape (n_samples, n_features)
        y with shape (n_samples,)
    """
    X = check_array(X, "X", ndim=2, dtype=dtype_X, allow_nan=allow_nan)
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D with shape (n_samples,), got shape {y.shape}")
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            f"X and y have inconsistent numbers of samples: X has "
            f"{X.shape[0]}, y has {y.shape[0]}")
    if y.dtype.kind == "f" and np.isnan(y).any():
        raise ValueError("y contains NaN labels; drop or impute them first")
    return X, y


def check_same_length(a, b, name_a: str = "y_true", name_b: str = "y_pred"):
    """Check that two 1-D arrays have the same length, and return both as arrays."""
    a = np.asarray(a).ravel() if np.ndim(a) == 2 and np.shape(a)[1] == 1 else np.asarray(a)
    b = np.asarray(b).ravel() if np.ndim(b) == 2 and np.shape(b)[1] == 1 else np.asarray(b)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"{name_a} and {name_b} must be 1-D, got shapes {a.shape} and {b.shape}")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"{name_a} and {name_b} have different lengths: "
                         f"{a.shape[0]} vs {b.shape[0]}")
    return a, b


def nbytes_of(obj, _seen=None) -> int:
    """Rough memory use (in bytes) of all NumPy arrays that can be reached from obj.

    Looks inside __dict__, lists, tuples and dicts, and inside anything they
    hold. Only array data is counted, since that is most of the memory for a
    numeric model.
    """
    if _seen is None:
        _seen = set()
    if id(obj) in _seen:
        return 0
    _seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(nbytes_of(v, _seen) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return sum(nbytes_of(v, _seen) for v in obj)
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return nbytes_of(vars(obj), _seen)
    return 0
