"""Shared validation helpers and exceptions for NumCompute.

Shape conventions used throughout the package
---------------------------------------------
* ``X`` : 2-D array of shape ``(n_samples, n_features)``. Rows are samples,
  columns are features; every "per-feature" statistic is computed along
  ``axis=0``.
* ``y`` : 1-D array of shape ``(n_samples,)``.
* A 1-D ``X`` is *not* silently reshaped – an explicit error is raised so
  that ``(n,)`` vs ``(1, n)`` ambiguity never goes unnoticed.
"""
from __future__ import annotations

import numpy as np


class NotFittedError(RuntimeError):
    """Raised when a method that needs learned state is called too early."""


def check_array(X, name: str = "X", ndim: int = 2, dtype=float,
                allow_nan: bool = True, allow_empty: bool = False) -> np.ndarray:
    """Convert ``X`` to a NumPy array and validate its shape.

    Parameters
    ----------
    X : array-like
    name : str
        Name used in error messages.
    ndim : int
        Required number of dimensions (1 or 2).
    dtype : numpy dtype or None
        Target dtype. ``None`` keeps the inferred dtype (e.g. object arrays
        holding categorical strings).
    allow_nan : bool
        If False, NaN values raise ``ValueError``.
    allow_empty : bool
        If False, arrays with zero samples raise ``ValueError``.

    Returns
    -------
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
    """Validate a feature matrix and label vector of matching length.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
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
    """Validate two 1-D arrays have the same length; returns both as arrays."""
    a = np.asarray(a).ravel() if np.ndim(a) == 2 and np.shape(a)[1] == 1 else np.asarray(a)
    b = np.asarray(b).ravel() if np.ndim(b) == 2 and np.shape(b)[1] == 1 else np.asarray(b)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"{name_a} and {name_b} must be 1-D, got shapes {a.shape} and {b.shape}")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"{name_a} and {name_b} have different lengths: "
                         f"{a.shape[0]} vs {b.shape[0]}")
    return a, b


def nbytes_of(obj, _seen=None) -> int:
    """Rough memory footprint (bytes) of NumPy arrays reachable from ``obj``.

    Walks ``__dict__``, lists, tuples and dicts recursively. Only array
    buffers are counted, which dominates memory for numerical models.
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
