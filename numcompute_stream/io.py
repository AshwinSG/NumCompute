"""CSV input/output for NumCompute (plain Python ``csv`` + NumPy only).

Two entry points:

* :func:`load_csv` – read a whole file into ``(X, y)`` arrays.
* :func:`iter_csv_chunks` – stream a file in fixed-size chunks without ever
  holding it in memory, yielding ``(X_chunk, y_chunk)``.

Missing-value tokens (``""``, ``"NA"``, ``"NaN"``, ``"?"`` ...) become ``np.nan``
in numeric columns and ``None`` in categorical (string) columns.
If every feature column is numeric, ``X`` is ``float64``; if any column is
categorical, ``X`` is an ``object`` array so strings survive until a
:class:`~numcompute_stream.preprocessing.OneHotEncoder` converts them.
"""
from __future__ import annotations

import csv
from collections import namedtuple
from itertools import islice
from typing import Iterator, Optional, Sequence

import numpy as np

DEFAULT_MISSING = ("", "na", "nan", "null", "none", "?")

CSVData = namedtuple("CSVData", ["X", "y", "feature_names", "categorical_idx"])
CSVData.__doc__ = """Result of :func:`load_csv`.

X : ndarray (n_samples, n_features), float64 or object
y : ndarray (n_samples,) or None
feature_names : list[str]
categorical_idx : list[int] – column indices of X holding strings
"""


def _is_missing(tokens: np.ndarray, missing) -> np.ndarray:
    lowered = np.char.lower(np.char.strip(tokens.astype(str)))
    return np.isin(lowered, [m.lower() for m in missing])


def _convert_column(tokens: np.ndarray, missing, force_categorical: bool):
    """Return (array, is_categorical) for a column of raw string tokens."""
    miss = _is_missing(tokens, missing)
    if not force_categorical:
        filled = np.where(miss, "nan", tokens)
        try:
            return filled.astype(float), False
        except ValueError:
            pass
    out = np.char.strip(tokens.astype(str)).astype(object)
    out[miss] = None
    return out, True


def _rows_to_arrays(rows, header, target, missing, categorical):
    if not rows:
        raise ValueError("CSV chunk contains no data rows")
    widths = {len(r) for r in rows}
    if len(widths) != 1 or widths.pop() != len(header):
        bad = next(i for i, r in enumerate(rows) if len(r) != len(header))
        raise ValueError(
            f"Ragged CSV: data row {bad} has {len(rows[bad])} fields but the "
            f"header has {len(header)}")
    raw = np.array(rows, dtype=str)  # (n_rows, n_cols)

    target_idx = None
    if target is not None:
        if isinstance(target, int):
            target_idx = target if target >= 0 else len(header) + target
        elif target in header:
            target_idx = header.index(target)
        else:
            raise KeyError(f"target column {target!r} not found; columns are {header}")

    categorical = set(categorical or ())
    cols, names, cat_idx = [], [], []
    y = None
    for j, name in enumerate(header):
        if j == target_idx:
            col, is_cat = _convert_column(raw[:, j], missing, False)
            y = col if not is_cat else raw[:, j].astype(str)
            continue
        col, is_cat = _convert_column(raw[:, j], missing, name in categorical)
        if is_cat:
            cat_idx.append(len(cols))
        cols.append(col)
        names.append(name)
    if cat_idx:
        X = np.empty((raw.shape[0], len(cols)), dtype=object)
        for j, c in enumerate(cols):
            X[:, j] = c
    else:
        X = np.column_stack(cols).astype(float) if cols else np.empty((raw.shape[0], 0))
    if y is not None and y.dtype.kind == "f":
        if np.isnan(y).any():
            raise ValueError(f"target column {header[target_idx]!r} contains missing labels")
        if np.all(y == np.round(y)):
            y = y.astype(np.int64)
    return CSVData(X, y, names, cat_idx)


def load_csv(path, target: Optional[object] = None, delimiter: str = ",",
             missing_values: Sequence[str] = DEFAULT_MISSING,
             categorical: Optional[Sequence[str]] = None) -> CSVData:
    """Load an entire CSV file (first line must be a header).

    Parameters
    ----------
    path : str or Path
    target : str or int, optional
        Label column (name or index). Excluded from ``X``; returned as ``y``.
        Integer-valued float labels are cast to ``int64``.
    delimiter : str
    missing_values : sequence of str
        Case-insensitive tokens treated as missing.
    categorical : sequence of str, optional
        Column names to force as categorical. Non-numeric columns are
        detected as categorical automatically.

    Returns
    -------
    CSVData(X, y, feature_names, categorical_idx)
    """
    with open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration:
            raise ValueError(f"{path} is empty") from None
        rows = [r for r in reader if r]
    return _rows_to_arrays(rows, header, target, missing_values, categorical)


def iter_csv_chunks(path, chunk_size: int, target: Optional[object] = None,
                    delimiter: str = ",",
                    missing_values: Sequence[str] = DEFAULT_MISSING,
                    categorical: Optional[Sequence[str]] = None) -> Iterator[CSVData]:
    """Stream a CSV file in chunks of ``chunk_size`` rows.

    Only one chunk is held in memory at a time. Pass ``categorical`` so that
    column types are consistent across chunks (a chunk where a categorical
    column happens to be all missing would otherwise look numeric).

    Yields
    ------
    CSVData for each chunk (the last chunk may be smaller).
    """
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
    with open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration:
            raise ValueError(f"{path} is empty") from None
        rows_iter = (r for r in reader if r)
        while True:
            rows = list(islice(rows_iter, chunk_size))
            if not rows:
                return
            yield _rows_to_arrays(rows, header, target, missing_values, categorical)


def iter_array_chunks(X, y=None, chunk_size: int = 100):
    """Split in-memory arrays into consecutive chunks along ``axis=0``.

    Yields ``(X_chunk, y_chunk)`` (``y_chunk`` is None if ``y`` is None).
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    X = np.asarray(X)
    if y is not None and len(y) != len(X):
        raise ValueError(f"X and y lengths differ: {len(X)} vs {len(y)}")
    for start in range(0, len(X), chunk_size):
        sl = slice(start, start + chunk_size)
        yield X[sl], (None if y is None else np.asarray(y)[sl])


def save_csv(path, X, header: Sequence[str], y=None, target_name: str = "target",
             delimiter: str = ",") -> None:
    """Write ``X`` (and optionally ``y`` as the last column) to CSV.

    NaN / None are written as empty fields.
    """
    X = np.asarray(X, dtype=object)
    if X.ndim != 2 or X.shape[1] != len(header):
        raise ValueError(f"X shape {X.shape} does not match header of length {len(header)}")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter=delimiter)
        w.writerow(list(header) + ([target_name] if y is not None else []))
        for i in range(X.shape[0]):
            row = ["" if (v is None or (isinstance(v, float) and np.isnan(v))) else v
                   for v in X[i]]
            if y is not None:
                row.append(y[i])
            w.writerow(row)
