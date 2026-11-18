"""Preprocessing transformers that can learn from data one chunk at a time.

Every transformer has:
    partial_fit(X_chunk)  update what it has learned with one chunk
    fit(X)                reset(), then partial_fit(X)
    transform(X)          apply what it has learned so far
    fit_transform(X)      fit, then transform
    reset()

X has shape (n_samples, n_features). Statistics are worked out for each
feature (axis=0). NaNs are skipped while learning and passed straight through
by the scalers (fill them with SimpleImputer).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .stats import StreamingQuantile, StreamingStats, chunk_moments
from .utils import NotFittedError, check_array


class _Transformer:
    """Base class that adds fit and fit_transform on top of partial_fit."""

    def fit(self, X, y=None):
        self.reset()
        return self.partial_fit(X, y)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def _check_width(self, X, expected: int, growable: bool = False):
        if X.shape[1] != expected and not (growable and X.shape[1] > expected):
            raise ValueError(f"{type(self).__name__} expected {expected} features, "
                             f"got {X.shape[1]}")


# --------------------------------------------------------------------------- #
class StandardScaler(_Transformer):
    """Scale features to (X - mean) / std using running statistics.

    Parameters:
        method: "welford" or "ema", default "welford".
            "welford" gives the exact running mean and variance over the
            whole stream (Welford's method, done one chunk at a time).
            "ema" gives a moving average of the mean and variance that
            follows data that changes over time. Each chunk is mixed in with
            weight alpha.
        alpha: float, default 0.1. How much weight each new chunk gets when
            method="ema" (0 < alpha <= 1).
        with_mean, with_std: bool
        allow_growth: bool, default False. Accept chunks with extra columns
            at the end (for example new one-hot columns). Earlier rows are
            taken to be 0 in those columns.

    Notes:
        Features with zero variance get scale_ = 1, so they become 0 instead
        of causing a divide by zero. NaNs pass through unchanged.
    """

    def __init__(self, method: str = "welford", alpha: float = 0.1,
                 with_mean: bool = True, with_std: bool = True,
                 allow_growth: bool = False):
        if method not in ("welford", "ema"):
            raise ValueError(f"method must be 'welford' or 'ema', got {method!r}")
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.method = method
        self.alpha = alpha
        self.with_mean = with_mean
        self.with_std = with_std
        self.allow_growth = allow_growth
        self.reset()

    def reset(self):
        self._stats = StreamingStats(growth_fill=0.0 if self.allow_growth else None)
        self._ema_mean = None
        self._ema_var = None
        return self

    def partial_fit(self, X, y=None):
        X = check_array(X)
        if self.method == "welford":
            self._stats.update_stats(X)
            return self
        cnt, cmean, cM2 = chunk_moments(X)
        cvar = np.where(cnt > 0, cM2 / np.where(cnt > 0, cnt, 1), 0.0)
        if self._ema_mean is None:
            self._ema_mean = np.where(cnt > 0, cmean, 0.0)
            self._ema_var = cvar
            self._ema_seen = cnt > 0
            return self
        if X.shape[1] != self._ema_mean.shape[0]:
            if self.allow_growth and X.shape[1] > self._ema_mean.shape[0]:
                extra = X.shape[1] - self._ema_mean.shape[0]
                self._ema_mean = np.concatenate([self._ema_mean, np.zeros(extra)])
                self._ema_var = np.concatenate([self._ema_var, np.zeros(extra)])
                self._ema_seen = np.concatenate([self._ema_seen, np.ones(extra, bool)])
            else:
                self._check_width(X, self._ema_mean.shape[0])
        a = np.where(cnt > 0, np.where(self._ema_seen, self.alpha, 1.0), 0.0)
        delta = cmean - self._ema_mean
        # update the moving mean and variance (West, 1979)
        self._ema_var = (1 - a) * (self._ema_var + a * delta * delta) + a * cvar
        self._ema_mean = self._ema_mean + a * delta
        self._ema_seen = self._ema_seen | (cnt > 0)
        return self

    @property
    def mean_(self) -> np.ndarray:
        if self.method == "ema":
            if self._ema_mean is None:
                raise NotFittedError("StandardScaler is not fitted; call partial_fit first")
            return self._ema_mean.copy()
        m = self._stats.mean
        return np.where(np.isnan(m), 0.0, m)

    @property
    def var_(self) -> np.ndarray:
        if self.method == "ema":
            if self._ema_var is None:
                raise NotFittedError("StandardScaler is not fitted; call partial_fit first")
            return self._ema_var.copy()
        v = self._stats.var()
        return np.where(np.isnan(v), 0.0, v)

    @property
    def scale_(self) -> np.ndarray:
        s = np.sqrt(self.var_)
        return np.where(s > np.finfo(float).eps * np.maximum(1.0, np.abs(self.mean_)), s, 1.0)

    @property
    def n_features_in_(self) -> int:
        return self.mean_.shape[0]

    def transform(self, X):
        X = check_array(X)
        mean, scale = self.mean_, self.scale_
        self._check_width(X, mean.shape[0])
        out = X.copy()
        if self.with_mean:
            out -= mean
        if self.with_std:
            out /= scale
        return out

    def inverse_transform(self, X):
        X = check_array(X)
        self._check_width(X, self.n_features_in_)
        out = X.copy()
        if self.with_std:
            out *= self.scale_
        if self.with_mean:
            out += self.mean_
        return out


# --------------------------------------------------------------------------- #
class MinMaxScaler(_Transformer):
    """Scale features into feature_range using the running min and max.

    Features that never change (max == min) map to the low end of the range.
    """

    def __init__(self, feature_range: tuple = (0.0, 1.0)):
        if feature_range[1] <= feature_range[0]:
            raise ValueError(f"invalid feature_range {feature_range}")
        self.feature_range = feature_range
        self.reset()

    def reset(self):
        self._stats = StreamingStats()
        return self

    def partial_fit(self, X, y=None):
        self._stats.update_stats(X)
        return self

    def transform(self, X):
        X = check_array(X)
        lo, hi = self._stats.min, self._stats.max
        self._check_width(X, lo.shape[0])
        lo = np.where(np.isnan(lo), 0.0, lo)
        rng = np.where(np.isnan(hi), 0.0, hi) - lo
        rng = np.where(rng > 0, rng, 1.0)
        a, b = self.feature_range
        return a + (X - lo) / rng * (b - a)


# --------------------------------------------------------------------------- #
class SimpleImputer(_Transformer):
    """Fill NaNs with values that are updated as new chunks arrive.

    Parameters:
        strategy: "mean", "median", "most_frequent" or "constant".
            mean uses exact running means. median uses a random sample
            (exact until capacity rows). most_frequent keeps running counts
            of each value (on a tie, the smallest value wins). constant
            always uses fill_value.
        fill_value: float, default 0.0. Used for strategy="constant" and for
            features that have not had any value yet.
        capacity: int. Size of the random sample for strategy="median".
        allow_growth: bool. Accept chunks with extra columns at the end
            (mean only).

    Attributes:
        statistics_: array (n_features,), the current fill values
    """

    _STRATEGIES = ("mean", "median", "most_frequent", "constant")

    def __init__(self, strategy: str = "mean", fill_value: float = 0.0,
                 capacity: int = 10_000, allow_growth: bool = False,
                 random_state: Optional[int] = None):
        if strategy not in self._STRATEGIES:
            raise ValueError(f"strategy must be one of {self._STRATEGIES}, got {strategy!r}")
        self.strategy = strategy
        self.fill_value = float(fill_value)
        self.capacity = capacity
        self.allow_growth = allow_growth
        self.random_state = random_state
        self.reset()

    def reset(self):
        self._n_features = None
        self._stats = StreamingStats(growth_fill=0.0 if self.allow_growth else None)
        self._quant = StreamingQuantile(self.capacity, self.random_state)
        self._counts = None  # one dict per column: value -> count
        return self

    def partial_fit(self, X, y=None):
        X = check_array(X)
        if self._n_features is not None:
            self._check_width(X, self._n_features,
                              growable=self.allow_growth and self.strategy == "mean")
        self._n_features = X.shape[1]
        if self.strategy == "mean":
            self._stats.update_stats(X)
        elif self.strategy == "median":
            self._quant.update_stats(X)
        elif self.strategy == "most_frequent":
            if self._counts is None:
                self._counts = [dict() for _ in range(X.shape[1])]
            for j in range(X.shape[1]):  # loop over features, not samples
                col = X[:, j]
                vals, cnts = np.unique(col[~np.isnan(col)], return_counts=True)
                d = self._counts[j]
                for v, c in zip(vals.tolist(), cnts.tolist()):
                    d[v] = d.get(v, 0) + c
        return self

    @property
    def statistics_(self) -> np.ndarray:
        if self._n_features is None:
            raise NotFittedError("SimpleImputer is not fitted; call partial_fit first")
        if self.strategy == "constant":
            return np.full(self._n_features, self.fill_value)
        if self.strategy == "mean":
            s = self._stats.mean
        elif self.strategy == "median":
            s = self._quant.median()
        else:
            s = np.array([min(d.items(), key=lambda kv: (-kv[1], kv[0]))[0] if d else np.nan
                          for d in self._counts])
        return np.where(np.isnan(s), self.fill_value, s)

    def transform(self, X):
        X = check_array(X)
        stats = self.statistics_
        self._check_width(X, stats.shape[0])
        mask = np.isnan(X)
        return np.where(mask, stats[None, :], X)


# --------------------------------------------------------------------------- #
class OneHotEncoder(_Transformer):
    """Turn category columns into 0/1 columns, adding new categories as they appear.

    The output is [other columns..., one-hot columns]. New categories are
    always added at the end of the one-hot columns (in the order they first
    appear), so columns that were already output never move. Later steps
    only see new columns added at the end (see allow_growth on the scalers
    and imputer, and the padding in the tree's buffer).

    Parameters:
        columns: list of int or None. Indices of the category columns. None
            means every column is a category column.
        handle_unknown: "ignore" or "error". What transform does with a
            category that partial_fit has not seen. "ignore" outputs all
            zeros.

    Missing values (None or NaN) always become all zeros.

    Attributes:
        categories_: dict of column index to a list of categories (str)
        feature_map_: list of (column index, category), one for each one-hot
            column
    """

    def __init__(self, columns: Optional[Sequence[int]] = None,
                 handle_unknown: str = "ignore"):
        if handle_unknown not in ("ignore", "error"):
            raise ValueError("handle_unknown must be 'ignore' or 'error'")
        self.columns = None if columns is None else list(columns)
        self.handle_unknown = handle_unknown
        self.reset()

    def reset(self):
        self.categories_ = None
        self.feature_map_ = []
        self._index = {}  # (col, cat) -> position in the one-hot columns
        self._n_features = None
        return self

    @staticmethod
    def _as_str(col: np.ndarray):
        """Return (string values, missing mask) for a raw column."""
        col = np.asarray(col, dtype=object)
        miss = np.array([v is None or (isinstance(v, float) and v != v) for v in col],
                        dtype=bool)
        tokens = np.array([str(v) for v in col], dtype=object)
        return tokens, miss

    def _cat_cols(self, n_features):
        cols = list(range(n_features)) if self.columns is None else self.columns
        bad = [c for c in cols if not -n_features <= c < n_features]
        if bad:
            raise ValueError(f"categorical column indices {bad} out of range for "
                             f"{n_features} features")
        return [c % n_features for c in cols]

    def partial_fit(self, X, y=None):
        X = check_array(X, dtype=None)
        if self._n_features is not None:
            self._check_width(X, self._n_features)
        self._n_features = X.shape[1]
        cols = self._cat_cols(X.shape[1])
        if self.categories_ is None:
            self.categories_ = {c: [] for c in cols}
        for c in cols:
            tokens, miss = self._as_str(X[:, c])
            # np.unique sorts the values; put them back in order of first
            # appearance so the result is always the same
            uniq, first = np.unique(tokens[~miss].astype(str), return_index=True)
            for cat in uniq[np.argsort(first)].tolist():
                if (c, cat) not in self._index:
                    self._index[(c, cat)] = len(self.feature_map_)
                    self.feature_map_.append((c, cat))
                    self.categories_[c].append(cat)
        return self

    @property
    def n_output_features_(self) -> int:
        if self._n_features is None:
            raise NotFittedError("OneHotEncoder is not fitted")
        return self._n_features - len(self.categories_) + len(self.feature_map_)

    def transform(self, X):
        if self._n_features is None:
            raise NotFittedError("OneHotEncoder is not fitted; call partial_fit first")
        X = check_array(X, dtype=None)
        self._check_width(X, self._n_features)
        cols = self._cat_cols(X.shape[1])
        pass_cols = [j for j in range(X.shape[1]) if j not in set(cols)]
        n = X.shape[0]
        if pass_cols:
            raw = X[:, pass_cols]
            try:
                passthrough = np.array(
                    [[np.nan if v is None else v for v in row] for row in raw]
                    if raw.dtype == object else raw, dtype=float)
            except (TypeError, ValueError) as exc:
                raise TypeError("non-categorical columns must be numeric; pass their "
                                f"indices in `columns` to encode them ({exc})") from exc
        else:
            passthrough = np.empty((n, 0))
        block = np.zeros((n, len(self.feature_map_)))
        for c in cols:
            tokens, miss = self._as_str(X[:, c])
            uniq, inv = np.unique(tokens.astype(str), return_inverse=True)
            pos = np.array([self._index.get((c, u), -1) for u in uniq.tolist()], dtype=np.int64)
            out_pos = pos[inv]
            unknown = (out_pos < 0) & ~miss
            if unknown.any() and self.handle_unknown == "error":
                raise ValueError(f"unknown categories {sorted(set(tokens[unknown]))} "
                                 f"in column {c}")
            ok = (out_pos >= 0) & ~miss
            block[np.nonzero(ok)[0], out_pos[ok]] = 1.0
        return np.hstack([passthrough, block])

    def get_feature_names(self, input_names: Optional[Sequence[str]] = None):
        if self._n_features is None:
            raise NotFittedError("OneHotEncoder is not fitted")
        names = list(input_names) if input_names is not None else \
            [f"x{j}" for j in range(self._n_features)]
        cols = set(self._cat_cols(self._n_features))
        return [names[j] for j in range(self._n_features) if j not in cols] + \
               [f"{names[c]}={cat}" for c, cat in self.feature_map_]
