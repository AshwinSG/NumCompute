"""Running statistics that are updated one chunk at a time.

All classes have the same methods:
    update_stats(X_chunk) adds a new chunk of shape (n_samples, n_features)
    to the running totals (update is another name for it).
    reset() forgets everything.

Statistics are always worked out for each feature, along axis=0.
NaNs are skipped (each feature keeps its own count of non-missing values).

Classes:
    StreamingStats      count, mean, variance, min and max (Chan et al. merge)
    StreamingQuantile   quantiles from a random sample (exact until the sample is full)
    StreamingHistogram  histograms with fixed bins, can count only the last few chunks
    StreamingSummary    the three above, updated together
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import numpy as np

from .utils import NotFittedError, check_array


# --------------------------------------------------------------------------- #
# Merging counts, means and variances
# --------------------------------------------------------------------------- #
def chunk_moments(X: np.ndarray):
    """Count, mean and M2 for each feature of a 2-D chunk, skipping NaNs.

    M2 is the sum of squared distances from the chunk mean. Columns with no
    values get count=0, mean=0, M2=0.
    """
    mask = ~np.isnan(X)
    n = mask.sum(axis=0).astype(float)
    safe_n = np.where(n > 0, n, 1.0)
    mean = np.where(mask, X, 0.0).sum(axis=0) / safe_n
    dev = np.where(mask, X - mean, 0.0)
    M2 = (dev * dev).sum(axis=0)
    return n, mean, M2


def combine_moments(n_a, mean_a, M2_a, n_b, mean_b, M2_b):
    """Merge two sets of (count, mean, M2), using the method of Chan, Golub
    and LeVeque (1979).

    It works with distances from the mean instead of raw sums of squares, so
    it stays accurate even when all the values are very large. It handles all
    features at once and copes with a count of zero on either side.
    """
    n = n_a + n_b
    safe_n = np.where(n > 0, n, 1.0)
    delta = mean_b - mean_a
    mean = mean_a + delta * (n_b / safe_n)
    M2 = M2_a + M2_b + delta * delta * (n_a * n_b / safe_n)
    return n, np.where(n > 0, mean, 0.0), np.where(n > 0, M2, 0.0)


def welford_loop(X) -> tuple:
    """Slow loop version of the Welford update (one sample at a time).

    Used in the speed tests to compare against StreamingStats, which uses
    NumPy array operations. Returns (count, mean, var) for each feature
    (variance divided by n, not n - 1).
    """
    X = check_array(X)
    n = np.zeros(X.shape[1])
    mean = np.zeros(X.shape[1])
    M2 = np.zeros(X.shape[1])
    for row in X:
        for j in range(X.shape[1]):
            x = row[j]
            if x != x:  # NaN
                continue
            n[j] += 1
            d = x - mean[j]
            mean[j] += d / n[j]
            M2[j] += d * (x - mean[j])
    var = np.where(n > 0, M2 / np.where(n > 0, n, 1), np.nan)
    return n, np.where(n > 0, mean, np.nan), var


class StreamingStats:
    """Running count, mean, variance, min and max for each feature.

    Parameters:
        growth_fill: float or None, default None. If not None, chunks with
            more columns than seen so far are accepted (for example after a
            OneHotEncoder finds a new category). The new columns are set up
            as if every earlier row held growth_fill. This is exact for
            one-hot columns with growth_fill=0. If None, a change in the
            number of columns raises ValueError.

    Attributes:
        n_: array (n_features,), count of non-missing values per feature
        n_rows_: int, total rows seen
        mean_, M2_, min_, max_: arrays (n_features,)
    """

    def __init__(self, growth_fill: Optional[float] = None):
        self.growth_fill = growth_fill
        self.reset()

    def reset(self):
        self.n_ = None
        self.mean_ = None
        self.M2_ = None
        self.min_ = None
        self.max_ = None
        self.n_rows_ = 0
        return self

    @property
    def n_features_in_(self) -> int:
        return 0 if self.n_ is None else self.n_.shape[0]

    def _grow(self, width: int):
        extra = width - self.n_features_in_
        fill = float(self.growth_fill)
        self.n_ = np.concatenate([self.n_, np.full(extra, float(self.n_rows_))])
        self.mean_ = np.concatenate([self.mean_, np.full(extra, fill)])
        self.M2_ = np.concatenate([self.M2_, np.zeros(extra)])
        mm = fill if self.n_rows_ else np.inf
        self.min_ = np.concatenate([self.min_, np.full(extra, mm)])
        self.max_ = np.concatenate([self.max_, np.full(extra, -mm)])

    def update_stats(self, X_chunk):
        """Add X_chunk (n_samples, n_features) to the running totals."""
        X = check_array(X_chunk, "X_chunk")
        width = X.shape[1]
        if self.n_ is None:
            self.n_ = np.zeros(width)
            self.mean_ = np.zeros(width)
            self.M2_ = np.zeros(width)
            self.min_ = np.full(width, np.inf)
            self.max_ = np.full(width, -np.inf)
        elif width != self.n_features_in_:
            if width > self.n_features_in_ and self.growth_fill is not None:
                self._grow(width)
            else:
                raise ValueError(
                    f"X_chunk has {width} features but previous chunks had "
                    f"{self.n_features_in_}")
        n_b, mean_b, M2_b = chunk_moments(X)
        self.n_, self.mean_, self.M2_ = combine_moments(
            self.n_, self.mean_, self.M2_, n_b, mean_b, M2_b)
        nan = np.isnan(X)
        self.min_ = np.minimum(self.min_, np.where(nan, np.inf, X).min(axis=0))
        self.max_ = np.maximum(self.max_, np.where(nan, -np.inf, X).max(axis=0))
        self.n_rows_ += X.shape[0]
        return self

    update = update_stats

    def merge(self, other: "StreamingStats") -> "StreamingStats":
        """Combine with another StreamingStats (for example one filled by a parallel worker)."""
        if other.n_ is None:
            return self
        if self.n_ is None:
            for k in ("n_", "mean_", "M2_", "min_", "max_"):
                setattr(self, k, getattr(other, k).copy())
            self.n_rows_ = other.n_rows_
            return self
        if other.n_features_in_ != self.n_features_in_:
            raise ValueError("cannot merge accumulators with different widths")
        self.n_, self.mean_, self.M2_ = combine_moments(
            self.n_, self.mean_, self.M2_, other.n_, other.mean_, other.M2_)
        self.min_ = np.minimum(self.min_, other.min_)
        self.max_ = np.maximum(self.max_, other.max_)
        self.n_rows_ += other.n_rows_
        return self

    def _check(self):
        if self.n_ is None:
            raise NotFittedError("StreamingStats has not seen any data; call update_stats first")

    @property
    def count(self) -> np.ndarray:
        self._check()
        return self.n_.copy()

    @property
    def mean(self) -> np.ndarray:
        """Mean of each feature. NaN for features with no values yet."""
        self._check()
        return np.where(self.n_ > 0, self.mean_, np.nan)

    def var(self, ddof: int = 0) -> np.ndarray:
        """Variance of each feature, dividing by count - ddof.

        Returns NaN where count - ddof <= 0. Tiny negative values caused by
        rounding are set to 0.
        """
        self._check()
        denom = self.n_ - ddof
        out = np.where(denom > 0, self.M2_ / np.where(denom > 0, denom, 1.0), np.nan)
        return np.where(np.isnan(out), out, np.maximum(out, 0.0))

    def std(self, ddof: int = 0) -> np.ndarray:
        return np.sqrt(self.var(ddof))

    @property
    def min(self) -> np.ndarray:
        self._check()
        return np.where(self.n_ > 0, self.min_, np.nan)

    @property
    def max(self) -> np.ndarray:
        self._check()
        return np.where(self.n_ > 0, self.max_, np.nan)


class StreamingQuantile:
    """Rough quantiles for each feature, taken from a random sample of rows.

    The sample (called a reservoir) keeps a fair random pick of at most
    capacity rows, so quantiles are exact while fewer than capacity rows have
    been seen, and a fair estimate after that. Memory stays at
    capacity * n_features values no matter how long the stream is.

    Parameters:
        capacity: int, default 10_000
        random_state: int or None
    """

    def __init__(self, capacity: int = 10_000, random_state: Optional[int] = None):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.random_state = random_state
        self.reset()

    def reset(self):
        self._rng = np.random.default_rng(self.random_state)
        self.reservoir_ = None
        self.n_seen_ = 0
        self._filled = 0
        return self

    def update_stats(self, X_chunk):
        X = check_array(X_chunk, "X_chunk")
        if self.reservoir_ is None:
            self.reservoir_ = np.full((self.capacity, X.shape[1]), np.nan)
        elif X.shape[1] != self.reservoir_.shape[1]:
            raise ValueError(f"X_chunk has {X.shape[1]} features, expected "
                             f"{self.reservoir_.shape[1]}")
        # 1) put rows straight into empty slots
        free = self.capacity - self._filled
        k = min(free, X.shape[0])
        if k:
            self.reservoir_[self._filled:self._filled + k] = X[:k]
            self._filled += k
        rest = X[k:]
        if rest.shape[0]:
            # 2) reservoir sampling for the whole chunk at once: the row at
            #    overall position i (counting from 1) picks a random slot j
            #    from 0 to i-1, and replaces it if j < capacity.
            idx = self.n_seen_ + k + np.arange(1, rest.shape[0] + 1)
            j = (self._rng.random(rest.shape[0]) * idx).astype(np.int64)
            keep = j < self.capacity
            # if two rows pick the same slot, the later row wins, the same as
            # going one row at a time. NumPy index assignment does this for us.
            self.reservoir_[j[keep]] = rest[keep]
        self.n_seen_ += X.shape[0]
        return self

    update = update_stats

    def quantile(self, q) -> np.ndarray:
        """Quantile(s) q between 0 and 1. Returns shape (n_features,) for a
        single q, or (len(q), n_features) for a list. NaNs are skipped."""
        if self.reservoir_ is None:
            raise NotFittedError("StreamingQuantile has not seen any data")
        q_arr = np.asarray(q, dtype=float)
        if np.any((q_arr < 0) | (q_arr > 1)):
            raise ValueError(f"quantiles must lie in [0, 1], got {q}")
        sample = self.reservoir_[:self._filled]
        all_nan = np.isnan(sample).all(axis=0)
        filled = np.where(all_nan, 0.0, sample)  # avoid warnings for all-NaN columns
        out = np.nanquantile(filled, q_arr, axis=0)
        return np.where(all_nan, np.nan, out)

    def median(self) -> np.ndarray:
        return self.quantile(0.5)


class StreamingHistogram:
    """Histograms for each feature, with fixed bins of equal width.

    Parameters:
        bins: int, default 20
        range: (float, float) or None. Bin range used for all features. If
            None, each feature gets its own range from the first chunk (a
            feature with only one value gets a range of width 1 around it).
            Values outside the range are put in the first or last bin and
            counted in n_underflow_ and n_overflow_.
        window: int or None. If set, only the last window chunks are
            counted.

    Attributes:
        edges_: array (n_features, bins + 1)
        counts: array (n_features, bins) (a property)
    """

    def __init__(self, bins: int = 20, range: Optional[tuple] = None,
                 window: Optional[int] = None):
        if bins < 1:
            raise ValueError(f"bins must be >= 1, got {bins}")
        if window is not None and window < 1:
            raise ValueError(f"window must be >= 1 or None, got {window}")
        self.bins = int(bins)
        self.range = range
        self.window = window
        self.reset()

    def reset(self):
        self.edges_ = None
        self._total = None
        self._chunks = deque(maxlen=self.window) if self.window else None
        self.n_underflow_ = None
        self.n_overflow_ = None
        return self

    def _init_edges(self, X):
        d = X.shape[1]
        if self.range is not None:
            lo, hi = map(float, self.range)
            if not hi > lo:
                raise ValueError(f"range must satisfy max > min, got {self.range}")
            lo, hi = np.full(d, lo), np.full(d, hi)
        else:
            nan = np.isnan(X)
            lo = np.where(nan, np.inf, X).min(axis=0)
            hi = np.where(nan, -np.inf, X).max(axis=0)
            empty = ~np.isfinite(lo)
            lo = np.where(empty, 0.0, lo)
            hi = np.where(empty, 1.0, hi)
            flat = hi <= lo
            lo = np.where(flat, lo - 0.5, lo)
            hi = np.where(flat, hi + 0.5, hi)
        t = np.linspace(0.0, 1.0, self.bins + 1)
        self.edges_ = lo[:, None] + (hi - lo)[:, None] * t[None, :]
        self._total = np.zeros((d, self.bins), dtype=np.int64)
        self.n_underflow_ = np.zeros(d, dtype=np.int64)
        self.n_overflow_ = np.zeros(d, dtype=np.int64)

    def update_stats(self, X_chunk):
        X = check_array(X_chunk, "X_chunk")
        if self.edges_ is None:
            self._init_edges(X)
        elif X.shape[1] != self.edges_.shape[0]:
            raise ValueError(f"X_chunk has {X.shape[1]} features, expected {self.edges_.shape[0]}")
        d = X.shape[1]
        lo, hi = self.edges_[:, 0], self.edges_[:, -1]
        width = (hi - lo) / self.bins
        valid = ~np.isnan(X)
        self.n_underflow_ += (valid & (X < lo)).sum(axis=0)
        self.n_overflow_ += (valid & (X > hi)).sum(axis=0)
        idx = np.floor((np.where(valid, X, lo) - lo) / width).astype(np.int64)
        idx = np.clip(idx, 0, self.bins - 1)
        flat = (idx + np.arange(d)[None, :] * self.bins)[valid]
        counts = np.bincount(flat, minlength=d * self.bins).reshape(d, self.bins)
        if self._chunks is not None:
            if len(self._chunks) == self._chunks.maxlen:
                self._total -= self._chunks[0]
            self._chunks.append(counts)
        self._total += counts
        return self

    update = update_stats

    @property
    def counts(self) -> np.ndarray:
        if self._total is None:
            raise NotFittedError("StreamingHistogram has not seen any data")
        return self._total.copy()

    def density(self) -> np.ndarray:
        """Histogram scaled so its area is 1 for each feature (all 0 if empty)."""
        c = self.counts.astype(float)
        widths = np.diff(self.edges_, axis=1)
        tot = c.sum(axis=1, keepdims=True)
        return np.where(tot > 0, c / np.where(tot > 0, tot, 1) / widths, 0.0)

    def quantile(self, q) -> np.ndarray:
        """Rough quantile, found by drawing a straight line inside the bin."""
        q = float(q)
        if not 0 <= q <= 1:
            raise ValueError(f"q must lie in [0, 1], got {q}")
        c = self.counts.astype(float)
        cum = np.cumsum(c, axis=1)
        tot = cum[:, -1]
        target = q * tot
        b = np.minimum((cum < target[:, None]).sum(axis=1), self.bins - 1)
        rows = np.arange(c.shape[0])
        prev = np.where(b > 0, cum[rows, np.maximum(b - 1, 0)], 0.0)
        frac = np.where(c[rows, b] > 0, (target - prev) / np.where(c[rows, b] > 0, c[rows, b], 1), 0.0)
        lo = self.edges_[rows, b]
        hi = self.edges_[rows, b + 1]
        return np.where(tot > 0, lo + frac * (hi - lo), np.nan)


class StreamingSummary:
    """StreamingStats, StreamingQuantile and StreamingHistogram, all updated
    by one update_stats call.

    Example:
        s = StreamingSummary(quantiles=(0.25, 0.5, 0.75))
        for chunk in chunks:
            s.update_stats(chunk)
        s.result()["mean"]
    """

    def __init__(self, quantiles: Sequence[float] = (0.25, 0.5, 0.75),
                 bins: int = 20, window: Optional[int] = None,
                 capacity: int = 10_000, random_state: Optional[int] = None):
        self.quantiles = tuple(quantiles)
        self.moments = StreamingStats()
        self.quantile_sketch = StreamingQuantile(capacity, random_state)
        self.histogram = StreamingHistogram(bins=bins, window=window)

    def update_stats(self, X_chunk):
        X = check_array(X_chunk, "X_chunk")
        self.moments.update_stats(X)
        self.quantile_sketch.update_stats(X)
        self.histogram.update_stats(X)
        return self

    update = update_stats

    def reset(self):
        self.moments.reset()
        self.quantile_sketch.reset()
        self.histogram.reset()
        return self

    def result(self) -> dict:
        return {
            "count": self.moments.count,
            "mean": self.moments.mean,
            "var": self.moments.var(),
            "std": self.moments.std(),
            "min": self.moments.min,
            "max": self.moments.max,
            "quantiles": dict(zip(self.quantiles,
                                  self.quantile_sketch.quantile(self.quantiles))),
            "histogram": self.histogram.counts,
        }
