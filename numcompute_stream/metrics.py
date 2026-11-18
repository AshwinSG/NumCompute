"""Classification metrics: plain functions and running (streaming) versions.

Functions that take whole arrays:
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score.

Streaming metrics all have these methods:
    m.update(y_true_chunk, y_pred_chunk)   # or y_score_chunk for AUC
    m.result()                             # current value
    m.reset()

Pass window=k to work out the metric over only the last k chunks.
window=None counts the whole stream. Labels can be ints or strings, and new
labels can appear at any time. The confusion matrix grows when needed.

Ratios that cannot be worked out (for example precision of a class that was
never predicted) are returned as 0.0 (zero_division=0) instead of NaN or a
warning.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .utils import check_same_length

_AVERAGES = ("macro", "micro", "weighted", "binary", None)


# --------------------------------------------------------------------------- #
# Helpers based on the confusion matrix
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    """Confusion matrix built with NumPy array operations (no Python loop).

    Returns:
        cm: array (n_labels, n_labels). cm[i, j] is the number of samples
        with true label labels[i] that were predicted as labels[j]. By
        default labels are all values from both arrays, sorted. Samples
        whose label is not in labels are skipped.
    """
    y_true, y_pred = check_same_length(y_true, y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    k = len(labels)
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]

    def codes(y):
        pos = np.searchsorted(sorted_labels, y)
        pos = np.clip(pos, 0, k - 1)
        valid = sorted_labels[pos] == y
        return order[pos], valid

    t, vt = codes(y_true)
    p, vp = codes(y_pred)
    v = vt & vp
    return np.bincount(t[v] * k + p[v], minlength=k * k).reshape(k, k)


def confusion_matrix_loop(y_true, y_pred, labels=None) -> np.ndarray:
    """Slow loop version of confusion_matrix (only used in the speed tests)."""
    y_true, y_pred = check_same_length(y_true, y_pred)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    index = {lab: i for i, lab in enumerate(list(labels))}
    cm = np.zeros((len(index), len(index)), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if t in index and p in index:
            cm[index[t], index[p]] += 1
    return cm


def _safe_div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def _prf_from_cm(cm, labels, kind, average, pos_label):
    if average not in _AVERAGES:
        raise ValueError(f"average must be one of {_AVERAGES}, got {average!r}")
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(axis=0).astype(float)
    true_pos = cm.sum(axis=1).astype(float)
    if cm.size == 0:
        return 0.0

    def score(tp_, pp_, ap_):
        p = _safe_div(tp_, pp_)
        r = _safe_div(tp_, ap_)
        if kind == "precision":
            return p
        if kind == "recall":
            return r
        return _safe_div(2 * p * r, p + r)

    if average == "micro":
        return float(score(tp.sum(), pred_pos.sum(), true_pos.sum()))
    per_class = score(tp, pred_pos, true_pos)
    if average is None:
        return per_class
    if average == "binary":
        labels = list(np.asarray(labels).tolist())
        if pos_label not in labels:
            # positive class never seen: precision and recall are taken as 0
            return 0.0
        return float(per_class[labels.index(pos_label)])
    if average == "weighted":
        w = true_pos
        return float(_safe_div((per_class * w).sum(), w.sum()))
    return float(per_class.mean())


def accuracy_score(y_true, y_pred) -> float:
    """Share of predictions that match exactly. Empty input returns 0.0."""
    y_true, y_pred = check_same_length(y_true, y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def precision_score(y_true, y_pred, average="macro", pos_label=1, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels)
    labs = labels if labels is not None else np.unique(np.concatenate(
        [np.asarray(y_true), np.asarray(y_pred)]))
    return _prf_from_cm(cm, labs, "precision", average, pos_label)


def recall_score(y_true, y_pred, average="macro", pos_label=1, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels)
    labs = labels if labels is not None else np.unique(np.concatenate(
        [np.asarray(y_true), np.asarray(y_pred)]))
    return _prf_from_cm(cm, labs, "recall", average, pos_label)


def f1_score(y_true, y_pred, average="macro", pos_label=1, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels)
    labs = labels if labels is not None else np.unique(np.concatenate(
        [np.asarray(y_true), np.asarray(y_pred)]))
    return _prf_from_cm(cm, labs, "f1", average, pos_label)


def _binary_targets(y_true, pos_label):
    y_true = np.asarray(y_true)
    if pos_label is None:
        uniq = np.unique(y_true)
        if uniq.size > 2:
            raise ValueError(f"AUC is binary-only; y_true has {uniq.size} classes")
        pos_label = uniq.max() if uniq.size else 1
    return (y_true == pos_label)


def _score_1d(y_score):
    s = np.asarray(y_score, dtype=float)
    if s.ndim == 2 and s.shape[1] == 2:
        s = s[:, 1]
    if s.ndim != 1:
        raise ValueError(f"y_score must be 1-D (or (n, 2) probabilities), got shape {s.shape}")
    if np.isnan(s).any():
        raise ValueError("y_score contains NaN")
    return s


def roc_auc_score(y_true, y_score, pos_label=None) -> float:
    """Exact ROC AUC for two classes, worked out from ranks (Mann-Whitney U).

    Equal scores get the average rank, so a tie between a positive and a
    negative counts as 1/2. Returns NaN if only one class is present.
    """
    s = _score_1d(y_score)
    pos = _binary_targets(y_true, pos_label)
    if pos.shape[0] != s.shape[0]:
        raise ValueError(f"y_true and y_score lengths differ: {pos.shape[0]} vs {s.shape[0]}")
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    # average ranks for ties
    _, first, counts = np.unique(s_sorted, return_index=True, return_counts=True)
    avg_rank = first + (counts + 1) / 2.0          # average rank, counting from 1
    ranks = np.empty_like(s)
    ranks[order] = np.repeat(avg_rank, counts)
    u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


# --------------------------------------------------------------------------- #
# Streaming metrics
# --------------------------------------------------------------------------- #
class StreamingMetric:
    """Base class with update, result and reset.

    requires_score tells callers such as StreamTrainer whether update expects
    predicted classes (False) or scores for the positive class (True).
    """

    requires_score = False

    def update(self, y_true_chunk, y_pred_chunk):  # pragma: no cover - abstract
        raise NotImplementedError

    def result(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def reset(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}(window={getattr(self, 'window', None)})"


class StreamingConfusionMatrix(StreamingMetric):
    """Confusion matrix added up over a stream (can be limited to the last few chunks).

    Attributes:
        labels: array of sorted labels, in the same order as the rows and
            columns of result()
    """

    def __init__(self, window: Optional[int] = None):
        if window is not None and window < 1:
            raise ValueError(f"window must be >= 1 or None, got {window}")
        self.window = window
        self.reset()

    def reset(self):
        self._labels = []           # in the order they were first seen
        self._index = {}
        self._total = np.zeros((0, 0), dtype=np.int64)
        self._chunks = deque(maxlen=self.window) if self.window else None
        self.n_updates_ = 0
        return self

    @staticmethod
    def _pad(m, k):
        if m.shape[0] == k:
            return m
        out = np.zeros((k, k), dtype=m.dtype)
        out[:m.shape[0], :m.shape[1]] = m
        return out

    def update(self, y_true_chunk, y_pred_chunk):
        y_true, y_pred = check_same_length(y_true_chunk, y_pred_chunk)
        for lab in np.unique(np.concatenate([y_true, y_pred])).tolist():
            if lab not in self._index:
                self._index[lab] = len(self._labels)
                self._labels.append(lab)
        k = len(self._labels)
        cm = confusion_matrix(y_true, y_pred, labels=np.array(self._labels))
        if self._chunks is not None:
            self._chunks.append(cm)
            self._total = sum((self._pad(c, k) for c in self._chunks),
                              np.zeros((k, k), dtype=np.int64))
        else:
            self._total = self._pad(self._total, k) + cm
        self.n_updates_ += 1
        return self

    @property
    def labels(self) -> np.ndarray:
        return np.array(sorted(self._labels)) if self._labels else np.array([])

    def result(self) -> np.ndarray:
        if not self._labels:
            return np.zeros((0, 0), dtype=np.int64)
        order = np.argsort(np.array(self._labels), kind="stable")
        return self._total[np.ix_(order, order)].copy()


class _CMMetric(StreamingMetric):
    def __init__(self, window: Optional[int] = None):
        self.window = window
        self.cm = StreamingConfusionMatrix(window)

    def update(self, y_true_chunk, y_pred_chunk):
        self.cm.update(y_true_chunk, y_pred_chunk)
        return self

    def reset(self):
        self.cm.reset()
        return self


class StreamingAccuracy(_CMMetric):
    """Running accuracy (or accuracy over the last few chunks). 0.0 before any update."""

    def result(self) -> float:
        m = self.cm.result()
        tot = m.sum()
        return float(np.trace(m) / tot) if tot else 0.0


class StreamingErrorRate(StreamingAccuracy):
    """1 - accuracy (0.0 before any update)."""

    def result(self) -> float:
        return 0.0 if self.cm.result().sum() == 0 else 1.0 - super().result()


class _PRF(_CMMetric):
    kind = None

    def __init__(self, average="macro", pos_label=1, window: Optional[int] = None):
        if average not in _AVERAGES:
            raise ValueError(f"average must be one of {_AVERAGES}, got {average!r}")
        super().__init__(window)
        self.average = average
        self.pos_label = pos_label

    def result(self):
        return _prf_from_cm(self.cm.result(), self.cm.labels, self.kind,
                            self.average, self.pos_label)


class StreamingPrecision(_PRF):
    """Running precision (average can be macro, micro, weighted, binary or None)."""
    kind = "precision"


class StreamingRecall(_PRF):
    """Running recall (average can be macro, micro, weighted, binary or None)."""
    kind = "recall"


class StreamingF1(_PRF):
    """Running F1 (average can be macro, micro, weighted, binary or None)."""
    kind = "f1"


class StreamingAUC(StreamingMetric):
    """ROC AUC for two classes, added up over a stream using a fixed amount of memory.

    Scores (expected between 0 and 1, for example predict_proba[:, 1]) are
    put into n_bins bins of equal width for each class. Pairs that land in
    the same bin count as ties (1/2), so the result is within about
    1/n_bins of the exact AUC, while memory stays at n_bins counts.

    Call update(y_true_chunk, y_score_chunk). result() is NaN until both
    classes have been seen.
    """

    requires_score = True

    def __init__(self, n_bins: int = 1000, pos_label=1, window: Optional[int] = None):
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self.n_bins = n_bins
        self.pos_label = pos_label
        self.window = window
        self.reset()

    def reset(self):
        self._pos = np.zeros(self.n_bins, dtype=np.int64)
        self._neg = np.zeros(self.n_bins, dtype=np.int64)
        self._chunks = deque(maxlen=self.window) if self.window else None
        return self

    def update(self, y_true_chunk, y_score_chunk):
        s = _score_1d(y_score_chunk)
        y = np.asarray(y_true_chunk)
        if y.shape[0] != s.shape[0]:
            raise ValueError(f"y_true and y_score lengths differ: {y.shape[0]} vs {s.shape[0]}")
        is_pos = y == self.pos_label
        b = np.clip((np.clip(s, 0.0, 1.0) * self.n_bins).astype(np.int64), 0, self.n_bins - 1)
        pos = np.bincount(b[is_pos], minlength=self.n_bins)
        neg = np.bincount(b[~is_pos], minlength=self.n_bins)
        if self._chunks is not None:
            if len(self._chunks) == self._chunks.maxlen:
                old_p, old_n = self._chunks[0]
                self._pos -= old_p
                self._neg -= old_n
            self._chunks.append((pos, neg))
        self._pos += pos
        self._neg += neg
        return self

    def result(self) -> float:
        P, N = self._pos.sum(), self._neg.sum()
        if P == 0 or N == 0:
            return float("nan")
        neg_below = np.concatenate([[0], np.cumsum(self._neg)[:-1]])
        u = (self._pos * neg_below).sum() + 0.5 * (self._pos * self._neg).sum()
        return float(u / (P * N))
