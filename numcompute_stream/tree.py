"""CART decision tree classifier with buffered retraining for streams.

Algorithm
---------
* Binary splits ``x[f] <= threshold`` chosen greedily by impurity decrease
  (Gini or entropy).
* **Vectorised split search**: for a node, all candidate features are sorted
  at once (``argsort(axis=0)``), class counts to the left of every candidate
  threshold come from a single ``cumsum`` over a one-hot label tensor, and the
  impurity of every (threshold, feature) pair is evaluated in one shot. Only
  the loop over *nodes* is in Python (an explicit stack, no recursion).
* **Missing values**: NaNs are ignored when scoring a split; the gain is
  scaled by the fraction of non-missing rows (C4.5 style) and missing rows are
  routed to the larger child (``nan_left_``), both at fit and predict time.
* **Ties**: thresholds are only placed *between distinct values*; among
  equally good splits the lowest feature index, then the lowest threshold,
  wins; among equally likely classes the smallest label wins. Results are
  therefore deterministic for a fixed ``random_state``.

Streaming
---------
``partial_fit`` appends each chunk to an internal buffer. Once the buffer
holds at least ``buffer_size`` samples the tree is **retrained from scratch
on the buffered data** and the buffer is cleared. Between retrains,
``predict`` uses the most recently trained tree. ``flush()`` forces a retrain
on whatever is buffered (e.g. at the end of a stream).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .utils import NotFittedError, check_array, check_X_y

_LEAF = -1
_SPLIT_BLOCK_ELEMS = 4_000_000  # cap on n * features * classes per block


# --------------------------------------------------------------------------- #
# Impurity
# --------------------------------------------------------------------------- #
def impurity(counts: np.ndarray, criterion: str = "gini") -> np.ndarray:
    """Impurity of class-count vectors along the last axis.

    Parameters
    ----------
    counts : ndarray (..., n_classes)
    criterion : {"gini", "entropy"}

    Returns
    -------
    ndarray (...) – 0 for empty nodes.
    """
    counts = np.asarray(counts, dtype=float)
    tot = counts.sum(axis=-1, keepdims=True)
    p = counts / np.where(tot > 0, tot, 1.0)
    if criterion == "gini":
        out = 1.0 - (p * p).sum(axis=-1)
    elif criterion == "entropy":
        logp = np.log2(np.where(p > 0, p, 1.0))
        out = -(p * logp).sum(axis=-1)
    else:
        raise ValueError(f"criterion must be 'gini' or 'entropy', got {criterion!r}")
    return np.where(tot[..., 0] > 0, out, 0.0)


def _midpoint(a, b):
    """Overflow-safe midpoint that is guaranteed to satisfy a <= m < b."""
    m = a + (b - a) / 2.0
    return np.where(m < b, m, a)


# --------------------------------------------------------------------------- #
# Split search
# --------------------------------------------------------------------------- #
def best_split(X, y, n_classes, criterion="gini", min_samples_leaf=1, features=None):
    """Vectorised search for the best binary split.

    Parameters
    ----------
    X : ndarray (n_samples, n_features), may contain NaN
    y : ndarray (n_samples,) of int class codes in ``[0, n_classes)``
    features : ndarray of candidate feature indices (default: all)

    Returns
    -------
    (feature, threshold, gain, nan_left) or None if no valid split exists.
    """
    n = X.shape[0]
    features = np.arange(X.shape[1]) if features is None else np.asarray(features)
    if n < 2 or features.size == 0:
        return None
    onehot = np.zeros((n, n_classes))
    onehot[np.arange(n), y] = 1.0

    best = (-np.inf, None)
    block = max(1, _SPLIT_BLOCK_ELEMS // max(1, n * n_classes))
    for start in range(0, features.size, block):
        feats = features[start:start + block]
        Xf = X[:, feats]                                        # (n, f)
        order = np.argsort(Xf, axis=0, kind="stable")           # NaNs sorted last
        Xs = np.take_along_axis(Xf, order, axis=0)
        valid = ~np.isnan(Xs)
        n_valid = valid.sum(axis=0).astype(float)               # (f,)
        Ys = onehot[order] * valid[..., None]                   # (n, f, K)
        left = np.cumsum(Ys, axis=0)[:-1]                       # (n-1, f, K)
        total = left[-1] + Ys[-1] if n > 1 else Ys[-1]          # (f, K)
        right = total[None] - left
        nl = np.arange(1, n, dtype=float)[:, None]              # (n-1, 1)
        nr = n_valid[None, :] - nl
        ok = (valid[1:] & (Xs[1:] > np.where(valid[:-1], Xs[:-1], np.inf))
              & (nl >= min_samples_leaf) & (nr >= min_samples_leaf))
        if not ok.any():
            continue
        safe_nv = np.where(n_valid > 0, n_valid, 1.0)
        child = (nl * impurity(left, criterion) + nr * impurity(right, criterion)) / safe_nv
        gain = (n_valid / n) * (impurity(total, criterion)[None, :] - child)
        gain = np.where(ok, gain, -np.inf)
        # transpose so ties resolve to lowest feature, then lowest threshold
        flat = int(np.argmax(gain.T))
        fi, pos = divmod(flat, n - 1)
        g = gain[pos, fi]
        if g > best[0]:
            thr = float(_midpoint(Xs[pos, fi], Xs[pos + 1, fi]))
            nan_left = bool(nl[pos, 0] >= nr[pos, fi])
            best = (g, (int(feats[fi]), thr, float(g), nan_left))
    return best[1]


def best_split_loop(X, y, n_classes, criterion="gini", min_samples_leaf=1, features=None):
    """Reference **loop-based** split search with identical semantics to
    :func:`best_split` (used for benchmarking and correctness tests)."""
    n = X.shape[0]
    features = range(X.shape[1]) if features is None else features
    best_gain, best = -np.inf, None
    for f in features:
        col = X[:, f]
        idx = [i for i in np.argsort(col, kind="stable") if not np.isnan(col[i])]
        nv = len(idx)
        if nv < 2:
            continue
        total = [0.0] * n_classes
        for i in idx:
            total[y[i]] += 1
        parent = float(impurity(np.array(total), criterion))
        left = [0.0] * n_classes
        for k in range(nv - 1):
            left[y[idx[k]]] += 1
            a, b = col[idx[k]], col[idx[k + 1]]
            n_l, n_r = k + 1, nv - k - 1
            if b <= a or n_l < min_samples_leaf or n_r < min_samples_leaf:
                continue
            right = [t - l for t, l in zip(total, left)]
            child = (n_l * float(impurity(np.array(left), criterion))
                     + n_r * float(impurity(np.array(right), criterion))) / nv
            g = (nv / n) * (parent - child)
            if g > best_gain:
                best_gain = g
                best = (int(f), float(_midpoint(a, b)), float(g), n_l >= n_r)
    return best


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #
class DecisionTreeClassifier:
    """Depth-limited CART classifier supporting streaming via buffered retraining.

    Parameters
    ----------
    max_depth : int or None
        Maximum depth (root has depth 0). None = grow until pure/min sizes.
    min_samples_split : int, default 2
        Minimum samples in a node to consider splitting it.
    min_samples_leaf : int, default 1
        Minimum (non-missing) samples on each side of a split.
    max_features : None, int, float, "sqrt" or "log2"
        Number of features sampled (without replacement) at each node.
    criterion : {"gini", "entropy"}
    buffer_size : int or None, default None
        Streaming buffer size. ``partial_fit`` retrains once the buffer holds
        ``>= buffer_size`` samples; None retrains on every ``partial_fit``.
    min_impurity_decrease : float, default 0.0
        A split must improve impurity by more than this.
    pad_value : float, default 0.0
        If a chunk arrives with more (trailing) features than the buffered
        chunks – e.g. after a one-hot encoder discovered a new category – the
        buffered rows are padded with this value.
    random_state : int or None

    Attributes
    ----------
    classes_ : ndarray – sorted labels seen so far (grows while streaming)
    n_features_in_ : int – width of the data the current tree was trained on
    n_retrains_ : int, retrain_times_ : list[float] (seconds)
    feature_, threshold_, left_, right_, nan_left_, value_ : node arrays

    Shapes
    ------
    fit / partial_fit : X (n_samples, n_features), y (n_samples,)
    predict : X (n_samples, >= n_features_in_) -> (n_samples,)
    predict_proba : -> (n_samples, len(classes_))
    """

    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, max_features=None, criterion: str = "gini",
                 buffer_size: Optional[int] = None, min_impurity_decrease: float = 0.0,
                 pad_value: float = 0.0, random_state: Optional[int] = None):
        if max_depth is not None and (not isinstance(max_depth, (int, np.integer)) or max_depth < 0):
            raise ValueError(f"max_depth must be a non-negative int or None, got {max_depth!r}")
        if min_samples_split < 2:
            raise ValueError(f"min_samples_split must be >= 2, got {min_samples_split}")
        if min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf}")
        if criterion not in ("gini", "entropy"):
            raise ValueError(f"criterion must be 'gini' or 'entropy', got {criterion!r}")
        if buffer_size is not None and (not isinstance(buffer_size, (int, np.integer)) or buffer_size < 1):
            raise ValueError(f"buffer_size must be a positive int or None, got {buffer_size!r}")
        if isinstance(max_features, str) and max_features not in ("sqrt", "log2"):
            raise ValueError(f"max_features string must be 'sqrt' or 'log2', got {max_features!r}")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.buffer_size = buffer_size
        self.min_impurity_decrease = min_impurity_decrease
        self.pad_value = pad_value
        self.random_state = random_state
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        """Forget the tree, the buffer, the classes and all counters."""
        self.classes_ = None
        self.feature_ = None
        self.n_features_in_ = None
        self._rng = np.random.default_rng(self.random_state)
        self._buf_X, self._buf_y = [], []
        self._buf_n = 0
        self._buf_width = None
        self.n_retrains_ = 0
        self.retrain_times_ = []
        self.n_samples_seen_ = 0
        return self

    @property
    def is_fitted(self) -> bool:
        return self.feature_ is not None

    @property
    def buffer_count(self) -> int:
        """Number of samples currently waiting in the buffer."""
        return self._buf_n

    @property
    def buffer_nbytes(self) -> int:
        return int(sum(a.nbytes for a in self._buf_X) + sum(a.nbytes for a in self._buf_y))

    # --------------------------------------------------------------- training
    def _update_classes(self, y, classes=None):
        new = np.unique(y)
        if classes is not None:
            new = np.union1d(new, np.asarray(classes))
        self.classes_ = new if self.classes_ is None else np.union1d(self.classes_, new)

    def _n_split_features(self, n_features):
        mf = self.max_features
        if mf is None:
            k = n_features
        elif mf == "sqrt":
            k = int(np.sqrt(n_features))
        elif mf == "log2":
            k = int(np.log2(n_features)) if n_features > 1 else 1
        elif isinstance(mf, (float, np.floating)):
            if not 0 < mf <= 1:
                raise ValueError(f"float max_features must be in (0, 1], got {mf}")
            k = int(mf * n_features)
        else:
            k = int(mf)
            if k < 1:
                raise ValueError(f"max_features must be >= 1, got {mf}")
        return int(np.clip(k, 1, n_features))

    def fit(self, X, y):
        """Train a tree from scratch on ``(X, y)`` (clears buffer and classes)."""
        X, y = check_X_y(X, y)
        self.reset()
        self._update_classes(y)
        self._train(X, y)
        self.n_samples_seen_ = X.shape[0]
        return self

    def partial_fit(self, X_chunk, y_chunk, classes=None):
        """Buffer a chunk; retrain when the buffer reaches ``buffer_size``.

        Parameters
        ----------
        X_chunk : (n_samples, n_features)
        y_chunk : (n_samples,)
        classes : array-like, optional
            Full set of labels if known up-front, so ``predict_proba`` has a
            stable width from the first retrain.

        Returns
        -------
        self
        """
        X, y = check_X_y(X_chunk, y_chunk)
        self._update_classes(y, classes)
        width = X.shape[1]
        if self._buf_width is not None and width != self._buf_width:
            if width < self._buf_width:
                raise ValueError(f"X_chunk has {width} features but buffered chunks have "
                                 f"{self._buf_width}; features may only be appended")
            self._buf_X = [np.pad(b, ((0, 0), (0, width - b.shape[1])),
                                  constant_values=self.pad_value) for b in self._buf_X]
        if self.n_features_in_ is not None and width < self.n_features_in_:
            raise ValueError(f"X_chunk has {width} features; the current tree was trained "
                             f"on {self.n_features_in_}")
        self._buf_width = width
        self._buf_X.append(X)
        self._buf_y.append(y)
        self._buf_n += X.shape[0]
        self.n_samples_seen_ += X.shape[0]
        if self.buffer_size is None or self._buf_n >= self.buffer_size:
            self._retrain_from_buffer()
        return self

    def flush(self):
        """Retrain on the buffered samples now (no-op if the buffer is empty)."""
        if self._buf_n:
            self._retrain_from_buffer()
        return self

    def _retrain_from_buffer(self):
        X = np.vstack(self._buf_X)
        y = np.concatenate(self._buf_y)
        self._buf_X, self._buf_y, self._buf_n = [], [], 0
        self._train(X, y)

    def _train(self, X, y):
        t0 = time.perf_counter()
        y_enc = np.searchsorted(self.classes_, y)
        self._build(X, y_enc, len(self.classes_))
        self.tree_classes_ = self.classes_.copy()
        self.n_features_in_ = X.shape[1]
        dt = time.perf_counter() - t0
        self.retrain_times_.append(dt)
        self.n_retrains_ += 1

    def _build(self, X, y, K):
        n_feat_split = self._n_split_features(X.shape[1])
        feature, threshold, nan_left, left, right, value, imp, nsamp = ([] for _ in range(8))
        # stack items: (row indices, depth, parent id, is_left)
        stack = [(np.arange(X.shape[0]), 0, -1, False)]
        max_depth_seen = 0
        while stack:
            idx, depth, parent, is_left = stack.pop()
            node = len(feature)
            counts = np.bincount(y[idx], minlength=K).astype(float)
            node_imp = float(impurity(counts, self.criterion))
            feature.append(_LEAF); threshold.append(np.nan); nan_left.append(True)
            left.append(_LEAF); right.append(_LEAF)
            value.append(counts); imp.append(node_imp); nsamp.append(idx.size)
            if parent >= 0:
                (left if is_left else right)[parent] = node
            max_depth_seen = max(max_depth_seen, depth)

            if (node_imp <= 0.0 or idx.size < self.min_samples_split
                    or idx.size < 2 * self.min_samples_leaf
                    or (self.max_depth is not None and depth >= self.max_depth)):
                continue
            if n_feat_split < X.shape[1]:
                feats = np.sort(self._rng.choice(X.shape[1], n_feat_split, replace=False))
            else:
                feats = None
            split = best_split(X[idx], y[idx], K, self.criterion,
                               self.min_samples_leaf, feats)
            if split is None:
                continue
            f, thr, gain, nl = split
            if gain <= max(self.min_impurity_decrease, 1e-12):
                continue
            col = X[idx, f]
            go_left = np.where(np.isnan(col), nl, col <= thr)
            feature[node], threshold[node], nan_left[node] = f, thr, nl
            # push right first so the left subtree gets the lower node ids
            stack.append((idx[~go_left], depth + 1, node, False))
            stack.append((idx[go_left], depth + 1, node, True))

        self.feature_ = np.array(feature, dtype=np.int64)
        self.threshold_ = np.array(threshold, dtype=float)
        self.nan_left_ = np.array(nan_left, dtype=bool)
        self.left_ = np.array(left, dtype=np.int64)
        self.right_ = np.array(right, dtype=np.int64)
        self.value_ = np.array(value, dtype=float).reshape(len(feature), K)
        self.impurity_ = np.array(imp)
        self.n_node_samples_ = np.array(nsamp, dtype=np.int64)
        self.depth_ = max_depth_seen

    # ------------------------------------------------------------- inference
    def _check_predict_X(self, X):
        if not self.is_fitted:
            extra = ""
            if self._buf_n:
                extra = (f" ({self._buf_n}/{self.buffer_size} samples buffered; call "
                         "flush() to train on them now)")
            raise NotFittedError("DecisionTreeClassifier has not been trained yet" + extra)
        X = check_array(X)
        if X.shape[1] < self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but the tree was trained on "
                             f"{self.n_features_in_}")
        return X

    def apply(self, X) -> np.ndarray:
        """Leaf index for each sample, shape (n_samples,).

        Vectorised level-by-level traversal: one NumPy step per tree level.
        """
        X = self._check_predict_X(X)
        node = np.zeros(X.shape[0], dtype=np.int64)
        for _ in range(self.depth_ + 1):
            f = self.feature_[node]
            internal = np.nonzero(f != _LEAF)[0]
            if internal.size == 0:
                break
            nd = node[internal]
            x = X[internal, f[internal]]
            go_left = np.where(np.isnan(x), self.nan_left_[nd], x <= self.threshold_[nd])
            node[internal] = np.where(go_left, self.left_[nd], self.right_[nd])
        return node

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities, shape (n_samples, len(classes_)).

        Columns follow ``classes_``; classes first seen after the last retrain
        get probability 0 until the next retrain.
        """
        leaf = self.apply(X)
        counts = self.value_[leaf]
        tot = counts.sum(axis=1, keepdims=True)
        p_tree = counts / np.where(tot > 0, tot, 1.0)
        out = np.zeros((p_tree.shape[0], len(self.classes_)))
        out[:, np.searchsorted(self.classes_, self.tree_classes_)] = p_tree
        return out

    def predict(self, X) -> np.ndarray:
        """Predicted labels, shape (n_samples,). Ties -> smallest label."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_loop(self, X) -> np.ndarray:
        """Reference per-sample (loop-based) prediction, for benchmarking."""
        X = self._check_predict_X(X)
        out = np.empty(X.shape[0], dtype=np.int64)
        for i in range(X.shape[0]):
            node = 0
            while self.feature_[node] != _LEAF:
                x = X[i, self.feature_[node]]
                go_left = self.nan_left_[node] if x != x else x <= self.threshold_[node]
                node = self.left_[node] if go_left else self.right_[node]
            out[i] = node
        counts = self.value_[out]
        return self.tree_classes_[np.argmax(counts, axis=1)]

    def score(self, X, y) -> float:
        """Accuracy on ``(X, y)``."""
        X, y = check_X_y(X, y)
        return float(np.mean(self.predict(X) == y))

    # ----------------------------------------------------------- inspection
    @property
    def n_nodes(self) -> int:
        return 0 if self.feature_ is None else int(self.feature_.size)

    @property
    def n_leaves(self) -> int:
        return 0 if self.feature_ is None else int((self.feature_ == _LEAF).sum())

    def get_depth(self) -> int:
        if not self.is_fitted:
            raise NotFittedError("tree not trained")
        return int(self.depth_)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Normalised total impurity decrease per feature (sums to 1, or all 0)."""
        if not self.is_fitted:
            raise NotFittedError("tree not trained")
        internal = np.nonzero(self.feature_ != _LEAF)[0]
        w = self.n_node_samples_.astype(float)
        dec = (w[internal] * self.impurity_[internal]
               - w[self.left_[internal]] * self.impurity_[self.left_[internal]]
               - w[self.right_[internal]] * self.impurity_[self.right_[internal]])
        imp = np.zeros(self.n_features_in_)
        np.add.at(imp, self.feature_[internal], np.maximum(dec, 0.0))
        s = imp.sum()
        return imp / s if s > 0 else imp

    def export_text(self, feature_names=None, decimals: int = 3) -> str:
        """Human-readable text rendering of the current tree."""
        if not self.is_fitted:
            raise NotFittedError("tree not trained")
        names = feature_names or [f"x{j}" for j in range(self.n_features_in_)]
        lines = []

        def rec(node, depth):  # depth bounded by max_depth; recursion is fine here
            pad = "|   " * depth
            if self.feature_[node] == _LEAF:
                cls = self.tree_classes_[np.argmax(self.value_[node])]
                lines.append(f"{pad}class: {cls} (n={self.n_node_samples_[node]})")
                return
            f, t = names[self.feature_[node]], round(self.threshold_[node], decimals)
            lines.append(f"{pad}{f} <= {t}")
            rec(self.left_[node], depth + 1)
            lines.append(f"{pad}{f} >  {t}")
            rec(self.right_[node], depth + 1)

        rec(0, 0)
        return "\n".join(lines)

    def __repr__(self):
        return (f"DecisionTreeClassifier(max_depth={self.max_depth}, criterion="
                f"'{self.criterion}', buffer_size={self.buffer_size})")
