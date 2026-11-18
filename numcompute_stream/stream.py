"""Run a model or pipeline over a stream of data and log metrics for each chunk.

The usual way to score a model that learns as data arrives is test-then-train:
each chunk is first used to test the current model and then to train it, so
every prediction is made on data the model has not seen yet.

Example:
    trainer = StreamTrainer(pipe, metrics={"f1": StreamingF1()})
    for X_chunk, y_chunk in stream:
        trainer.process_chunk(X_chunk, y_chunk)   # score_chunk, then fit_chunk
    trainer.history["cumulative_accuracy"]
"""
from __future__ import annotations

import logging
import math
import time
from typing import Dict, Iterable, Optional

import numpy as np

from .metrics import StreamingAccuracy, StreamingMetric
from .utils import NotFittedError, nbytes_of

logger = logging.getLogger("numcompute_stream.stream")

_BASE_KEYS = ("chunk", "n_samples", "chunk_accuracy", "cumulative_accuracy",
              "fit_time", "predict_time", "memory_bytes", "buffer_count", "n_retrains")


def _final(model):
    return model.final_estimator if hasattr(model, "final_estimator") else model


class StreamTrainer:
    """Holds a model (or Pipeline), its running metrics and a log for each chunk.

    Parameters:
        model: any object with partial_fit and predict (and predict_proba if
            you use score-based metrics such as AUC).
        metrics: optional dict of name to StreamingMetric. Their result() is
            logged after each scored chunk. StreamingAccuracy is always
            tracked as well.
        classes: optional list of all labels, passed on to
            partial_fit(classes=...).
        pos_label: the label whose probability column is used by metrics
            with requires_score=True.
        verbose: if True, write one INFO log line per chunk.

    Attributes:
        history: dict of lists with one entry per chunk (NaN where a value
            does not apply).
        last_y_true_, last_y_pred_: labels and predictions from the last
            scored chunk.
    """

    def __init__(self, model, metrics: Optional[Dict[str, StreamingMetric]] = None,
                 classes=None, pos_label=1, name: Optional[str] = None,
                 verbose: bool = False):
        for meth in ("partial_fit", "predict"):
            if not hasattr(model, meth):
                raise TypeError(f"model must implement {meth}()")
        self.model = model
        self.metrics = dict(metrics or {})
        for k, m in self.metrics.items():
            if not isinstance(m, StreamingMetric):
                raise TypeError(f"metric {k!r} must be a StreamingMetric, got {type(m).__name__}")
        self.classes = classes
        self.pos_label = pos_label
        self.name = name or type(_final(model)).__name__
        self.verbose = verbose
        self.reset_logs()

    def reset_logs(self):
        self._cum_acc = StreamingAccuracy()
        for m in self.metrics.values():
            m.reset()
        self._records = []
        self._chunk = 0
        self.last_y_true_ = None
        self.last_y_pred_ = None
        return self

    # ------------------------------------------------------------- records
    def _record(self) -> dict:
        if not self._records or self._records[-1]["chunk"] != self._chunk:
            rec = {k: math.nan for k in _BASE_KEYS}
            rec.update({k: math.nan for k in self.metrics})
            rec["chunk"] = self._chunk
            self._records.append(rec)
        return self._records[-1]

    @property
    def history(self) -> Dict[str, list]:
        keys = list(_BASE_KEYS) + list(self.metrics)
        return {k: [r[k] for r in self._records] for k in keys}

    # ------------------------------------------------------------- actions
    def score_chunk(self, X_chunk, y_chunk) -> Optional[dict]:
        """Test the current model on a chunk and update all metrics.

        Returns the logged values, or None if the model has not been trained
        yet (for example a buffered tree before its first retrain).
        """
        y = np.asarray(y_chunk)
        rec = self._record()
        rec["n_samples"] = int(y.shape[0])
        t0 = time.perf_counter()
        try:
            y_pred = self.model.predict(X_chunk)
            proba = None
            if any(m.requires_score for m in self.metrics.values()):
                P = self.model.predict_proba(X_chunk)
                classes = list(np.asarray(self.model.classes_).tolist())
                proba = (P[:, classes.index(self.pos_label)] if self.pos_label in classes
                         else np.zeros(P.shape[0]))
        except NotFittedError:
            logger.debug("chunk %d: model not trained yet, skipping evaluation", self._chunk)
            return None
        rec["predict_time"] = time.perf_counter() - t0
        rec["chunk_accuracy"] = float(np.mean(y_pred == y)) if y.size else math.nan
        self._cum_acc.update(y, y_pred)
        rec["cumulative_accuracy"] = self._cum_acc.result()
        for k, m in self.metrics.items():
            m.update(y, proba if m.requires_score else y_pred)
            rec[k] = m.result()
        self.last_y_true_, self.last_y_pred_ = y, y_pred
        return dict(rec)

    def fit_chunk(self, X_chunk, y_chunk) -> dict:
        """Train the model on a chunk and log the time, memory and retrain count."""
        rec = self._record()
        rec["n_samples"] = int(np.asarray(y_chunk).shape[0])
        kw = {"classes": self.classes} if self.classes is not None else {}
        t0 = time.perf_counter()
        self.model.partial_fit(X_chunk, y_chunk, **kw)
        rec["fit_time"] = time.perf_counter() - t0
        est = _final(self.model)
        rec["memory_bytes"] = nbytes_of(self.model)
        rec["buffer_count"] = getattr(est, "buffer_count", math.nan)
        rec["n_retrains"] = getattr(est, "n_retrains_", math.nan)
        if self.verbose:
            logger.info("[%s] chunk %3d | n=%4d | acc=%.3f | cum_acc=%.3f | fit=%.4fs | "
                        "mem=%.1f KB | retrains=%s", self.name, self._chunk, rec["n_samples"],
                        rec["chunk_accuracy"], rec["cumulative_accuracy"], rec["fit_time"],
                        rec["memory_bytes"] / 1024, rec["n_retrains"])
        out = dict(rec)
        self._chunk += 1
        return out

    def process_chunk(self, X_chunk, y_chunk) -> dict:
        """One test-then-train step: score_chunk, then fit_chunk."""
        self.score_chunk(X_chunk, y_chunk)
        return self.fit_chunk(X_chunk, y_chunk)

    def run(self, stream: Iterable, flush: bool = False) -> Dict[str, list]:
        """Go through (X_chunk, y_chunk) pairs, testing and then training on each.

        flush=True retrains on any samples still left in the buffer at the end.
        """
        for X_chunk, y_chunk in stream:
            self.process_chunk(X_chunk, y_chunk)
        if flush and hasattr(self.model, "flush"):
            self.model.flush()
        return self.history

    def summary(self) -> dict:
        h = self.history
        acc = np.array(h["chunk_accuracy"], dtype=float)
        fit = np.array(h["fit_time"], dtype=float)
        est = _final(self.model)
        return {
            "model": self.name,
            "chunks": len(self._records),
            "final_cumulative_accuracy": self._cum_acc.result(),
            "mean_chunk_accuracy": float(np.nanmean(acc)) if np.isfinite(acc).any() else math.nan,
            "total_fit_time": float(np.nansum(fit)),
            "n_retrains": getattr(est, "n_retrains_", None),
            "total_retrain_time": float(np.sum(getattr(est, "retrain_times_", [0.0]))),
            "peak_memory_bytes": int(np.nanmax(h["memory_bytes"])) if h["memory_bytes"] else 0,
            **{k: m.result() for k, m in self.metrics.items()},
        }
