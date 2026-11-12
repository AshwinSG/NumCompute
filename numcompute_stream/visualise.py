"""Reusable matplotlib plots for streaming experiments.

Every function

* accepts an optional ``ax`` (draw into an existing figure) – otherwise a new
  figure is created,
* accepts ``save_path`` (write PNG/SVG/PDF, inferred from the extension) and
  ``show`` (call ``plt.show()`` for inline / interactive display),
* returns ``(fig, ax)`` (or ``(fig, axes)``) so callers can customise further.

Colours use a fixed, colour-blind-validated categorical order; series identity
is always also carried by a legend / direct label and a marker shape, never by
colour alone.
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

# Fixed categorical order (never cycled); validated for CVD separation.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SEQ_CMAP = "Blues"

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
    "lines.linewidth": 2.0,
    "font.size": 10,
}


def _prepare(ax, figsize):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _finish(fig, save_path: Optional[str], show: bool):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    if show:
        plt.show()


def _as_series(values, name):
    v = np.asarray(values, dtype=float)
    if v.ndim != 1:
        raise ValueError(f"{name} must be 1-D (one value per chunk), got shape {v.shape}")
    if v.size == 0:
        raise ValueError(f"{name} is empty")
    return v


def plot_metric_over_time(metric_values: Sequence[float], title: str = "Metric over time",
                          ylabel: str = "Accuracy", xlabel: str = "Chunk",
                          events: Optional[Sequence[int]] = None, event_label: str = "retrain",
                          ylim: Optional[tuple] = None, ax=None, figsize=(8, 4),
                          save_path: Optional[str] = None, show: bool = False):
    """Line plot of one metric across chunks.

    Parameters
    ----------
    metric_values : 1-D sequence, one value per chunk. NaN = not evaluated
        (drawn as a gap, e.g. before the first retrain).
    events : chunk indices to mark with faint vertical lines (e.g. retrains).
    """
    v = _as_series(metric_values, "metric_values")
    x = np.arange(v.size)
    with plt.rc_context(STYLE):
        fig, ax = _prepare(ax, figsize)
        if events is not None:
            for i, e in enumerate(events):
                ax.axvline(e, color=AXIS, lw=1, ls=":", zorder=0,
                           label=event_label if i == 0 else None)
        ax.plot(x, v, color=SERIES_COLORS[0], marker="o", markersize=4)
        finite = np.isfinite(v)
        if finite.any():
            last = np.nonzero(finite)[0][-1]
            ax.annotate(f"{v[last]:.3f}", (x[last], v[last]), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK, fontsize=9)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if events is not None and len(events):
            ax.legend(loc="lower right")
        _finish(fig, save_path, show)
    return fig, ax


def compare_models(metric1: Sequence[float], metric2: Sequence[float],
                   labels: Sequence[str] = ("Model A", "Model B"),
                   title: str = "Model comparison", ylabel: str = "Accuracy",
                   xlabel: str = "Chunk", *more_metrics, ylim: Optional[tuple] = None,
                   ax=None, figsize=(8, 4), save_path: Optional[str] = None,
                   show: bool = False):
    """Overlay two (or more) streaming metric curves, e.g. trees with
    different ``buffer_size`` values.

    Extra series can be passed positionally after ``xlabel``; ``labels`` must
    then have one entry per series.
    """
    series = [_as_series(metric1, "metric1"), _as_series(metric2, "metric2")]
    series += [_as_series(m, f"metric{i + 3}") for i, m in enumerate(more_metrics)]
    if len(labels) != len(series):
        raise ValueError(f"need {len(series)} labels, got {len(labels)}")
    if len(series) > len(SERIES_COLORS):
        raise ValueError(f"at most {len(SERIES_COLORS)} series are supported; facet instead")
    with plt.rc_context(STYLE):
        fig, ax = _prepare(ax, figsize)
        ends = []
        for i, (s, lab) in enumerate(zip(series, labels)):
            x = np.arange(s.size)
            ax.plot(x, s, color=SERIES_COLORS[i], marker=MARKERS[i], markersize=4, label=lab)
            finite = np.isfinite(s)
            if finite.any():
                last = np.nonzero(finite)[0][-1]
                ends.append((s[last], x[last], i))
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(loc="best")
        ax.margins(x=0.12)
        if ends and len(series) <= 4:
            # direct value labels at line ends, nudged apart so they never overlap
            lo, hi = ax.get_ylim()
            gap = 0.06 * (hi - lo)
            ends.sort()
            placed = []
            for v, xl, i in ends:
                y = v if not placed else max(v, placed[-1] + gap)
                placed.append(y)
                ax.annotate(f"{labels[i]}: {v:.3f}", (xl, v), xytext=(xl + 0.8, y),
                            textcoords="data", va="center", fontsize=8, color=INK_2)
        _finish(fig, save_path, show)
    return fig, ax


def plot_predictions_vs_ground_truth(y_true, y_pred, title: str = "Latest chunk: predictions vs. ground truth",
                                     labels: Optional[Sequence] = None, figsize=(10, 4),
                                     save_path: Optional[str] = None, show: bool = False):
    """Two panels for a classification chunk:

    1. confusion matrix heat-map (true rows × predicted columns, counts shown),
    2. per-class counts, actual vs. predicted, as grouped bars.

    Returns ``(fig, (ax_cm, ax_bar))``.
    """
    from .metrics import confusion_matrix  # local import avoids a cycle at import time

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError(f"y_true and y_pred must be 1-D of equal length, got "
                         f"{y_true.shape} and {y_pred.shape}")
    labs = np.unique(np.concatenate([y_true, y_pred])) if labels is None else np.asarray(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labs)
    acc = float(np.mean(y_true == y_pred)) if y_true.size else float("nan")
    with plt.rc_context(STYLE):
        fig, (ax_cm, ax_bar) = plt.subplots(1, 2, figsize=figsize,
                                            gridspec_kw={"width_ratios": [1, 1.3]})
        im = ax_cm.imshow(cm, cmap=SEQ_CMAP, vmin=0)
        ax_cm.grid(False)
        thresh = cm.max() / 2 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10,
                           color="#ffffff" if cm[i, j] > thresh else INK)
        ax_cm.set_xticks(range(len(labs)), [str(l) for l in labs])
        ax_cm.set_yticks(range(len(labs)), [str(l) for l in labs])
        ax_cm.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix (acc={acc:.3f})")
        for s in ax_cm.spines.values():
            s.set_visible(False)
        fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

        x = np.arange(len(labs))
        w = 0.38
        true_c, pred_c = cm.sum(axis=1), cm.sum(axis=0)
        ax_bar.bar(x - w / 2 - 0.01, true_c, w, color=SERIES_COLORS[0], label="Actual")
        ax_bar.bar(x + w / 2 + 0.01, pred_c, w, color=SERIES_COLORS[1], label="Predicted",
                   hatch="//", edgecolor=SURFACE, linewidth=0)
        ax_bar.set_xticks(x, [str(l) for l in labs])
        ax_bar.set(xlabel="Class", ylabel="Samples", title="Class counts")
        ax_bar.legend()
        fig.suptitle(title, color=INK, fontweight="semibold")
        _finish(fig, save_path, show)
    return fig, (ax_cm, ax_bar)


def plot_tradeoff(x_values: Sequence, left_values: Sequence[float], right_values: Sequence[float],
                  xlabel: str = "buffer_size", left_label: str = "Prequential accuracy",
                  right_label: str = "Total retrain time (s)",
                  title: str = "buffer_size trade-off", figsize=(10, 4),
                  save_path: Optional[str] = None, show: bool = False):
    """Two side-by-side panels sharing the x values (no dual y-axis), e.g.
    accuracy and retraining cost as a function of ``buffer_size``."""
    xs = [str(v) for v in x_values]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for ax, vals, lab, c, fmt in ((axes[0], left_values, left_label, SERIES_COLORS[0], "{:.3f}"),
                                      (axes[1], right_values, right_label, SERIES_COLORS[1], "{:.3g}")):
            vals = np.asarray(vals, dtype=float)
            ax.plot(xs, vals, color=c, marker="o")
            for xi, v in zip(xs, vals):
                ax.annotate(fmt.format(v), (xi, v), xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=8, color=INK_2)
            ax.set(xlabel=xlabel, ylabel=lab, title=lab)
            ax.margins(y=0.15)
        axes[1].set_ylim(bottom=0)  # costs are magnitudes: never truncate the axis
        fig.suptitle(title, color=INK, fontweight="semibold")
        _finish(fig, save_path, show)
    return fig, axes
