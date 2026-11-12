# NumCompute 2.2: streaming decision-tree framework

A NumPy + matplotlib only framework for **online learning** with a single
decision tree that adapts to streams through **buffered retraining**.

```
numcompute_stream/
  io.py            CSV loading (load_csv) and chunked streaming (iter_csv_chunks, iter_array_chunks)
  stats.py         StreamingStats (mean/var/min/max), StreamingQuantile, StreamingHistogram, update_stats()
  preprocessing.py StandardScaler (Welford or EMA), MinMaxScaler, SimpleImputer, OneHotEncoder – all partial_fit
  tree.py          DecisionTreeClassifier (Gini/entropy, max_depth, min_samples_split, max_features, buffer_size)
  metrics.py       accuracy/precision/recall/F1/confusion/AUC – functional + streaming (update/result/reset, window=)
  pipeline.py      Pipeline with partial_fit / fit / predict / predict_proba
  stream.py        StreamTrainer – fit_chunk / score_chunk / process_chunk, per-chunk logs (accuracy, memory, time)
  visualise.py     plot_metric_over_time, compare_models, plot_predictions_vs_ground_truth, plot_tradeoff
  utils.py         validation helpers, NotFittedError
data/              sensor_stream.csv + make_dataset.py (drift at row 3 500, new category after row 3 000)
demo/              stream_demo.ipynb
tests/stream/      74 pytest tests for numcompute_stream
benchmark/         stream_benchmark.py → stream_results.md + figures/
```

This branch extends the group's Assignment 2.1 package `numcompute/`, which is
kept unchanged (see [Base package](#base-package-assignment-21) below).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook demo/stream_demo.ipynb     # run the streaming demo
```

## Quick start

```python
from numcompute_stream import (io, Pipeline, OneHotEncoder, SimpleImputer, StandardScaler,
                               DecisionTreeClassifier, StreamTrainer, StreamingF1)

pipe = Pipeline([
    ("onehot", OneHotEncoder(columns=[5])),                 # machine_type
    ("impute", SimpleImputer(allow_growth=True)),
    ("scale",  StandardScaler(allow_growth=True)),
    ("model",  DecisionTreeClassifier(max_depth=5, buffer_size=500)),
])
trainer = StreamTrainer(pipe, metrics={"f1": StreamingF1()})
for chunk in io.iter_csv_chunks("data/sensor_stream.csv", 100, target="fault",
                                categorical=["machine_type"]):
    trainer.process_chunk(chunk.X, chunk.y)     # test-then-train
print(trainer.summary())
```

```bash
python -m pytest -q tests/stream     # streaming package only
python -m pytest -q                  # everything (incl. 2.1 base package)
```

```bash
python benchmark/stream_benchmark.py
```

## Design notes

**Buffered retraining (`tree.py`).** `partial_fit` appends the chunk to a buffer.
When the buffer holds `>= buffer_size` samples the tree is rebuilt **from scratch**
on the buffer and the buffer is cleared. Between retrains `predict` uses the latest
tree. Before the first retrain `predict` raises `NotFittedError` with the buffer
fill level, and `flush()` forces a retrain early. `buffer_size=None` retrains on
every call. `classes_` grows as new labels appear, and `partial_fit(..., classes=)`
fixes the `predict_proba` width up front.

**Vectorisation.** Split search sorts all candidate features at once, takes class
counts left of every threshold from a single `cumsum` over a one-hot tensor, and
scores every (threshold, feature) pair in one expression. Features are processed in
blocks so the temporary arrays stay small. Prediction walks the tree one level at a
time for all samples together. The only Python loop in training is over nodes.
Loop-based reference versions (`best_split_loop`, `predict_loop`, `welford_loop`,
`confusion_matrix_loop`) exist for the benchmarks and the equivalence tests.

**Numerical stability.**
* Running mean and variance use Chan et al.'s parallel merge of Welford's algorithm.
  It stays exact at a 1e9 offset, where a naive sum of squares fails (tested).
  Tiny negative variances from round-off are clipped to 0.
* NaNs are ignored by all statistics. A feature with no observed values reports NaN
  and doesn't raise. Imputers fall back to `fill_value` for those features.
* Zero-variance features get `scale_ = 1` in the scalers. A constant feature gets a
  unit-width histogram range. A tree on all-constant features returns one leaf.
* Ties: split thresholds only go between distinct values (an overflow-safe midpoint).
  Equal gains go to the lowest feature index and then the lowest threshold. Equal
  class probabilities go to the smallest label. Equal AUC scores count as ½.
* Missing values in the tree: gain is computed on non-missing rows and scaled by the
  non-missing fraction (C4.5 style). Missing rows go to the larger child.
* Ratios that are ill-defined (for example the precision of a class never predicted)
  return 0.0 instead of NaN or a warning.

**Growing feature width.** `OneHotEncoder` always appends new categories at the end,
so existing columns never move. With `allow_growth=True`, scalers and imputers accept
the new trailing columns and assume earlier rows held 0 there, which is exact for
one-hot data. The tree pads buffered rows to match.

**API conventions.** `X` is `(n_samples, n_features)` and statistics run along
`axis=0`. A 1-D `X` raises an error that suggests the correct reshape. Mismatched
lengths and widths raise errors that name both shapes.

**Known limitation.** Inside a `Pipeline`, the tree buffers rows that are already
transformed. Rows buffered early were therefore scaled with slightly older
statistics. This matters only while the scaler is still warming up.

## Benchmark highlights (see `benchmark/stream_results.md`)

| Operation | Speed-up of vectorised over loop |
|---|---:|
| Best-split search (n=1000, 8 features) | ~100x |
| Running mean/var | ~45x |
| Tree predict (20k samples) | ~20x |
| Confusion matrix (200k) | ~6x |

On the drifting stream, `buffer_size` of 250 to 1000 gives the best prequential
accuracy (0.81 to 0.83). At 100, each tree sees too little data. At 2000 or more,
the start is slow and the model reacts slowly to drift: accuracy falls to 0.54 at
3000. Total retrain time hardly changes with buffer size because every sample is
trained on exactly once, but the cost of a single retrain grows with `buffer_size`.

## Base package (Assignment 2.1)

`numcompute/` is the group's batch toolkit from Assignment 2.1 (I/O, preprocessing,
sorting/searching, ranking, statistics, metrics, finite-difference optimisation,
pipelines). It is unchanged on this branch. Its tests are in `tests/test_*.py`, and its
benchmarks are `benchmark/run_benchmarks.py` → `benchmark/results.md`.
`numcompute_stream/` is a separate package, so code that uses the 2.1 API keeps working.
