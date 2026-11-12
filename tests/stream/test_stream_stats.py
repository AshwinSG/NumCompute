import numpy as np
import pytest

from numcompute_stream.io import iter_array_chunks
from numcompute_stream.stats import (StreamingHistogram, StreamingQuantile, StreamingStats,
                              StreamingSummary, welford_loop)
from numcompute_stream.utils import NotFittedError


def test_streaming_mean_var_match_batch(rng):
    X = rng.normal(5, 3, size=(1000, 4))
    s = StreamingStats()
    for chunk, _ in iter_array_chunks(X, chunk_size=97):
        s.update_stats(chunk)
    np.testing.assert_allclose(s.mean, X.mean(axis=0))
    np.testing.assert_allclose(s.var(), X.var(axis=0))
    np.testing.assert_allclose(s.var(ddof=1), X.var(axis=0, ddof=1))
    np.testing.assert_allclose(s.min, X.min(axis=0))
    np.testing.assert_allclose(s.max, X.max(axis=0))


def test_streaming_stats_ignore_nan(rng):
    X = rng.normal(size=(300, 3))
    X[rng.random(X.shape) < 0.2] = np.nan
    s = StreamingStats()
    for chunk, _ in iter_array_chunks(X, chunk_size=50):
        s.update_stats(chunk)
    np.testing.assert_allclose(s.mean, np.nanmean(X, axis=0))
    np.testing.assert_allclose(s.var(), np.nanvar(X, axis=0))
    np.testing.assert_array_equal(s.count, (~np.isnan(X)).sum(axis=0))


def test_all_nan_column_gives_nan_not_error():
    s = StreamingStats().update_stats(np.array([[1.0, np.nan], [3.0, np.nan]]))
    assert s.mean[0] == 2.0 and np.isnan(s.mean[1])
    assert np.isnan(s.var()[1]) and np.isnan(s.min[1])


def test_numerical_stability_large_offset(rng):
    """Naive sum-of-squares fails at 1e9 offset; Chan/Welford merge must not."""
    X = 1e9 + rng.normal(0, 1, size=(10_000, 1))
    s = StreamingStats()
    for chunk, _ in iter_array_chunks(X, chunk_size=128):
        s.update_stats(chunk)
    np.testing.assert_allclose(s.var(), X.var(axis=0), rtol=1e-6)
    assert s.var()[0] > 0.9


def test_zero_variance_chunk():
    s = StreamingStats().update_stats(np.full((10, 2), 7.0))
    np.testing.assert_array_equal(s.var(), [0.0, 0.0])
    assert np.isnan(StreamingStats().update_stats(np.ones((1, 1))).var(ddof=1)[0])


def test_welford_loop_agrees_with_vectorised(rng):
    X = rng.normal(size=(200, 3))
    _, m, v = welford_loop(X)
    s = StreamingStats().update_stats(X)
    np.testing.assert_allclose(m, s.mean)
    np.testing.assert_allclose(v, s.var())


def test_merge_equals_sequential(rng):
    A, B = rng.normal(size=(100, 2)), rng.normal(3, 2, size=(50, 2))
    s1 = StreamingStats().update_stats(A).merge(StreamingStats().update_stats(B))
    s2 = StreamingStats().update_stats(np.vstack([A, B]))
    np.testing.assert_allclose(s1.mean, s2.mean)
    np.testing.assert_allclose(s1.var(), s2.var())


def test_stats_width_mismatch_raises():
    s = StreamingStats().update_stats(np.ones((3, 2)))
    with pytest.raises(ValueError, match="features"):
        s.update_stats(np.ones((3, 3)))


def test_stats_reject_1d_with_hint():
    with pytest.raises(ValueError, match="reshape"):
        StreamingStats().update_stats(np.ones(5))


def test_not_fitted_stats():
    with pytest.raises(NotFittedError):
        StreamingStats().mean


def test_quantile_exact_below_capacity(rng):
    X = rng.normal(size=(500, 2))
    q = StreamingQuantile(capacity=1000)
    for chunk, _ in iter_array_chunks(X, chunk_size=64):
        q.update_stats(chunk)
    np.testing.assert_allclose(q.quantile([0.1, 0.5, 0.9]),
                               np.quantile(X, [0.1, 0.5, 0.9], axis=0))


def test_quantile_reservoir_approximation(rng):
    X = rng.uniform(0, 1, size=(50_000, 1))
    q = StreamingQuantile(capacity=2000, random_state=1)
    for chunk, _ in iter_array_chunks(X, chunk_size=1000):
        q.update_stats(chunk)
    assert abs(q.median()[0] - 0.5) < 0.05
    assert q.reservoir_.shape == (2000, 1)


def test_histogram_counts_and_sliding_window():
    h = StreamingHistogram(bins=4, range=(0, 4), window=2)
    h.update_stats(np.array([[0.5], [1.5]]))
    h.update_stats(np.array([[2.5]]))
    h.update_stats(np.array([[3.5], [3.6], [9.0]]))  # first chunk falls out of window
    np.testing.assert_array_equal(h.counts[0], [0, 0, 1, 3])
    assert h.n_overflow_[0] == 1


def test_histogram_quantile_and_zero_variance(rng):
    X = rng.uniform(0, 10, size=(5000, 1))
    h = StreamingHistogram(bins=100, range=(0, 10)).update_stats(X)
    assert abs(h.quantile(0.5)[0] - 5) < 0.3
    flat = StreamingHistogram(bins=5).update_stats(np.full((4, 1), 3.0))
    assert flat.counts.sum() == 4 and flat.edges_[0, 0] < 3.0 < flat.edges_[0, -1]


def test_summary_bundle(rng):
    X = rng.normal(size=(100, 3))
    r = StreamingSummary().update_stats(X[:50]).update_stats(X[50:]).result()
    np.testing.assert_allclose(r["mean"], X.mean(axis=0))
    assert r["histogram"].shape == (3, 20)
