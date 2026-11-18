import numpy as np
import pytest

from numcompute_stream.io import iter_array_chunks
from numcompute_stream.preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler
from numcompute_stream.utils import NotFittedError


def test_standard_scaler_partial_equals_batch(rng):
    X = rng.normal(10, 4, size=(600, 3))
    sc = StandardScaler()
    for chunk, _ in iter_array_chunks(X, chunk_size=55):
        sc.partial_fit(chunk)
    np.testing.assert_allclose(sc.transform(X), (X - X.mean(0)) / X.std(0), atol=1e-10)
    np.testing.assert_allclose(sc.inverse_transform(sc.transform(X)), X)


def test_standard_scaler_zero_variance_no_div_by_zero():
    X = np.column_stack([np.full(20, 5.0), np.arange(20.0)])
    out = StandardScaler().fit_transform(X)
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out[:, 0], 0.0)


def test_standard_scaler_ema_tracks_drift(rng):
    sc_w, sc_e = StandardScaler(), StandardScaler(method="ema", alpha=0.5)
    for mu in [0.0] * 10 + [100.0] * 10:
        chunk = rng.normal(mu, 1, size=(50, 1))
        sc_w.partial_fit(chunk)
        sc_e.partial_fit(chunk)
    assert abs(sc_e.mean_[0] - 100) < 1.0      # EMA follows the new values
    assert abs(sc_w.mean_[0] - 50) < 1.0       # Welford gives the mean over all data


def test_scaler_propagates_nan_and_validates_width():
    sc = StandardScaler().fit(np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]]))
    assert np.isnan(sc.transform(np.array([[np.nan, 1.0]]))[0, 0])
    with pytest.raises(ValueError, match="expected 2 features"):
        sc.transform(np.ones((2, 3)))
    with pytest.raises(NotFittedError):
        StandardScaler().transform(np.ones((2, 2)))


def test_minmax_scaler_streaming(rng):
    X = rng.uniform(-5, 5, size=(200, 2))
    mm = MinMaxScaler()
    for chunk, _ in iter_array_chunks(X, chunk_size=33):
        mm.partial_fit(chunk)
    out = mm.transform(X)
    np.testing.assert_allclose(out.min(0), 0)
    np.testing.assert_allclose(out.max(0), 1)


@pytest.mark.parametrize("strategy,expected", [("mean", 2.0), ("median", 2.0),
                                               ("most_frequent", 1.0), ("constant", -1.0)])
def test_imputer_strategies(strategy, expected):
    imp = SimpleImputer(strategy=strategy, fill_value=-1.0)
    imp.partial_fit(np.array([[1.0], [2.0]]))
    imp.partial_fit(np.array([[3.0], [np.nan]]))
    out = imp.transform(np.array([[np.nan], [7.0]]))
    assert out[0, 0] == pytest.approx(expected) and out[1, 0] == 7.0


def test_imputer_updates_on_the_fly_and_handles_all_nan_column():
    imp = SimpleImputer(fill_value=0.0)
    imp.partial_fit(np.array([[1.0, np.nan]]))
    assert imp.statistics_.tolist() == [1.0, 0.0]          # unseen column -> fill_value
    imp.partial_fit(np.array([[3.0, 10.0]]))
    assert imp.statistics_.tolist() == [2.0, 10.0]


def test_onehot_expands_incrementally_and_keeps_column_order():
    enc = OneHotEncoder(columns=[1])
    a = np.array([[1.0, "red"], [2.0, "blue"]], dtype=object)
    enc.partial_fit(a)
    first = enc.transform(a)
    assert first.shape == (2, 3)
    enc.partial_fit(np.array([[3.0, "green"]], dtype=object))
    second = enc.transform(a)
    assert second.shape == (2, 4)
    np.testing.assert_array_equal(second[:, :3], first)   # old columns unchanged
    assert enc.get_feature_names(["num", "col"]) == ["num", "col=red", "col=blue", "col=green"]


def test_onehot_unknown_and_missing():
    enc = OneHotEncoder().fit(np.array([["a"], ["b"]], dtype=object))
    out = enc.transform(np.array([["c"], [None]], dtype=object))
    np.testing.assert_array_equal(out, 0.0)
    strict = OneHotEncoder(handle_unknown="error").fit(np.array([["a"]], dtype=object))
    with pytest.raises(ValueError, match="unknown categories"):
        strict.transform(np.array([["z"]], dtype=object))


def test_scaler_growth_matches_zero_padded_batch():
    """allow_growth treats past rows as 0 in new columns, which is exact for one-hot."""
    sc = StandardScaler(allow_growth=True)
    sc.partial_fit(np.array([[1.0], [2.0]]))
    sc.partial_fit(np.array([[3.0, 1.0], [4.0, 0.0]]))
    full = np.array([[1.0, 0], [2.0, 0], [3.0, 1.0], [4.0, 0.0]])
    np.testing.assert_allclose(sc.mean_, full.mean(0))
    np.testing.assert_allclose(sc.var_, full.var(0))
