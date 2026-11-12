import numpy as np
import pytest

from numcompute_stream.io import iter_array_chunks
from numcompute_stream.tree import DecisionTreeClassifier, best_split, best_split_loop, impurity
from numcompute_stream.utils import NotFittedError


def test_impurity_values():
    assert impurity(np.array([5, 5]), "gini") == pytest.approx(0.5)
    assert impurity(np.array([5, 5]), "entropy") == pytest.approx(1.0)
    assert impurity(np.array([10, 0]), "entropy") == 0.0
    assert impurity(np.array([0, 0]), "gini") == 0.0  # empty node, no NaN


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_tree_solves_xor(xor_data, criterion):
    X, y = xor_data
    t = DecisionTreeClassifier(max_depth=4, criterion=criterion).fit(X, y)
    assert t.score(X, y) > 0.97


def test_max_depth_respected(rng):
    X = rng.normal(size=(500, 5))
    y = rng.integers(0, 3, 500)
    for d in (0, 1, 3):
        t = DecisionTreeClassifier(max_depth=d).fit(X, y)
        assert t.get_depth() <= d
    assert DecisionTreeClassifier(max_depth=0).fit(X, y).n_nodes == 1


def test_min_samples_leaf_respected(rng):
    X = rng.normal(size=(300, 3))
    y = rng.integers(0, 2, 300)
    t = DecisionTreeClassifier(min_samples_leaf=20).fit(X, y)
    leaves = t.feature_ == -1
    assert t.n_node_samples_[leaves].min() >= 20


def test_vectorised_split_matches_loop(rng):
    for seed in range(5):
        r = np.random.default_rng(seed)
        X = r.normal(size=(150, 4))
        X[r.random(X.shape) < 0.1] = np.nan
        y = r.integers(0, 3, 150)
        for crit in ("gini", "entropy"):
            a = best_split(X, y, 3, crit, min_samples_leaf=3)
            b = best_split_loop(X, y, 3, crit, min_samples_leaf=3)
            assert a[0] == b[0] and a[3] == b[3]
            assert a[1] == pytest.approx(b[1]) and a[2] == pytest.approx(b[2])


def test_ties_never_split_equal_values_and_are_deterministic():
    X = np.array([[1.0], [1.0], [1.0], [2.0], [2.0]])
    y = np.array([0, 1, 0, 1, 1])
    f, thr, _, _ = best_split(X, y, 2)
    assert 1.0 <= thr < 2.0   # threshold strictly between distinct values
    # identical duplicate features -> lowest index wins
    X2 = np.hstack([X, X])
    assert best_split(X2, y, 2)[0] == 0
    # leaf class tie -> smallest label
    t = DecisionTreeClassifier(max_depth=0).fit(np.zeros((4, 1)), np.array([3, 3, 1, 1]))
    assert t.predict(np.zeros((1, 1)))[0] == 1


def test_constant_features_produce_single_leaf():
    t = DecisionTreeClassifier().fit(np.ones((50, 3)), np.r_[np.zeros(25), np.ones(25)])
    assert t.n_nodes == 1
    np.testing.assert_allclose(t.predict_proba(np.ones((1, 3))), [[0.5, 0.5]])


def test_nan_in_features_train_and_predict(xor_data, rng):
    X, y = xor_data
    X = X.copy()
    X[rng.random(X.shape) < 0.1] = np.nan
    t = DecisionTreeClassifier(max_depth=4).fit(X, y)
    pred = t.predict(np.array([[np.nan, np.nan], [0.5, np.nan]]))
    assert pred.shape == (2,)
    assert t.score(X, y) > 0.8


def test_predict_proba_rows_sum_to_one_and_string_labels(rng):
    X = rng.normal(size=(200, 2))
    y = np.where(X[:, 0] > 0, "pos", "neg")
    t = DecisionTreeClassifier(max_depth=3).fit(X, y)
    p = t.predict_proba(X)
    np.testing.assert_allclose(p.sum(1), 1.0)
    assert set(t.predict(X)) <= {"pos", "neg"}


def test_vectorised_predict_matches_loop(rng):
    X = rng.normal(size=(400, 5))
    X[rng.random(X.shape) < 0.05] = np.nan
    y = (np.nan_to_num(X[:, 0]) + np.nan_to_num(X[:, 3]) > 0).astype(int)
    t = DecisionTreeClassifier(max_depth=8).fit(X, y)
    np.testing.assert_array_equal(t.predict(X), t.predict_loop(X))


def test_max_features_options(rng):
    X = rng.normal(size=(200, 9))
    y = (X[:, 0] > 0).astype(int)
    for mf in ("sqrt", "log2", 2, 0.5, None):
        t = DecisionTreeClassifier(max_depth=3, max_features=mf, random_state=0).fit(X, y)
        assert t.is_fitted


def test_feature_importances(rng):
    X = rng.normal(size=(500, 4))
    y = (X[:, 2] > 0).astype(int)
    imp = DecisionTreeClassifier(max_depth=3).fit(X, y).feature_importances_
    assert imp.argmax() == 2 and imp.sum() == pytest.approx(1.0)


# ------------------------------------------------------------- streaming
def test_buffered_retraining_schedule(xor_data):
    X, y = xor_data
    t = DecisionTreeClassifier(max_depth=4, buffer_size=150)
    retrain_counts = []
    for Xc, yc in iter_array_chunks(X, y, chunk_size=50):
        t.partial_fit(Xc, yc)
        retrain_counts.append(t.n_retrains_)
    # 400 samples / 50 per chunk; retrain after chunks 3 and 6 (150 each)
    assert retrain_counts == [0, 0, 1, 1, 1, 2, 2, 2]
    assert t.buffer_count == 100
    t.flush()
    assert t.n_retrains_ == 3 and t.buffer_count == 0


def test_buffer_overshoot_trains_on_all_buffered(rng):
    t = DecisionTreeClassifier(buffer_size=100)
    X = rng.normal(size=(130, 2))
    y = (X[:, 0] > 0).astype(int)
    t.partial_fit(X[:70], y[:70])
    t.partial_fit(X[70:], y[70:])           # buffer = 130 >= 100
    assert t.n_retrains_ == 1 and t.n_node_samples_[0] == 130 and t.buffer_count == 0


def test_predict_uses_last_tree_between_retrains(xor_data, rng):
    X, y = xor_data
    t = DecisionTreeClassifier(max_depth=4, buffer_size=200).fit(X, y)
    before = t.predict(X).copy()
    t.partial_fit(rng.normal(size=(50, 2)), rng.integers(0, 2, 50))  # buffered only
    np.testing.assert_array_equal(t.predict(X), before)


def test_predict_before_first_retrain_raises_informative_error(rng):
    t = DecisionTreeClassifier(buffer_size=500)
    t.partial_fit(rng.normal(size=(10, 2)), rng.integers(0, 2, 10))
    with pytest.raises(NotFittedError, match="10/500 samples buffered"):
        t.predict(np.zeros((1, 2)))


def test_buffer_size_none_retrains_every_call(rng):
    t = DecisionTreeClassifier()
    for _ in range(3):
        t.partial_fit(rng.normal(size=(20, 2)), rng.integers(0, 2, 20))
    assert t.n_retrains_ == 3


def test_new_class_in_stream_and_classes_argument(rng):
    t = DecisionTreeClassifier(buffer_size=10)
    t.partial_fit(rng.normal(size=(10, 2)), np.zeros(10, int), classes=[0, 1])
    assert t.predict_proba(np.zeros((1, 2))).shape == (1, 2)
    t.partial_fit(rng.normal(size=(5, 2)), np.full(5, 2))        # new class, buffered
    p = t.predict_proba(np.zeros((1, 2)))
    assert p.shape == (1, 3) and p[0, 2] == 0.0                  # unseen by current tree


def test_single_class_chunk_and_growing_features(rng):
    t = DecisionTreeClassifier(buffer_size=20)
    t.partial_fit(rng.normal(size=(10, 2)), np.ones(10, int))
    t.partial_fit(rng.normal(size=(10, 3)), np.ones(10, int))   # one-hot grew -> padded
    assert t.n_features_in_ == 3 and (t.predict(np.zeros((2, 3))) == 1).all()
    with pytest.raises(ValueError, match="features may only be appended"):
        t.partial_fit(rng.normal(size=(5, 3)), np.ones(5, int))
        t.partial_fit(rng.normal(size=(5, 2)), np.ones(5, int))


@pytest.mark.parametrize("kwargs", [{"max_depth": -1}, {"buffer_size": 0},
                                    {"criterion": "mse"}, {"min_samples_split": 1},
                                    {"max_features": "all"}])
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(**kwargs)


def test_shape_errors():
    t = DecisionTreeClassifier()
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        t.fit(np.ones((5, 2)), np.ones(4))
    with pytest.raises(ValueError, match="2-D"):
        t.fit(np.ones(5), np.ones(5))
    t.fit(np.ones((5, 2)), np.ones(5))
    with pytest.raises(ValueError, match="trained on 2"):
        t.predict(np.ones((3, 1)))
