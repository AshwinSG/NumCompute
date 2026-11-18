import numpy as np
import pytest

from numcompute_stream.io import iter_array_chunks
from numcompute_stream.metrics import (StreamingAccuracy, StreamingAUC, StreamingConfusionMatrix,
                                StreamingErrorRate, StreamingF1, StreamingPrecision,
                                StreamingRecall, accuracy_score, confusion_matrix,
                                confusion_matrix_loop, f1_score, precision_score,
                                recall_score, roc_auc_score)

Y_T = np.array([0, 1, 2, 2, 1, 0, 1, 2])
Y_P = np.array([0, 2, 2, 2, 1, 0, 0, 1])


def test_confusion_matrix_vectorised_matches_loop(rng):
    a, b = rng.integers(0, 5, 1000), rng.integers(0, 5, 1000)
    np.testing.assert_array_equal(confusion_matrix(a, b), confusion_matrix_loop(a, b))
    np.testing.assert_array_equal(confusion_matrix(Y_T, Y_P),
                                  [[2, 0, 0], [1, 1, 1], [0, 1, 2]])


def test_functional_prf_known_values():
    assert accuracy_score(Y_T, Y_P) == pytest.approx(5 / 8)
    # per-class precision: [2/3, 1/2, 2/3], recall: [1, 1/3, 2/3]
    np.testing.assert_allclose(precision_score(Y_T, Y_P, average=None), [2 / 3, 1 / 2, 2 / 3])
    np.testing.assert_allclose(recall_score(Y_T, Y_P, average=None), [1, 1 / 3, 2 / 3])
    assert precision_score(Y_T, Y_P, average="micro") == pytest.approx(5 / 8)
    assert f1_score(Y_T, Y_P, average="macro") == pytest.approx(
        np.mean([0.8, 0.4, 2 / 3]))


def test_zero_division_returns_zero():
    assert precision_score([0, 0], [0, 0], average="binary", pos_label=1) == 0.0
    assert recall_score([1, 1], [0, 0], average="binary", pos_label=1) == 0.0


@pytest.mark.parametrize("cls,fn", [(StreamingAccuracy, None),
                                    (StreamingPrecision, precision_score),
                                    (StreamingRecall, recall_score),
                                    (StreamingF1, f1_score)])
def test_streaming_equals_batch(rng, cls, fn):
    yt, yp = rng.integers(0, 3, 500), rng.integers(0, 3, 500)
    m = cls() if fn is None else cls(average="weighted")
    for (a, _), (b, _) in zip(iter_array_chunks(yt, chunk_size=37), iter_array_chunks(yp, chunk_size=37)):
        m.update(a, b)
    expected = accuracy_score(yt, yp) if fn is None else fn(yt, yp, average="weighted")
    assert m.result() == pytest.approx(expected)


def test_rolling_window_metric():
    m = StreamingAccuracy(window=2)
    m.update([1, 1], [0, 0])        # 0%, will drop out of the window
    m.update([1, 1], [1, 1])        # 100%
    m.update([1, 1], [1, 0])        # 50%
    assert m.result() == pytest.approx(0.75)
    m.reset()
    assert m.result() == 0.0
    assert StreamingErrorRate().update([1, 0], [1, 1]).result() == pytest.approx(0.5)


def test_confusion_matrix_accumulates_with_new_labels():
    cm = StreamingConfusionMatrix()
    cm.update(["b", "b"], ["b", "a"])
    cm.update(["c"], ["c"])            # new label mid-stream
    np.testing.assert_array_equal(cm.labels, ["a", "b", "c"])
    np.testing.assert_array_equal(cm.result(), [[0, 0, 0], [1, 1, 0], [0, 0, 1]])
    w = StreamingConfusionMatrix(window=1)
    w.update([0], [1]); w.update([1, 1], [1, 1])
    assert w.result().sum() == 2


def test_roc_auc_exact_and_ties():
    assert roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert roc_auc_score([0, 1], [0.5, 0.5]) == pytest.approx(0.5)  # tie counts 1/2
    assert np.isnan(roc_auc_score([1, 1], [0.2, 0.9]))              # single class


def test_streaming_auc_close_to_exact(rng):
    y = rng.integers(0, 2, 3000)
    s = np.clip(y * 0.3 + rng.uniform(0, 0.7, 3000), 0, 1)
    m = StreamingAUC(n_bins=2000)
    for (a, _), (b, _) in zip(iter_array_chunks(y, chunk_size=300), iter_array_chunks(s, chunk_size=300)):
        m.update(a, b)
    assert m.result() == pytest.approx(roc_auc_score(y, s), abs=2e-3)
    assert np.isnan(StreamingAUC().update([1, 1], [0.3, 0.4]).result())


def test_metric_shape_errors():
    with pytest.raises(ValueError, match="different lengths"):
        accuracy_score([1, 2, 3], [1, 2])
    with pytest.raises(ValueError, match="average"):
        StreamingF1(average="bogus")
