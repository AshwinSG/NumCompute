from pathlib import Path

import matplotlib
import numpy as np
import pytest

from numcompute_stream import visualise as vis
from numcompute_stream.io import iter_array_chunks, iter_csv_chunks, load_csv, save_csv
from numcompute_stream.metrics import StreamingAUC, StreamingF1
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier

DATA = Path(__file__).resolve().parents[2] / "data" / "sensor_stream.csv"


def make_pipe(buffer_size=200):
    return Pipeline([
        ("impute", SimpleImputer()),
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=5, buffer_size=buffer_size)),
    ])


# ---------------------------------------------------------------- pipeline
def test_pipeline_partial_fit_chains_steps(xor_data):
    X, y = xor_data
    pipe = make_pipe(buffer_size=100)
    for Xc, yc in iter_array_chunks(X, y, 50):
        pipe.partial_fit(Xc, yc)
    assert pipe["model"].n_retrains_ == 4
    np.testing.assert_allclose(pipe["scale"].mean_, X.mean(axis=0))  # running mean of full stream
    assert pipe.score(X, y) > 0.9


def test_pipeline_validation_and_batch_fit(xor_data):
    with pytest.raises(TypeError, match="lacks"):
        Pipeline([("bad", object()), ("m", DecisionTreeClassifier())])
    with pytest.raises(ValueError, match="unique"):
        Pipeline([("a", StandardScaler()), ("a", DecisionTreeClassifier())])
    X, y = xor_data
    assert make_pipe().fit(X, y).predict(X).shape == (len(X),)


def test_pipeline_with_growing_onehot_width():
    pipe = Pipeline([
        ("onehot", OneHotEncoder(columns=[1])),
        ("impute", SimpleImputer(allow_growth=True)),
        ("scale", StandardScaler(allow_growth=True)),
        ("model", DecisionTreeClassifier(max_depth=3, buffer_size=6)),
    ])
    pipe.partial_fit(np.array([[1.0, "a"], [2.0, "b"], [np.nan, "a"]], dtype=object), [0, 1, 0])
    pipe.partial_fit(np.array([[3.0, "c"], [4.0, "c"], [5.0, None]], dtype=object), [1, 1, 0])
    assert pipe["model"].n_features_in_ == 4 and pipe["model"].n_retrains_ == 1
    assert pipe.predict(np.array([[2.0, "zzz"]], dtype=object)).shape == (1,)


# ------------------------------------------------------------ stream trainer
def test_stream_trainer_prequential_logging(xor_data):
    X, y = xor_data
    tr = StreamTrainer(make_pipe(100), metrics={"f1": StreamingF1(), "auc": StreamingAUC()})
    hist = tr.run(iter_array_chunks(X, y, 50))
    assert len(hist["chunk"]) == 8
    # first two chunks: no tree yet -> NaN accuracy, but fit is still logged
    assert np.isnan(hist["chunk_accuracy"][0]) and hist["fit_time"][0] >= 0
    assert np.isfinite(hist["cumulative_accuracy"][-1])
    assert hist["memory_bytes"][-1] > 0 and hist["n_retrains"][-1] == 4
    assert tr.last_y_pred_.shape == (50,)
    assert 0.5 < tr.summary()["auc"] <= 1.0


def test_stream_trainer_score_then_fit_api(xor_data):
    X, y = xor_data
    tr = StreamTrainer(DecisionTreeClassifier(max_depth=3))
    assert tr.score_chunk(X[:10], y[:10]) is None       # untrained -> skipped
    tr.fit_chunk(X[:200], y[:200])
    rec = tr.score_chunk(X[200:], y[200:])
    assert rec["chunk"] == 1 and rec["chunk_accuracy"] > 0.9
    with pytest.raises(TypeError):
        StreamTrainer(object())


# ---------------------------------------------------------------------- io
def test_load_csv_types_and_missing():
    d = load_csv(DATA, target="fault")
    assert d.X.shape == (6000, 6) and d.y.dtype.kind == "i"
    assert d.categorical_idx == [5] and d.X.dtype == object
    assert any(v is not None and isinstance(v, float) and np.isnan(v) for v in d.X[:, 0])


def test_iter_csv_chunks_round_trip(tmp_path):
    X = np.array([[1.5, np.nan], [2.0, 3.0], [4.0, 5.0]])
    p = tmp_path / "t.csv"
    save_csv(p, X, ["a", "b"], y=[0, 1, 1], target_name="lbl")
    chunks = list(iter_csv_chunks(p, chunk_size=2, target="lbl"))
    assert [c.X.shape[0] for c in chunks] == [2, 1]
    np.testing.assert_array_equal(np.vstack([c.X for c in chunks]), X)
    with pytest.raises(KeyError):
        load_csv(p, target="missing")
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n3\n")
    with pytest.raises(ValueError, match="Ragged"):
        load_csv(tmp_path / "bad.csv")


# --------------------------------------------------------------- visualise
def test_visualise_functions_return_figures_and_save(tmp_path):
    matplotlib.use("Agg")
    fig, ax = vis.plot_metric_over_time([0.5, np.nan, 0.7, 0.8], "Acc", "accuracy",
                                        events=[1, 3], save_path=tmp_path / "a.png")
    assert (tmp_path / "a.png").exists() and ax.get_title() == "Acc"
    fig, ax = vis.compare_models([0.5, 0.6], [0.4, 0.7], labels=["b=100", "b=500"])
    assert len(ax.get_lines()) == 2
    fig, (a1, a2) = vis.plot_predictions_vs_ground_truth(np.array([0, 1, 1]), np.array([0, 1, 0]),
                                                         save_path=tmp_path / "p.svg")
    assert (tmp_path / "p.svg").exists()
    with pytest.raises(ValueError):
        vis.compare_models([0.1], [0.2], labels=["only one"])
    import matplotlib.pyplot as plt
    plt.close("all")
