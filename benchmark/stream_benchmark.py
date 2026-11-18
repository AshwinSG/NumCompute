"""Speed tests for NumCompute 2.2.

(a) Python loop versions against NumPy array versions of the slowest parts:
    split search, tree prediction, running mean and variance, confusion matrix.
(b) Effect of buffer_size on a streaming run over data/sensor_stream.csv
    (test-then-train accuracy, accuracy after the drift, number of retrains,
    retrain time, memory).

Run:  python benchmark/stream_benchmark.py
Outputs: benchmark/stream_results.md and PNG figures in benchmark/figures/.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from numcompute_stream import visualise as vis  # noqa: E402
from numcompute_stream.io import iter_csv_chunks  # noqa: E402
from numcompute_stream.metrics import StreamingAccuracy, confusion_matrix, confusion_matrix_loop  # noqa: E402
from numcompute_stream.pipeline import Pipeline  # noqa: E402
from numcompute_stream.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler  # noqa: E402
from numcompute_stream.stats import StreamingStats, welford_loop  # noqa: E402
from numcompute_stream.stream import StreamTrainer  # noqa: E402
from numcompute_stream.tree import DecisionTreeClassifier, best_split, best_split_loop  # noqa: E402

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
DATA = ROOT / "data" / "sensor_stream.csv"
DRIFT_AT = 3500


def timeit(fn, repeat=3):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# --------------------------------------------------------------------------- #
def bench_loop_vs_vectorised(rng):
    rows = []

    for n in (250, 1000, 4000):
        X = rng.normal(size=(n, 8))
        y = (X[:, 0] + rng.normal(size=n) > 0).astype(int)
        assert best_split(X, y, 2)[:2] == best_split_loop(X, y, 2)[:2]
        rows.append((f"Best-split search (n={n}, 8 features)",
                     timeit(lambda: best_split_loop(X, y, 2), 1),
                     timeit(lambda: best_split(X, y, 2))))

    X = rng.normal(size=(20_000, 8))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    tree = DecisionTreeClassifier(max_depth=10).fit(X[:5000], y[:5000])
    assert (tree.predict(X) == tree.predict_loop(X)).all()
    rows.append(("Tree predict (20k samples, depth 10)",
                 timeit(lambda: tree.predict_loop(X), 1), timeit(lambda: tree.predict(X))))

    Xs = rng.normal(size=(20_000, 8))

    def vec_stats():
        s = StreamingStats()
        for i in range(0, len(Xs), 500):
            s.update_stats(Xs[i:i + 500])
    rows.append(("Running mean/var (20k x 8, chunks of 500)",
                 timeit(lambda: welford_loop(Xs), 1), timeit(vec_stats)))

    a, b = rng.integers(0, 5, 200_000), rng.integers(0, 5, 200_000)
    rows.append(("Confusion matrix (200k labels, 5 classes)",
                 timeit(lambda: confusion_matrix_loop(a, b), 1),
                 timeit(lambda: confusion_matrix(a, b))))
    return rows


# --------------------------------------------------------------------------- #
def make_pipeline(buffer_size):
    return Pipeline([
        ("onehot", OneHotEncoder(columns=[5])),
        ("impute", SimpleImputer(strategy="mean", allow_growth=True)),
        ("scale", StandardScaler(allow_growth=True)),
        ("tree", DecisionTreeClassifier(max_depth=6, min_samples_leaf=5,
                                        buffer_size=buffer_size, random_state=0)),
    ])


def run_stream(buffer_size, chunk_size=100):
    trainer = StreamTrainer(make_pipeline(buffer_size),
                            metrics={"rolling_acc": StreamingAccuracy(window=5)})
    post = StreamingAccuracy()
    seen = 0
    for chunk in iter_csv_chunks(DATA, chunk_size, target="fault", categorical=["machine_type"]):
        rec = trainer.score_chunk(chunk.X, chunk.y)
        if rec is not None and seen >= DRIFT_AT:
            post.update(chunk.y, trainer.last_y_pred_)
        trainer.fit_chunk(chunk.X, chunk.y)
        seen += len(chunk.y)
    tree = trainer.model["tree"]
    s = trainer.summary()
    return {
        "buffer_size": buffer_size,
        "prequential_acc": s["final_cumulative_accuracy"],
        "post_drift_acc": post.result(),
        "n_retrains": tree.n_retrains_,
        "total_retrain_s": float(np.sum(tree.retrain_times_)),
        "mean_retrain_ms": 1e3 * float(np.mean(tree.retrain_times_)),
        "peak_mem_kb": s["peak_memory_bytes"] / 1024,
        "history": trainer.history,
    }


def bench_buffer_sizes(sizes=(100, 250, 500, 1000, 2000, 3000)):
    return [run_stream(b) for b in sizes]


# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(0)
    print("(a) loop vs vectorised ...")
    lv = bench_loop_vs_vectorised(rng)
    print("(b) buffer_size sweep ...")
    bs = bench_buffer_sizes()

    lines = ["# NumCompute 2.2 benchmarks", "",
             f"Machine: {os.uname().machine}, NumPy {np.__version__}. "
             "Loop timings: single run; vectorised: best of 3.", "",
             "## (a) Loop-based vs. vectorised", "",
             "| Operation | Loop (s) | Vectorised (s) | Speed-up |",
             "|---|---:|---:|---:|"]
    for name, tl, tv in lv:
        lines.append(f"| {name} | {tl:.4f} | {tv:.5f} | {tl / tv:,.0f}x |")
    lines += ["", "## (b) Effect of `buffer_size` (6 000-row stream, chunks of 100, "
              f"concept drift at row {DRIFT_AT})", "",
              "| buffer_size | Prequential acc | Acc after drift | Retrains | "
              "Total retrain (s) | Mean retrain (ms) | Peak memory (KB) |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in bs:
        lines.append(f"| {r['buffer_size']} | {r['prequential_acc']:.3f} | "
                     f"{r['post_drift_acc']:.3f} | {r['n_retrains']} | "
                     f"{r['total_retrain_s']:.3f} | {r['mean_retrain_ms']:.1f} | "
                     f"{r['peak_mem_kb']:.0f} |")
    (OUT / "stream_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    vis.plot_tradeoff([r["buffer_size"] for r in bs], [r["prequential_acc"] for r in bs],
                      [r["total_retrain_s"] for r in bs],
                      save_path=FIG / "buffer_tradeoff.png")
    small, large = bs[1], bs[-2]
    vis.compare_models(small["history"]["rolling_acc"], large["history"]["rolling_acc"],
                       labels=[f"buffer_size={small['buffer_size']}",
                               f"buffer_size={large['buffer_size']}"],
                       title="Rolling accuracy (last 5 chunks) – drift at chunk 35",
                       ylabel="Rolling accuracy", save_path=FIG / "buffer_compare.png")
    labels = [n.split(" (")[0] for n, _, _ in lv]
    speed = [tl / tv for _, tl, tv in lv]
    import matplotlib.pyplot as plt
    with plt.rc_context(vis.STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(labels[::-1], speed[::-1], color=vis.SERIES_COLORS[0], height=0.6)
        ax.set_xscale("log")
        ax.grid(axis="x"); ax.grid(axis="y", visible=False)
        for i, v in enumerate(speed[::-1]):
            ax.annotate(f"{v:,.0f}x", (v, i), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=vis.INK_2)
        ax.set(title="Vectorised speed-up over loop implementation (log scale)",
               xlabel="Speed-up (x)")
        fig.tight_layout()
        fig.savefig(FIG / "vectorisation_speedup.png", dpi=150, facecolor=vis.SURFACE)
    print(f"\nwrote {OUT / 'stream_results.md'} and figures in {FIG}")


if __name__ == "__main__":
    main()
