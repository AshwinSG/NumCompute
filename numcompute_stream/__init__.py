"""NumCompute stream: a NumPy-only package for training decision trees on
data that arrives in chunks.

Modules:
    io             load CSV files, or read them in chunks
    stats          running mean, variance, quantiles and histograms (update_stats)
    preprocessing  StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder
    tree           DecisionTreeClassifier that retrains from a buffer
    metrics        metric functions and running metrics
    pipeline       Pipeline with partial_fit
    stream         StreamTrainer (tests on each chunk, then trains on it)
    visualise      matplotlib plot helpers
"""
from .io import iter_array_chunks, iter_csv_chunks, load_csv, save_csv
from .metrics import (StreamingAccuracy, StreamingAUC, StreamingConfusionMatrix,
                      StreamingErrorRate, StreamingF1, StreamingPrecision,
                      StreamingRecall, accuracy_score, confusion_matrix, f1_score,
                      precision_score, recall_score, roc_auc_score)
from .pipeline import Pipeline
from .preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler
from .stats import (StreamingHistogram, StreamingQuantile, StreamingStats,
                    StreamingSummary)
from .stream import StreamTrainer
from .tree import DecisionTreeClassifier
from .utils import NotFittedError

__version__ = "2.2.0"

__all__ = [
    "load_csv", "iter_csv_chunks", "iter_array_chunks", "save_csv",
    "StreamingStats", "StreamingQuantile", "StreamingHistogram", "StreamingSummary",
    "StandardScaler", "MinMaxScaler", "SimpleImputer", "OneHotEncoder",
    "DecisionTreeClassifier", "Pipeline", "StreamTrainer",
    "accuracy_score", "precision_score", "recall_score", "f1_score",
    "confusion_matrix", "roc_auc_score",
    "StreamingAccuracy", "StreamingErrorRate", "StreamingPrecision", "StreamingRecall",
    "StreamingF1", "StreamingConfusionMatrix", "StreamingAUC", "NotFittedError",
]
