"""Chain transformers and a final model, with support for data in chunks.

Pipeline.partial_fit(X_chunk, y_chunk) goes through each transformer in
order, calls partial_fit(chunk) and then transform(chunk), and passes the
result to the next step. The final model gets
partial_fit(transformed_chunk, y_chunk).

Note on buffered models: a DecisionTreeClassifier stores the transformed
rows, so rows stored early were scaled with slightly older statistics than
rows stored later. A tree only compares values within one feature, so this
mostly matters in the first few chunks while the scaler statistics are still
settling.
"""
from __future__ import annotations

from typing import List, Tuple

from .utils import NotFittedError


class Pipeline:
    """Runs steps in order: [(name, transformer), ..., (name, model)].

    Every step except the last must have partial_fit and transform (and fit
    for use on the whole dataset). The last step must have partial_fit, fit
    and predict.
    """

    def __init__(self, steps: List[Tuple[str, object]]):
        if not steps:
            raise ValueError("Pipeline needs at least one step")
        names = [n for n, _ in steps]
        if len(set(names)) != len(names):
            raise ValueError(f"step names must be unique, got {names}")
        for name, step in steps[:-1]:
            for meth in ("partial_fit", "transform"):
                if not hasattr(step, meth):
                    raise TypeError(f"step {name!r} ({type(step).__name__}) lacks {meth}()")
        last_name, last = steps[-1]
        if not hasattr(last, "partial_fit"):
            raise TypeError(f"final step {last_name!r} lacks partial_fit()")
        self.steps = list(steps)

    @property
    def named_steps(self) -> dict:
        return dict(self.steps)

    @property
    def transformers(self):
        return self.steps[:-1]

    @property
    def final_estimator(self):
        return self.steps[-1][1]

    def __getitem__(self, key):
        return self.named_steps[key] if isinstance(key, str) else self.steps[key][1]

    # ----------------------------------------------------------- streaming
    def partial_fit(self, X_chunk, y_chunk=None, **fit_params):
        """Update every step with one chunk."""
        Xt = X_chunk
        for _, step in self.transformers:
            step.partial_fit(Xt, y_chunk)
            Xt = step.transform(Xt)
        self.final_estimator.partial_fit(Xt, y_chunk, **fit_params)
        return self

    def flush(self):
        """Pass flush() on to the final model if it buffers data."""
        if hasattr(self.final_estimator, "flush"):
            self.final_estimator.flush()
        return self

    # -------------------------------------------------------- whole dataset
    def fit(self, X, y=None):
        Xt = X
        for _, step in self.transformers:
            Xt = step.fit(Xt, y).transform(Xt)
        self.final_estimator.fit(Xt, y)
        return self

    def transform(self, X):
        """Run all transformers (not the final model)."""
        Xt = X
        for name, step in self.transformers:
            try:
                Xt = step.transform(Xt)
            except NotFittedError as exc:
                raise NotFittedError(f"pipeline step {name!r} is not fitted: {exc}") from exc
        return Xt

    def predict(self, X):
        return self.final_estimator.predict(self.transform(X))

    def predict_proba(self, X):
        if not hasattr(self.final_estimator, "predict_proba"):
            raise AttributeError("final estimator has no predict_proba")
        return self.final_estimator.predict_proba(self.transform(X))

    def score(self, X, y):
        import numpy as np
        return float(np.mean(self.predict(X) == np.asarray(y)))

    @property
    def classes_(self):
        return self.final_estimator.classes_

    def __repr__(self):
        inner = ",\n  ".join(f"('{n}', {s!r})" for n, s in self.steps)
        return f"Pipeline([\n  {inner}\n])"
