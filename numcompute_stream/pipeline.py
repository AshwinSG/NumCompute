"""Chain transformers and a final estimator, with streaming support.

``Pipeline.partial_fit(X_chunk, y_chunk)`` runs, for each transformer in
order, ``partial_fit(chunk)`` followed by ``transform(chunk)`` and feeds the
result to the next step; the final estimator receives
``partial_fit(transformed_chunk, y_chunk)``.

Note on buffered models: a :class:`DecisionTreeClassifier` buffers the
*transformed* rows, so rows buffered early were transformed with slightly
older statistics than rows buffered later. Trees only compare values within
a feature, so this mainly matters during the first few chunks while scaler
statistics are still settling.
"""
from __future__ import annotations

from typing import List, Tuple

from .utils import NotFittedError


class Pipeline:
    """Sequential pipeline: ``[(name, transformer), ..., (name, estimator)]``.

    Every step except the last must implement ``partial_fit`` / ``transform``
    (and ``fit`` for batch use). The last step must implement
    ``partial_fit`` / ``fit`` / ``predict``.
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
        """Incrementally update every step with one chunk."""
        Xt = X_chunk
        for _, step in self.transformers:
            step.partial_fit(Xt, y_chunk)
            Xt = step.transform(Xt)
        self.final_estimator.partial_fit(Xt, y_chunk, **fit_params)
        return self

    def flush(self):
        """Forward ``flush()`` to the final estimator if it buffers data."""
        if hasattr(self.final_estimator, "flush"):
            self.final_estimator.flush()
        return self

    # --------------------------------------------------------------- batch
    def fit(self, X, y=None):
        Xt = X
        for _, step in self.transformers:
            Xt = step.fit(Xt, y).transform(Xt)
        self.final_estimator.fit(Xt, y)
        return self

    def transform(self, X):
        """Apply all transformers (not the final estimator)."""
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
