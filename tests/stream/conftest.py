import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def rng():
    """Seeded generator (overrides the base-package fixture for these tests)."""
    return np.random.default_rng(0)


@pytest.fixture
def xor_data(rng):
    """Non-linearly separable 2-class data a depth-2 tree solves perfectly."""
    X = rng.uniform(-1, 1, size=(400, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X, y
