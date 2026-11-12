"""Generate ``sensor_stream.csv`` – a synthetic machine-fault stream.

* 6 000 rows, 5 numeric sensor readings + 1 categorical ``machine_type``
* ~3 % missing values in the numeric columns
* a new category (``"D"``) only appears after row 3 000 (exercises the
  incremental one-hot encoder)
* **concept drift** at row 3 500: the fault rule changes, so a model must
  retrain to recover (exercises ``buffer_size`` trade-offs)
* binary label ``fault`` (0/1)

Run:  python data/make_dataset.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from numcompute_stream.io import save_csv  # noqa: E402

N, DRIFT_AT, NEW_CAT_AT = 6000, 3500, 3000
rng = np.random.default_rng(42)

temp = rng.normal(70, 8, N)
pressure = rng.normal(30, 5, N)
humidity = rng.uniform(20, 80, N)
vibration = rng.gamma(2.0, 1.5, N)
voltage = rng.normal(230, 4, N)
mtype = rng.choice(["A", "B", "C"], N, p=[0.5, 0.3, 0.2]).astype(object)
late = np.arange(N) >= NEW_CAT_AT
mtype[late & (rng.random(N) < 0.25)] = "D"

type_risk = np.select([mtype == "A", mtype == "B", mtype == "C", mtype == "D"],
                      [0.0, 0.8, -0.5, 1.5])
# regime 1: heat + vibration cause faults
logit1 = 0.25 * (temp - 75) + 0.9 * (vibration - 3.5) + type_risk
# regime 2 (after drift): pressure + humidity cause faults, temperature reversed
logit2 = 0.45 * (pressure - 31) + 0.08 * (humidity - 55) - 0.12 * (temp - 70) + type_risk
logit = np.where(np.arange(N) < DRIFT_AT, logit1, logit2)
fault = (rng.random(N) < 1 / (1 + np.exp(-2.0 * logit))).astype(int)

X = np.column_stack([temp.round(2), pressure.round(2), humidity.round(2),
                     vibration.round(3), voltage.round(2)]).astype(object)
X[rng.random(X.shape) < 0.03] = np.nan
X = np.column_stack([X, mtype])

out = Path(__file__).with_name("sensor_stream.csv")
save_csv(out, X, ["temperature", "pressure", "humidity", "vibration", "voltage",
                  "machine_type"], y=fault, target_name="fault")
print(f"wrote {out} – {N} rows, fault rate {fault.mean():.3f}")
