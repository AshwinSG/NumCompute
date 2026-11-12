# NumCompute 2.2 benchmarks

Machine: arm64, NumPy 2.0.2. Loop timings: single run; vectorised: best of 3.

## (a) Loop-based vs. vectorised

| Operation | Loop (s) | Vectorised (s) | Speed-up |
|---|---:|---:|---:|
| Best-split search (n=250, 8 features) | 0.0158 | 0.00015 | 107x |
| Best-split search (n=1000, 8 features) | 0.0632 | 0.00052 | 121x |
| Best-split search (n=4000, 8 features) | 0.2534 | 0.00261 | 97x |
| Tree predict (20k samples, depth 10) | 0.0181 | 0.00086 | 21x |
| Running mean/var (20k x 8, chunks of 500) | 0.0661 | 0.00154 | 43x |
| Confusion matrix (200k labels, 5 classes) | 0.0310 | 0.00536 | 6x |

## (b) Effect of `buffer_size` (6 000-row stream, chunks of 100, concept drift at row 3500)

| buffer_size | Prequential acc | Acc after drift | Retrains | Total retrain (s) | Mean retrain (ms) | Peak memory (KB) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.805 | 0.768 | 60 | 0.033 | 0.5 | 2 |
| 250 | 0.828 | 0.780 | 20 | 0.032 | 1.6 | 19 |
| 500 | 0.821 | 0.756 | 12 | 0.031 | 2.6 | 36 |
| 1000 | 0.814 | 0.744 | 6 | 0.029 | 4.8 | 77 |
| 2000 | 0.774 | 0.714 | 3 | 0.028 | 9.2 | 156 |
| 3000 | 0.540 | 0.474 | 2 | 0.026 | 12.8 | 234 |
