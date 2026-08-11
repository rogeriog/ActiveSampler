# Paper Support Data

This folder contains the data, scripts, and figures supporting the retrospective benchmark presented in the manuscript:

**"Data-Driven Phase Selectivity in Cs₃Sb₂I₉ via Robotic LARP and ActiveSampler"**

## Contents

### `data/`
- `consolidated_data.csv` — Unified dataset of 88 experiments (initial screening + AL Round 1 + AL Round 2), with standardized columns and a `phase` column indicating the origin of each data point.

### `scripts/`
- `retrospective_benchmark.py` — Python script that reproduces all benchmark results and figures. Trains a surrogate oracle on initial screening data only (avoiding circularity), then simulates random search, grid search, factorial design (3 tiers), random+model ablation, and replays the actual ActiveSampler selections.

### `figures/`
- `benchmark_surrogate_based.png` — **Figure S5**: Cumulative trigonal discovery curves for all strategies.
- `benchmark_hit_rates.png` — **Figure S6**: Hit rate bar chart comparison.
- `benchmark_ablation.png` — **Figure S7**: Ablation study (AL acquisition vs random acquisition, same model).
- `benchmark_prediction_map.png` — Surrogate model prediction map (supplementary).
- `benchmark_parameter_space.png` — Experimental coverage of parameter space (supplementary).

## Reproducing Results

```bash
pip install pandas numpy scikit-learn matplotlib
python scripts/retrospective_benchmark.py
```

## Key Results

| Strategy | Trigonal found (48 exp) | Hit rate (%) | vs Random |
|---|---|---|---|
| Active Learning (actual) | 35 | 72.9 | 3.3× |
| Random Search (mean) | 10.7 | 22.4 | 1.0× |
| Random + Model (ablation) | 10.8 | 22.5 | 1.0× |
| Grid Search (best ordering) | 17 | 35.4 | 1.6× |
| Factorial (proper 2-level) | 0 | 0.0 | 0.0× |
| Factorial (best-case mid) | 16 | 33.3 | 1.5× |
| Factorial (cheating, all-data) | 27 | 56.2 | 2.5× |
