"""
Retrospective benchmark: Active Learning vs Random Search vs Grid Search
vs Factorial Design vs Random+Model (ablation).

Addresses Reviewer 2 & 3 requirement for quantitative benchmarking of the
active learning workflow against conventional strategies.

Key design decisions to avoid circularity:
  - Surrogate model trained ONLY on initial screening data (not AL data)
  - Validated against actual AL Round 1 & 2 outcomes
  - Factorial design uses proper 2-level corners + center points (no retrospective knowledge)
  - Random+Model ablation: same model, random acquisition, iterative retraining
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from itertools import product
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# 1. Load consolidated data
# ============================================================
data = pd.read_csv('consolidated_data.csv')
data['is_trigonal'] = (data['structural_response'] == 2).astype(int)

print("=" * 70)
print(" DATASET SUMMARY")
print("=" * 70)
print(f"Total experiments: {len(data)}")
for phase in ['initial', 'AL_round1', 'AL_round2']:
    subset = data[data.phase == phase]
    n_tri = subset.is_trigonal.sum()
    print(f"  {phase:12s}: {len(subset):3d} experiments, {n_tri:2d} trigonal ({100*n_tri/len(subset):.1f}%)")
print(f"  {'Total':12s}: {len(data):3d} experiments, {data.is_trigonal.sum():2d} trigonal")

# ============================================================
# 2. Define full parameter grid (from Table S2)
# ============================================================
OLA_LEVELS = [0, 5, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
BR_LEVELS = [0, 1, 2, 3, 4]
OA_LEVELS = [0, 300, 600, 900, 1200]
TOLUENE_LEVELS = [3, 6, 9, 12]

grid_points = list(product(OLA_LEVELS, BR_LEVELS, OA_LEVELS, TOLUENE_LEVELS))
full_grid = pd.DataFrame(grid_points, columns=['ola_uL', 'y_br', 'oa_uL', 'toluene_mL'])
print(f"\nFull parameter grid: {len(full_grid)} combinations")

features = ['ola_uL', 'y_br', 'oa_uL', 'toluene_mL']

# ============================================================
# 3. BUILD SURROGATE ORACLE (trained on INITIAL data only)
# ============================================================
print("\n" + "=" * 70)
print(" BUILDING SURROGATE ORACLE (initial screening only)")
print("=" * 70)

initial_data = data[data.phase == 'initial'].reset_index(drop=True)
al_data = data[data.phase.isin(['AL_round1', 'AL_round2'])].reset_index(drop=True)

# Train ONLY on initial screening data
X_init = initial_data[features].values
y_init = initial_data['is_trigonal'].values

print(f"Training set (initial screening): {len(initial_data)} points ({y_init.sum()} trigonal)")
print(f"Validation set (AL rounds):       {len(al_data)} points ({al_data.is_trigonal.sum()} trigonal)")

# Cross-validation on initial data
rf_init = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=5)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_init, X_init, y_init, cv=cv, scoring='accuracy')
print(f"\nSurrogate model (RF, trained on initial screening only)")
print(f"  5-fold CV accuracy on initial data: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

# Train on initial data, validate on AL outcomes
rf_init.fit(X_init, y_init)
X_al = al_data[features].values
y_al = al_data['is_trigonal'].values
al_pred = rf_init.predict(X_al)
al_proba = rf_init.predict_proba(X_al)[:, 1]
al_accuracy = (al_pred == y_al).mean()
print(f"  Accuracy on AL outcomes (out-of-sample): {al_accuracy:.3f}")

# Per-round validation
for phase in ['AL_round1', 'AL_round2']:
    subset = al_data[al_data.phase == phase]
    X_sub = subset[features].values
    y_sub = subset['is_trigonal'].values
    pred_sub = rf_init.predict(X_sub)
    acc = (pred_sub == y_sub).mean()
    print(f"    {phase}: accuracy = {acc:.3f} ({(pred_sub == y_sub).sum()}/{len(subset)})")

# Predict on full grid
X_grid = full_grid[features].values
grid_proba = rf_init.predict_proba(X_grid)[:, 1]
grid_pred = (grid_proba >= 0.5).astype(int)
full_grid['pred_trigonal'] = grid_pred
full_grid['pred_proba'] = grid_proba

print(f"\n  Predicted trigonal in full grid: {grid_pred.sum()} / {len(full_grid)} ({100*grid_pred.sum()/len(full_grid):.1f}%)")

# Build oracle: actual outcomes for ALL tested points (initial + AL), predictions for untested
in_grid = data[data.in_grid == 'yes'].reset_index(drop=True)
tested_map = {}
for _, row in in_grid.iterrows():
    key = (row.ola_uL, row.y_br, row.oa_uL, row.toluene_mL)
    tested_map[key] = row.is_trigonal

full_grid['tested'] = False
full_grid['oracle'] = grid_pred  # Model prediction as default
for idx, row in full_grid.iterrows():
    key = (row.ola_uL, row.y_br, row.oa_uL, row.toluene_mL)
    if key in tested_map:
        full_grid.loc[idx, 'tested'] = True
        full_grid.loc[idx, 'oracle'] = tested_map[key]  # Override with actual outcome

n_tested = int(full_grid.tested.sum())
n_oracle_tri = int(full_grid.oracle.sum())
print(f"  Grid points actually tested: {n_tested}")
print(f"  Oracle trigonal (actual + predicted): {n_oracle_tri} / {len(full_grid)} ({100*n_oracle_tri/len(full_grid):.1f}%)")

# Also build an "all-data" oracle (trained on all 88 points including AL outcomes)
# This is an optimistic upper bound -- gives baselines access to AL-discovered knowledge.
# Used only for the "cheating" factorial tier to show even that doesn't beat AL.
X_all = data[features].values
y_all = data['is_trigonal'].values
rf_all = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=6)
rf_all.fit(X_all, y_all)
grid_proba_all = rf_all.predict_proba(X_grid)[:, 1]
grid_pred_all = (grid_proba_all >= 0.5).astype(int)
full_grid['oracle_alldata'] = grid_pred_all
for idx, row in full_grid.iterrows():
    key = (row.ola_uL, row.y_br, row.oa_uL, row.toluene_mL)
    if key in tested_map:
        full_grid.loc[idx, 'oracle_alldata'] = tested_map[key]
n_oracle_all_tri = int(full_grid.oracle_alldata.sum())
print(f"\n  All-data oracle (optimistic upper bound):")
print(f"    Predicted trigonal: {n_oracle_all_tri} / {len(full_grid)} ({100*n_oracle_all_tri/len(full_grid):.1f}%)")
print(f"    (Trained on all 88 pts incl. AL outcomes -- used only for cheating factorial)")

# ============================================================
# 4. SIMULATE STRATEGIES ON FULL GRID
# ============================================================
print("\n" + "=" * 70)
print(" SURROGATE-BASED RETROSPECTIVE BENCHMARK")
print("=" * 70)

# Starting point: in-grid initial screening points
initial_in_grid = in_grid[in_grid.phase == 'initial'].reset_index(drop=True)
initial_keys = set(
    initial_in_grid.apply(lambda r: (r.ola_uL, r.y_br, r.oa_uL, r.toluene_mL), axis=1)
)
n_initial = len(initial_in_grid)
initial_tri = int(initial_in_grid.is_trigonal.sum())
print(f"\nStarting point: {n_initial} initial in-grid experiments ({initial_tri} trigonal)")

remaining = full_grid[
    full_grid.apply(lambda r: (r.ola_uL, r.y_br, r.oa_uL, r.toluene_mL) not in initial_keys, axis=1)
].reset_index(drop=True)
n_remaining_tri = int(remaining.oracle.sum())
print(f"Remaining grid: {len(remaining)} points ({n_remaining_tri} oracle-trigonal, {100*n_remaining_tri/len(remaining):.1f}%)")

# --- AL actual trajectory ---
al_selections = in_grid[in_grid.phase.isin(['AL_round1', 'AL_round2'])].reset_index(drop=True)
n_al = len(al_selections)
al_tri = int(al_selections.is_trigonal.sum())
print(f"\nAL actual: {n_al} additional experiments, {al_tri} trigonal found")

al_cumulative = np.zeros(n_al + 1)
al_cumulative[0] = initial_tri
for i in range(n_al):
    al_cumulative[i + 1] = al_cumulative[i] + al_selections.is_trigonal.iloc[i]

# --- Strategy 1: Random Search (Monte Carlo, 10000 runs) ---
N_RUNS = 10000
max_budget = n_al
random_curves = np.zeros((N_RUNS, max_budget + 1))
for run in range(N_RUNS):
    sampled = remaining.sample(n=max_budget, random_state=run)
    cum = sampled.oracle.cumsum().values
    random_curves[run, 0] = initial_tri
    random_curves[run, 1:] = initial_tri + cum

rand_mean = random_curves.mean(axis=0)
rand_std = random_curves.std(axis=0)
rand_p5 = np.percentile(random_curves, 5, axis=0)
rand_p25 = np.percentile(random_curves, 25, axis=0)
rand_p75 = np.percentile(random_curves, 75, axis=0)
rand_p95 = np.percentile(random_curves, 95, axis=0)

print(f"Random search (mean): {rand_mean[-1]:.1f} +/- {rand_std[-1]:.1f} trigonal in {max_budget} experiments")

# --- Strategy 2: Grid Search (systematic traversal, multiple orderings) ---
grid_orderings = [
    ['ola_uL', 'y_br', 'oa_uL', 'toluene_mL'],
    ['toluene_mL', 'oa_uL', 'y_br', 'ola_uL'],
    ['y_br', 'ola_uL', 'toluene_mL', 'oa_uL'],
    ['oa_uL', 'toluene_mL', 'ola_uL', 'y_br'],
]
grid_curves = []
for ordering in grid_orderings:
    gs = remaining.sort_values(ordering).reset_index(drop=True)
    gc = np.zeros(max_budget + 1)
    gc[0] = initial_tri
    for i in range(min(max_budget, len(gs))):
        gc[i + 1] = gc[i] + gs.oracle.iloc[i]
    grid_curves.append(gc)
grid_curves = np.array(grid_curves)
grid_best = grid_curves.max(axis=0)
grid_worst = grid_curves.min(axis=0)
grid_mean = grid_curves.mean(axis=0)

# How many experiments does grid search need to find first trigonal?
grid_first_tri = []
for ordering in grid_orderings:
    gs = remaining.sort_values(ordering).reset_index(drop=True)
    oracle_vals = gs.oracle.values
    first_idx = np.argmax(oracle_vals > 0) + 1 if oracle_vals.sum() > 0 else len(gs)
    grid_first_tri.append(first_idx)

print(f"Grid search (best of 4 orderings):  {grid_best[-1]:.0f} trigonal in {max_budget} experiments")
print(f"Grid search (worst of 4 orderings): {grid_worst[-1]:.0f} trigonal in {max_budget} experiments")
print(f"Grid search: experiments to 1st trigonal: {min(grid_first_tri)}-{max(grid_first_tri)} (depending on ordering)")
print(f"  (vs AL: 1 experiment to find 1st trigonal)")

# --- Strategy 3: Factorial Design (proper 2-level + center points) ---
# 2^4 = 16 corner points: min/max of each variable
# No retrospective knowledge of which mid-levels are productive
corners = remaining[
    (remaining.ola_uL.isin([0, 1000])) &
    (remaining.y_br.isin([0, 4])) &
    (remaining.oa_uL.isin([0, 1200])) &
    (remaining.toluene_mL.isin([3, 12]))
].reset_index(drop=True)

# Center points: geometric center of each variable
# OLA center: 500 (midpoint of 0-1000)
# Br center: 2 (midpoint of 0-4)
# OA center: 600 (midpoint of 0-1200)
# Toluene center: 7.5 -> use 6 or 9 (closest grid levels)
center_points = remaining[
    (remaining.ola_uL == 500) &
    (remaining.y_br == 2) &
    (remaining.oa_uL == 600) &
    (remaining.toluene_mL.isin([6, 9]))
].reset_index(drop=True)

# Axial points: vary one factor at a time from center
axial_points = remaining[
    (
        ((remaining.ola_uL.isin([0, 1000])) & (remaining.y_br == 2) & (remaining.oa_uL == 600) & (remaining.toluene_mL == 6)) |
        ((remaining.ola_uL == 500) & (remaining.y_br.isin([0, 4])) & (remaining.oa_uL == 600) & (remaining.toluene_mL == 6)) |
        ((remaining.ola_uL == 500) & (remaining.y_br == 2) & (remaining.oa_uL.isin([0, 1200])) & (remaining.toluene_mL == 6)) |
        ((remaining.ola_uL == 500) & (remaining.y_br == 2) & (remaining.oa_uL == 600) & (remaining.toluene_mL.isin([3, 12])))
    )
].reset_index(drop=True)

# Combine: corners first, then center, then axial (standard DoE approach)
factorial_all = pd.concat([corners, center_points, axial_points], ignore_index=True)
# Remove duplicates
factorial_all = factorial_all.drop_duplicates(subset=features).reset_index(drop=True)
factorial_all = factorial_all.head(max_budget)

fact_cumulative = np.full(max_budget + 1, initial_tri, dtype=float)
for i in range(len(factorial_all)):
    fact_cumulative[i + 1] = fact_cumulative[i] + factorial_all.oracle.iloc[i]
# If factorial has fewer points than budget, fill remaining with last value
if len(factorial_all) < max_budget:
    fact_cumulative[len(factorial_all) + 1:] = fact_cumulative[len(factorial_all)]

n_corners_tri = int(corners.oracle.sum())
n_center_tri = int(center_points.oracle.sum())
n_axial_tri = int(axial_points.oracle.sum()) if len(axial_points) > 0 else 0

print(f"Factorial design (proper 2-level + center + axial):")
print(f"  Corners: {len(corners)} points, {n_corners_tri} trigonal")
print(f"  Center:  {len(center_points)} points, {n_center_tri} trigonal")
print(f"  Axial:   {len(axial_points)} points, {n_axial_tri} trigonal")
print(f"  Total in {max_budget} experiments: {fact_cumulative[-1]:.0f} trigonal")

# --- Factorial design (best-case / optimistic) ---
# Uses mid-level points that happen to land in the trigonal zone.
# This is an optimistic upper bound: in practice, choosing these mid-levels
# requires prior knowledge of where the trigonal region lies.
factorial_mid = remaining[
    (remaining.ola_uL.isin([200, 300, 500])) &
    (remaining.y_br.isin([1, 2, 3])) &
    (remaining.oa_uL.isin([600, 900])) &
    (remaining.toluene_mL.isin([6, 9]))
].reset_index(drop=True)

factorial_best = pd.concat([corners, center_points, axial_points, factorial_mid], ignore_index=True)
factorial_best = factorial_best.drop_duplicates(subset=features).reset_index(drop=True)
factorial_best = factorial_best.head(max_budget)

fact_best_cumulative = np.full(max_budget + 1, initial_tri, dtype=float)
for i in range(len(factorial_best)):
    fact_best_cumulative[i + 1] = fact_best_cumulative[i] + factorial_best.oracle.iloc[i]
if len(factorial_best) < max_budget:
    fact_best_cumulative[len(factorial_best) + 1:] = fact_best_cumulative[len(factorial_best)]

n_fact_best_tri = int(fact_best_cumulative[-1] - initial_tri)
print(f"\nFactorial design (best-case, corners + center + axial + mid-level):")
print(f"  Mid-level points: {len(factorial_mid)} points, {int(factorial_mid.oracle.sum())} trigonal")
print(f"  Total in {max_budget} experiments: {fact_best_cumulative[-1]:.0f} trigonal ({n_fact_best_tri} additional)")
print(f"  (Note: best-case uses mid-levels that require prior knowledge of trigonal region)")

# --- Factorial design (cheating / optimistic upper bound) ---
# Uses the all-data oracle (trained on all 88 points incl. AL outcomes) AND
# optimally placed mid-levels. This is the most generous possible scenario:
# the factorial has access to knowledge that AL discovered. If even this
# doesn't beat AL, the case is closed.
factorial_cheat = pd.concat([corners, center_points, axial_points, factorial_mid], ignore_index=True)
factorial_cheat = factorial_cheat.drop_duplicates(subset=features).reset_index(drop=True)
factorial_cheat = factorial_cheat.head(max_budget)

fact_cheat_cumulative = np.full(max_budget + 1, initial_tri, dtype=float)
for i in range(len(factorial_cheat)):
    fact_cheat_cumulative[i + 1] = fact_cheat_cumulative[i] + factorial_cheat.oracle_alldata.iloc[i]
if len(factorial_cheat) < max_budget:
    fact_cheat_cumulative[len(factorial_cheat) + 1:] = fact_cheat_cumulative[len(factorial_cheat)]

n_fact_cheat_tri = int(fact_cheat_cumulative[-1] - initial_tri)
print(f"\nFactorial design (cheating: all-data oracle + optimized mid-levels):")
print(f"  Total in {max_budget} experiments: {fact_cheat_cumulative[-1]:.0f} trigonal ({n_fact_cheat_tri} additional)")
print(f"  (Note: this is an optimistic upper bound -- oracle trained on ALL data incl. AL outcomes)")

# --- Strategy 4: Random + Model (ablation) ---
# Same model, same retraining schedule, but RANDOM acquisition
# This isolates the contribution of the AL acquisition function
print(f"\nRunning Random+Model ablation (1000 runs)...")

N_RUNS_RM = 1000
rm_curves = np.zeros((N_RUNS_RM, max_budget + 1))
batch_size = 24  # Same as AL (2 batches of 24)

for run in range(N_RUNS_RM):
    # Start with initial screening data
    X_train = initial_data[features].values.copy()
    y_train = initial_data['is_trigonal'].values.copy()
    trained_keys = set(
        initial_data.apply(lambda r: (r.ola_uL, r.y_br, r.oa_uL, r.toluene_mL), axis=1)
    )

    # Available pool: all grid points not in initial screening
    pool = remaining.copy()
    pool['oracle_val'] = pool.oracle.values

    cumulative = initial_tri
    rm_curves[run, 0] = cumulative
    step = 0

    for batch in range(max_budget // batch_size):
        # Train model on current training set
        if len(np.unique(y_train)) < 2:
            # Not enough class diversity, random sample
            available = pool[~pool.apply(lambda r: (r.ola_uL, r.y_br, r.oa_uL, r.toluene_mL) in trained_keys, axis=1)]
            if len(available) == 0:
                break
            sampled = available.sample(n=min(batch_size, len(available)), random_state=run + batch)
        else:
            rf_rm = RandomForestClassifier(n_estimators=100, random_state=run, max_depth=5)
            rf_rm.fit(X_train, y_train)

            # Random acquisition: sample randomly from available pool
            available = pool[~pool.apply(lambda r: (r.ola_uL, r.y_br, r.oa_uL, r.toluene_mL) in trained_keys, axis=1)]
            if len(available) == 0:
                break
            sampled = available.sample(n=min(batch_size, len(available)), random_state=run + batch)

        # "Run experiments" - get oracle outcomes
        for _, s in sampled.iterrows():
            step += 1
            cumulative += s.oracle_val
            if step <= max_budget:
                rm_curves[run, step] = cumulative
            # Add to training set
            X_train = np.vstack([X_train, s[features].values])
            y_train = np.append(y_train, s.oracle_val)
            trained_keys.add((s.ola_uL, s.y_br, s.oa_uL, s.toluene_mL))

    # Fill remaining steps
    if step < max_budget:
        rm_curves[run, step + 1:] = cumulative

rm_mean = rm_curves.mean(axis=0)
rm_std = rm_curves.std(axis=0)
rm_p5 = np.percentile(rm_curves, 5, axis=0)
rm_p25 = np.percentile(rm_curves, 25, axis=0)
rm_p75 = np.percentile(rm_curves, 75, axis=0)
rm_p95 = np.percentile(rm_curves, 95, axis=0)

print(f"Random+Model (mean): {rm_mean[-1]:.1f} +/- {rm_std[-1]:.1f} trigonal in {max_budget} experiments")

# ============================================================
# 5. KEY METRICS
# ============================================================
print("\n" + "=" * 70)
print(" KEY COMPARISON METRICS")
print("=" * 70)

al_hit_rate = al_tri / n_al * 100
rand_hit_rate = (rand_mean[-1] - initial_tri) / max_budget * 100
grid_hit_rate_best = (grid_best[-1] - initial_tri) / max_budget * 100
fact_hit_rate = (fact_cumulative[-1] - initial_tri) / max_budget * 100
fact_best_hit_rate = (fact_best_cumulative[-1] - initial_tri) / max_budget * 100
fact_cheat_hit_rate = (fact_cheat_cumulative[-1] - initial_tri) / max_budget * 100
rm_hit_rate = (rm_mean[-1] - initial_tri) / max_budget * 100
baseline_rate = n_remaining_tri / len(remaining) * 100

print(f"\nHit rate (% trigonal found per experiment):")
print(f"  Baseline (random over full grid):     {baseline_rate:.1f}%")
print(f"  Active Learning (actual):             {al_hit_rate:.1f}%")
print(f"  Random Search (simulated):            {rand_hit_rate:.1f}%")
print(f"  Random + Model (ablation):            {rm_hit_rate:.1f}%")
print(f"  Grid Search (best ordering):          {grid_hit_rate_best:.1f}%")
print(f"  Factorial Design (proper):            {fact_hit_rate:.1f}%")
print(f"  Factorial Design (best-case):         {fact_best_hit_rate:.1f}%")
print(f"  Factorial Design (cheating):          {fact_cheat_hit_rate:.1f}%")

print(f"\nEfficiency improvement over random search:")
print(f"  AL vs Random:         {al_hit_rate / max(rand_hit_rate, 0.1):.1f}x")
print(f"  AL vs Random+Model:   {al_hit_rate / max(rm_hit_rate, 0.1):.1f}x")
print(f"  AL vs Grid:           {al_hit_rate / max(grid_hit_rate_best, 0.1):.1f}x")
print(f"  AL vs Factorial (proper):    {al_hit_rate / max(fact_hit_rate, 0.1):.1f}x")
print(f"  AL vs Factorial (best-case): {al_hit_rate / max(fact_best_hit_rate, 0.1):.1f}x")
print(f"  AL vs Factorial (cheating):  {al_hit_rate / max(fact_cheat_hit_rate, 0.1):.1f}x")

print(f"\nExperiments needed to find N additional trigonal samples:")
print(f"  {'Target':>8} | {'AL':>6} | {'Random':>12} | {'Rand+Model':>12} | {'Grid(best)':>12} | {'Factorial':>10}")
print("  " + "-" * 75)
for target in [1, 5, 10, 15, 20, 25, 30, 35]:
    target_abs = target + initial_tri
    al_idx = np.argmax(al_cumulative >= target_abs) if al_cumulative.max() >= target_abs else None
    rand_idx = np.argmax(rand_mean >= target_abs) if rand_mean.max() >= target_abs else None
    rm_idx = np.argmax(rm_mean >= target_abs) if rm_mean.max() >= target_abs else None
    grid_idx = np.argmax(grid_best >= target_abs) if grid_best.max() >= target_abs else None
    fact_idx = np.argmax(fact_cumulative >= target_abs) if fact_cumulative.max() >= target_abs else None

    al_str = f"{al_idx}" if al_idx and al_idx > 0 else "N/A"
    rand_str = f"{rand_idx}+/-{rand_std[rand_idx]:.0f}" if rand_idx and rand_idx > 0 else "N/A"
    rm_str = f"{rm_idx}+/-{rm_std[rm_idx]:.0f}" if rm_idx and rm_idx > 0 else "N/A"
    grid_str = f"{grid_idx}" if grid_idx and grid_idx > 0 else "N/A"
    fact_str = f"{fact_idx}" if fact_idx and fact_idx > 0 else "N/A"
    print(f"  {target:>8} | {al_str:>6} | {rand_str:>12} | {rm_str:>12} | {grid_str:>12} | {fact_str:>10}")

# ============================================================
# 6. PLOTTING
# ============================================================
print("\n" + "=" * 70)
print(" Generating plots...")
print("=" * 70)

# --- Plot 1: Main surrogate-based comparison ---
fig, ax = plt.subplots(figsize=(10, 6.5))
x = np.arange(0, max_budget + 1)

ax.plot(x, al_cumulative, 'r-', linewidth=2.5, marker='o', markersize=4,
        label=f'Active Learning (actual, n={n_al})', zorder=5)

# Random search
ax.plot(x, rand_mean, 'b--', linewidth=2, label=f'Random Search (mean of {N_RUNS} runs)')
ax.fill_between(x, rand_p5, rand_p95, alpha=0.12, color='blue', label='Random (5th-95th pct)')
ax.fill_between(x, rand_p25, rand_p75, alpha=0.22, color='blue', label='Random (25th-75th pct)')

# Random + Model
ax.plot(x, rm_mean, 'c:', linewidth=2, label=f'Random + Model (mean of {N_RUNS_RM} runs)')
ax.fill_between(x, rm_p25, rm_p75, alpha=0.15, color='cyan', label='Rand+Model (25th-75th pct)')

# Grid search
ax.plot(x, grid_best, 'g-', linewidth=1.5, alpha=0.7, label='Grid Search (best ordering)')
ax.plot(x, grid_worst, 'g-', linewidth=1.5, alpha=0.3, label='Grid Search (worst ordering)')
ax.fill_between(x, grid_worst, grid_best, alpha=0.1, color='green')

# Factorial
ax.plot(x, fact_cumulative, 'm-.', linewidth=2, label='Factorial Design (proper 2-level)')
ax.plot(x, fact_best_cumulative, 'm:', linewidth=2, alpha=0.7,
        label='Factorial Design (best-case, optimized mid-levels)')
ax.plot(x, fact_cheat_cumulative, 'm--', linewidth=2, alpha=0.5,
        label='Factorial Design (cheating: all-data oracle + mid-levels)')

ax.axhline(y=initial_tri, color='gray', linestyle=':', alpha=0.7,
           label=f'Initial screening baseline ({initial_tri} trigonal)')
ax.set_xlabel('Additional Experiments (after initial screening)', fontsize=13)
ax.set_ylabel('Cumulative Trigonal Samples Found', fontsize=13)
ax.set_title('Retrospective Benchmark: Active Learning vs Conventional Strategies\n'
             f'(Surrogate trained on initial screening only, {len(full_grid)}-point grid)', fontsize=11)
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim(0, max_budget)
ax.set_ylim(0, max(al_cumulative.max() + 2, rand_p95.max() + 2))
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('benchmark_surrogate_based.png', dpi=300, bbox_inches='tight')
print("  Saved: benchmark_surrogate_based.png")

# --- Plot 2: Hit rate bar chart ---
fig, ax = plt.subplots(figsize=(9, 5))
strategies = ['Random\nSearch', 'Random\n+Model', 'Grid Search\n(best)', 'Factorial\n(proper)', 'Factorial\n(best-case)', 'Factorial\n(cheating)', 'AL\nRound 1', 'AL\nRound 2']
hit_rates = [rand_hit_rate, rm_hit_rate, grid_hit_rate_best, fact_hit_rate, fact_best_hit_rate, fact_cheat_hit_rate, 45.8, 100.0]
colors = ['#4C72B0', '#76B7B2', '#55A868', '#8172B3', '#C8B2D8', '#E5D4E8', '#DD8452', '#C44E52']
bars = ax.bar(strategies, hit_rates, color=colors, edgecolor='black', linewidth=0.8)
for bar, rate in zip(bars, hit_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.axhline(y=baseline_rate, color='gray', linestyle='--', alpha=0.5,
           label=f'Baseline (grid avg: {baseline_rate:.1f}%)')
ax.set_ylabel('Trigonal Phase Hit Rate (%)', fontsize=13)
ax.set_title('Hit Rate Comparison: AL vs Conventional Strategies', fontsize=13)
ax.set_ylim(0, 115)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig('benchmark_hit_rates.png', dpi=300, bbox_inches='tight')
print("  Saved: benchmark_hit_rates.png")

# --- Plot 3: Parameter space coverage ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for phase, color, label, marker in zip(
    ['initial', 'AL_round1', 'AL_round2'],
    ['#4C72B0', '#DD8452', '#C44E52'],
    ['Initial screening', 'AL Round 1', 'AL Round 2'],
    ['o', 's', '^']
):
    subset = data[data.phase == phase]
    trigonal = subset[subset.is_trigonal == 1]
    non_trigonal = subset[subset.is_trigonal == 0]
    axes[0].scatter(non_trigonal.ola_uL, non_trigonal.oa_uL, c=color, marker='x',
                    s=50, alpha=0.5, label=f'{label} (non-trigonal)')
    axes[0].scatter(trigonal.ola_uL, trigonal.oa_uL, c=color, marker=marker,
                    s=80, alpha=0.8, edgecolors='black', linewidth=0.5,
                    label=f'{label} (trigonal)')
axes[0].set_xlabel('OLA Volume (uL)', fontsize=12)
axes[0].set_ylabel('OA Volume (uL)', fontsize=12)
axes[0].set_title('Parameter Space: OLA vs OA', fontsize=12)
axes[0].legend(fontsize=8, loc='upper left')
axes[0].grid(True, alpha=0.3)

for phase, color, label, marker in zip(
    ['initial', 'AL_round1', 'AL_round2'],
    ['#4C72B0', '#DD8452', '#C44E52'],
    ['Initial screening', 'AL Round 1', 'AL Round 2'],
    ['o', 's', '^']
):
    subset = data[data.phase == phase]
    trigonal = subset[subset.is_trigonal == 1]
    non_trigonal = subset[subset.is_trigonal == 0]
    axes[1].scatter(non_trigonal.ola_uL, non_trigonal.toluene_mL, c=color, marker='x',
                    s=50, alpha=0.5, label=f'{label} (non-trigonal)')
    axes[1].scatter(trigonal.ola_uL, trigonal.toluene_mL, c=color, marker=marker,
                    s=80, alpha=0.8, edgecolors='black', linewidth=0.5,
                    label=f'{label} (trigonal)')
axes[1].set_xlabel('OLA Volume (uL)', fontsize=12)
axes[1].set_ylabel('Toluene Volume (mL)', fontsize=12)
axes[1].set_title('Parameter Space: OLA vs Toluene', fontsize=12)
axes[1].legend(fontsize=8, loc='upper left')
axes[1].grid(True, alpha=0.3)

fig.suptitle('Experimental Coverage of Parameter Space (filled = trigonal, x = non-trigonal)',
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig('benchmark_parameter_space.png', dpi=300, bbox_inches='tight')
print("  Saved: benchmark_parameter_space.png")

# --- Plot 4: Surrogate model prediction map ---
fig, ax = plt.subplots(figsize=(8, 6))
pivot = full_grid.groupby(['ola_uL', 'oa_uL'])['pred_proba'].mean().unstack()
im = ax.imshow(pivot.values, aspect='auto', origin='lower', cmap='RdYlBu_r',
               extent=[pivot.columns.min(), pivot.columns.max(),
                       pivot.index.min(), pivot.index.max()])
ax.set_xlabel('OA Volume (uL)', fontsize=12)
ax.set_ylabel('OLA Volume (uL)', fontsize=12)
ax.set_title('Surrogate Model (trained on initial screening only):\nPredicted P(Trigonal), averaged over Br and toluene', fontsize=11)
plt.colorbar(im, ax=ax, label='P(trigonal)')
for phase, color, marker in zip(
    ['initial', 'AL_round1', 'AL_round2'],
    ['white', 'yellow', 'black'],
    ['o', 's', '^']
):
    subset = data[(data.phase == phase) & (data.in_grid == 'yes')]
    ax.scatter(subset.oa_uL, subset.ola_uL, c=color, marker=marker,
               s=50, edgecolors='gray', linewidth=0.5, label=phase)
ax.legend(fontsize=9, loc='upper left')
fig.tight_layout()
fig.savefig('benchmark_prediction_map.png', dpi=300, bbox_inches='tight')
print("  Saved: benchmark_prediction_map.png")

# --- Plot 5: AL ablation comparison (AL vs Random+Model) ---
fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(0, max_budget + 1)

ax.plot(x, al_cumulative, 'r-', linewidth=2.5, marker='o', markersize=4,
        label='Active Learning (acquisition function)', zorder=5)
ax.plot(x, rm_mean, 'c--', linewidth=2, label='Random + Model (random acquisition)')
ax.fill_between(x, rm_p5, rm_p95, alpha=0.15, color='cyan', label='Rand+Model (5th-95th pct)')
ax.fill_between(x, rm_p25, rm_p75, alpha=0.25, color='cyan', label='Rand+Model (25th-75th pct)')
ax.plot(x, rand_mean, 'b:', linewidth=1.5, alpha=0.7, label='Pure Random (no model)')

ax.axhline(y=initial_tri, color='gray', linestyle=':', alpha=0.7,
           label=f'Initial baseline ({initial_tri} trigonal)')
ax.set_xlabel('Additional Experiments', fontsize=13)
ax.set_ylabel('Cumulative Trigonal Samples Found', fontsize=13)
ax.set_title('Ablation: AL Acquisition Function vs Random Acquisition\n(Same model, same retraining schedule)', fontsize=11)
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(0, max_budget)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('benchmark_ablation.png', dpi=300, bbox_inches='tight')
print("  Saved: benchmark_ablation.png")

# ============================================================
# 7. SUMMARY TABLE FOR PAPER
# ============================================================
print("\n" + "=" * 70)
print(" SUMMARY TABLE FOR PAPER")
print("=" * 70)

print(f"""
Table: Retrospective benchmark of Active Learning vs conventional strategies.

  Parameter grid: {len(full_grid)} combinations (14 OLA x 5 Br x 5 OA x 4 toluene)
  Surrogate model: Random Forest, trained on initial screening only ({len(initial_data)} points)
    5-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}
    Out-of-sample accuracy on AL outcomes: {al_accuracy:.3f}
  Predicted trigonal in full grid: {n_oracle_tri} / {len(full_grid)} ({100*n_oracle_tri/len(full_grid):.1f}%)
  Initial screening: {n_initial} in-grid experiments ({initial_tri} trigonal)
  AL budget: {n_al} additional experiments (2 rounds of {batch_size})

  {'Strategy':<35} | {'Trigonal found':>15} | {'Hit rate (%)':>12} | {'vs Random':>10}
  {'-'*80}
  {'Active Learning (actual)':<35} | {al_tri:>15} | {al_hit_rate:>12.1f} | {al_hit_rate/max(rand_hit_rate,0.1):>9.1f}x
  {'Random Search (mean)':<35} | {rand_mean[-1]-initial_tri:>15.1f} | {rand_hit_rate:>12.1f} | {1.0:>9.1f}x
  {'Random + Model (ablation)':<35} | {rm_mean[-1]-initial_tri:>15.1f} | {rm_hit_rate:>12.1f} | {rm_hit_rate/max(rand_hit_rate,0.1):>9.1f}x
  {'Grid Search (best ordering)':<35} | {grid_best[-1]-initial_tri:>15.0f} | {grid_hit_rate_best:>12.1f} | {grid_hit_rate_best/max(rand_hit_rate,0.1):>9.1f}x
  {'Factorial Design (proper 2-level)':<35} | {fact_cumulative[-1]-initial_tri:>15.0f} | {fact_hit_rate:>12.1f} | {fact_hit_rate/max(rand_hit_rate,0.1):>9.1f}x
  {'Factorial Design (best-case mid)':<35} | {fact_best_cumulative[-1]-initial_tri:>15.0f} | {fact_best_hit_rate:>12.1f} | {fact_best_hit_rate/max(rand_hit_rate,0.1):>9.1f}x
  {'Factorial Design (cheating, all-data)':<35} | {fact_cheat_cumulative[-1]-initial_tri:>15.0f} | {fact_cheat_hit_rate:>12.1f} | {fact_cheat_hit_rate/max(rand_hit_rate,0.1):>9.1f}x
  {'Baseline (grid average)':<35} | {'--':>15} | {baseline_rate:>12.1f} | {'--':>10}

  Grid search experiments to 1st trigonal: {min(grid_first_tri)}-{max(grid_first_tri)} (vs AL: 1)
""")

print("Done. All plots and metrics generated.")
