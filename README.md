# Optimal Regret for Online Conformal Inference: Experiment Code

This repository contains the simulation code for the numerical experiments in the paper:

**Optimal training-conditional regret for online conformal prediction**  
Jiadong Liang, Zhimei Ren, Yuxin Chen

## Overview

We study online conformal prediction under distribution drift and compare drift-aware conformal methods against standard online calibration baselines.

This repository includes two experiment modules:

- `split_conformal/`: experiments for **online conformal prediction with pretrained scores**
- `full_conformal/`: experiments for **online conformal prediction with adaptively trained parametric scores**

The code is intended to reproduce the main numerical trends reported in Section 5 of the paper.

## Repository structure

```text
.
├── split_conformal/
│   ├── main.py                          # Main entry (4 settings × 6 algorithms)
│   ├── algorithm.py                     # ACI + DriftOCP algorithm classes
│   ├── data_generating.py               # Data generators for Settings 1–4
│   ├── drift_detection_conformal.py     # Drift detection conformal procedure
│   └── plot.py                          # Plotting utilities
├── full_conformal/
│   ├── online_vs_pretrain_both_drifts.py    # Well-specified model experiment
│   ├── misspecified_both_drifts.py          # Misspecified model experiment
│   └── regenerate_combined_2x4_plots.py    # Regenerate combined 2×4 figures
└── README.md
```

## Experimental settings

### 1. Pretrained-score experiments (`split_conformal/`)

This module studies the setting where the predictive model is trained offline and then kept fixed during the online phase.

The implementation includes:

- a drift-aware conformal calibration procedure based on round-wise updates and drift detection
- ACI baselines with both fixed and decaying step sizes
- multiple distribution-shift scenarios, including abrupt and smooth drift

Typical ingredients include:

- offline model fitting on an initial training sample
- residual-based nonconformity scores
- round-wise quantile calibration
- drift detection within each round
- Monte Carlo approximation of instantaneous coverage / cumulative regret

### 2. Online parametric-score experiments (`full_conformal/`)

This module studies the setting where the predictive model is updated online.

In the current implementation, the prediction set in round `(n, r)` is interpreted in the residual-threshold form

```math
\mathcal{C}_{n,r}(x) = \{y : s(x,y) \le q_{n,r}\},
```

where the score is induced by an online-updated parametric model. The code uses:

- online SGD / online parametric updates
- round-wise residual collection
- drift detection based on current-round coverage deviations
- stage restarts after detected drift
- comparisons with pretrained-model and model-free baselines

This implementation follows the **drift-detection protocol** of the round-based algorithm: drift is monitored using data accumulated from the current round onward, and a detected drift triggers a stage restart.

## Methodological note

The repository is designed to support the experiments, not to serve as a general-purpose conformal inference library.

In particular:

- the `split_conformal/` module corresponds to the **pretrained-score** experimental setting in the paper
- the `full_conformal/` module corresponds to an **online parametric residual-score** implementation of the adaptively trained setting

So, for the second module, the code should be read as implementing a **round-based online parametric conformal procedure with drift detection**, rather than a generic full conformal package.

## Installation

We recommend Python 3.10+ and using a conda environment.

```bash
conda create -n spci python=3.11
conda activate spci
pip install numpy scipy scikit-learn matplotlib pandas seaborn tqdm
```

> **Note on version compatibility:** matplotlib ≥ 3.9 is required if using NumPy ≥ 2.0. Earlier versions of matplotlib (e.g. 3.7) are compiled against NumPy 1.x and will fail to import under NumPy 2.

## How to run

### 1. Pretrained-score experiments (`split_conformal/`)

```bash
cd split_conformal
python main.py
```

This runs all 4 settings × 6 algorithms in parallel and produces:

- `experiment_results_all_settings.pkl` — full raw experiment data
- `experiment_summary.json` — summary statistics
- `comparison_Setting{1,2,3,4}_*.png` — per-setting comparison plots
- `split_conformal_combined_2x4.pdf` — combined 2×4 figure

### 2. Online parametric-score experiments (`full_conformal/`)

There is no single `main.py`; run the two experiment scripts separately:

```bash
cd full_conformal

# Step 1: Well-specified model (Y = Xβ* + ε)
python online_vs_pretrain_both_drifts.py

# Step 2: Misspecified model (Y = Xβ* + (1/C)||X||² + ε)
python misspecified_both_drifts.py
```

Each script runs 20 repetitions comparing three methods (Adaptive / Pre-trained / Model-free) under both mean drift and variance drift. Outputs:

- `results/well_specified_combined.pdf` + `results/well_specified_data.pkl`
- `results/misspec_combined.pdf` + `results/misspec_data.pkl`

After both experiments finish, you can regenerate the combined 2×4 figure:

```bash
python regenerate_combined_2x4_plots.py
```

This reads the `.pkl` files from both modules and produces `results/full_conformal_combined_2x4.pdf`.

## Outputs

The scripts generate the quantities reported in Section 5 of the paper:

- cumulative regret trajectories
- quantile evolution
- prediction interval widths
- rolling / local coverage curves

All plots are saved to disk as PDF or PNG files in the respective output directories.
