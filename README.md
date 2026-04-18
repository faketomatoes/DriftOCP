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
│   ├── ...
│   └── ...
├── full_conformal/
│   ├── ...
│   └── ...
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

We recommend Python 3.10+.

Install the main dependencies with:

```bash
pip install numpy scipy scikit-learn matplotlib pandas
```

Depending on the scripts, you may also need:

```bash
pip install seaborn tqdm
```

## How to run

Please enter the relevant subdirectory and run the main experiment script there.

A typical workflow is:

```bash
cd split_conformal
python main.py
```

or

```bash
cd full_conformal
python main.py
```

## Outputs

The scripts are designed to generate the quantities reported in the paper, such as:

- cumulative regret trajectories
- quantile evolution
- prediction interval widths
- rolling / local coverage curves

Plots are typically either displayed directly or saved to disk, depending on the script configuration.
