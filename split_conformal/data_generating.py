#!/usr/bin/env python3
"""
Data Generation Module

Contains data generation functions for 4 experimental settings,
as well as functions for computing the theoretical true quantiles.

Mathematical definitions for the 4 Settings:
  Setting 1: Y_i = 2X_{i,0} + X_{i,1} + σ_i·ε_i,  σ piecewise constant (0.5/2.0/3.5)
  Setting 2: Y_i = 2X_{i,0} + X_{i,1} + 0.001·i + 0.5·ε_i  (linear mean drift)
  Setting 3: Y_i = 2X_{i,0} + X_{i,1} + σ_i·ε_i,  σ_i²=1+40i/5000  (smooth variance growth)
  Setting 4: Y_i = 2X_{i,0} + X_{i,1} + 0.5·ε_i  (no drift, control group)
  
Where ε_i ~ N(0,1) is standard normal noise, X_{i,j} ~ N(0,1) are standard normal features.

Each generation function has two calling modes:
  Mode A (n_samples=None): Generate time-series test data, each row of X corresponds to one time step
  Mode B (n_samples≠None): Batch-generate n_samples independent samples at a given time t (for Monte Carlo regret estimation)
"""

import numpy as np                                           # Core numerical computing library
from scipy import stats                                      # Statistical distribution functions (for computing theoretical quantiles)


# ==================== Setting 1: Piecewise Variance Shift ====================
def generate_data_setting1(X, t=None, n_samples=None):
    """
    Setting 1: Piecewise Variance Shift (Jump Variance)
    
    Data model: Y_i = 2·X_{i,0} + X_{i,1} + σ_i · ε_i
    Variance undergoes abrupt changes (jumps) at two time points:
      t ∈ [0, 4000):    σ = 0.5  → variance=0.25  (low noise)
      t ∈ [4000, 7000):  σ = 2.0  → variance=4.0   (medium noise, sudden increase)
      t ∈ [7000, 10000): σ = 3.5  → variance=12.25  (high noise, another increase)
    
    Args:
        X:         Feature matrix, shape=(n, n_features)
        t:         Time step (used in Mode B)
        n_samples: Number of samples (None=Mode A, otherwise=Mode B)
    
    Returns:
        Y: Generated response values
    """
    if n_samples is None:                                    # Mode A: generate sequentially by time step (for test sequence)
        T_len = len(X)                                       # Get sequence length
        Y = np.zeros(T_len)                                  # Initialize output array
        for i in range(T_len):                               # Iterate over each time step
            if i < 4000:                                     # First segment: t ∈ [0, 4000)
                Y[i] = 2 * X[i, 0] + X[i, 1] + 0.5 * np.random.randn()   # Noise with σ=0.5
            elif i < 7000:                                   # Second segment: t ∈ [4000, 7000)
                Y[i] = 2 * X[i, 0] + X[i, 1] + 2.0 * np.random.randn()   # Noise with σ=2.0
            else:                                            # Third segment: t ∈ [7000, 10000)
                Y[i] = 2 * X[i, 0] + X[i, 1] + 3.5 * np.random.randn()   # Noise with σ=3.5
    else:                                                    # Mode B: batch-generate n_samples at time t
        if t < 4000:                                         # Determine noise level based on time t
            Y = 2 * X[:, 0] + X[:, 1] + 0.5 * np.random.randn(n_samples)  # σ=0.5
        elif t < 7000:                                       # t in second segment
            Y = 2 * X[:, 0] + X[:, 1] + 2.0 * np.random.randn(n_samples)  # σ=2.0
        else:                                                # t in third segment
            Y = 2 * X[:, 0] + X[:, 1] + 3.5 * np.random.randn(n_samples)  # σ=3.5
    return Y                                                 # Return generated Y values


# ==================== Setting 2: Linear Mean Drift ====================
def generate_data_setting2(X, t=None, n_samples=None, T=10000):
    """
    Setting 2: Linear Bias Drift
    
    Data model: Y_i = 2·X_{i,0} + X_{i,1} + μ_i + 0.5·ε_i
    Mean drift: μ_i = α · i,  α = 0.001
    i.e., the mean grows linearly from μ_0=0 to μ_10000=10
    Noise level is fixed at σ=0.5, but due to mean drift, model residuals grow over time
    
    Args:
        X:         Feature matrix
        t:         Time step (Mode B)
        n_samples: Number of samples
        T:         Total time steps (unused, kept for compatibility)
    
    Returns:
        Y: Generated response values
    """
    alpha = 0.001                                            # Drift coefficient: mean increases by 0.001 per step
    if n_samples is None:                                    # Mode A: generate time series
        T_len = len(X)                                       # Sequence length
        Y = np.zeros(T_len)                                  # Initialize output
        for i in range(T_len):                               # Iterate over each time step
            mu_t = alpha * i                                 # Mean drift at current time: 0.001 * i
            Y[i] = 2 * X[i, 0] + X[i, 1] + mu_t + 0.5 * np.random.randn()  # Signal + drift + noise
    else:                                                    # Mode B: batch sample at time t
        mu_t = alpha * t                                     # Drift amount at time t
        Y = 2 * X[:, 0] + X[:, 1] + mu_t + 0.5 * np.random.randn(n_samples)  # Batch generation
    return Y                                                 # Return Y


# ==================== Setting 3: Smooth Variance Growth ====================
def generate_data_setting3(X, t=None, n_samples=None, T=10000):
    """
    Setting 3: Smooth Variance Growth
    
    Data model: Y_i = 2·X_{i,0} + X_{i,1} + σ_i · ε_i
    Variance formula: σ_i² = 1.0 + 40.0 · i / 5000
    Variance grows smoothly from σ²=1 (σ=1) to σ²=81 (σ=9)
    Unlike Setting 1: there are no abrupt changes here, variance increases continuously
    
    Args:
        X:         Feature matrix
        t:         Time step (Mode B)
        n_samples: Number of samples
        T:         Total time steps (unused, kept for compatibility)
    
    Returns:
        Y: Generated response values
    """
    if n_samples is None:                                    # Mode A: generate time series
        T_len = len(X)                                       # Sequence length
        Y = np.zeros(T_len)                                  # Initialize output
        for i in range(T_len):                               # Iterate over each time step
            sigma_t_sq = 1.0 + 40.0 * i / 5000              # Variance at current time: linear growth
            sigma_t = np.sqrt(sigma_t_sq)                    # Square root of variance to get std dev
            Y[i] = 2 * X[i, 0] + X[i, 1] + sigma_t * np.random.randn()   # Signal + time-varying noise
    else:                                                    # Mode B: batch sample at time t
        sigma_t_sq = 1.0 + 40.0 * t / 5000                  # Variance at time t
        sigma_t = np.sqrt(sigma_t_sq)                        # Standard deviation
        Y = 2 * X[:, 0] + X[:, 1] + sigma_t * np.random.randn(n_samples)  # Batch generation
    return Y                                                 # Return Y


# ==================== Setting 4: Stationary Distribution (No Drift) ====================
def generate_data_setting4(X, t=None, n_samples=None):
    """
    Setting 4: Stationary Distribution (Control Group)
    
    Data model: Y_i = 2·X_{i,0} + X_{i,1} + 0.5·ε_i
    Data distribution remains constant, σ=0.5 is fixed, no drift of any kind
    Purpose: serves as a control group to verify algorithms do not introduce extra error under no-drift scenarios
    
    Args:
        X:         Feature matrix
        t:         Time step (Mode B; not actually used here since distribution does not change over time)
        n_samples: Number of samples
    
    Returns:
        Y: Generated response values
    """
    if n_samples is None:                                    # Mode A: generate time series
        T_len = len(X)                                       # Sequence length
        Y = 2 * X[:, 0] + X[:, 1] + 0.5 * np.random.randn(T_len)  # Vectorized in one step (distribution is constant)
    else:                                                    # Mode B: batch sample at time t
        Y = 2 * X[:, 0] + X[:, 1] + 0.5 * np.random.randn(n_samples)  # Batch generation
    return Y                                                 # Return Y


# ==================== Theoretical True Quantile Computation ====================
def compute_true_quantiles(T, alpha, setting):
    """
    Compute the theoretical true quantile value at each time step.
    
    When plotting, the true quantile serves as a black dashed reference baseline.
    It represents: if we knew the true parameters of the data distribution,
    what would the optimal quantile threshold be.
    
    Mathematical background:
      conformal score: s_t = |Y_t - Ŷ_t|
      true quantile:   q*(t) satisfies P(s_t > q*(t)) = α
      i.e., q*(t) is the (1-α) quantile of the distribution of s_t
    
    For different Settings:
      Setting 1: s_t ≈ σ_t·|ε|, so q*(t) = σ_t · z_{1-α/2}  (folded normal quantile)
      Setting 2: s_t ≈ |μ_t + σε|, requires numerical solution
      Setting 3: Same as Setting 1, but σ_t varies continuously
      Setting 4: q* is constant = 0.5 · z_{1-α/2}
    
    Args:
        T:       Total number of time steps
        alpha:   Target miscoverage rate (e.g., 0.1 means 90% coverage)
        setting: Data setting number (1~4)
    
    Returns:
        true_quantiles: True quantile value at each step, shape=(T,)
    """
    true_quantiles = np.zeros(T)                             # Initialize output array, length=T
    
    if setting == 1:
        # Setting 1: Piecewise variance shift
        # score ≈ σ·|ε|, where |ε| ~ folded normal
        # P(|ε| ≤ q/σ) = 2Φ(q/σ) - 1 = 1-α  →  q = σ · Φ^{-1}(1-α/2)
        for t in range(T):                                   # Iterate over each time step
            if t < 4000:                                     # First segment: low noise
                sigma = 0.5                                  # σ=0.5
            elif t < 7000:                                   # Second segment: medium noise
                sigma = 2.0                                  # σ=2.0
            else:                                            # Third segment: high noise
                sigma = 3.5                                  # σ=3.5
            # (1-α) quantile of folded normal = σ × (1-α/2) quantile of standard normal
            true_quantiles[t] = sigma * stats.norm.ppf(1 - alpha / 2)
    
    elif setting == 2:
        # Setting 2: Linear bias drift μ_t = 0.001·t
        # score ≈ |μ_t + σε| (model is unaware of drift, predictions do not include μ_t)
        # Need to solve P(|μ_t + σε| ≤ q) = 1-α
        alpha_coef = 0.001                                   # Drift coefficient
        sigma = 0.5                                          # Noise standard deviation
        for t in range(T):                                   # Iterate over each time step
            mu_t = alpha_coef * t                            # Mean drift at current time
            if mu_t > 3 * sigma:                             # When drift is much larger than noise (μ >> σ)
                # |μ_t + σε| is almost always positive (since μ_t >> σ), approximated as μ_t + σε
                # Its (1-α) quantile ≈ μ_t + σ·z_{1-α/2}
                true_quantiles[t] = abs(mu_t) + sigma * stats.norm.ppf(1 - alpha / 2)
            else:                                            # When drift and noise are of similar magnitude, need exact solution
                from scipy.optimize import brentq            # Import Brent's method (1D root-finding algorithm)
                def cdf_diff(q):                             # Define equation: P(|μ+σε| ≤ q) - (1-α) = 0
                    # P(|μ+σε| ≤ q) = P(-q ≤ μ+σε ≤ q) = Φ((q-μ)/σ) - Φ((-q-μ)/σ)
                    return (stats.norm.cdf((q - mu_t) / sigma) -    # P(μ+σε ≤ q)
                           stats.norm.cdf((-q - mu_t) / sigma) -   # minus P(μ+σε ≤ -q)
                           (1 - alpha))                             # minus target probability
                try:
                    # Find root in the interval [0, μ_t+5σ]
                    true_quantiles[t] = brentq(cdf_diff, 0, mu_t + 5 * sigma)
                except:                                      # If root-finding fails, fall back to approximation
                    true_quantiles[t] = abs(mu_t) + sigma * stats.norm.ppf(1 - alpha / 2)
    
    elif setting == 3:
        # Setting 3: Smooth variance growth σ_t² = 1 + 40t/5000
        # Same computation as Setting 1, but σ_t varies continuously
        for t in range(T):                                   # Iterate over each time step
            sigma_t_sq = 1.0 + 40.0 * t / 5000              # Current variance
            sigma_t = np.sqrt(sigma_t_sq)                    # Current standard deviation
            # (1-α) quantile of folded normal
            true_quantiles[t] = sigma_t * stats.norm.ppf(1 - alpha / 2)
    
    elif setting == 4:
        # Setting 4: Stationary, σ=0.5 remains constant
        sigma = 0.5                                          # Fixed noise level
        true_quantile_value = sigma * stats.norm.ppf(1 - alpha / 2)  # Constant quantile
        true_quantiles[:] = true_quantile_value              # Fill all time steps with the same value
    
    return true_quantiles                                    # Return true quantile array
