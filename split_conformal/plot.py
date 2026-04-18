#!/usr/bin/env python3
"""
Plotting Module

Contains two plotting functions:
  1. plot_comparison_results: Plot comparison charts for a single Setting (1 row × 2 cols: Regret + Quantile)
  2. plot_combined_2x4:      Plot combined chart for all 4 Settings (2 rows × 4 cols, top row Regret, bottom row Quantile)

Chart meanings:
  - Left column / Top row (Cumulative Regret):
    R_T = Σ_{t=1}^{T} |miscoverage_t - α|
    Measures the cumulative deviation of the algorithm's online quantile adjustments; lower is better.
    DriftOCP should be significantly lower than ACI in Settings with drift.
    
  - Right column / Bottom row (Quantile Evolution):
    Shows the trajectory of q_t over time, compared against the true quantile q*(t).
    A good algorithm should make q_t closely track q*(t) (black dashed line).
"""

import numpy as np                                           # Numerical computing
import matplotlib.pyplot as plt                              # Core plotting library
import matplotlib                                            # Matplotlib configuration
matplotlib.rcParams['pdf.fonttype'] = 42                     # Embed TrueType fonts in PDF (avoid Type3)
matplotlib.rcParams['ps.fonttype'] = 42                      # Same treatment for PostScript


# ==================== Plot Configuration for 6 Algorithms ====================
# Each element: (data key prefix, display name, color)
ALGO_CONFIGS = [
    ('drift',           'DriftOCP',                'blue'),      # Our method
    ('aci_06',          r'ACI $\eta_t=t^{-0.6}$', 'red'),       # ACI decay 0.6
    ('aci_05',          r'ACI $\eta_t=t^{-0.5}$', 'green'),     # ACI decay 0.5
    ('aci_fixed_001',   r'ACI $\eta=0.01$',        'cyan'),      # ACI fixed 0.01
    ('aci_fixed_01',    r'ACI $\eta=0.1$',         'magenta'),   # ACI fixed 0.1
    ('aci_fixed_05',    r'ACI $\eta=0.5$',         'orange'),    # ACI fixed 0.5
]

# Mapping from Setting number to short name
SETTING_NAMES = {
    1: 'Jump Variance',                                      # Piecewise variance jump
    2: 'Linear Bias Drift',                                  # Linear mean drift
    3: 'Smooth Variance',                                    # Smooth variance growth
    4: 'Stationary',                                         # Stationary distribution
}


# ==================== Single Setting Comparison Plot ====================
def plot_comparison_results(results, setting, setting_name, save_dir='.'):
    """
    Plot comparison chart for a single Setting: 1 row × 2 columns
    
    Left plot: Cumulative Regret (mean ± std)
    Right plot: Quantile Evolution (single experiment trajectory + true quantile)
    
    Args:
        results:      Result dictionary returned by run_parallel_comparison, containing:
                        - '{algo}_cumulative_trajectories': cumulative regret trajectories across all repeated experiments
                        - '{algo}_quantile': quantile trajectory from a single experiment
                        - 'true_quantiles': theoretical true quantile
        setting:      Setting number (1~4), used for file naming
        setting_name: Setting name string (used for title)
        save_dir:     Directory path to save the plot
    """
    T = len(results['true_quantiles'])                       # Get total number of time steps
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))         # Create 1×2 subplots, 16 wide × 6 tall inches

    # ======== Left Plot: Cumulative Regret ========
    ax1 = axes[0]                                            # Get left subplot handle
    time_steps = np.arange(1, T + 1)                         # Time axis: [1, 2, ..., T]

    # Key name suffix for cumulative regret trajectories
    traj_suffix = '_cumulative_trajectories'                 # Trajectory data key suffix
    for algo_key, label, color in ALGO_CONFIGS:              # Iterate over 6 algorithms
        traj_key = f'{algo_key}{traj_suffix}'                # Build full key name (e.g., 'drift_cumulative_trajectories')
        if traj_key in results and len(results[traj_key]) > 0:  # Confirm data exists for this algorithm
            trajectories = np.array(results[traj_key])       # Convert to matrix, shape=(n_experiments, T)
            mean_traj = np.mean(trajectories, axis=0)        # Mean along experiment dimension, shape=(T,)
            std_traj = np.std(trajectories, axis=0)          # Std dev along experiment dimension
            ax1.plot(time_steps, mean_traj,                  # Plot mean line
                     color=color, linewidth=2, label=label)
            ax1.fill_between(time_steps,                     # Plot ±1 std dev shaded region
                             mean_traj - std_traj,           # Lower bound
                             mean_traj + std_traj,           # Upper bound
                             color=color, alpha=0.2)         # Semi-transparent fill

    ax1.set_xlabel('Time Step t', fontsize=12)               # X-axis label
    ax1.set_ylabel('Cumulative Regret', fontsize=12)         # Y-axis label
    ax1.set_title(f'{setting_name}\nCumulative Regret (mean ± std)',  # Title (with Setting name)
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')                # Legend in upper left
    ax1.grid(True, alpha=0.3)                                # Show light grid lines

    # ======== Right Plot: Quantile Evolution ========
    ax2 = axes[1]                                            # Get right subplot handle

    sample_interval = 50                                     # Sample one point every 50 steps (avoid overly dense lines)
    sample_indices = np.arange(0, T, sample_interval)        # Sampling indices

    # Plot true quantile (black dashed line as reference baseline)
    true_q = results['true_quantiles']                       # Get true quantile array
    ax2.plot(sample_indices, true_q[sample_indices],         # Plot true quantile
             'k--', linewidth=2.5, label='True Quantile', alpha=0.8)  # Black dashed line

    # Plot quantile estimate trajectories for each algorithm
    for algo_key, label, color in ALGO_CONFIGS:              # Iterate over 6 algorithms
        q_key = f'{algo_key}_quantile'                       # Quantile data key name
        if q_key in results:                                 # If result contains this algorithm's quantile
            quantiles = np.array(results[q_key])             # Get quantile array, shape=(T,)
            ax2.plot(sample_indices, quantiles[sample_indices],  # Plot sampled quantile
                     color=color, linewidth=1.5, label=label, alpha=0.7)

    ax2.set_xlabel('Time Step t', fontsize=12)               # X-axis label
    ax2.set_ylabel('Quantile Estimate', fontsize=12)         # Y-axis label
    ax2.set_title(f'{setting_name}\nQuantile Evolution',     # Title
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')                # Legend
    ax2.grid(True, alpha=0.3)                                # Grid

    plt.tight_layout()                                       # Auto-adjust subplot spacing to avoid overlap

    # Generate safe filename (remove special characters)
    safe_name = (setting_name                                # Start from setting name
                 .replace(" ", "_")                          # Space → underscore
                 .replace(":", "")                           # Remove colons
                 .replace("(", "").replace(")", "")          # Remove parentheses
                 .replace("=", "")                           # Remove equals signs
                 .replace("/", "_")                          # Slash → underscore
                 .replace("→", "to"))                        # Arrow → "to"
    filename = f'{save_dir}/comparison_Setting{setting}_{safe_name}.png'  # Full file path
    plt.savefig(filename, dpi=300, bbox_inches='tight')      # Save as PNG, 300 DPI high resolution
    print(f"  ✓ Plot saved: {filename}")                     # Print save confirmation
    plt.close()                                              # Close figure to free memory


# ==================== 2×4 Combined Plot ====================
def plot_combined_2x4(all_results, save_dir='.'):
    """
    Plot a 2-row × 4-column combined chart:
    
    Row 1: Cumulative Regret for all 4 Settings
    Row 2: Quantile Evolution for all 4 Settings
    
    Output: split_conformal_combined_2x4.pdf
    
    Args:
        all_results: Dictionary {setting_id: results_dict}, containing results for all 4 Settings
        save_dir:    Directory path to save the plot
    """
    T = 10000                                                # Fixed total number of time steps
    alpha = 0.1                                              # Target miscoverage rate (for annotation, does not affect plotting)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))         # Create 2×4 subplot grid, 20 wide × 8 tall inches

    time_steps = np.arange(1, T + 1)                         # Time axis: [1, 2, ..., T]
    sample_interval = 50                                     # Sampling interval for quantile plot
    sample_indices = np.arange(0, T, sample_interval)        # Sampling indices for quantile plot

    for col, setting in enumerate([1, 2, 3, 4]):             # Iterate over 4 Settings, col=column index
        if setting not in all_results:                       # If data for this Setting is missing
            continue                                         # Skip this column
        data = all_results[setting]                          # Get result dictionary for this Setting

        # ============ Row 1: Cumulative Regret ============
        ax_regret = axes[0, col]                             # Get subplot at row 1, column col
        for algo_key, label, color in ALGO_CONFIGS:          # Iterate over 6 algorithms
            traj_key = f'{algo_key}_cumulative_trajectories' # Key for cumulative regret trajectories
            if traj_key in data and len(data[traj_key]) > 0: # If data exists
                trajectories = np.array(data[traj_key])      # shape=(n_experiments, T)
                mean_traj = np.mean(trajectories, axis=0)    # Mean across experiments
                std_traj = np.std(trajectories, axis=0)      # Std dev across experiments
                ax_regret.plot(time_steps, mean_traj,        # Plot mean line
                               color=color, linewidth=1.5,
                               label=label if col == 0 else "")  # Only show legend in first column
                ax_regret.fill_between(time_steps,           # Plot ±1 std dev shaded region
                                       mean_traj - std_traj,
                                       mean_traj + std_traj,
                                       color=color, alpha=0.15)

        ax_regret.set_title(SETTING_NAMES[setting],          # Column title
                           fontsize=12, fontweight='bold')
        if col == 0:                                         # Only add Y-axis label and legend in first column
            ax_regret.set_ylabel('Cumulative Regret', fontsize=11)
            ax_regret.legend(fontsize=10, loc='upper left')
        ax_regret.grid(True, alpha=0.3)                      # Grid lines

        # ============ Row 2: Quantile Evolution ============
        ax_quant = axes[1, col]                              # Get subplot at row 2, column col

        # True quantile (black dashed baseline)
        true_q = data['true_quantiles']                      # True quantile array
        ax_quant.plot(sample_indices, true_q[sample_indices], # Plot
                      'k--', linewidth=2,                    # Black dashed line
                      label='True Quantile' if col == 0 else "",  # Only show legend in first column
                      alpha=0.8)

        # Quantile trajectories for each algorithm
        for algo_key, label, color in ALGO_CONFIGS:          # Iterate over 6 algorithms
            q_key = f'{algo_key}_quantile'                   # Quantile data key name
            if q_key in data:                                # If data exists
                quantiles = np.array(data[q_key])            # shape=(T,)
                ax_quant.plot(sample_indices,                 # Plot sampled quantile trajectory
                              quantiles[sample_indices],
                              color=color, linewidth=1.2,
                              label=label if col == 0 else "",  # Only show legend in first column
                              alpha=0.7)

        ax_quant.set_xlabel('Time Step', fontsize=10)        # X-axis label
        if col == 0:                                         # Only add Y-axis label and legend in first column
            ax_quant.set_ylabel('Quantile Estimate', fontsize=11)
            ax_quant.legend(fontsize=10, loc='upper left')
        ax_quant.grid(True, alpha=0.3)                       # Grid lines

    plt.tight_layout()                                       # Auto-adjust subplot spacing

    filename = f'{save_dir}/split_conformal_combined_2x4.pdf'  # Output PDF file path
    plt.savefig(filename, dpi=300, bbox_inches='tight')      # Save as vector PDF, 300 DPI
    print(f"  ✓ Combined plot saved: {filename}")            # Print save confirmation
    plt.close()                                              # Close figure to free memory
