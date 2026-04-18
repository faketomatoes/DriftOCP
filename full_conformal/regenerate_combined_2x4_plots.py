#!/usr/bin/env python3
"""
Regenerate combined 2x4 plots with larger legend fonts (11pt equivalent)
- split_conformal_combined_2x4.pdf
- full_conformal_combined_2x4.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestRegressor

# ==================== SPLIT CONFORMAL ====================
def generate_data_setting1(X, t=None, n_samples=None):
    if n_samples is None:
        T_len = len(X)
        Y = np.zeros(T_len)
        for i in range(T_len):
            if i < 4000: Y[i] = 2 * X[i, 0] + X[i, 1] + 0.5 * np.random.randn()
            elif i < 7000: Y[i] = 2 * X[i, 0] + X[i, 1] + 2.0 * np.random.randn()
            else: Y[i] = 2 * X[i, 0] + X[i, 1] + 3.5 * np.random.randn()
    else:
        if t < 4000: Y = 2 * X[:, 0] + X[:, 1] + 0.5 * np.random.randn(n_samples)
        elif t < 7000: Y = 2 * X[:, 0] + X[:, 1] + 2.0 * np.random.randn(n_samples)
        else: Y = 2 * X[:, 0] + X[:, 1] + 3.5 * np.random.randn(n_samples)
    return Y

def generate_data_setting2(X, t=None, n_samples=None):
    alpha_drift = 0.002
    if n_samples is None:
        T_len = len(X)
        Y = np.zeros(T_len)
        for i in range(T_len):
            mu_t = alpha_drift * i
            Y[i] = 2 * X[i, 0] + X[i, 1] + mu_t + 0.5 * np.random.randn()
    else:
        mu_t = alpha_drift * t
        Y = 2 * X[:, 0] + X[:, 1] + mu_t + 0.5 * np.random.randn(n_samples)
    return Y

def generate_data_setting3(X, t=None, n_samples=None):
    if n_samples is None:
        T_len = len(X)
        Y = np.zeros(T_len)
        for i in range(T_len):
            sigma_t = np.sqrt(1.0 + 40.0 * i / 5000)
            Y[i] = 2 * X[i, 0] + X[i, 1] + sigma_t * np.random.randn()
    else:
        sigma_t = np.sqrt(1.0 + 40.0 * t / 5000)
        Y = 2 * X[:, 0] + X[:, 1] + sigma_t * np.random.randn(n_samples)
    return Y

def generate_data_setting4(X, t=None, n_samples=None):
    if n_samples is None:
        T_len = len(X)
        Y = 2 * X[:, 0] + X[:, 1] + 0.5 * np.random.randn(T_len)
    else:
        Y = 2 * X[:, 0] + X[:, 1] + 0.5 * np.random.randn(n_samples)
    return Y

def compute_empirical_true_quantiles(setting, T, alpha=0.1, n_samples=10000):
    print(f"  Computing empirical true quantiles for Setting {setting}...")
    np.random.seed(42)
    n_train, dim = 500, 5
    X_train = np.random.randn(n_train, dim)
    Y_train = 2 * X_train[:, 0] + X_train[:, 1] + 0.5 * np.random.randn(n_train)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, Y_train)
    
    sample_points = list(range(0, T, 100)) + [T-1]
    true_quantiles = np.zeros(T)
    generate_funcs = {1: generate_data_setting1, 2: generate_data_setting2, 
                      3: generate_data_setting3, 4: generate_data_setting4}
    generate_func = generate_funcs[setting]
    
    for t in sample_points:
        X_samples = np.random.randn(n_samples, dim)
        Y_samples = generate_func(X_samples, t=t, n_samples=n_samples)
        Y_pred = model.predict(X_samples)
        scores = np.abs(Y_samples - Y_pred)
        true_quantiles[t] = np.quantile(scores, 1 - alpha)
    
    for i in range(len(sample_points) - 1):
        t1, t2 = sample_points[i], sample_points[i+1]
        q1, q2 = true_quantiles[t1], true_quantiles[t2]
        for t in range(t1+1, t2):
            true_quantiles[t] = q1 + (q2 - q1) * (t - t1) / (t2 - t1)
    
    return true_quantiles

def rolling_coverage(coverages, window=500):
    T = len(coverages)
    rolling = np.zeros(T)
    for t in range(T):
        start = max(0, t - window + 1)
        rolling[t] = np.mean(coverages[start:t+1])
    return rolling

def smooth_curve(data, window=100):
    """Smooth curve with proper boundary handling to avoid edge effects."""
    # Use pandas rolling mean which handles boundaries better
    import pandas as pd
    return pd.Series(data).rolling(window=window, min_periods=1, center=True).mean().values


# ==================== MAIN ====================
if __name__ == "__main__":
    LEGEND_FONTSIZE = 13  # Larger legend font
    
    # ========== 1. SPLIT CONFORMAL COMBINED 2x4 ==========
    print("="*60)
    print("Generating split_conformal_combined_2x4.pdf")
    print("="*60)
    
    # Load data
    print("Loading split conformal data...")
    with open('experiment_results_6algorithms_parallel.pkl', 'rb') as f:
        main_results = pickle.load(f)
    with open('../comparison_results/setting1_jump_variance_results.pkl', 'rb') as f:
        setting1_results = pickle.load(f)
    
    all_results = {1: setting1_results}
    for setting in [2, 3, 4]:
        if setting in main_results:
            all_results[setting] = main_results[setting]
    
    setting_names = {
        1: 'Jump Variance',
        2: 'Linear Bias Drift',
        3: 'Smooth Variance',
        4: 'Stationary'
    }
    
    # Compute true quantiles
    T = 10000
    true_quantiles = {}
    for setting in [1, 2, 3, 4]:
        true_quantiles[setting] = compute_empirical_true_quantiles(setting, T)
    
    # Create 2x4 figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    
    algo_configs = [
        ('drift', 'DriftOCP', 'blue'),
        ('aci_06', r'ACI $\eta_t=t^{-0.6}$', 'red'),
        ('aci_05', r'ACI $\eta_t=t^{-0.5}$', 'green'),
        ('aci_fixed_001', r'ACI $\eta=0.01$', 'cyan'),
        ('aci_fixed_01', r'ACI $\eta=0.1$', 'magenta'),
        ('aci_fixed_05', r'ACI $\eta=0.5$', 'orange'),
    ]
    
    time_steps = np.arange(1, T + 1)
    sample_interval = 50
    sample_indices = np.arange(0, T, sample_interval)
    
    for col, setting in enumerate([1, 2, 3, 4]):
        data = all_results[setting]
        
        # Row 1: Cumulative Regret
        ax_regret = axes[0, col]
        for algo_key, label, color in algo_configs:
            traj_key = f'{algo_key}_cumulative_trajectories'
            if traj_key in data and len(data[traj_key]) > 0:
                trajectories = np.array(data[traj_key])
                mean_traj = np.mean(trajectories, axis=0)
                std_traj = np.std(trajectories, axis=0)
                ax_regret.plot(time_steps, mean_traj, color=color, linewidth=1.5,
                              label=label if col == 0 else "")
                ax_regret.fill_between(time_steps, mean_traj - std_traj, mean_traj + std_traj,
                                       color=color, alpha=0.15)
        
        ax_regret.set_title(setting_names[setting], fontsize=12, fontweight='bold')
        if col == 0:
            ax_regret.set_ylabel('Cumulative Regret', fontsize=11)
            ax_regret.legend(fontsize=LEGEND_FONTSIZE, loc='upper left')
        ax_regret.grid(True, alpha=0.3)
        
        # Row 2: Quantile Evolution
        ax_quant = axes[1, col]
        ax_quant.plot(sample_indices, true_quantiles[setting][sample_indices], 
                     'k--', linewidth=2, label='True Quantile' if col == 0 else "", alpha=0.8)
        
        for algo_key, label, color in algo_configs:
            q_key = f'{algo_key}_quantile'
            if q_key in data:
                quantiles = np.array(data[q_key])
                ax_quant.plot(sample_indices, quantiles[sample_indices], 
                             color=color, linewidth=1.2, 
                             label=label if col == 0 else "", alpha=0.7)
        
        ax_quant.set_xlabel('Time Step', fontsize=10)
        if col == 0:
            ax_quant.set_ylabel('Quantile Estimate', fontsize=11)
            ax_quant.legend(fontsize=LEGEND_FONTSIZE, loc='upper left')
        ax_quant.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../comparison_results/split_conformal_combined_2x4.pdf', dpi=300, bbox_inches='tight')
    print("✓ Saved: ../comparison_results/split_conformal_combined_2x4.pdf")
    plt.close()
    
    # ========== 2. FULL CONFORMAL COMBINED 2x4 ==========
    print("\n" + "="*60)
    print("Generating full_conformal_combined_2x4.pdf")
    print("="*60)
    
    # Load data
    print("Loading full conformal data...")
    with open('results/well_specified_data.pkl', 'rb') as f:
        well_data = pickle.load(f)
    with open('results/misspec_data.pkl', 'rb') as f:
        misspec_data = pickle.load(f)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    
    colors = {
        'Adaptive': '#3498db',      # Blue (formerly online)
        'Pre-trained': '#9b59b6',   # Purple (formerly pretrain)
        'Model-free': '#e74c3c',    # Red (formerly absY)
    }
    
    alpha = 0.1
    target_cov = 1 - alpha
    smooth_window = 100
    window = 500
    
    titles = [
        'Well-spec. Mean Drift', 'Well-spec. Variance Drift',
        'Misspec. Mean Drift', 'Misspec. Variance Drift'
    ]
    
    data_sources = [
        well_data['results_mean'],
        well_data['results_var'],
        misspec_data['results_mean'],
        misspec_data['results_var'],
    ]
    
    T = data_sources[0]['online_widths'].shape[1]
    t_axis = np.arange(T)
    
    for col, data in enumerate(data_sources):
        drift_points = data['drift_points']
        
        # Row 1: Interval Width
        ax_width = axes[0, col]
        
        # Adaptive (Online)
        online_widths_mean = np.mean(data['online_widths'], axis=0)
        online_widths_std = np.std(data['online_widths'], axis=0)
        online_smooth = smooth_curve(online_widths_mean, smooth_window)
        online_std_smooth = smooth_curve(online_widths_std, smooth_window)
        ax_width.plot(t_axis, online_smooth, color=colors['Adaptive'], 
                     label='Adaptive' if col == 0 else "", linewidth=1.5)
        ax_width.fill_between(t_axis, online_smooth - online_std_smooth, 
                             online_smooth + online_std_smooth, 
                             color=colors['Adaptive'], alpha=0.2)
        
        # Pre-trained
        pretrain_widths_mean = np.mean(data['pretrain_widths'], axis=0)
        pretrain_widths_std = np.std(data['pretrain_widths'], axis=0)
        pretrain_smooth = smooth_curve(pretrain_widths_mean, smooth_window)
        pretrain_std_smooth = smooth_curve(pretrain_widths_std, smooth_window)
        ax_width.plot(t_axis, pretrain_smooth, color=colors['Pre-trained'], 
                     label='Pre-trained' if col == 0 else "", linewidth=1.5)
        ax_width.fill_between(t_axis, pretrain_smooth - pretrain_std_smooth, 
                             pretrain_smooth + pretrain_std_smooth, 
                             color=colors['Pre-trained'], alpha=0.2)
        
        # Model-free (|Y|)
        absY_widths_mean = np.mean(data['absY_widths'], axis=0)
        absY_widths_std = np.std(data['absY_widths'], axis=0)
        absY_smooth = smooth_curve(absY_widths_mean, smooth_window)
        absY_std_smooth = smooth_curve(absY_widths_std, smooth_window)
        ax_width.plot(t_axis, absY_smooth, color=colors['Model-free'], 
                     label='Model-free' if col == 0 else "", linewidth=1.5)
        ax_width.fill_between(t_axis, absY_smooth - absY_std_smooth, 
                             absY_smooth + absY_std_smooth, 
                             color=colors['Model-free'], alpha=0.2)
        
        for dp in drift_points:
            ax_width.axvline(x=dp, color='gray', linestyle=':', alpha=0.5)
        
        ax_width.set_title(titles[col], fontsize=11)
        if col == 0:
            ax_width.set_ylabel('Interval Width', fontsize=11)
            ax_width.legend(fontsize=LEGEND_FONTSIZE, loc='upper left')
        ax_width.grid(True, alpha=0.3)
        
        # Row 2: Rolling Coverage
        ax_cov = axes[1, col]
        
        # Adaptive (Online)
        online_cov = np.array([rolling_coverage(c, window) for c in data['online_coverages']])
        online_cov_mean = np.mean(online_cov, axis=0)
        online_cov_std = np.std(online_cov, axis=0)
        ax_cov.plot(t_axis, online_cov_mean, color=colors['Adaptive'], 
                   label='Adaptive' if col == 0 else "", linewidth=1.5)
        ax_cov.fill_between(t_axis, online_cov_mean - online_cov_std, 
                           online_cov_mean + online_cov_std, 
                           color=colors['Adaptive'], alpha=0.2)
        
        # Pre-trained
        pretrain_cov = np.array([rolling_coverage(c, window) for c in data['pretrain_coverages']])
        pretrain_cov_mean = np.mean(pretrain_cov, axis=0)
        pretrain_cov_std = np.std(pretrain_cov, axis=0)
        ax_cov.plot(t_axis, pretrain_cov_mean, color=colors['Pre-trained'], 
                   label='Pre-trained' if col == 0 else "", linewidth=1.5)
        ax_cov.fill_between(t_axis, pretrain_cov_mean - pretrain_cov_std, 
                           pretrain_cov_mean + pretrain_cov_std, 
                           color=colors['Pre-trained'], alpha=0.2)
        
        # Model-free (|Y|)
        absY_cov = np.array([rolling_coverage(c, window) for c in data['absY_coverages']])
        absY_cov_mean = np.mean(absY_cov, axis=0)
        absY_cov_std = np.std(absY_cov, axis=0)
        ax_cov.plot(t_axis, absY_cov_mean, color=colors['Model-free'], 
                   label='Model-free' if col == 0 else "", linewidth=1.5)
        ax_cov.fill_between(t_axis, absY_cov_mean - absY_cov_std, 
                           absY_cov_mean + absY_cov_std, 
                           color=colors['Model-free'], alpha=0.2)
        
        ax_cov.axhline(y=target_cov, color='black', linestyle='--', alpha=0.7, 
                      label='Target (90%)' if col == 0 else "")
        
        for dp in drift_points:
            ax_cov.axvline(x=dp, color='gray', linestyle=':', alpha=0.5)
        
        ax_cov.set_xlabel('Time', fontsize=10)
        if col == 0:
            ax_cov.set_ylabel('Rolling Coverage', fontsize=11)
            ax_cov.legend(fontsize=LEGEND_FONTSIZE, loc='lower left')
        ax_cov.set_ylim([0.7, 1.0])
        ax_cov.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/full_conformal_combined_2x4.pdf', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/full_conformal_combined_2x4.pdf")
    plt.close()
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)

