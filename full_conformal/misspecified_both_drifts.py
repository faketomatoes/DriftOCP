#!/usr/bin/env python3
"""
Model Misspecification Experiment - Mean Drift vs Variance Drift
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_data_mean_drift(seed=42, n_train=500, n_test=10000, dim=10,
                              drift_points=None, n_pretrain=100, noise_std=1.0, C=100.0):
    """Mean drift data: X ~ N(μ, I), μ undergoes drift"""
    if drift_points is None:
        drift_points = [3333, 6667]
    
    mean_stage1 = 0.0
    mean_stage2 = 3.0
    mean_stage3 = -2.0
    
    np.random.seed(seed)
    beta_true = np.random.randn(dim)
    
    X_pretrain = np.random.randn(n_pretrain, dim) + mean_stage1
    Y_pretrain = (X_pretrain @ beta_true + 
                  (1/C) * np.sum(X_pretrain**2, axis=1) + 
                  np.random.randn(n_pretrain) * noise_std)
    
    model = Ridge(alpha=1.0)
    model.fit(X_pretrain, Y_pretrain)
    beta_estimated = model.coef_
    
    X_train = np.random.randn(n_train, dim) + mean_stage1
    Y_train = (X_train @ beta_true + 
               (1/C) * np.sum(X_train**2, axis=1) + 
               np.random.randn(n_train) * noise_std)
    
    X_test = np.zeros((n_test, dim))
    X_test[:drift_points[0]] = np.random.randn(drift_points[0], dim) + mean_stage1
    X_test[drift_points[0]:drift_points[1]] = np.random.randn(drift_points[1] - drift_points[0], dim) + mean_stage2
    X_test[drift_points[1]:] = np.random.randn(n_test - drift_points[1], dim) + mean_stage3
    
    Y_test = (X_test @ beta_true + 
              (1/C) * np.sum(X_test**2, axis=1) + 
              np.random.randn(n_test) * noise_std)
    
    return {
        'X_train': X_train, 'Y_train': Y_train,
        'X_test': X_test, 'Y_test': Y_test,
        'beta_true': beta_true, 'beta_estimated': beta_estimated,
        'drift_points': drift_points
    }


def generate_data_var_drift(seed=42, n_train=500, n_test=10000, dim=10,
                             drift_points=None, n_pretrain=100, noise_std=1.0, C=100.0):
    """Variance drift data: X ~ N(0, σ²I), σ undergoes drift"""
    if drift_points is None:
        drift_points = [3333, 6667]
    
    std_stage1 = 1.0
    std_stage2 = 5.0
    std_stage3 = 10.0
    
    np.random.seed(seed)
    beta_true = np.random.randn(dim)
    
    X_pretrain = np.random.randn(n_pretrain, dim) * std_stage1
    Y_pretrain = (X_pretrain @ beta_true + 
                  (1/C) * np.sum(X_pretrain**2, axis=1) + 
                  np.random.randn(n_pretrain) * noise_std)
    
    model = Ridge(alpha=1.0)
    model.fit(X_pretrain, Y_pretrain)
    beta_estimated = model.coef_
    
    X_train = np.random.randn(n_train, dim) * std_stage1
    Y_train = (X_train @ beta_true + 
               (1/C) * np.sum(X_train**2, axis=1) + 
               np.random.randn(n_train) * noise_std)
    
    X_test = np.zeros((n_test, dim))
    X_test[:drift_points[0]] = np.random.randn(drift_points[0], dim) * std_stage1
    X_test[drift_points[0]:drift_points[1]] = np.random.randn(drift_points[1] - drift_points[0], dim) * std_stage2
    X_test[drift_points[1]:] = np.random.randn(n_test - drift_points[1], dim) * std_stage3
    
    Y_test = (X_test @ beta_true + 
              (1/C) * np.sum(X_test**2, axis=1) + 
              np.random.randn(n_test) * noise_std)
    
    return {
        'X_train': X_train, 'Y_train': Y_train,
        'X_test': X_test, 'Y_test': Y_test,
        'beta_true': beta_true, 'beta_estimated': beta_estimated,
        'drift_points': drift_points
    }


class OnlineSGD:
    """Online SGD + drift detection"""
    def __init__(self, dim):
        self.dim = dim
        self.beta = np.zeros(dim)
        self.eta_base = 0.01
    
    def update(self, X, Y, t):
        eta = self.eta_base / (t + 1) ** 0.5
        residual = Y - X @ self.beta
        grad = -2 * X * residual
        self.beta = self.beta - eta * grad
    
    def run(self, X_test, Y_test, X_train, Y_train, alpha=0.1):
        for i in range(len(X_train)):
            self.update(X_train[i], Y_train[i], i)
        
        T = len(X_test)
        quantiles, lower_bounds, upper_bounds, coverages = [], [], [], []
        drift_points = []
        
        train_scores = np.abs(Y_train - X_train @ self.beta)
        q = np.quantile(train_scores, 1 - alpha)
        
        n, r, tau = 1, 1, 0
        
        while tau < T:
            T_n_r = min(3 ** r, T - tau)
            round_errors, round_scores = [], []
            t, drift_detected = 0, False
            
            while t < T_n_r and tau < T:
                X_t, Y_t = X_test[tau], Y_test[tau]
                pred = X_t @ self.beta
                
                lower_bounds.append(pred - q)
                upper_bounds.append(pred + q)
                quantiles.append(q)
                
                score = np.abs(Y_t - pred)
                round_scores.append(score)
                
                error_t = 1 if score > q else 0
                round_errors.append(error_t - alpha)
                coverages.append(1 if (Y_t >= pred - q) and (Y_t <= pred + q) else 0)
                
                drift_found = False
                if len(round_errors) >= 10:
                    for j in range(len(round_errors)):
                        sub = round_errors[j:]
                        if len(sub) < 10:
                            break
                        avg_error = np.mean(sub)
                        threshold = 4.0 / np.sqrt(len(sub))
                        if abs(avg_error) > threshold:
                            drift_found = True
                            break
                if drift_found:
                    drift_points.append(tau)
                    n, r, drift_detected = n + 1, 1, True
                    self.update(X_t, Y_t, tau + len(X_train))
                    tau += 1
                    break
                
                self.update(X_t, Y_t, tau + len(X_train))
                tau += 1
                t += 1
            
            if not drift_detected and round_scores:
                q = np.quantile(round_scores, 1 - alpha)
                r += 1
        
        return {
            'lower': np.array(lower_bounds), 'upper': np.array(upper_bounds),
            'quantiles': np.array(quantiles), 'coverages': np.array(coverages),
            'n_drifts': len(drift_points)
        }


class PretrainMethod:
    """Pretrain + drift detection"""
    def __init__(self, beta):
        self.beta = beta
    
    def run(self, X_test, Y_test, X_train, Y_train, alpha=0.1):
        T = len(X_test)
        train_scores = np.abs(Y_train - X_train @ self.beta)
        q = np.quantile(train_scores, 1 - alpha)
        
        quantiles, lower_bounds, upper_bounds, coverages = [], [], [], []
        drift_points = []
        n, r, tau = 1, 1, 0
        
        while tau < T:
            T_n_r = min(3 ** r, T - tau)
            round_errors, round_scores = [], []
            t, drift_detected = 0, False
            
            while t < T_n_r and tau < T:
                X_t, Y_t = X_test[tau], Y_test[tau]
                pred = X_t @ self.beta
                
                lower_bounds.append(pred - q)
                upper_bounds.append(pred + q)
                quantiles.append(q)
                
                score = np.abs(Y_t - pred)
                round_scores.append(score)
                
                error_t = 1 if score > q else 0
                round_errors.append(error_t - alpha)
                coverages.append(1 if (Y_t >= pred - q) and (Y_t <= pred + q) else 0)
                
                drift_found = False
                if len(round_errors) >= 10:
                    for j in range(len(round_errors)):
                        sub = round_errors[j:]
                        if len(sub) < 10:
                            break
                        avg_error = np.mean(sub)
                        threshold = 4.0 / np.sqrt(len(sub))
                        if abs(avg_error) > threshold:
                            drift_found = True
                            break
                if drift_found:
                    drift_points.append(tau)
                    n, r, drift_detected = n + 1, 1, True
                    tau += 1
                    break
                
                tau += 1
                t += 1
            
            if not drift_detected and round_scores:
                q = np.quantile(round_scores, 1 - alpha)
                r += 1
        
        return {
            'lower': np.array(lower_bounds), 'upper': np.array(upper_bounds),
            'quantiles': np.array(quantiles), 'coverages': np.array(coverages),
            'n_drifts': len(drift_points)
        }


class AbsoluteYMethod:
    """Score = |Y| + drift detection"""
    def run(self, X_test, Y_test, X_train, Y_train, alpha=0.1):
        T = len(X_test)
        train_scores = np.abs(Y_train)
        q = np.quantile(train_scores, 1 - alpha)
        
        quantiles, lower_bounds, upper_bounds, coverages = [], [], [], []
        drift_points = []
        n, r, tau = 1, 1, 0
        
        while tau < T:
            T_n_r = min(3 ** r, T - tau)
            round_errors, round_scores = [], []
            t, drift_detected = 0, False
            
            while t < T_n_r and tau < T:
                Y_t = Y_test[tau]
                
                lower_bounds.append(-q)
                upper_bounds.append(q)
                quantiles.append(q)
                
                score = np.abs(Y_t)
                round_scores.append(score)
                
                error_t = 1 if score > q else 0
                round_errors.append(error_t - alpha)
                coverages.append(1 if (Y_t >= -q) and (Y_t <= q) else 0)
                
                drift_found = False
                if len(round_errors) >= 10:
                    for j in range(len(round_errors)):
                        sub = round_errors[j:]
                        if len(sub) < 10:
                            break
                        avg_error = np.mean(sub)
                        threshold = 4.0 / np.sqrt(len(sub))
                        if abs(avg_error) > threshold:
                            drift_found = True
                            break
                if drift_found:
                    drift_points.append(tau)
                    n, r, drift_detected = n + 1, 1, True
                    tau += 1
                    break
                
                tau += 1
                t += 1
            
            if not drift_detected and round_scores:
                q = np.quantile(round_scores, 1 - alpha)
                r += 1
        
        return {
            'lower': np.array(lower_bounds), 'upper': np.array(upper_bounds),
            'quantiles': np.array(quantiles), 'coverages': np.array(coverages),
            'n_drifts': len(drift_points)
        }


def run_experiments(data_generator, n_exp=20, C=100.0):
    """Run experiments"""
    online_widths_all, pretrain_widths_all, absY_widths_all = [], [], []
    online_coverages_all, pretrain_coverages_all, absY_coverages_all = [], [], []
    
    for exp in range(n_exp):
        print(f"  Exp {exp+1}/{n_exp}...", end=" ", flush=True)
        data = data_generator(seed=42 + exp, C=C)
        dim = data['X_train'].shape[1]
        
        online = OnlineSGD(dim)
        result_online = online.run(data['X_test'], data['Y_test'], data['X_train'], data['Y_train'])
        
        pretrain = PretrainMethod(data['beta_estimated'])
        result_pretrain = pretrain.run(data['X_test'], data['Y_test'], data['X_train'], data['Y_train'])
        
        absY = AbsoluteYMethod()
        result_absY = absY.run(data['X_test'], data['Y_test'], data['X_train'], data['Y_train'])
        
        online_widths_all.append(result_online['upper'] - result_online['lower'])
        pretrain_widths_all.append(result_pretrain['upper'] - result_pretrain['lower'])
        absY_widths_all.append(result_absY['upper'] - result_absY['lower'])
        
        online_coverages_all.append(result_online['coverages'])
        pretrain_coverages_all.append(result_pretrain['coverages'])
        absY_coverages_all.append(result_absY['coverages'])
        
        print(f"Online:{result_online['n_drifts']} drifts, Pretrain:{result_pretrain['n_drifts']} drifts, |Y|:{result_absY['n_drifts']} drifts")
    
    return {
        'online_widths': np.array(online_widths_all),
        'pretrain_widths': np.array(pretrain_widths_all),
        'absY_widths': np.array(absY_widths_all),
        'online_coverages': np.array(online_coverages_all),
        'pretrain_coverages': np.array(pretrain_coverages_all),
        'absY_coverages': np.array(absY_coverages_all),
        'drift_points': [3333, 6667]
    }


def plot_two_figures(results, title, filename, drift_labels):
    """Plot only two figures: Interval Width and Local Coverage Rate"""
    drift_points = results['drift_points']
    
    online_width_mean = np.mean(results['online_widths'], axis=0)
    pretrain_width_mean = np.mean(results['pretrain_widths'], axis=0)
    absY_width_mean = np.mean(results['absY_widths'], axis=0)
    
    online_width_std = np.std(results['online_widths'], axis=0)
    pretrain_width_std = np.std(results['pretrain_widths'], axis=0)
    absY_width_std = np.std(results['absY_widths'], axis=0)
    
    T = len(online_width_mean)
    time_steps = np.arange(T)
    
    colors = {'online': '#3498db', 'pretrain': '#9b59b6', 'absY': '#e74c3c'}
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # ========== Plot 1: Interval Width ==========
    ax1 = axes[0]
    ax1.plot(time_steps, online_width_mean, color=colors['online'], linewidth=2, label='Online SGD')
    ax1.plot(time_steps, pretrain_width_mean, color=colors['pretrain'], linewidth=2, 
             linestyle='--', label='Pretrain |Y-Xβ̂|')
    ax1.plot(time_steps, absY_width_mean, color=colors['absY'], linewidth=2, 
             linestyle=':', label='Score = |Y|')
    
    ax1.fill_between(time_steps, online_width_mean - online_width_std, 
                     online_width_mean + online_width_std, alpha=0.15, color=colors['online'])
    ax1.fill_between(time_steps, pretrain_width_mean - pretrain_width_std, 
                     pretrain_width_mean + pretrain_width_std, alpha=0.15, color=colors['pretrain'])
    ax1.fill_between(time_steps, absY_width_mean - absY_width_std, 
                     absY_width_mean + absY_width_std, alpha=0.15, color=colors['absY'])
    
    for dp in drift_points:
        ax1.axvline(x=dp, linestyle='--', color='black', linewidth=2, alpha=0.5)
    
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Interval Width', fontsize=12)
    ax1.set_title('Prediction Interval Width', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # ========== Plot 2: Local Coverage Rate ==========
    ax2 = axes[1]
    window = 100
    
    def compute_local_coverage(coverages_all):
        local_cov = []
        for t in range(T):
            start = max(0, t - window + 1)
            local_cov.append(np.mean([cov[start:t+1].mean() for cov in coverages_all]))
        return np.array(local_cov)
    
    online_local = compute_local_coverage(results['online_coverages'])
    pretrain_local = compute_local_coverage(results['pretrain_coverages'])
    absY_local = compute_local_coverage(results['absY_coverages'])
    
    ax2.plot(time_steps, online_local, color=colors['online'], linewidth=2, label='Online SGD')
    ax2.plot(time_steps, pretrain_local, color=colors['pretrain'], linewidth=2, 
             linestyle='--', label='Pretrain')
    ax2.plot(time_steps, absY_local, color=colors['absY'], linewidth=2, 
             linestyle=':', label='|Y|')
    ax2.axhline(y=0.9, color='black', linewidth=1.5, label='Target (90%)')
    
    for dp in drift_points:
        ax2.axvline(x=dp, linestyle='--', color='black', linewidth=2, alpha=0.5)
    
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel(f'Local Coverage (Window={window})', fontsize=12)
    ax2.set_title('Local Coverage Rate', fontsize=14, fontweight='bold')
    ax2.set_ylim(0.7, 1.0)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()
    
    # Print statistics
    print(f"\n  Interval Width Statistics:")
    for i, (start, end) in enumerate([(0, drift_points[0]), 
                                       (drift_points[0], drift_points[1]), 
                                       (drift_points[1], T)]):
        print(f"    {drift_labels[i]}: Online={np.mean(online_width_mean[start:end]):.2f}, "
              f"Pretrain={np.mean(pretrain_width_mean[start:end]):.2f}, "
              f"|Y|={np.mean(absY_width_mean[start:end]):.2f}")
    
    print(f"  Coverage: Online={100*np.mean(results['online_coverages']):.1f}%, "
          f"Pretrain={100*np.mean(results['pretrain_coverages']):.1f}%, "
          f"|Y|={100*np.mean(results['absY_coverages']):.1f}%")


def plot_combined(results_mean, results_var, C, filename='results/misspec_combined.pdf'):
    """Combine mean drift and variance drift results into a single figure (2x2)"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = {'online': '#3498db', 'pretrain': '#9b59b6', 'absY': '#e74c3c'}
    window = 100
    
    def compute_local_coverage(coverages_all, T):
        local_cov = []
        for t in range(T):
            start = max(0, t - window + 1)
            local_cov.append(np.mean([cov[start:t+1].mean() for cov in coverages_all]))
        return np.array(local_cov)
    
    # ========== Row 1: Mean Drift ==========
    drift_points = results_mean['drift_points']
    T = len(np.mean(results_mean['online_widths'], axis=0))
    time_steps = np.arange(T)
    
    # Plot 1: Mean Drift - Interval Width
    ax1 = axes[0, 0]
    online_w = np.mean(results_mean['online_widths'], axis=0)
    pretrain_w = np.mean(results_mean['pretrain_widths'], axis=0)
    absY_w = np.mean(results_mean['absY_widths'], axis=0)
    
    ax1.plot(time_steps, online_w, color=colors['online'], linewidth=2, label='Adaptive')
    ax1.plot(time_steps, pretrain_w, color=colors['pretrain'], linewidth=2, linestyle='--', label='Pre-trained')
    ax1.plot(time_steps, absY_w, color=colors['absY'], linewidth=2, linestyle=':', label='Model-free')
    ax1.fill_between(time_steps, online_w - np.std(results_mean['online_widths'], axis=0),
                     online_w + np.std(results_mean['online_widths'], axis=0), alpha=0.15, color=colors['online'])
    ax1.fill_between(time_steps, pretrain_w - np.std(results_mean['pretrain_widths'], axis=0),
                     pretrain_w + np.std(results_mean['pretrain_widths'], axis=0), alpha=0.15, color=colors['pretrain'])
    ax1.fill_between(time_steps, absY_w - np.std(results_mean['absY_widths'], axis=0),
                     absY_w + np.std(results_mean['absY_widths'], axis=0), alpha=0.15, color=colors['absY'])
    for dp in drift_points:
        ax1.axvline(x=dp, linestyle='--', color='black', linewidth=2, alpha=0.5)
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Interval Width', fontsize=12)
    ax1.set_title('Mean Drift | Interval Width', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean Drift - Local Coverage
    ax2 = axes[0, 1]
    online_cov = compute_local_coverage(results_mean['online_coverages'], T)
    pretrain_cov = compute_local_coverage(results_mean['pretrain_coverages'], T)
    absY_cov = compute_local_coverage(results_mean['absY_coverages'], T)
    
    ax2.plot(time_steps, online_cov, color=colors['online'], linewidth=2, label='Adaptive')
    ax2.plot(time_steps, pretrain_cov, color=colors['pretrain'], linewidth=2, linestyle='--', label='Pre-trained')
    ax2.plot(time_steps, absY_cov, color=colors['absY'], linewidth=2, linestyle=':', label='Model-free')
    ax2.axhline(y=0.9, color='black', linewidth=1.5, label='Target (90%)')
    for dp in drift_points:
        ax2.axvline(x=dp, linestyle='--', color='black', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel(f'Local Coverage (Window={window})', fontsize=12)
    ax2.set_title('Mean Drift | Local Coverage', fontsize=14, fontweight='bold')
    ax2.set_ylim(0.7, 1.0)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # ========== Row 2: Variance Drift ==========
    drift_points = results_var['drift_points']
    T = len(np.mean(results_var['online_widths'], axis=0))
    time_steps = np.arange(T)
    
    # Plot 3: Variance Drift - Interval Width
    ax3 = axes[1, 0]
    online_w = np.mean(results_var['online_widths'], axis=0)
    pretrain_w = np.mean(results_var['pretrain_widths'], axis=0)
    absY_w = np.mean(results_var['absY_widths'], axis=0)
    
    ax3.plot(time_steps, online_w, color=colors['online'], linewidth=2, label='Adaptive')
    ax3.plot(time_steps, pretrain_w, color=colors['pretrain'], linewidth=2, linestyle='--', label='Pre-trained')
    ax3.plot(time_steps, absY_w, color=colors['absY'], linewidth=2, linestyle=':', label='Model-free')
    ax3.fill_between(time_steps, online_w - np.std(results_var['online_widths'], axis=0),
                     online_w + np.std(results_var['online_widths'], axis=0), alpha=0.15, color=colors['online'])
    ax3.fill_between(time_steps, pretrain_w - np.std(results_var['pretrain_widths'], axis=0),
                     pretrain_w + np.std(results_var['pretrain_widths'], axis=0), alpha=0.15, color=colors['pretrain'])
    ax3.fill_between(time_steps, absY_w - np.std(results_var['absY_widths'], axis=0),
                     absY_w + np.std(results_var['absY_widths'], axis=0), alpha=0.15, color=colors['absY'])
    for dp in drift_points:
        ax3.axvline(x=dp, linestyle='--', color='black', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Time Step', fontsize=12)
    ax3.set_ylabel('Interval Width', fontsize=12)
    ax3.set_title('Variance Drift | Interval Width', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Variance Drift - Local Coverage
    ax4 = axes[1, 1]
    online_cov = compute_local_coverage(results_var['online_coverages'], T)
    pretrain_cov = compute_local_coverage(results_var['pretrain_coverages'], T)
    absY_cov = compute_local_coverage(results_var['absY_coverages'], T)
    
    ax4.plot(time_steps, online_cov, color=colors['online'], linewidth=2, label='Adaptive')
    ax4.plot(time_steps, pretrain_cov, color=colors['pretrain'], linewidth=2, linestyle='--', label='Pre-trained')
    ax4.plot(time_steps, absY_cov, color=colors['absY'], linewidth=2, linestyle=':', label='Model-free')
    ax4.axhline(y=0.9, color='black', linewidth=1.5, label='Target (90%)')
    for dp in drift_points:
        ax4.axvline(x=dp, linestyle='--', color='black', linewidth=2, alpha=0.5)
    ax4.set_xlabel('Time Step', fontsize=12)
    ax4.set_ylabel(f'Local Coverage (Window={window})', fontsize=12)
    ax4.set_title('Variance Drift | Local Coverage', fontsize=14, fontweight='bold')
    ax4.set_ylim(0.7, 1.0)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


if __name__ == "__main__":
    C = 100.0
    n_exp = 20
    
    print("="*70)
    print("Model Misspecification Experiment")
    print("True model: Y = Xβ* + (1/C)||X||² + ε")
    print("Fitted model: Linear model Y = Xβ")
    print(f"C = {C}, dim = 10")
    print("="*70)
    
    # ========== Mean Drift ==========
    print("\n" + "="*70)
    print("[Mean Drift] X ~ N(μ, I), μ: 0 → 3 → -2")
    print("="*70)
    
    results_mean = run_experiments(generate_data_mean_drift, n_exp=n_exp, C=C)
    
    # Print statistics
    print("\n  Mean Drift - Interval Width Statistics:")
    online_w = np.mean(results_mean['online_widths'], axis=0)
    pretrain_w = np.mean(results_mean['pretrain_widths'], axis=0)
    absY_w = np.mean(results_mean['absY_widths'], axis=0)
    dp = results_mean['drift_points']
    print(f"    Seg 1 (μ=0): Online={np.mean(online_w[:dp[0]]):.2f}, Pretrain={np.mean(pretrain_w[:dp[0]]):.2f}, |Y|={np.mean(absY_w[:dp[0]]):.2f}")
    print(f"    Seg 2 (μ=3): Online={np.mean(online_w[dp[0]:dp[1]]):.2f}, Pretrain={np.mean(pretrain_w[dp[0]:dp[1]]):.2f}, |Y|={np.mean(absY_w[dp[0]:dp[1]]):.2f}")
    print(f"    Seg 3 (μ=-2): Online={np.mean(online_w[dp[1]:]):.2f}, Pretrain={np.mean(pretrain_w[dp[1]:]):.2f}, |Y|={np.mean(absY_w[dp[1]:]):.2f}")
    print(f"  Coverage: Online={100*np.mean(results_mean['online_coverages']):.1f}%, "
          f"Pretrain={100*np.mean(results_mean['pretrain_coverages']):.1f}%, "
          f"|Y|={100*np.mean(results_mean['absY_coverages']):.1f}%")
    
    # ========== Variance Drift ==========
    print("\n" + "="*70)
    print("[Variance Drift] X ~ N(0, σ²I), σ: 1 → 5 → 10")
    print("="*70)
    
    results_var = run_experiments(generate_data_var_drift, n_exp=n_exp, C=C)
    
    # Print statistics
    print("\n  Variance Drift - Interval Width Statistics:")
    online_w = np.mean(results_var['online_widths'], axis=0)
    pretrain_w = np.mean(results_var['pretrain_widths'], axis=0)
    absY_w = np.mean(results_var['absY_widths'], axis=0)
    dp = results_var['drift_points']
    print(f"    Seg 1 (σ=1): Online={np.mean(online_w[:dp[0]]):.2f}, Pretrain={np.mean(pretrain_w[:dp[0]]):.2f}, |Y|={np.mean(absY_w[:dp[0]]):.2f}")
    print(f"    Seg 2 (σ=5): Online={np.mean(online_w[dp[0]:dp[1]]):.2f}, Pretrain={np.mean(pretrain_w[dp[0]:dp[1]]):.2f}, |Y|={np.mean(absY_w[dp[0]:dp[1]]):.2f}")
    print(f"    Seg 3 (σ=10): Online={np.mean(online_w[dp[1]:]):.2f}, Pretrain={np.mean(pretrain_w[dp[1]:]):.2f}, |Y|={np.mean(absY_w[dp[1]:]):.2f}")
    print(f"  Coverage: Online={100*np.mean(results_var['online_coverages']):.1f}%, "
          f"Pretrain={100*np.mean(results_var['pretrain_coverages']):.1f}%, "
          f"|Y|={100*np.mean(results_var['absY_coverages']):.1f}%")
    
    # ========== Combined Plot ==========
    print("\n" + "="*70)
    print("Generating combined plot...")
    print("="*70)
    plot_combined(results_mean, results_var, C, 'results/misspec_combined.pdf')
    
    # Save experiment data
    import pickle
    import os
    os.makedirs('results', exist_ok=True)
    data_to_save = {
        'results_mean': results_mean,
        'results_var': results_var,
        'C': C,
        'n_exp': n_exp,
        'experiment_type': 'misspecified'
    }
    with open('results/misspec_data.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)
    print("✓ Data saved: results/misspec_data.pkl")
    
    print("\n" + "="*70)
    print("Experiment complete!")
    print("="*70)

