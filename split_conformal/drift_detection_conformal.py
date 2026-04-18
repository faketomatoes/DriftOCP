"""
Online Conformal Inference with Drift Detection Algorithm Implementation
Based on Algorithm 1 from the paper
"""

import numpy as np
import pandas as pd
from typing import Tuple, Callable, Optional, Dict, List
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


class DriftDetectionConformal:
    """
    Online conformal inference algorithm with drift detection
    
    This algorithm detects distribution drift by monitoring coverage errors and resets the quantile upon drift detection
    """
    
    def __init__(self, model, X_train: np.ndarray, Y_train: np.ndarray, 
                 X_predict: np.ndarray, Y_predict: np.ndarray, threshold_c: float = 4.0):
        """
        Initialize
        
        Args:
            model: Base prediction model (e.g., RandomForestRegressor)
            X_train: Training features
            Y_train: Training labels
            X_predict: Prediction features
            Y_predict: Prediction labels (observed incrementally during online updates)
            threshold_c: Drift detection threshold parameter, threshold = threshold_c / sqrt(window_size)
        """
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_predict = X_predict
        self.threshold_c = threshold_c
        self.Y_predict = Y_predict
        
        # Train the model
        self.model.fit(X_train, Y_train)
        
    def get_conformal_score(self, X: np.ndarray, Y: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
        """
        Compute conformal score
        Uses absolute error as the nonconformity score
        
        Args:
            X: Features
            Y: True values
            Y_pred: Predicted values
            
        Returns:
            scores: nonconformity scores
        """
        return np.abs(Y - Y_pred)
    
    def compute_drift_detection_intervals(self, alpha: float = 0.1, T: Optional[int] = None,
                                        initial_q: float = None, compute_regret: bool = False,
                                        n_regret_samples: int = 500) -> Dict:
        """
        Main algorithm for online conformal inference with drift detection
        
        Args:
            alpha: Target miscoverage rate (1-alpha is the target coverage rate)
            T: Total number of steps; if None, uses all prediction data
            initial_q: Initial quantile value; if None, estimated from the training set
            compute_regret: Whether to compute regret metrics
            n_regret_samples: Number of samples used for regret estimation
            
        Returns:
            results: Dictionary containing coverage, width, drift detection info, etc.
        """
        if T is None:
            T = len(self.X_predict)
        else:
            T = min(T, len(self.X_predict))
            
        # Initialize quantile
        if initial_q is None:
            # Estimate initial quantile from the training set
            train_pred = self.model.predict(self.X_train)
            train_scores = self.get_conformal_score(self.X_train, self.Y_train, train_pred)
            initial_q = np.quantile(train_scores, 1 - alpha)
        
        # Initialize variables
        n = 1  # Current stage number
        r = 1  # Current round number
        tau = 0  # Total iteration count
        
        # Record results
        lower_bounds = []
        upper_bounds = []
        predictions = []
        quantiles = []
        stage_history = []
        round_history = []
        drift_points = []  # Record drift detection points
        coverage_errors = []  # Record coverage errors
        
        # Regret-related variables
        regret_values = [] if compute_regret else None
        coverage_gaps = [] if compute_regret else None
        
        # Quantile for the current stage
        q_n_r = initial_q
        
        # Start position of current stage and range of previous round
        stage_start_tau = 0
        prev_round_start = 0  # Start position of previous round
        prev_round_end = 0    # End position of previous round
        stage_errors = []  # Kept for statistics, but quantile estimation only uses the previous round
        
        # Progress tracking
        import time as time_module
        start_time = time_module.time()
        last_print_time = start_time
        
        while tau < T:
            # Length of each stage: T_n_r = 4^r
            T_n_r = min(3**r, T - tau)
            
            # Errors for the current round (used for drift detection)
            round_errors = []  # Reset at the start of each round, used only for drift detection
            
            for t in range(T_n_r):
                if tau >= T:
                    break
                    
                # Get the current data point
                X_t = self.X_predict[tau:tau+1]
                
                # Predict
                Y_pred_t = self.model.predict(X_t)[0]
                predictions.append(Y_pred_t)
                
                # Construct prediction interval C_n,r,t(X_n,r,t) = {y : s_n,r,t(X_n,r,t, y) ≤ q_n,r}
                # For absolute error score, this is equivalent to [Y_pred - q, Y_pred + q]
                lower = Y_pred_t - q_n_r
                upper = Y_pred_t + q_n_r
                lower_bounds.append(lower)
                upper_bounds.append(upper)
                
                # Record current quantile and stage/round info
                quantiles.append(q_n_r)
                stage_history.append(n)
                round_history.append(r)
                
                # Observe the true response
                Y_t = self.Y_predict[tau]
                
                # Compute score and check if it falls within the prediction interval
                score_t = self.get_conformal_score(X_t, np.array([Y_t]), np.array([Y_pred_t]))[0]
                error_t = 1 if score_t > q_n_r else 0  # Miscoverage indicator
                coverage_error = error_t - alpha  # Coverage error
                
                # Add to both lists
                stage_errors.append(coverage_error)  # For quantile estimation (cumulative)
                round_errors.append(coverage_error)  # For drift detection (current round)
                
                # Compute Regret (if needed)
                if compute_regret:
                    # Simulate sampling multiple scores from the current distribution to estimate conditional coverage
                    # Here we use local data to approximate the current distribution
                    window_size = min(50, tau + 1)  # Use the most recent 50 points to estimate the current distribution
                    if tau >= window_size - 1:
                        # Get local window data
                        local_X = self.X_predict[max(0, tau - window_size + 1):tau + 1]
                        local_Y = self.Y_predict[max(0, tau - window_size + 1):tau + 1]
                        
                        # Sample from local data to simulate the score distribution
                        sampled_coverage_errors = []
                        for _ in range(n_regret_samples):
                            # Randomly select a data point
                            idx = np.random.randint(len(local_X))
                            X_sample = local_X[idx:idx+1]
                            Y_sample = local_Y[idx]
                            
                            # Add noise to simulate distribution variability
                            noise_scale = np.std(local_Y) * 0.1  # Use 10% of local std as noise
                            Y_sample_noisy = Y_sample + np.random.normal(0, noise_scale)
                            
                            # Predict and compute score
                            Y_pred_sample = self.model.predict(X_sample)[0]
                            score_sample = self.get_conformal_score(
                                X_sample, np.array([Y_sample_noisy]), np.array([Y_pred_sample])
                            )[0]
                            
                            # Check if it exceeds the current quantile
                            sampled_coverage_errors.append(1 if score_sample > q_n_r else 0)
                        
                        # Compute conditional coverage error
                        empirical_miscoverage = np.mean(sampled_coverage_errors)
                        coverage_gap = abs(empirical_miscoverage - alpha)
                        coverage_gaps.append(coverage_gap)
                        regret_values.append(coverage_gap)  # Regret contribution of a single time step
                    else:
                        # Insufficient data in the initial phase, use 0
                        coverage_gaps.append(0)
                        regret_values.append(0)
                
                # Drift detection: only use data from the current round
                for j in range(1, min(t+1, len(round_errors)+1)):
                    # Compute average coverage error from j to t (within the current round only)
                    if j-1 < len(round_errors):
                        avg_error = np.mean(round_errors[j-1:])
                        
                        # Detection condition: |average error| > threshold_c / sqrt(t-j+1)
                        threshold = self.threshold_c / np.sqrt(t - j + 1)
                        
                        # Drift detected; only trigger restart if window length t-j+1 >= 10
                        if abs(avg_error) > threshold and (t - j + 1) >= 10:
                            # Output drift detection info
                            print(f"\n    ⚠️  Drift detected! τ={tau}, Stage {n}, Round {r}, position in round t={t}")
                            print(f"       |avg_error|={abs(avg_error):.4f} > threshold={threshold:.4f}")
                            print(f"       Detection window: round_errors[{j-1}:] (length={t-j+1})", flush=True)
                            
                            # Record drift point
                            drift_points.append(tau)
                            
                            # Update stage and round
                            n = n + 1
                            r = 1
                            q_n_r = quantiles[-1]  # Use the last quantile as the initial value for the new stage
                            
                            # Reset stage-related variables (start new stage)
                            stage_start_tau = tau + 1
                            prev_round_start = tau + 1  # New stage starts, reset prev_round range
                            prev_round_end = tau + 1
                            stage_errors = []  # New stage starts, reset cumulative errors
                            
                            # Break out of the inner loop, enter the next stage
                            tau += 1
                            
                            # Show progress (also shown upon drift detection)
                            current_time = time_module.time()
                            elapsed = current_time - start_time
                            progress = tau / T * 100
                            eta = (elapsed / tau * (T - tau)) if tau > 0 else 0
                            print(f"    >>> Stage restart! {tau}/{T} ({progress:.1f}%) | "
                                  f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s", flush=True)
                            last_print_time = current_time
                            
                            break
                else:
                    # If no drift detected, continue the current stage
                    tau += 1
                    
                    # Show progress (every 100 steps or every 5 seconds)
                    current_time = time_module.time()
                    if tau % 100 == 0 or (current_time - last_print_time) >= 5:
                        elapsed = current_time - start_time
                        progress = tau / T * 100
                        eta = (elapsed / tau * (T - tau)) if tau > 0 else 0
                        # Compute mean error of the current round (for monitoring)
                        round_avg_err = np.mean(np.abs(round_errors)) if len(round_errors) > 0 else 0
                        print(f"    Drift Detection progress: {tau}/{T} ({progress:.1f}%) | "
                              f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s | "
                              f"Stage: {n}, Round: {r} | Round error: {round_avg_err:.4f}", flush=True)
                        last_print_time = current_time
                    
                # If drift was detected, break out of the outer loop
                if tau > 0 and len(drift_points) > 0 and drift_points[-1] == tau - 1:
                    break
            
            # If the current round is completed (no early exit due to drift)
            if tau > 0 and (len(drift_points) == 0 or drift_points[-1] != tau - 1):
                # Compute the range of the round that just ended
                current_round_start = tau - T_n_r
                current_round_end = tau
                
                # Update quantile: use data from the current round
                if current_round_end > current_round_start:
                    current_round_X = self.X_predict[current_round_start:current_round_end]
                    current_round_Y = self.Y_predict[current_round_start:current_round_end]
                    current_round_pred = self.model.predict(current_round_X)
                    current_round_scores = self.get_conformal_score(current_round_X, current_round_Y, current_round_pred)
                    
                    # Use argmin to find the optimal quantile
                    sorted_scores = np.sort(current_round_scores)
                    best_q = sorted_scores[0]
                    best_error = float('inf')
                    
                    for q_candidate in sorted_scores:
                        error = abs(np.mean(current_round_scores <= q_candidate) - (1 - alpha))
                        if error < best_error:
                            best_error = error
                            best_q = q_candidate
                    
                    q_n_r = best_q
                    n_data_used = len(current_round_scores)
                else:
                    n_data_used = 0
                
                r = r + 1  # Move to the next round
                
                # Show progress (at end of round)
                current_time = time_module.time()
                elapsed = current_time - start_time
                progress = tau / T * 100
                eta = (elapsed / tau * (T - tau)) if tau > 0 else 0
                print(f"    >>> Quantile updated! {tau}/{T} ({progress:.1f}%) | "
                      f"Stage: {n}, Round: {r} | Used {n_data_used} data points from prev round | "
                      f"New q={q_n_r:.2f}", flush=True)
                last_print_time = current_time
        
        # Compute final statistics
        print(f"    Drift Detection complete! Total time: {time_module.time() - start_time:.1f}s", flush=True)
        n_test = len(lower_bounds)
        Y_test = self.Y_predict[:n_test]
        
        # Compute coverage rate
        coverage_indicators = []
        for i in range(n_test):
            covered = (Y_test[i] >= lower_bounds[i]) and (Y_test[i] <= upper_bounds[i])
            coverage_indicators.append(covered)
        
        coverage = np.mean(coverage_indicators)
        
        # Compute average width
        widths = np.array(upper_bounds) - np.array(lower_bounds)
        avg_width = np.mean(widths)
        
        # Create results DataFrame
        PIs_df = pd.DataFrame({
            'lower': lower_bounds,
            'upper': upper_bounds,
            'prediction': predictions,
            'width': widths,
            'quantile': quantiles,
            'stage': stage_history,
            'round': round_history,
            'covered': coverage_indicators
        })
        
        # Compute total Regret
        total_regret = None
        avg_regret = None
        if compute_regret and regret_values:
            total_regret = np.sum(regret_values)
            avg_regret = np.mean(regret_values)
        
        results = {
            'coverage': coverage,
            'width': avg_width,
            'final_quantile': quantiles[-1] if quantiles else initial_q,
            'n_stages': n,
            'n_rounds': r,
            'drift_points': drift_points,
            'n_drifts': len(drift_points),
            'PIs': PIs_df,
            'quantile_history': quantiles,
            'stage_history': stage_history,
            'round_history': round_history,
            'alpha': alpha,
            'regret_values': regret_values,
            'coverage_gaps': coverage_gaps,
            'total_regret': total_regret,
            'avg_regret': avg_regret
        }
        
        # Verbose output disabled for regret experiments
        # print(f"\n📈 Drift detection conformal inference complete:")
        # print(f"  Coverage: {100*coverage:.1f}%")
        # print(f"  Average width: {avg_width:.3f}")
        # print(f"  Number of drifts detected: {len(drift_points)}")
        # if drift_points:
        #     print(f"  Drift locations: {drift_points}")
        # print(f"  Total stages: {n}")
        # print(f"  Final quantile: {results['final_quantile']:.3f}")
        
        # if compute_regret and total_regret is not None:
        #     print(f"\n📊 Regret metrics:")
        #     print(f"  Total Regret: {total_regret:.4f}")
        #     print(f"  Average Regret: {avg_regret:.4f}")
        #     print(f"  Regret sample count: {n_regret_samples}")
        
        return results
    
    def plot_results(self, results: Dict, save_path: str = 'drift_detection_results.png'):
        """
        Visualize results
        
        Args:
            results: Return value from compute_drift_detection_intervals
            save_path: File save path
        """
        # Create 4 subplots if regret data is available, otherwise 3
        n_subplots = 4 if results.get('regret_values') is not None else 3
        fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 10 + (2 if n_subplots == 4 else 0)))
        
        PIs_df = results['PIs']
        n_points = len(PIs_df)
        x = np.arange(n_points)
        
        # Subplot 1: Prediction intervals and true values
        ax1 = axes[0]
        Y_test = self.Y_predict[:n_points]
        
        # Plot prediction intervals
        ax1.fill_between(x, PIs_df['lower'], PIs_df['upper'], 
                        alpha=0.3, color='blue', label='Prediction Intervals')
        ax1.plot(x, Y_test, 'k-', linewidth=1, alpha=0.8, label='True Values')
        ax1.plot(x, PIs_df['prediction'], 'r--', linewidth=1, alpha=0.6, label='Predictions')
        
        # Mark drift points
        for drift_point in results['drift_points']:
            if drift_point < n_points:
                ax1.axvline(x=drift_point, color='red', linestyle='--', alpha=0.7, linewidth=2)
                ax1.text(drift_point, ax1.get_ylim()[1]*0.95, 'Drift', 
                        rotation=90, va='top', ha='right', color='red', fontweight='bold')
        
        ax1.set_title('Prediction Intervals with Drift Detection', fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Quantile evolution and Stage/Round info
        ax2 = axes[1]
        ax2.plot(x, PIs_df['quantile'], 'b-', linewidth=2, label='Quantile')
        
        # Mark different stages with different colors
        unique_stages = PIs_df['stage'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_stages)))
        
        for i, stage in enumerate(unique_stages):
            stage_mask = PIs_df['stage'] == stage
            stage_x = x[stage_mask]
            if len(stage_x) > 0:
                ax2.axvspan(stage_x[0], stage_x[-1], alpha=0.2, color=colors[i], 
                           label=f'Stage {stage}')
        
        # Mark drift points
        for drift_point in results['drift_points']:
            if drift_point < n_points:
                ax2.axvline(x=drift_point, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        ax2.set_title('Quantile Evolution with Stage Transitions', fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Quantile Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Rolling coverage rate
        ax3 = axes[2]
        window_size = min(100, n_points // 10)
        if window_size > 1:
            rolling_coverage = pd.Series(PIs_df['covered']).rolling(window=window_size).mean()
            ax3.plot(x, rolling_coverage, 'g-', linewidth=2, label=f'Rolling Coverage (window={window_size})')
            ax3.axhline(y=1-results.get('alpha', 0.1), color='red', linestyle='--', 
                       alpha=0.7, label=f'Target: {100*(1-results.get("alpha", 0.1)):.0f}%')
            
            # Mark drift points
            for drift_point in results['drift_points']:
                if drift_point < n_points:
                    ax3.axvline(x=drift_point, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        ax3.set_title('Rolling Coverage Rate', fontweight='bold')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Coverage Rate')
        ax3.set_ylim([0, 1.1])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Regret evolution (if data is available)
        if results.get('regret_values') is not None and len(results['regret_values']) > 0:
            ax4 = axes[3]
            regret_values = results['regret_values'][:n_points]
            
            # Plot per-time-step regret
            ax4.plot(x[:len(regret_values)], regret_values, 'b-', linewidth=1.5, 
                    alpha=0.6, label='Instantaneous Regret')
            
            # Plot cumulative average regret
            cumulative_avg_regret = np.cumsum(regret_values) / (np.arange(len(regret_values)) + 1)
            ax4.plot(x[:len(regret_values)], cumulative_avg_regret, 'r-', linewidth=2,
                    label='Cumulative Average Regret')
            
            # Mark drift points
            for drift_point in results['drift_points']:
                if drift_point < len(regret_values):
                    ax4.axvline(x=drift_point, color='red', linestyle='--', alpha=0.7, linewidth=2)
            
            ax4.set_title('Regret Evolution', fontweight='bold')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Regret (Coverage Gap)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Add average regret text annotation
            ax4.text(0.95, 0.95, f'Avg Regret: {results["avg_regret"]:.4f}',
                    transform=ax4.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\n✅ Visualization saved to: {save_path}")


def test_drift_detection():
    """Test the drift detection algorithm"""
    print("🚀 Testing online conformal inference algorithm with drift detection")
    print("="*60)
    
    # Generate synthetic data with distribution drift
    np.random.seed(42)
    n_train = 500
    n_test = 1000
    
    # Training data: standard setup
    X_train = np.random.randn(n_train, 5)
    Y_train = 2 * X_train[:, 0] + X_train[:, 1] + 0.5 * np.random.randn(n_train)
    
    # Test data: contains two distribution drifts
    X_test = np.random.randn(n_test, 5)
    Y_test = np.zeros(n_test)
    
    # Segment 1: same distribution as training data
    Y_test[:400] = 2 * X_test[:400, 0] + X_test[:400, 1] + 0.5 * np.random.randn(400)
    
    # Segment 2: strong mean drift
    Y_test[400:700] = 2 * X_test[400:700, 0] + X_test[400:700, 1] + 5 + 0.5 * np.random.randn(300)
    
    # Segment 3: strong variance drift
    Y_test[700:] = 2 * X_test[700:, 0] + X_test[700:, 1] + 3.0 * np.random.randn(300)
    
    # Initialize model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Run drift detection algorithm
    drift_detector = DriftDetectionConformal(model, X_train, Y_train, X_test, Y_test)
    results = drift_detector.compute_drift_detection_intervals(alpha=0.1)
    
    # Visualize results
    drift_detector.plot_results(results)
    
    return results


def test_regret_with_multiple_runs():
    """
    Run multiple experiments to estimate the expected value of Regret
    """
    print("🎲 Running multiple experiments to compute expected Regret...")
    print("="*60)
    
    n_experiments = 20  # Run 20 experiments
    all_regrets = []
    all_coverages = []
    all_widths = []
    
    for exp in range(n_experiments):
        print(f"\nExperiment {exp+1}/{n_experiments}:")
        
        # Set random seed to ensure reproducible but distinct results
        np.random.seed(42 + exp)
        
        # Generate data (same setup as test_drift_detection)
        n_train = 500
        n_test = 1000
        
        # Training data
        X_train = np.random.randn(n_train, 5)
        Y_train = 2 * X_train[:, 0] + X_train[:, 1] + 0.5 * np.random.randn(n_train)
        
        # Test data: contains drift
        X_test = np.random.randn(n_test, 5)
        Y_test = np.zeros(n_test)
        
        # Segment 1: same distribution as training data
        Y_test[:400] = 2 * X_test[:400, 0] + X_test[:400, 1] + 0.5 * np.random.randn(400)
        
        # Segment 2: mean drift
        Y_test[400:700] = 2 * X_test[400:700, 0] + X_test[400:700, 1] + 5 + 0.5 * np.random.randn(300)
        
        # Segment 3: variance drift
        Y_test[700:] = 2 * X_test[700:, 0] + X_test[700:, 1] + 3.0 * np.random.randn(300)
        
        # Initialize model
        model = RandomForestRegressor(n_estimators=100, random_state=42 + exp)
        
        # Run algorithm and compute regret
        drift_detector = DriftDetectionConformal(model, X_train, Y_train, X_test, Y_test)
        results = drift_detector.compute_drift_detection_intervals(
            alpha=0.1, 
            compute_regret=True,
            n_regret_samples=500
        )
        
        # Collect statistics
        if results['total_regret'] is not None:
            all_regrets.append(results['total_regret'])
            all_coverages.append(results['coverage'])
            all_widths.append(results['width'])
    
    # Compute and display statistical results
    print("\n" + "="*60)
    print("📊 Multi-experiment statistical results:")
    print(f"  Number of experiments: {n_experiments}")
    print(f"\n  Regret statistics:")
    print(f"    Mean Total Regret: {np.mean(all_regrets):.4f}")
    print(f"    Std Dev: {np.std(all_regrets):.4f}")
    print(f"    Min: {np.min(all_regrets):.4f}")
    print(f"    Max: {np.max(all_regrets):.4f}")
    print(f"\n  Coverage statistics:")
    print(f"    Mean coverage: {100*np.mean(all_coverages):.1f}%")
    print(f"    Std Dev: {100*np.std(all_coverages):.1f}%")
    print(f"\n  Width statistics:")
    print(f"    Mean width: {np.mean(all_widths):.3f}")
    print(f"    Std Dev: {np.std(all_widths):.3f}")
    
    # Only visualize the results of the last experiment
    if exp == n_experiments - 1:
        drift_detector.plot_results(results, save_path='drift_detection_with_regret.png')
    
    return {
        'all_regrets': all_regrets,
        'all_coverages': all_coverages,
        'all_widths': all_widths,
        'mean_regret': np.mean(all_regrets),
        'std_regret': np.std(all_regrets)
    }


def compute_rigorous_regret():
    """
    Rigorous Regret computation experiment:
    1. Pre-sample 500 fixed samples for each time step (to estimate conditional coverage)
    2. Run 1000 independent experiments
    3. Compute cumulative regret for each experiment
    4. Average the 1000 cumulative regrets
    """
    print("🎯 Rigorous Regret computation experiment (two variance drifts)")
    print("="*60)
    print("Data setup:")
    print("  [0, 2000): Normal distribution (σ=0.5)")
    print("  [2000, 3500): Variance drift (σ=2.0, variance increased 16x)")
    print("  [3500, 5000): Larger variance drift (σ=3.5, variance increased 49x)")
    print("="*60)
    
    np.random.seed(42)
    n_train = 500
    T = 5000  # Time horizon
    n_samples_per_t = 500  # Number of samples per time step
    n_experiments = 40  # Number of experiments
    
    # Step 1: Pre-generate fixed sample sets for each time step
    print(f"📦 Step 1: Pre-sampling {n_samples_per_t} samples for each time step t=1,...,{T}...")
    
    # Generate fixed training data
    X_train = np.random.randn(n_train, 5)
    Y_train = 2 * X_train[:, 0] + X_train[:, 1] + 0.5 * np.random.randn(n_train)
    
    # Pre-generate samples for each time step (these samples come from the true distribution)
    fixed_samples_X = []
    fixed_samples_Y = []
    
    for t in range(T):
        # Determine the current distribution based on the time step
        X_samples = np.random.randn(n_samples_per_t, 5)
        Y_samples = np.zeros(n_samples_per_t)
        
        if t < 2000:
            # Segment 1: normal distribution
            Y_samples = 2 * X_samples[:, 0] + X_samples[:, 1] + 0.5 * np.random.randn(n_samples_per_t)
        elif t < 3500:
            # Segment 2: variance drift (moderate)
            Y_samples = 2 * X_samples[:, 0] + X_samples[:, 1] + 2.0 * np.random.randn(n_samples_per_t)
        else:
            # Segment 3: variance drift (larger)
            Y_samples = 2 * X_samples[:, 0] + X_samples[:, 1] + 3.5 * np.random.randn(n_samples_per_t)
        
        fixed_samples_X.append(X_samples)
        fixed_samples_Y.append(Y_samples)
    
    print(f"✓ Done: generated {n_samples_per_t} samples for each of {T} time steps")
    
    # Step 2: Run multiple independent experiments
    print(f"\n🔄 Step 2: Running {n_experiments} independent experiments...")
    
    all_cumulative_regrets = []
    all_regret_trajectories = []  # Store regret trajectory for each experiment
    
    for exp_id in range(n_experiments):
        print(f"Experiment [{exp_id + 1}/{n_experiments}]", end=" ")
        
        # Generate new observation data sequence for this experiment
        np.random.seed(42 + exp_id)
        
        X_test = np.random.randn(T, 5)
        Y_test = np.zeros(T)
        Y_test[:2000] = 2 * X_test[:2000, 0] + X_test[:2000, 1] + 0.5 * np.random.randn(2000)
        Y_test[2000:3500] = 2 * X_test[2000:3500, 0] + X_test[2000:3500, 1] + 2.0 * np.random.randn(1500)
        Y_test[3500:] = 2 * X_test[3500:, 0] + X_test[3500:, 1] + 3.5 * np.random.randn(1500)
        
        # Run drift detection algorithm (without computing regret, only get quantile sequence)
        model = RandomForestRegressor(n_estimators=100, random_state=42 + exp_id)
        drift_detector = DriftDetectionConformal(model, X_train, Y_train, X_test, Y_test)
        results = drift_detector.compute_drift_detection_intervals(alpha=0.1, compute_regret=False)
        
        # Get quantile sequence
        quantile_history = results['quantile_history']
        
        # Step 3: Compute conditional coverage error for each time step using pre-generated samples
        regret_per_t = []
        for t in range(len(quantile_history)):
            q_t = quantile_history[t]
            
            # Use pre-generated fixed samples
            X_samples_t = fixed_samples_X[t]
            Y_samples_t = fixed_samples_Y[t]
            
            # Predict these samples with the model
            Y_pred_samples = model.predict(X_samples_t)
            
            # Compute scores
            scores = np.abs(Y_samples_t - Y_pred_samples)
            
            # Compute conditional miscoverage rate
            miscoverage = np.mean(scores > q_t)
            
            # Coverage gap
            coverage_gap = abs(miscoverage - 0.1)  # alpha = 0.1
            
            regret_per_t.append(coverage_gap)
        
        # Compute cumulative regret (cumulative sum)
        cumulative_regret_trajectory = np.cumsum(regret_per_t)
        cumulative_regret = cumulative_regret_trajectory[-1]
        
        all_cumulative_regrets.append(cumulative_regret)
        all_regret_trajectories.append(cumulative_regret_trajectory)
        
        # Get drift detection info and coverage
        n_drifts = results['n_drifts']
        drift_points = results['drift_points']
        coverage = results['coverage']
        
        # Format output
        if n_drifts == 0:
            drift_info = "None detected"
        else:
            drift_info = f"{n_drifts} times at {drift_points}"
        
        print(f"→ Regret={cumulative_regret:.4f}, Coverage={100*coverage:.1f}%, Drifts: {drift_info}")
    
    # Step 4: Compute average regret
    mean_regret = np.mean(all_cumulative_regrets)
    std_regret = np.std(all_cumulative_regrets)
    
    print(f"\n📊 Rigorous Regret experiment results:")
    print(f"  Number of experiments: {n_experiments}")
    print(f"  Samples per time step: {n_samples_per_t}")
    print(f"  Time horizon: T = {T}")
    print(f"\n  Mean Cumulative Regret: {mean_regret:.4f}")
    print(f"  Std Dev: {std_regret:.4f}")
    print(f"  Min: {np.min(all_cumulative_regrets):.4f}")
    print(f"  Max: {np.max(all_cumulative_regrets):.4f}")
    
    # Step 5: Plot cumulative regret curves
    print(f"\n📈 Plotting Cumulative Regret curves...")
    
    # Convert all trajectories to numpy array
    all_regret_trajectories = np.array(all_regret_trajectories)
    
    # Compute mean trajectory and standard deviation
    mean_trajectory = np.mean(all_regret_trajectories, axis=0)
    std_trajectory = np.std(all_regret_trajectories, axis=0)
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Time axis
    time_steps = np.arange(1, T + 1)
    
    # Plot all experiment trajectories (semi-transparent)
    for i, traj in enumerate(all_regret_trajectories):
        plt.plot(time_steps, traj, 'gray', alpha=0.1, linewidth=0.5)
    
    # Plot mean trajectory
    plt.plot(time_steps, mean_trajectory, 'b-', linewidth=2.5, label='Mean Cumulative Regret')
    
    # Plot confidence interval (±1 std dev)
    plt.fill_between(time_steps, 
                     mean_trajectory - std_trajectory, 
                     mean_trajectory + std_trajectory,
                     alpha=0.3, color='blue', label='±1 Std Dev')
    
    # Mark true drift locations
    plt.axvline(x=2000, color='red', linestyle='--', linewidth=2, label='1st Variance Drift (t=2000)')
    plt.axvline(x=3500, color='darkred', linestyle='--', linewidth=2, label='2nd Variance Drift (t=3500)')
    
    # Set labels and title
    plt.xlabel('Time Step t', fontsize=12)
    plt.ylabel('Cumulative Regret', fontsize=12)
    plt.title(f'Cumulative Regret Evolution (n={n_experiments} experiments, T={T})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('cumulative_regret_curve.png', dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved as: cumulative_regret_curve.png")
    plt.show()
    
    return {
        'mean_regret': mean_regret,
        'std_regret': std_regret,
        'all_regrets': all_cumulative_regrets,
        'mean_trajectory': mean_trajectory,
        'std_trajectory': std_trajectory,
        'n_experiments': n_experiments
    }


if __name__ == "__main__":
    # Run rigorous Regret computation experiment
    print("🚀 Running rigorous Regret computation experiment...")
    regret_results = compute_rigorous_regret()
