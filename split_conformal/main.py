#!/usr/bin/env python3
"""
DriftOCP Full Experiment Main Program

Integrates functionality from run_setting1_only.py and compare_algorithms_parallel.py,
completing 4 Settings x 6 algorithms parallel comparison experiments in a single run.

Memory optimization: Uses pool initializer to set fixed Monte Carlo samples as worker
global variables, avoiding repeated copying of ~240MB data per task.

Usage:
  cd driftocp/split
  python main.py

Output files (saved in current directory):
  Data files:
    experiment_results_all_settings.pkl  — Full raw experiment data (for further analysis)
    experiment_summary.json              — Summary statistics (mean regret per algorithm, etc.)
  Image files:
    comparison_Setting{1,2,3,4}_*.png    — Individual comparison plots per Setting
    split_conformal_combined_2x4.pdf     — Combined 2x4 figure
"""

import numpy as np                                           # Core numerical computing library
import time                                                  # Timing for displaying runtime
import os                                                    # Path operations
import pickle                                                # Serialization (save experiment data to .pkl)
import json                                                  # JSON format (save summary statistics)
from multiprocessing import Pool, cpu_count                  # Multiprocessing for parallel acceleration
from sklearn.ensemble import RandomForestRegressor           # Random forest regression model

# ---- Import from modules in the same directory ----
from data_generating import (                                # Import from data generation module
    generate_data_setting1,                                  # Setting 1 data generator
    generate_data_setting2,                                  # Setting 2 data generator
    generate_data_setting3,                                  # Setting 3 data generator
    generate_data_setting4,                                  # Setting 4 data generator
    compute_true_quantiles,                                  # Theoretical true quantile computation
)
from algorithm import (                                      # Import from algorithm module
    QuantileAdaptiveCI,                                      # ACI baseline algorithm class
    DriftDetectionConformal,                                 # DriftOCP algorithm class
)
from plot import (                                           # Import from plotting module
    plot_comparison_results,                                 # Single Setting comparison plot
    plot_combined_2x4,                                       # Combined 2x4 plot
)


# ==================== Global Shared Variables (for multiprocessing) ====================
# These variables are set once in each worker process via pool initializer,
# avoiding repeated serialization of ~240MB Monte Carlo sample data per task.
_shared_fixed_X = None                                       # Global shared fixed Monte Carlo X samples
_shared_fixed_Y = None                                       # Global shared fixed Monte Carlo Y samples


def _worker_init(fixed_X, fixed_Y):
    """
    Process pool worker initialization function.
    
    Called once when each worker process starts, setting fixed Monte Carlo samples
    as global variables. This way all tasks share the same data without repeatedly
    passing it in each task's arguments.
    
    Args:
        fixed_X: Fixed Monte Carlo sample X features list, length=T
        fixed_Y: Fixed Monte Carlo sample Y labels list, length=T
    """
    global _shared_fixed_X, _shared_fixed_Y                  # Declare global variable modification
    _shared_fixed_X = fixed_X                                # Set global X in worker process
    _shared_fixed_Y = fixed_Y                                # Set global Y in worker process


# ==================== Single Experiment Function (called by multiprocessing) ====================
def run_single_experiment(args):
    """
    Run a single experiment: run one algorithm on one dataset and compute cumulative regret.
    
    This function is called by multiprocessing.Pool, so it must:
      1. Be defined at module top level (pickle-serializable)
      2. Pass all parameters via a single args tuple
      3. Read Monte Carlo samples from global variables _shared_fixed_X/_shared_fixed_Y
    
    Args (unpacked from args tuple):
        exp_id:      Experiment ID (0 to n_experiments-1), used for random seed
        algo_name:   Algorithm identifier ('drift', 'aci_06', 'aci_05', 'aci_001', 'aci_01', 'aci_05_fixed')
        algo_config: Algorithm hyperparameter dict (step_type and fixed_eta for ACI)
        X_train:     Training set features
        Y_train:     Training set labels
        X_test:      Test set features (online data stream)
        Y_test:      Test set labels
        alpha:       Target miscoverage rate
    
    Returns:
        (exp_id, algo_name, cumulative_regret, quantiles, cumulative_regret_trajectory)
    """
    # Unpack args tuple (no longer includes fixed_samples_X/Y, read from globals)
    exp_id, algo_name, algo_config, X_train, Y_train, X_test, Y_test, alpha = args

    T = len(X_test)                                          # Total number of test time steps

    # Get Monte Carlo samples from global variables (set by _worker_init)
    fixed_samples_X = _shared_fixed_X                        # Read global shared X samples
    fixed_samples_Y = _shared_fixed_Y                        # Read global shared Y samples

    try:
        # ---- Step 1: Run algorithm to obtain quantile sequence ----
        if algo_name == 'drift':                             # If DriftOCP algorithm
            model = RandomForestRegressor(                   # Create random forest regressor
                n_estimators=100,                            # 100 trees
                random_state=42 + exp_id                     # Random seed (different per experiment)
            )
            detector = DriftDetectionConformal(              # Instantiate DriftOCP detector
                model, X_train, Y_train, X_test, Y_test     # Pass training/test data
            )
            result = detector.compute_drift_detection_intervals(  # Run drift detection algorithm
                alpha=alpha,                                 # Target miscoverage rate
                compute_regret=False                         # Don't compute regret internally (we compute it externally)
            )
            quantiles = np.array(result['quantile_history']) # Extract quantile history, shape=(T,)
        else:                                                # If ACI algorithm (one of 5 variants)
            model = RandomForestRegressor(                   # Create random forest regressor
                n_estimators=100,                            # 100 trees
                random_state=42 + exp_id                     # Random seed
            )
            aci = QuantileAdaptiveCI(                        # Instantiate ACI algorithm
                model, X_train, Y_train, X_test, Y_test     # Pass data
            )
            quantiles = aci.compute_quantile_adaptive_intervals(  # Run ACI
                alpha=alpha,                                 # Target miscoverage rate
                step_type=algo_config['step_type'],          # Step size type
                fixed_eta=algo_config.get('fixed_eta', 0.01) # Fixed step size value (only for fixed type)
            )

        # ---- Step 2: Compute cumulative regret using fixed Monte Carlo samples ----
        # Regret definition: R_T = sum_{t=1}^{T} |miscoverage_t - alpha|
        # where miscoverage_t = (1/M) sum_{j=1}^{M} 1{s_t^{(j)} > q_t}
        cumulative_regret_trajectory = np.zeros(T)           # Initialize cumulative regret trajectory array
        cumulative_regret = 0.0                              # Initialize cumulative regret accumulator

        for t in range(T):                                   # Iterate over each time step
            q_t = quantiles[t]                               # Current step's quantile threshold
            # Predict Y values for Monte Carlo samples using model
            Y_pred_samples = model.predict(fixed_samples_X[t])  # Y_hat_{t}^{(j)}, shape=(M,)
            # Compute conformal score for each sample
            scores = np.abs(fixed_samples_Y[t] - Y_pred_samples)  # s_t^{(j)} = |Y^{(j)} - Y_hat^{(j)}|
            # Monte Carlo estimate of current step's miscoverage rate
            miscoverage_t = np.mean(scores > q_t)            # (1/M) sum 1{s > q}
            # Current step's regret contribution
            error_t = abs(miscoverage_t - alpha)             # |miscoverage_t - alpha|
            cumulative_regret += error_t                     # Add to total regret
            cumulative_regret_trajectory[t] = cumulative_regret  # Record current cumulative value

        return (exp_id, algo_name, cumulative_regret,        # Return result tuple
                quantiles, cumulative_regret_trajectory)

    except Exception as e:                                   # Catch any exception during execution
        print(f"Experiment {exp_id} algorithm {algo_name} error: {e}")  # Print error message
        import traceback                                     # Import traceback module
        traceback.print_exc()                                # Print full error traceback
        return (exp_id, algo_name, None, None, None)         # Return None to indicate failure


# ==================== Parallel Experiment Execution Framework ====================
def run_parallel_comparison(setting, data_generator, n_experiments=40, n_workers=None):
    """
    Run n_experiments x 6 algorithms comparison experiments in parallel for one Setting.
    
    Memory optimization: Uses Pool(initializer=...) to share ~240MB Monte Carlo samples
    with worker processes via fork, instead of pickling per task.
    
    Args:
        setting:        Setting number (1-4)
        data_generator: Data generation function (generate_data_setting1, etc.)
        n_experiments:  Number of repeated experiments (default 40, for mean and CI)
        n_workers:      Number of parallel processes (None = CPU cores - 1)
    
    Returns:
        (organized_results, elapsed_time)
    """
    if n_workers is None:                                    # If worker count not specified
        n_workers = max(1, cpu_count() - 1)                  # Use CPU cores - 1 (reserve one for system)
    
    print(f"\n🔧 Parallel config: {n_workers} workers (CPU cores: {cpu_count()})")

    # ---- Experiment hyperparameters ----
    T = 10000                                                # Total time steps for online prediction
    n_train = 500                                            # Training set size
    n_features = 5                                           # Feature dimension (X in R^5)
    alpha = 0.1                                              # Target miscoverage rate (10%)
    n_regret_samples = 500                                   # Monte Carlo sample count (for per-step regret estimation)

    # ---- Configuration for 6 algorithms ----
    algorithms = {                                           # Algorithm name -> config mapping dict
        'drift':        {'name': 'Drift Detection',         # DriftOCP algorithm
                         'config': {}},                      # No extra hyperparameters
        'aci_06':       {'name': 'ACI η_t=t^(-0.6)',        # ACI decaying step size 0.6
                         'config': {'step_type': 'decaying_0.6'}},
        'aci_05':       {'name': 'ACI η_t=t^(-0.5)',        # ACI decaying step size 0.5
                         'config': {'step_type': 'decaying_0.5'}},
        'aci_001':      {'name': 'ACI η=0.01',              # ACI fixed step size 0.01
                         'config': {'step_type': 'fixed', 'fixed_eta': 0.01}},
        'aci_01':       {'name': 'ACI η=0.1',               # ACI fixed step size 0.1
                         'config': {'step_type': 'fixed', 'fixed_eta': 0.1}},
        'aci_05_fixed': {'name': 'ACI η=0.5',               # ACI fixed step size 0.5
                         'config': {'step_type': 'fixed', 'fixed_eta': 0.5}},
    }

    # ======== Step 1: Generate fixed Monte Carlo samples ========
    print(f"\n📦 Step 1: Generating fixed sample set (for regret computation)...")
    np.random.seed(42)                                       # Fix seed for reproducible Monte Carlo samples

    fixed_samples_X = []                                     # Store X samples per step
    fixed_samples_Y = []                                     # Store Y samples per step

    for t in range(T):                                       # Iterate over T=10000 time steps
        X_samples = np.random.randn(n_regret_samples, n_features)  # Generate 500x5 random features
        Y_samples = data_generator(X_samples, t=t, n_samples=n_regret_samples)  # Generate Y per current Setting
        fixed_samples_X.append(X_samples)                    # Append to list
        fixed_samples_Y.append(Y_samples)                    # Append to list

    print("✓ Done")                                          # Step 1 complete

    # ======== Step 2: Prepare parallel task list (without fixed_samples) ========
    total_tasks = n_experiments * len(algorithms)             # Total tasks = 40 experiments x 6 algorithms = 240
    print(f"\n🔄 Step 2: Preparing {total_tasks} parallel tasks...")
    tasks = []                                               # Task list

    for exp_id in range(n_experiments):                      # Iterate over 40 repeated experiments
        np.random.seed(1000 + exp_id)                        # Independent random seed per experiment
        X_train = np.random.randn(n_train, n_features)       # Generate training features 500x5
        X_test = np.random.randn(T, n_features)              # Generate test features 10000x5
        Y_train = data_generator(X_train)                    # Generate training labels (mode A)
        Y_test = data_generator(X_test)                      # Generate test labels (mode A)

        for algo_key, algo_info in algorithms.items():       # Iterate over 6 algorithms
            # Note: No longer passing fixed_samples_X/Y, read from global variables instead
            task = (exp_id, algo_key, algo_info['config'],   # Construct task argument tuple
                    X_train, Y_train, X_test, Y_test,        # Training + test data
                    alpha)                                    # Target miscoverage rate
            tasks.append(task)                               # Add to task list

    print(f"✓ Done, {len(tasks)} tasks total")               # Step 2 complete

    # ======== Step 3: Parallel execution with multiprocessing (using initializer for data sharing) ========
    print(f"\n⚡ Step 3: Parallel execution (using {n_workers} workers)...")
    start_time = time.time()                                 # Record start time

    # Use initializer to pass fixed_samples to worker processes
    # Each worker receives data only once at startup, not per task
    with Pool(
        processes=n_workers,                                 # Number of worker processes
        initializer=_worker_init,                            # Called when each worker starts
        initargs=(fixed_samples_X, fixed_samples_Y),         # Arguments passed to initializer
        maxtasksperchild=50                                  # Restart each child after 50 tasks
    ) as pool:
        results = []                                         # Store all task return values
        completed = 0                                        # Completed task counter
        total = len(tasks)                                   # Total number of tasks

        for result in pool.imap_unordered(run_single_experiment, tasks):  # Async unordered results
            results.append(result)                           # Add to results list
            completed += 1                                   # Increment counter

            # Print progress periodically (every 5 tasks or every 5%)
            progress_interval = min(5, max(1, total // 20))  # Compute print interval
            if completed % progress_interval == 0 or completed == total:  # At interval or completion
                progress = completed / total * 100           # Percentage progress
                elapsed = time.time() - start_time           # Elapsed time
                eta = elapsed / completed * (total - completed) if completed > 0 else 0  # Estimated time remaining
                print(f"  Progress: {completed}/{total} ({progress:.1f}%) | "
                      f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", flush=True)

    elapsed_time = time.time() - start_time                  # Compute total elapsed time
    print(f"✓ Done! Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}min)")

    # ======== Step 4: Organize results ========
    print(f"\n📊 Step 4: Organizing experiment results...")

    # Initialize result containers
    organized_results = {                                    # Dict storing regrets and trajectories for 6 algorithms
        'drift_regrets': [],                                 # DriftOCP regret list
        'aci_06_regrets': [],                                # ACI eta_t=t^(-0.6) regret list
        'aci_05_regrets': [],                                # ACI eta_t=t^(-0.5) regret list
        'aci_fixed_001_regrets': [],                         # ACI eta=0.01 regret list
        'aci_fixed_01_regrets': [],                          # ACI eta=0.1 regret list
        'aci_fixed_05_regrets': [],                          # ACI eta=0.5 regret list
        'drift_cumulative_trajectories': [],                 # DriftOCP cumulative regret trajectory list
        'aci_06_cumulative_trajectories': [],                # ACI eta_t=t^(-0.6) cumulative regret trajectory
        'aci_05_cumulative_trajectories': [],                # ACI eta_t=t^(-0.5) cumulative regret trajectory
        'aci_fixed_001_cumulative_trajectories': [],         # ACI eta=0.01 cumulative regret trajectory
        'aci_fixed_01_cumulative_trajectories': [],          # ACI eta=0.1 cumulative regret trajectory
        'aci_fixed_05_cumulative_trajectories': [],          # ACI eta=0.5 cumulative regret trajectory
    }

    # Organize raw results by (exp_id, algo_name)
    results_dict = {}                                        # {exp_id: {algo_name: {regret, quantiles, ...}}}
    for exp_id, algo_name, regret, quantiles, cumulative_traj in results:  # Iterate over all results
        if regret is None:                                   # Skip failed tasks
            continue
        if exp_id not in results_dict:                       # If this experiment ID appears first time
            results_dict[exp_id] = {}                        # Create empty dict
        results_dict[exp_id][algo_name] = {                  # Store result for this experiment-algorithm
            'regret': regret,                                # Final cumulative regret
            'quantiles': quantiles,                          # Quantile sequence
            'cumulative_trajectory': cumulative_traj          # Cumulative regret trajectory
        }

    # Mapping from internal algorithm key -> organized_results key prefix
    algo_key_map = {                                         # Mapping table
        'drift':        'drift',                             # drift -> drift
        'aci_06':       'aci_06',                            # aci_06 -> aci_06
        'aci_05':       'aci_05',                            # aci_05 -> aci_05
        'aci_001':      'aci_fixed_001',                     # aci_001 -> aci_fixed_001
        'aci_01':       'aci_fixed_01',                      # aci_01 -> aci_fixed_01
        'aci_05_fixed': 'aci_fixed_05',                      # aci_05_fixed -> aci_fixed_05
    }

    # Extract data sorted by experiment ID
    for exp_id in sorted(results_dict.keys()):               # Iterate over sorted experiment IDs
        exp_results = results_dict[exp_id]                   # Get all algorithm results for this experiment

        for algo_internal, algo_prefix in algo_key_map.items():  # Iterate over 6 algorithms
            if algo_internal in exp_results:                 # If this algorithm has results
                organized_results[f'{algo_prefix}_regrets'].append(               # Append regret
                    exp_results[algo_internal]['regret'])
                organized_results[f'{algo_prefix}_cumulative_trajectories'].append(  # Append trajectory
                    exp_results[algo_internal]['cumulative_trajectory'])

    # Save quantile trajectory from experiment #0 (for Quantile Evolution plot)
    if 0 in results_dict:                                    # If experiment #0 exists
        for algo_internal, algo_prefix in algo_key_map.items():  # Iterate over 6 algorithms
            if algo_internal in results_dict[0]:             # If experiment #0 has this algorithm's result
                organized_results[f'{algo_prefix}_quantile'] = \
                    list(results_dict[0][algo_internal]['quantiles'])  # Convert to list and save

    # Compute theoretical true quantiles (reference baseline for plotting)
    organized_results['true_quantiles'] = compute_true_quantiles(T, alpha, setting)

    print("✓ Done")                                          # Step 4 complete

    # ---- Print statistics report ----
    print(f"\n{'='*70}")                                     # Separator
    print(f"📊 Experiment Statistics ({len(organized_results['drift_regrets'])} repeated experiments):")
    print('='*70)

    algo_display = [                                         # (key, display name) list
        ('drift_regrets',           'Drift Detection'),      # DriftOCP
        ('aci_06_regrets',          'ACI η_t = t^(-0.6)'),   # ACI decaying 0.6
        ('aci_05_regrets',          'ACI η_t = t^(-0.5)'),   # ACI decaying 0.5
        ('aci_fixed_001_regrets',   'ACI η = 0.01'),         # ACI fixed 0.01
        ('aci_fixed_01_regrets',    'ACI η = 0.1'),          # ACI fixed 0.1
        ('aci_fixed_05_regrets',    'ACI η = 0.5'),          # ACI fixed 0.5
    ]

    best_regret = float('inf')                               # Track best (lowest) regret
    best_algo = None                                         # Track best algorithm name

    for key, name in algo_display:                           # Iterate over each algorithm
        if key in organized_results and len(organized_results[key]) > 0:  # Has data
            regrets = np.array(organized_results[key])       # Convert to numpy array
            mean_r = np.mean(regrets)                        # Mean
            std_r = np.std(regrets)                          # Standard deviation
            min_r = np.min(regrets)                          # Minimum
            max_r = np.max(regrets)                          # Maximum

            print(f"\n  [{name}]")                           # Print algorithm name
            print(f"    Mean Regret: {mean_r:.2f} ± {std_r:.2f}")
            print(f"    Range: [{min_r:.1f}, {max_r:.1f}]")

            if mean_r < best_regret:                         # If current algorithm is better
                best_regret = mean_r                         # Update best regret
                best_algo = name                             # Update best algorithm name

    if best_algo:                                            # If a best algorithm was found
        print(f"\n  💡 Best algorithm: {best_algo} (Regret = {best_regret:.2f})")

    print('='*70)                                            # Separator

    return organized_results, elapsed_time                   # Return results and elapsed time


# ==================== Main Function ====================
def main():
    """
    Main entry point: sequentially run experiments for Settings 1-4, save data and plot.
    """
    print("=" * 80)                                          # Print top separator
    print("🚀 DriftOCP Full Experiment: 6 Algorithms x 4 Data Settings Comparison")
    print("=" * 80)                                          # Separator

    # Output directory = directory of this script (i.e., driftocp/split/)
    output_dir = os.path.dirname(os.path.abspath(__file__))  # Get absolute path directory of current file
    os.makedirs(output_dir, exist_ok=True)                   # Ensure output directory exists

    # Configuration for 4 Settings: {id: (name, data_generator)}
    settings = {                                             # Settings config dict
        1: ('Setting 1: Piecewise Variance Shift', generate_data_setting1),  # Piecewise variance jump
        2: ('Setting 2: Linear Bias Drift', generate_data_setting2),          # Linear mean drift
        3: ('Setting 3: Smooth Variance Growth', generate_data_setting3),     # Smooth variance growth
        4: ('Setting 4: Stationary Distribution', generate_data_setting4),    # Stationary distribution
    }

    # Experiment parameters
    n_experiments = 40                                       # 40 repeated experiments per Setting
    n_workers = None                                         # Parallel workers (None = auto-detect CPU cores - 1)

    # Detailed Setting names for chart titles
    setting_names_display = {                                # Detailed name mapping
        1: "Setting 1: Piecewise Variance Shift (0.5→2.0→3.5)",
        2: "Setting 2: Linear Bias Drift (mu_t=0.001t)",
        3: "Setting 3: Smooth Variance Growth (sigma^2=1+40t/5000)",
        4: "Setting 4: Stationary Distribution"
    }

    # Print experiment configuration summary
    print(f"\nExperiment configuration:")                     # Config title
    print(f"  • Horizon: T = 10000")                         # Time steps
    print(f"  • Repeated experiments: {n_experiments}")       # Repetition count
    print(f"  • Parallel workers: {'auto-detect' if n_workers is None else n_workers}")
    print(f"  • Settings: 1, 2, 3, 4 (all)")                # Settings list
    print(f"  • Output directory: {output_dir}")             # Output path
    print("=" * 80)                                          # Separator

    all_results = {}                                         # Store results for all Settings
    total_times = {}                                         # Store runtime for all Settings

    overall_start = time.time()                              # Record overall start time

    # ======== Run experiments for each Setting ========
    for setting_id, (setting_name, data_gen) in settings.items():  # Iterate over 4 Settings
        print(f"\n{'='*80}")                                 # Separator
        print(f"Starting Setting {setting_id}")              # Print current Setting number
        print('='*80)                                        # Separator
        print(f"\n🎯 {setting_name}")                        # Print Setting name

        results, elapsed = run_parallel_comparison(          # Run parallel comparison experiment
            setting_id, data_gen,                            # Setting number + data generator
            n_experiments=n_experiments,                      # Number of repeated experiments
            n_workers=n_workers                              # Number of parallel workers
        )

        all_results[setting_id] = results                    # Save this Setting's results
        total_times[setting_id] = elapsed                    # Save this Setting's runtime

        # Plot individual comparison chart immediately after each Setting completes
        print(f"\n📈 Plotting Setting {setting_id} comparison chart...")
        plot_comparison_results(                             # Call plotting function
            results, setting_id,                             # Pass results and Setting number
            setting_names_display.get(setting_id, f"Setting {setting_id}"),
            save_dir=output_dir                              # Save directory
        )

    overall_elapsed = time.time() - overall_start            # Compute overall total time

    print("\n" + "="*80)                                     # Separator
    print("✅ All experiments completed!")
    print(f"⏱️  Total runtime: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}min)")
    print("="*80)                                            # Separator

    # ======== Save experiment data ========
    print("\n💾 Saving experiment data...")

    # Save complete experiment data (pickle format)
    pkl_path = os.path.join(output_dir, 'experiment_results_all_settings.pkl')
    with open(pkl_path, 'wb') as f:                          # Open in binary write mode
        pickle.dump(all_results, f)                          # Serialize all results to file
    print(f"✓ Complete data saved: {pkl_path}")

    # Save summary statistics (JSON format, human-readable)
    summary = {}                                             # Initialize summary dict
    for setting, results in all_results.items():             # Iterate over each Setting
        summary[f'Setting_{setting}'] = {                    # Statistics per Setting
            'Drift_Detection': {
                'mean_regret': float(np.mean(results['drift_regrets'])),
                'std_regret': float(np.std(results['drift_regrets'])),
            },
            'ACI_eta_t_0.6': {
                'mean_regret': float(np.mean(results['aci_06_regrets'])),
                'std_regret': float(np.std(results['aci_06_regrets'])),
            },
            'ACI_eta_t_0.5': {
                'mean_regret': float(np.mean(results['aci_05_regrets'])),
                'std_regret': float(np.std(results['aci_05_regrets'])),
            },
            'ACI_eta_fixed_0.01': {
                'mean_regret': float(np.mean(results['aci_fixed_001_regrets'])),
                'std_regret': float(np.std(results['aci_fixed_001_regrets'])),
            },
            'ACI_eta_fixed_0.1': {
                'mean_regret': float(np.mean(results['aci_fixed_01_regrets'])),
                'std_regret': float(np.std(results['aci_fixed_01_regrets'])),
            },
            'ACI_eta_fixed_0.5': {
                'mean_regret': float(np.mean(results['aci_fixed_05_regrets'])),
                'std_regret': float(np.std(results['aci_fixed_05_regrets'])),
            },
            'runtime_seconds': total_times[setting]
        }

    summary['overall_runtime_seconds'] = overall_elapsed     # Overall runtime (seconds)
    summary['overall_runtime_minutes'] = overall_elapsed / 60  # Overall runtime (minutes)

    json_path = os.path.join(output_dir, 'experiment_summary.json')
    with open(json_path, 'w') as f:                          # Open in text write mode
        json.dump(summary, f, indent=2)                      # Write JSON with indentation
    print(f"✓ Summary statistics saved: {json_path}")

    # ======== Plot 2x4 combined figure ========
    print("\n📈 Plotting 2x4 combined figure...")
    plot_combined_2x4(all_results, save_dir=output_dir)      # Call combined plot function

    # ======== Print final summary ========
    print("\n" + "="*80)                                     # Separator
    print("🎉 Experiment complete!")                         # Completion marker
    print("="*80)                                            # Separator
    print(f"\nGenerated files (in {output_dir}/):")          # File list title
    print("  Data files:")                                   # Data files subtitle
    print("    • experiment_results_all_settings.pkl (complete data)")
    print("    • experiment_summary.json (summary statistics)")
    print("  Image files:")                                  # Image files subtitle
    for setting_id in [1, 2, 3, 4]:                          # Iterate over 4 Settings
        print(f"    • comparison_Setting{setting_id}_*.png")
    print("    • split_conformal_combined_2x4.pdf (2x4 combined figure)")
    print("="*80)                                            # Final separator


# ==================== Script Entry Point ====================
if __name__ == "__main__":                                   # When running this script directly
    main()                                                   # Call main function
